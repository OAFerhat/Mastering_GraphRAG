from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging
import argparse
from tabulate import tabulate
import openai
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from textblob import TextBlob
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')
    nltk.download('wordnet')
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for database configuration
def get_remote_config():
    """Get remote database configuration from environment variables"""
    config = {
        "NEO4J_URI": os.getenv("REMOTE_NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("REMOTE_NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("REMOTE_NEO4J_PASSWORD")
    }
    # Debug logging
    logger.info("Remote config values:")
    for key, value in config.items():
        logger.info(f"{key}: {'[SET]' if value else '[MISSING]'}")
    return config

def get_local_config():
    """Get local database configuration from environment variables"""
    return {
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD")
    }

# Current database configuration (defaults to local)
CURRENT_DB_CONFIG = get_local_config()

def choose_database():
    """Function to choose between local and remote database"""
    # Reload environment variables
    load_dotenv(override=True)
    
    while True:
        print("\n=== Database Selection ===")
        print("1. Local Database (from .env)")
        print("2. Remote Database")
        print("Q. Return to Main Menu")
        
        choice = input("\nEnter your choice (1-2 or Q): ").strip().upper()
        
        if choice == 'Q':
            return
            
        if choice == '1':
            global CURRENT_DB_CONFIG
            CURRENT_DB_CONFIG = get_local_config()
            if not all(CURRENT_DB_CONFIG.values()):
                logger.error("❌ Missing local database configuration in .env file")
                return
            logger.info("✅ Switched to Local Database")
            return
            
        elif choice == '2':
            remote_config = get_remote_config()
            if not all(remote_config.values()):
                logger.error("❌ Missing remote database configuration in .env file")
                return
            CURRENT_DB_CONFIG.update(remote_config)
            logger.info("✅ Switched to Remote Database")
            return
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or Q.")

class Neo4jConnection:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get Neo4j credentials from environment variables
        self.uri = CURRENT_DB_CONFIG["NEO4J_URI"]
        self.username = CURRENT_DB_CONFIG["NEO4J_USERNAME"]
        self.password = CURRENT_DB_CONFIG["NEO4J_PASSWORD"]
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            logger.info("✅ Successfully connected to Neo4j database!")
            return True
        except ServiceUnavailable:
            logger.error("❌ Failed to connect to Neo4j database!")
            logger.error("Please check if:")
            logger.error("- The database is running")
            logger.error("- The credentials in .env are correct")
            logger.error(f"- The URI ({self.uri}) is correct")
            return False
        except Exception as e:
            logger.error(f"❌ An error occurred: {str(e)}")
            return False

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Database connection closed.")

    def execute_query(self, query, parameters=None):
        """Execute a Cypher query and return the results"""
        if not self.driver:
            raise Exception("Database connection not established. Call connect() first.")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

def get_node_type_summary(session):
    """Get summary of all node types, their counts, and embedding status"""
    # First get all labels
    labels_query = "CALL db.labels() YIELD label RETURN label"
    labels = [record["label"] for record in session.run(labels_query)]
    
    nodes_info = []
    for label in labels:
        # For each label, get the summary information
        query = f"""
            MATCH (n:`{label}`)
            WITH COUNT(n) as totalCount,
                 COUNT(n.embedding) as withEmbedding,
                 CASE 
                     WHEN COUNT(n.embedding) > 0 
                     THEN apoc.agg.first(n.embedding) 
                     ELSE null 
                 END as sample
            WHERE totalCount > 0
            RETURN 
                totalCount,
                withEmbedding,
                CASE 
                    WHEN sample IS NOT NULL 
                    THEN size(sample) 
                    ELSE 0 
                END as embedding_dim
        """
        
        try:
            result = session.run(query)
            record = result.single()
            if record:
                nodes_info.append({
                    "label": label,
                    "total_count": record["totalCount"],
                    "with_embedding": record["withEmbedding"],
                    "embedding_dim": record["embedding_dim"]
                })
        except Exception as e:
            logger.error(f"Error getting summary for label {label}: {str(e)}")
            continue
    
    return nodes_info

def check_vector_indexes(session):
    """Get all vector indexes in the database"""
    query = """
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE type = 'VECTOR'
        RETURN name, labelsOrTypes[0] as label
    """
    try:
        result = session.run(query)
        return {record["label"]: record["name"] for record in result}
    except Exception as e:
        logger.error(f"Error checking vector indexes: {str(e)}")
        return {}

def create_vector_index(session, label):
    """Create a vector index for a given label if it doesn't exist"""
    try:
        # Check if any nodes of this type have embeddings
        check_query = f"""
            MATCH (n:`{label}`)
            WHERE n.embedding IS NOT NULL
            RETURN COUNT(n) as count
        """
        result = session.run(check_query)
        count = result.single()["count"]
        
        if count == 0:
            logger.warning(f"No nodes of type {label} have embeddings. Skipping index creation.")
            return False
            
        # Create the index
        index_name = f"{label.lower()}_embedding_idx"
        create_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:`{label}`)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: 1536,
                `vector.similarity_function`: 'cosine'
            }}}}
        """
        session.run(create_query)
        logger.info(f"✅ Created vector index for {label}")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating vector index for {label}: {str(e)}")
        return False

def drop_vector_index(session, label):
    """Drop the vector index for a given label if it exists"""
    try:
        # Get the index name for this label
        query = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR'
            AND labelsOrTypes[0] = $label
            RETURN name
        """
        result = session.run(query, label=label)
        record = result.single()
        
        if not record:
            logger.warning(f"No vector index found for {label}")
            return False
            
        # Drop the index
        index_name = record["name"]
        drop_query = f"DROP INDEX {index_name}"
        session.run(drop_query)
        logger.info(f"✅ Dropped vector index for {label}")
        return True
    except Exception as e:
        logger.error(f"❌ Error dropping vector index for {label}: {str(e)}")
        return False

def create_vector_indexes(uri, user, password):
    """Create vector indexes for node types with embeddings"""
    logger.info("Starting vector index creation process...")
    
    db = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with db.session() as session:
            # Get summary of all node types
            nodes_info = get_node_type_summary(session)
            
            if not nodes_info:
                logger.info("No nodes found in the database.")
                return
                
            # Get existing vector indexes
            existing_indexes = check_vector_indexes(session)
            
            # Prepare table data
            table_data = []
            for idx, info in enumerate(nodes_info, 1):
                label = info["label"]
                has_index = label in existing_indexes
                table_data.append([
                    idx,
                    label,
                    info["total_count"],
                    f"{info['with_embedding']}/{info['total_count']}",
                    "✅" if has_index else "❌",
                    info["embedding_dim"] if info["with_embedding"] > 0 else "N/A"
                ])
            
            # Display table
            headers = ["#", "Node Type", "Total Nodes", "With Embeddings", "Indexed", "Embedding Dim"]
            print("\nNode Types Summary:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Get user input for which node types to index
            while True:
                choice = input("\nEnter the number of the node type to index (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                    
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(nodes_info):
                        label = nodes_info[idx]["label"]
                        if label in existing_indexes:
                            logger.info(f"Vector index already exists for {label}")
                        else:
                            create_vector_index(session, label)
                    else:
                        logger.warning("Invalid choice. Please try again.")
                except ValueError:
                    logger.warning("Please enter a valid number or 'q' to quit.")
                    
    except Exception as e:
        logger.error(f"❌ Error during vector index creation: {str(e)}")
    finally:
        db.close()

def manage_vector_indexes(uri, user, password):
    """Manage vector indexes for node types with embeddings"""
    logger.info("Starting vector index management...")
    
    db = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with db.session() as session:
            while True:
                # Get summary of all node types
                nodes_info = get_node_type_summary(session)
                
                if not nodes_info:
                    logger.info("No nodes found in the database.")
                    return
                    
                # Get existing vector indexes
                existing_indexes = check_vector_indexes(session)
                
                # Prepare table data
                table_data = []
                for idx, info in enumerate(nodes_info, 1):
                    label = info["label"]
                    has_index = label in existing_indexes
                    table_data.append([
                        idx,
                        label,
                        info["total_count"],
                        f"{info['with_embedding']}/{info['total_count']}",
                        "✅" if has_index else "❌",
                        info["embedding_dim"] if info["with_embedding"] > 0 else "N/A"
                    ])
                
                # Display table
                headers = ["#", "Node Type", "Total Nodes", "With Embeddings", "Indexed", "Embedding Dim"]
                print("\nNode Types Summary:")
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
                
                # Show menu
                print("\nChoose an action:")
                print("[C] Create index")
                print("[D] Drop index")
                print("[Q] Quit")
                
                action = input("Your choice: ").upper()
                
                if action == 'Q':
                    break
                elif action in ['C', 'D']:
                    try:
                        idx = int(input("\nEnter the number of the node type to process: ")) - 1
                        if 0 <= idx < len(nodes_info):
                            label = nodes_info[idx]["label"]
                            has_index = label in existing_indexes
                            
                            if action == 'C':
                                if has_index:
                                    logger.info(f"Vector index already exists for {label}")
                                else:
                                    create_vector_index(session, label)
                            else:  # action == 'D'
                                if not has_index:
                                    logger.info(f"No vector index exists for {label}")
                                else:
                                    drop_vector_index(session, label)
                        else:
                            logger.warning("Invalid choice. Please try again.")
                    except ValueError:
                        logger.warning("Please enter a valid number.")
                else:
                    logger.warning("Invalid choice. Please enter 'C', 'D', or 'Q'.")
                    
    except Exception as e:
        logger.error(f"❌ Error during vector index management: {str(e)}")
    finally:
        db.close()

def get_nodes_with_embeddings(session):
    """Get all nodes that have embeddings and their counts."""
    query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        WITH labels(n)[0] as label, n
        WITH label, 
             COUNT(n) as nodeCount,
             n.embedding as sample
        RETURN label, 
               nodeCount as count,
               size(sample) as embedding_dim
        ORDER BY label
    """
    
    try:
        result = session.run(query)
        records = list(result)
        if not records:
            logger.info("No nodes with embeddings found in the database.")
            return []
        
        # Format the results into a list of dictionaries
        nodes_info = [
            {
                "label": record["label"],
                "count": record["count"],
                "embedding_dim": record["embedding_dim"]
            }
            for record in records
        ]
        
        return nodes_info
    except Exception as e:
        logger.error(f"Error getting nodes with embeddings: {str(e)}")
        return []

def check_vector_index(tx, label):
    """Check if a vector index exists for a given label"""
    result = tx.run("""
        SHOW INDEXES
        YIELD name, type, labelsOrTypes, properties
        WHERE type = 'VECTOR'
          AND $label IN labelsOrTypes 
          AND 'embedding' IN properties
        RETURN count(*) > 0 as has_index
    """, label=label).single()
    return result and result["has_index"]

def get_embedding(text, api_key):
    """Get embedding from OpenAI API"""
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"❌ Error getting embedding from OpenAI: {str(e)}")
        raise

def wrap_text(text, max_width):
    """Wrap text to two lines with max width"""
    if len(text) <= max_width:
        return text
    
    # Find a good breaking point near the middle
    mid_point = len(text) // 2
    space_before = text.rfind(' ', 0, mid_point)
    space_after = text.find(' ', mid_point)
    
    break_point = space_before if space_before != -1 else space_after
    if break_point == -1:
        # If no good breaking point, just truncate
        return text[:max_width-3] + "..."
    
    first_line = text[:break_point]
    second_line = text[break_point+1:]
    
    # Truncate second line if too long
    if len(second_line) > max_width:
        second_line = second_line[:max_width-3] + "..."
    
    return f"{first_line}\n{second_line}"

def format_table_data(results, max_question=60, max_answer=100):
    """Format results into table data with specified maximum widths"""
    table_data = []
    headers = ["Similar Question", "Answer", "Score"]
    
    for record in results:
        table_data.append([
            wrap_text(record["question"], max_question),
            wrap_text(record["answer"], max_answer),
            f"{record['score']:.4f}"
        ])
    
    return headers, table_data

def get_course_data(chunk):
    """Extract course, module, and lesson data from chunk metadata"""
    data = {}
    path = chunk.metadata['source'].split(os.path.sep)
    
    # Extract hierarchy information from path
    data['course'] = path[-6]
    data['module'] = path[-4]
    data['lesson'] = path[-2]
    data['url'] = f"https://graphacademy.neo4j.com/courses/{data['course']}/{data['module']}/{data['lesson']}"
    data['text'] = chunk.page_content
    data['topics'] = TextBlob(data['text']).noun_phrases
    
    return data

def create_document_hierarchy(tx, data):
    """Create the document hierarchy in Neo4j"""
    return tx.run("""
        MERGE (c:Course {name: $course})
        MERGE (c)-[:HAS_MODULE]->(m:Module {name: $module})
        MERGE (m)-[:HAS_LESSON]->(l:Lesson {name: $lesson, url: $url})
        MERGE (l)-[:CONTAINS]->(p:Paragraph {text: $text})
        WITH p
        CALL db.create.setNodeVectorProperty(p, "embedding", $embedding)
        
        FOREACH (topic in $topics |
            MERGE (t:Topic {name: topic})
            MERGE (p)-[:MENTIONS]->(t)
        )
        """, 
        data
    )

def load_structured_documents():
    """Load documents and create a hierarchical structure in Neo4j"""
    logger.info("Starting structured document loading process...")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ Missing OpenAI API key!")
        return
    
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize Neo4j connection
    uri = CURRENT_DB_CONFIG["NEO4J_URI"]
    user = CURRENT_DB_CONFIG["NEO4J_USERNAME"]
    password = CURRENT_DB_CONFIG["NEO4J_PASSWORD"]
    
    if not all([uri, user, password]):
        logger.error("❌ Missing Neo4j credentials!")
        return
        
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        while True:
            # Get directory path from user
            dir_path = input("\nEnter the absolute path to your documents directory (or 'Q' to quit): ")
            if dir_path.upper() == 'Q':
                logger.info("Exiting document loading...")
                break
                
            if not os.path.exists(dir_path):
                logger.error(f"❌ Directory not found: {dir_path}")
                continue
                
            try:
                # Initialize document loader
                logger.info(f"Loading documents from: {dir_path}")
                loader = DirectoryLoader(dir_path, glob="**/lesson.adoc", loader_cls=TextLoader)
                docs = loader.load()
                logger.info(f"✅ Loaded {len(docs)} documents")
                
                # Initialize text splitter
                text_splitter = CharacterTextSplitter(
                    separator="\n\n",
                    chunk_size=1500,
                    chunk_overlap=200,
                )
                
                # Split documents into chunks
                logger.info("Splitting documents into chunks...")
                chunks = text_splitter.split_documents(docs)
                logger.info(f"✅ Created {len(chunks)} chunks")
                
                # Process chunks and create graph structure
                logger.info("Creating document hierarchy in Neo4j...")
                with driver.session(database="neo4j") as session:
                    for chunk in chunks:
                        # Get document structure data
                        data = get_course_data(chunk)
                        
                        # Generate embedding
                        try:
                            response = client.embeddings.create(
                                model="text-embedding-ada-002",
                                input=data['text']
                            )
                            data['embedding'] = response.data[0].embedding
                        except Exception as e:
                            logger.error(f"❌ Error generating embedding: {str(e)}")
                            continue
                        
                        # Create nodes and relationships
                        session.execute_write(create_document_hierarchy, data)
                        
                logger.info("✅ Successfully created document hierarchy with embeddings")
                
                # Ask if user wants to continue
                while True:
                    choice = input("\nWould you like to:\n[L] Load another directory\n[Q] Quit\nYour choice: ")
                    if choice.upper() in ['L', 'Q']:
                        if choice.upper() == 'Q':
                            logger.info("Exiting document loading...")
                            return
                        break
                    print("Invalid choice. Please enter 'L' to load another directory or 'Q' to quit.")
                    
            except Exception as e:
                logger.error(f"❌ An error occurred: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        raise
    finally:
        driver.close()

def search_paragraphs():
    """Search through paragraphs and get their lesson context"""
    logger.info("Starting paragraph search...")
    
    # Initialize Neo4j connection
    uri = CURRENT_DB_CONFIG["NEO4J_URI"]
    user = CURRENT_DB_CONFIG["NEO4J_USERNAME"]
    password = CURRENT_DB_CONFIG["NEO4J_PASSWORD"]
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not all([uri, user, password, api_key]):
        logger.error("❌ Missing required environment variables!")
        return
        
    db = GraphDatabase.driver(uri, auth=(user, password))
    client = openai.OpenAI(api_key=api_key)
    
    try:
        with db.session() as session:
            # Check existing vector indexes
            vector_indexes = check_vector_indexes(session)
            logger.info(f"Found vector indexes: {vector_indexes}")
            
            # Find the index name for Paragraph nodes
            paragraph_index = None
            for label, index_name in vector_indexes.items():
                if label.lower() == 'paragraph':
                    paragraph_index = index_name
                    break
            
            if not paragraph_index:
                logger.error("❌ No vector index found for Paragraph nodes! Please create one using the 'Manage Indexes' option.")
                return
            
            while True:
                # Get search query from user
                query = input("\nEnter your search query (or 'q' to quit): ")
                if query.lower() == 'q':
                    logger.info("Exiting search...")
                    break
                
                # Generate embedding for the query
                try:
                    response = client.embeddings.create(
                        model="text-embedding-ada-002",
                        input=query
                    )
                    embedding = response.data[0].embedding
                except Exception as e:
                    logger.error(f"❌ Error generating embedding: {str(e)}")
                    continue
                
                # Search for similar paragraphs with lesson context
                search_query = f"""
                    CALL db.index.vector.queryNodes('{paragraph_index}', 5, $embedding)
                    YIELD node, score
                    MATCH (l:Lesson)-[:CONTAINS]->(node)
                    RETURN 
                        l.name as lesson_name,
                        l.url as lesson_url,
                        node.text as paragraph_text,
                        score
                    ORDER BY score DESC
                """
                
                try:
                    result = session.run(search_query, embedding=embedding)
                    results = list(result)
                    
                    if results:
                        # Format results into table
                        table_data = []
                        headers = ["Lesson", "URL", "Relevance", "Content"]
                        
                        for r in results:
                            table_data.append([
                                r['lesson_name'],
                                r['lesson_url'],
                                f"{r['score']:.4f}",
                                wrap_text(r['paragraph_text'], 80)
                            ])
                        
                        print("\nSearch Results:")
                        print(tabulate(
                            table_data,
                            headers=headers,
                            tablefmt="grid",
                            maxcolwidths=[30, 30, 10, 110]
                        ))
                        logger.info("✅ Search completed successfully!")
                    else:
                        logger.info("No matching paragraphs found.")
                        
                except Exception as e:
                    logger.error(f"❌ Error during search: {str(e)}")
                    continue
                
                # Ask if user wants to continue
                while True:
                    choice = input("\nWould you like to:\n[S] Search again\n[Q] Quit\nYour choice: ").upper()
                    if choice in ['S', 'Q']:
                        if choice == 'Q':
                            logger.info("Exiting search...")
                            return
                        break
                    print("Invalid choice. Please enter 'S' to search again or 'Q' to quit.")
                    
    except Exception as e:
        logger.error(f"❌ Error during paragraph search: {str(e)}")
    finally:
        db.close()

def get_database_statistics(session):
    """Get node and relationship statistics from database"""
    # Get all distinct label combinations and their counts
    node_stats = session.run("""
        MATCH (n)
        WITH labels(n) as labels
        WITH CASE 
            WHEN size(labels) > 0 THEN labels 
            ELSE ['(no label)'] 
        END as labelSet
        RETURN labelSet, count(*) as count
        ORDER BY labelSet
    """)
    
    # Get regular relationship statistics
    rel_types = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    
    rel_stats = []
    for record in rel_types:
        rel_type = record['relationshipType']
        # For ACTED_IN and DIRECTED, exclude cases where both relationships exist
        if rel_type in ['ACTED_IN', 'DIRECTED']:
            result = session.run("""
                MATCH (a)-[r:`""" + rel_type + """`]->(b:Movie)
                WHERE NOT (
                    r.type = 'ACTED_IN' AND exists((a)-[:DIRECTED]->(b))
                    OR
                    r.type = 'DIRECTED' AND exists((a)-[:ACTED_IN]->(b))
                )
                RETURN 
                    $rel_type as relationshipType,
                    labels(a) as fromLabels,
                    labels(b) as toLabels,
                    count(r) as count
            """, rel_type=rel_type)
        else:
            result = session.run("""
                MATCH (a)-[r:`""" + rel_type + """`]->(b)
                RETURN 
                    $rel_type as relationshipType,
                    labels(a) as fromLabels,
                    labels(b) as toLabels,
                    count(r) as count
            """, rel_type=rel_type)
        stats = result.single()
        if stats and stats['count'] > 0:
            rel_stats.append(stats)
    
    # Add special case for dual ACTED_IN/DIRECTED relationships
    dual_rel_result = session.run("""
        MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
        MATCH (p)-[:DIRECTED]->(m)
        RETURN 
            'ACTED_IN+DIRECTED' as relationshipType,
            labels(p) as fromLabels,
            labels(m) as toLabels,
            count(*) as count
    """)
    dual_stats = dual_rel_result.single()
    if dual_stats and dual_stats['count'] > 0:
        rel_stats.append(dual_stats)
    
    return list(node_stats), rel_stats

def display_migration_menu(node_stats, rel_stats):
    """Display formatted table of nodes and relationships with counts"""
    from tabulate import tabulate
    
    # Combine nodes and relationships in a single list
    all_rows = []
    
    # Add nodes first
    for i, stat in enumerate(node_stats, 1):
        label_str = ':'.join(stat['labelSet'])
        all_rows.append([
            i, 
            "Node",
            label_str,
            "-",
            "-",
            stat['count']
        ])
    
    # Add relationships
    start_idx = len(node_stats) + 1
    for i, stat in enumerate(rel_stats, start_idx):
        from_labels = ':'.join(stat['fromLabels'])
        to_labels = ':'.join(stat['toLabels'])
        rel_type = stat['relationshipType']
        # Add special formatting for dual relationships
        if rel_type == 'ACTED_IN+DIRECTED':
            rel_display = "ACTED_IN+DIRECTED (same movie)"
        else:
            rel_display = rel_type
            
        all_rows.append([
            i,
            "Relationship",
            from_labels,
            rel_display,
            to_labels,
            stat['count']
        ])
    
    # Display table
    print("\nAvailable Items for Migration:")
    headers = ['#', 'Type', 'From', 'Relationship/Label', 'To', 'Count']
    print(tabulate(all_rows, headers=headers, tablefmt='grid'))
    return all_rows

def ensure_unique_constraint(session, labels):
    """Ensure unique constraint exists for remote_id on given labels"""
    # Convert labels list to string if necessary
    if isinstance(labels, list):
        labels = labels
    else:
        labels = [labels]
    
    try:
        # Get existing constraints using schema()
        constraints_result = session.run("""
            CALL db.schema.constraints()
            YIELD name, description
        """)
        
        # Parse the constraints to find which labels already have remote_id constraints
        existing_constraints = set()
        for record in constraints_result:
            description = record.get('description', '')
            # Parse the constraint description to extract the label
            if 'remote_id IS UNIQUE' in description:
                # Extract label from the description using string manipulation
                start_idx = description.find('( ') + 2
                end_idx = description.find(':')
                if start_idx > 1 and end_idx > start_idx:
                    label_part = description[end_idx+1:description.find(' )', end_idx)]
                    existing_constraints.add(label_part.strip())
        
        # Create individual constraints for each label if they don't exist
        for label in labels:
            if label not in existing_constraints:
                constraint_name = f"unique_{label.lower()}_remote_id"
                query = f"""
                    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                    FOR (n:{label})
                    REQUIRE n.remote_id IS UNIQUE
                """
                session.run(query)
                logger.info(f"✓ Created unique constraint for {label}")
            else:
                logger.debug(f"✓ Unique constraint already exists for {label}")
                
    except Exception as e:
        # If we can't check constraints, just try to create them
        logger.debug(f"Could not check existing constraints: {str(e)}")
        logger.debug("Falling back to direct constraint creation")
        for label in labels:
            try:
                constraint_name = f"unique_{label.lower()}_remote_id"
                query = f"""
                    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                    FOR (n:{label})
                    REQUIRE n.remote_id IS UNIQUE
                """
                session.run(query)
            except Exception as create_error:
                logger.warning(f"Could not create constraint for {label}: {str(create_error)}")

def migrate_selected_item(selection, node_stats, rel_stats, remote_session, local_session):
    """Migrate selected node type or relationship"""
    all_rows = node_stats + rel_stats
    if selection < 1 or selection > len(all_rows):
        logger.error("Invalid selection")
        return
    
    # Adjust index for 0-based list
    idx = selection - 1
    
    # Check if it's a node or relationship based on the selection
    if idx < len(node_stats):
        # Node migration code remains the same...
        labels = node_stats[idx]['labelSet']
        label_str = ':'.join(labels)
        logger.info(f"Migrating nodes with labels: {label_str}")
        
        # Ensure unique constraint exists
        ensure_unique_constraint(local_session, labels)
        
        result = remote_session.run(f"""
            MATCH (n:{label_str})
            RETURN 
                elementId(n) as id,
                properties(n) as props
        """)
        
        nodes_data = list(result)
        if not nodes_data:
            logger.info(f"No nodes found with labels {label_str}")
            return
            
        logger.info(f"Found {len(nodes_data)} nodes with labels {label_str}")
        
        # Create nodes in batches
        batch_size = 1000
        for i in range(0, len(nodes_data), batch_size):
            batch = nodes_data[i:i + batch_size]
            for node in batch:
                # Extract all properties and add remote_id
                props = dict(node['props'])
                remote_id = str(node['id'])  # Convert to string to ensure consistent handling
                props['remote_id'] = remote_id
                
                # Create a parameter dict for each property
                params = {f"prop_{k}": v for k, v in props.items()}
                params['remote_id'] = remote_id
                
                # Build the SET clause
                set_clause = ", ".join(f"n.{k} = ${f'prop_{k}'}" for k in props.keys())
                
                merge_query = f"""
                MERGE (n:{label_str} {{remote_id: $remote_id}})
                ON CREATE SET {set_clause}
                ON MATCH SET {set_clause}
                """
                
                try:
                    local_session.run(merge_query, params)
                except Exception as e:
                    logger.error(f"Error creating node: {str(e)}")
                    continue
                    
            logger.info(f"Processed {min(i + batch_size, len(nodes_data))} nodes...")
    else:
        # Relationship migration
        rel_idx = idx - len(node_stats)
        rel_type = rel_stats[rel_idx]['relationshipType']
        from_label = rel_stats[rel_idx]['fromLabels']
        to_label = rel_stats[rel_idx]['toLabels']
        
        if rel_type == 'ACTED_IN+DIRECTED':
            logger.info(f"Migrating dual relationships: {':'.join(from_label)}-[ACTED_IN+DIRECTED]->{':'.join(to_label)}")
            
            rel_result = remote_session.run("""
                MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
                MATCH (p)-[:DIRECTED]->(m)
                RETURN 
                    elementId(p) as start_id,
                    elementId(m) as end_id,
                    properties(p) as start_props,
                    properties(m) as end_props
            """)
        elif rel_type in ['ACTED_IN', 'DIRECTED']:
            logger.info(f"Migrating relationships: {':'.join(from_label)}-[{rel_type}]->{':'.join(to_label)} (excluding dual relationships)")
            
            rel_result = remote_session.run(f"""
                MATCH (a)-[r:{rel_type}]->(b:Movie)
                WHERE NOT (
                    r.type = 'ACTED_IN' AND exists((a)-[:DIRECTED]->(b))
                    OR
                    r.type = 'DIRECTED' AND exists((a)-[:ACTED_IN]->(b))
                )
                RETURN 
                    elementId(a) as start_id,
                    elementId(b) as end_id,
                    properties(r) as props
            """)
        else:
            logger.info(f"Migrating relationships: {':'.join(from_label)}-[{rel_type}]->{':'.join(to_label)}")
            
            rel_result = remote_session.run(f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN 
                    elementId(a) as start_id,
                    elementId(b) as end_id,
                    properties(r) as props
            """)
        
        rels_data = list(rel_result)
        if not rels_data:
            logger.info(f"No relationships found")
            return
            
        logger.info(f"Found {len(rels_data)} relationships")
        
        # Create relationships in batches
        batch_size = 1000
        for i in range(0, len(rels_data), batch_size):
            batch = rels_data[i:i + batch_size]
            for rel in batch:
                # Extract all properties
                props = dict(rel.get('props', {}))
                start_id = str(rel['start_id'])
                end_id = str(rel['end_id'])
                
                params = {
                    'start_id': start_id,
                    'end_id': end_id,
                    **{f'prop_{k}': v for k, v in props.items()}
                }
                
                set_clause = ", ".join(f"r.{k} = ${f'prop_{k}'}" for k in props.keys()) if props else ""
                
                if rel_type == 'ACTED_IN+DIRECTED':
                    # Create both relationships
                    rel_queries = [
                        f"""
                        MATCH (a:{':'.join(from_label)} {{remote_id: $start_id}})
                        MATCH (b:{':'.join(to_label)} {{remote_id: $end_id}})
                        MERGE (a)-[r:ACTED_IN]->(b)
                        """ + (f"ON CREATE SET {set_clause}" if set_clause else ""),
                        f"""
                        MATCH (a:{':'.join(from_label)} {{remote_id: $start_id}})
                        MATCH (b:{':'.join(to_label)} {{remote_id: $end_id}})
                        MERGE (a)-[r:DIRECTED]->(b)
                        """ + (f"ON CREATE SET {set_clause}" if set_clause else "")
                    ]
                    for query in rel_queries:
                        try:
                            local_session.run(query, params)
                        except Exception as e:
                            logger.error(f"Error creating relationship: {str(e)}")
                            continue
                else:
                    rel_query = f"""
                    MATCH (a:{':'.join(from_label)} {{remote_id: $start_id}})
                    MATCH (b:{':'.join(to_label)} {{remote_id: $end_id}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """ + (f"ON CREATE SET {set_clause}" if set_clause else "")
                    
                    try:
                        local_session.run(rel_query, params)
                    except Exception as e:
                        logger.error(f"Error creating relationship: {str(e)}")
                        continue
                    
            logger.info(f"Processed {min(i + batch_size, len(rels_data))} relationships...")
            
def migrate_database_content():
    """Migrate data from remote to local database by appending content"""
    logger.info("Starting database migration process...")
    
    # Get remote connection details
    remote_config = get_remote_config()
    remote_uri = remote_config["NEO4J_URI"]
    remote_username = remote_config["NEO4J_USERNAME"]
    remote_password = remote_config["NEO4J_PASSWORD"]
    
    # Get local connection details
    local_config = get_local_config()
    local_uri = local_config["NEO4J_URI"]
    local_username = local_config["NEO4J_USERNAME"]
    local_password = local_config["NEO4J_PASSWORD"]
    
    try:
        # Connect to remote database
        remote_driver = GraphDatabase.driver(remote_uri, auth=(remote_username, remote_password))
        local_driver = GraphDatabase.driver(local_uri, auth=(local_username, local_password))
        
        with remote_driver.session() as remote_session, local_driver.session() as local_session:
            while True:
                # Get statistics from remote database
                node_stats, rel_stats = get_database_statistics(remote_session)
                
                # Display menu and get selection
                all_rows = display_migration_menu(node_stats, rel_stats)
                
                print("\nEnter the number of the item you want to migrate (0 to exit):")
                try:
                    selection = int(input("> "))
                    if selection == 0:
                        break
                    
                    # Migrate selected item
                    migrate_selected_item(selection, node_stats, rel_stats, remote_session, local_session)
                    
                    # Ask if user wants to continue
                    print("\nDo you want to migrate another item? (y/n):")
                    if input("> ").lower() != 'y':
                        break
                        
                except ValueError:
                    logger.error("Please enter a valid number")
                    continue
            
            # Clean up temporary properties
            logger.info("Cleaning up temporary properties...")
            local_session.run("""
                MATCH (n)
                WHERE n._remote_id IS NOT NULL
                REMOVE n._remote_id
            """)
            
            # Show final statistics
            local_nodes = local_session.run("MATCH (n) RETURN count(n) as count").single()['count']
            local_rels = local_session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            logger.info(f"✅ Migration completed!")
            logger.info(f"Total nodes in local database: {local_nodes}")
            logger.info(f"Total relationships in local database: {local_rels}")
            
    except Exception as e:
        logger.error(f"❌ Error during migration: {str(e)}")
        raise
    finally:
        remote_driver.close()
        local_driver.close()

def get_embedding_stats(session):
    """Get statistics about embeddings in the database"""
    # First, let's check what node types and properties exist
    print("\nDiagnostic Information:")
    
    # Check total nodes
    total_nodes_query = session.run("MATCH (n) RETURN count(n) as count")
    total_nodes = total_nodes_query.single()['count']
    print(f"Total nodes in database: {total_nodes}")
    
    # Check node labels
    labels_query = session.run("CALL db.labels()")
    labels = [record['label'] for record in labels_query]
    print(f"Node labels in database: {labels}")
    
    # Check properties for each label
    print("\nProperties by label:")
    for label in labels:
        props_query = session.run(f"MATCH (n:`{label}`) RETURN keys(n) as props LIMIT 1")
        props = props_query.single()['props'] if props_query.peek() else []
        embedding_props = [p for p in props if 'embedding' in p.lower()]
        if embedding_props:
            print(f"{label}: {embedding_props}")
    
    print("\nGathering embedding statistics...")
    
    # Initialize results with all labels
    results = []
    for label in labels:
        # Get total nodes for this label
        label_count_query = session.run(f"""
            MATCH (n:`{label}`)
            RETURN count(n) as total_nodes
        """)
        label_total_nodes = label_count_query.single()['total_nodes']
        
        # Get embedding properties for this label
        props_query = session.run(f"""
            MATCH (n:`{label}`)
            WITH keys(n) as props
            UNWIND [prop in props WHERE toLower(prop) CONTAINS 'embedding'] as embeddingProp
            RETURN DISTINCT embeddingProp
        """)
        
        embedding_props = [record['embeddingProp'] for record in props_query]
        
        if embedding_props:  # If the label has embedding properties
            for prop in embedding_props:
                # Get stats for this label and property
                stats_query = session.run(f"""
                    MATCH (n:`{label}`)
                    WHERE n.`{prop}` IS NOT NULL
                    RETURN count(n) as embeddings_count
                """)
                
                if stats_query.peek():
                    stat = stats_query.single()
                    results.append({
                        'label': label,
                        'embedding_property': prop,
                        'embeddings_count': stat['embeddings_count'],
                        'total_nodes': label_total_nodes,
                        'coverage': (stat['embeddings_count'] / label_total_nodes * 100) if label_total_nodes > 0 else 0
                    })
        else:  # If the label has no embedding properties
            results.append({
                'label': label,
                'embedding_property': '',
                'embeddings_count': 0,
                'total_nodes': label_total_nodes,
                'coverage': 0
            })
    
    # Get vector index information
    index_info = session.run("""
        SHOW VECTOR INDEXES
        YIELD name, type, labelsOrTypes, properties
        RETURN name, labelsOrTypes[0] as label, properties[0] as property
    """)
    
    # Create a set of indexed label-property pairs
    indexed_pairs = {
        (record['label'], record['property'])
        for record in index_info
    }
    
    # Add index information to results
    for result in results:
        result['indexed'] = (result['label'], result['embedding_property']) in indexed_pairs if result['embedding_property'] else False
    
    # Sort results by label and embedding property
    results.sort(key=lambda x: (x['label'], x['embedding_property']))
    
    return results

def display_embedding_stats(stats):
    """Display embedding statistics in a formatted table"""
    from tabulate import tabulate
    import colorama
    from colorama import Fore, Style
    
    # Initialize colorama for Windows compatibility
    colorama.init()
    
    # Prepare rows for the table
    rows = []
    for i, stat in enumerate(stats, 1):
        # Use colored checkmarks/crosses for index status
        index_status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if stat['indexed'] else f"{Fore.RED}✗{Style.RESET_ALL}"
        
        rows.append([
            i,  # Row number
            stat['label'],
            stat['embedding_property'],
            stat['embeddings_count'],
            stat['total_nodes'],
            f"{stat['coverage']:.1f}%",
            index_status
        ])
    
    # Display table
    headers = ['#', 'Label', 'Embedding Property', 'With Embedding', 'Total Nodes', 'Coverage', 'Indexed']
    print("\nEmbedding Statistics:")
    print(tabulate(rows, headers=headers, tablefmt='grid'))

def manage_indexes():
    """Manage vector indexes in the database"""
    try:
        # Create and connect to database
        connection = Neo4jConnection()
        if not connection.connect():
            print("Failed to connect to database")
            return
            
        with connection.driver.session() as session:
            while True:
                # Get current stats
                stats = get_embedding_stats(session)
                display_embedding_stats(stats)
                
                print("\nOptions:")
                print("1. Create vector index")
                print("2. Drop vector index")
                print("3. Back to main menu")
                
                choice = input("\nEnter your choice (1-3): ")
                
                if choice == '1':
                    # Show only unindexed embeddings
                    unindexed = [s for s in stats if not s['indexed']]
                    if not unindexed:
                        print("All embedding properties are already indexed.")
                        continue
                    
                    print("\nAvailable embedding properties to index:")
                    for i, stat in enumerate(unindexed, 1):
                        print(f"{i}. {stat['label']}.{stat['embedding_property']}")
                    
                    try:
                        idx = int(input("\nEnter the number of the embedding to index (0 to cancel): "))
                        if idx == 0:
                            continue
                        if 1 <= idx <= len(unindexed):
                            stat = unindexed[idx-1]
                            index_name = f"vector_{stat['label'].lower()}_{stat['embedding_property'].lower()}"
                            
                            # Create vector index
                            session.run(f"""
                                CREATE VECTOR INDEX {index_name}
                                FOR (n:`{stat['label']}`)
                                ON (n.{stat['embedding_property']})
                                OPTIONS {{indexConfig: {{
                                    `vector.dimensions`: 1536,
                                    `vector.similarity_function`: 'cosine'
                                }}}}
                            """)
                            print(f"✓ Created vector index {index_name}")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except Exception as e:
                        print(f"Error creating index: {str(e)}")
                
                elif choice == '2':
                    # Show only indexed embeddings
                    indexed = [s for s in stats if s['indexed']]
                    if not indexed:
                        print("No indexed embedding properties found.")
                        continue
                    
                    print("\nIndexed embedding properties:")
                    for i, stat in enumerate(indexed, 1):
                        print(f"{i}. {stat['label']}.{stat['embedding_property']}")
                    
                    try:
                        idx = int(input("\nEnter the number of the index to drop (0 to cancel): "))
                        if idx == 0:
                            continue
                        if 1 <= idx <= len(indexed):
                            stat = indexed[idx-1]
                            index_name = f"vector_{stat['label'].lower()}_{stat['embedding_property'].lower()}"
                            
                            # Drop vector index
                            session.run(f"DROP INDEX {index_name}")
                            print(f"✓ Dropped vector index {index_name}")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except Exception as e:
                        print(f"Error dropping index: {str(e)}")
                
                elif choice == '3':
                    break
                
                else:
                    print("Invalid choice. Please enter a number between 1 and 3.")
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
    finally:
        if 'connection' in locals():
            connection.close()

def main():
    """Main function with interactive menu"""
    # Initialize with local database
    CURRENT_DB_CONFIG.update(get_local_config())
    logger.info("✅ Successfully connected to local Neo4j database!")
    
    while True:
        print("\n=== Neo4j Vector Operations Menu ===")
        print("1. Choose Database")
        print("2. Load Structured Documents")
        print("3. Manage Indexes")
        print("4. Search Paragraphs")
        print("5. Migrate Database Content")
        print("Q. Quit")
        
        choice = input("\nEnter your choice: ").strip().upper()
        
        try:
            if choice == '1':
                choose_database()
            elif choice == '2':
                load_structured_documents()
            elif choice == '3':
                manage_indexes()
            elif choice == '4':
                search_paragraphs()
            elif choice == '5':
                migrate_database_content()
            elif choice == 'Q':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            logger.error(f"Error in main menu: {str(e)}")

if __name__ == "__main__":
    main()
