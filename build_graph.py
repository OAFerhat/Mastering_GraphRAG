import asyncio
import logging.config
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.openai_llm import OpenAILLM

load_dotenv()

# Set log level to DEBUG for all neo4j_graphrag.* loggers
logging.config.dictConfig(
    {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
            }
        },
        "loggers": {
            "root": {
                "handlers": ["console"],
            },
            "neo4j_graphrag": {
                "level": "DEBUG",
            },
        },
    }
)

# Connect to the Neo4j database
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
driver = GraphDatabase.driver(URI, auth=AUTH)

# 1. Chunk the text
text_splitter = FixedSizeSplitter(chunk_size=150, chunk_overlap=20)

# 2. Embed the chunks
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# 3. List entities and relationships to extract
entities = ["Person", "House", "Planet", "Organization"]
relations = ["SON_OF", "HEIR_OF", "RULES", "MEMBER_OF"]
potential_schema = [
    ("Person", "SON_OF", "Person"),
    ("Person", "HEIR_OF", "House"),
    ("House", "RULES", "Planet"),
    ("Person", "MEMBER_OF", "Organization"),
]

# 4. Extract nodes and relationships from the chunks
llm = OpenAILLM(
    model_name="gpt-4o",
    model_params={
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "seed": 123
    },
)

# 5. Create the pipeline
pipeline = SimpleKGPipeline(
    driver=driver,
    text_splitter=text_splitter,
    embedder=embedder,
    entities=entities,
    relations=relations,
    potential_schema=potential_schema,
    llm=llm,
    on_error="IGNORE",
    from_pdf=False,
)

# 6. Run the pipeline
asyncio.run(
    pipeline.run_async(
        text=(
            "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of "
            "House Atreides, an aristocratic family that rules the planet Caladan. Lady "
            "Jessica is a Bene Gesserit and an important key in the Bene Gesserit "
            "breeding program."
        )
    )
)

driver.close()
