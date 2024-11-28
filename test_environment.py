# This will test the environment to ensure that the .env file is set up 
# correctly and that the OpenAI and Neo4j connections are working.
import os
import unittest
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()

class TestEnvironment(unittest.TestCase):
    """Test environment configuration and connections"""

    def test_env_file_exists(self):
        """Check if .env file exists"""
        env_file_exists = os.path.exists('.env')
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        """Helper method to check environment variables"""
        value = os.getenv(variable_name)
        self.assertIsNotNone(value, f"{variable_name} not found in .env file")
        return value

    def test_openai_variables(self):
        """Verify OpenAI API key in .env"""
        self.env_variable_exists('OPENAI_API_KEY')

    def test_local_neo4j_variables(self):
        """Verify local Neo4j credentials in .env"""
        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')

    def test_remote_neo4j_variables(self):
        """Verify remote Neo4j credentials in .env"""
        self.env_variable_exists('REMOTE_NEO4J_URI')
        self.env_variable_exists('REMOTE_NEO4J_USERNAME')
        self.env_variable_exists('REMOTE_NEO4J_PASSWORD')

    def test_openai_connection(self):
        """Test OpenAI API connectivity"""
        from openai import OpenAI, AuthenticationError

        api_key = self.env_variable_exists('OPENAI_API_KEY')
        llm = OpenAI(api_key=api_key)
        
        try:
            models = llm.models.list()
            self.assertGreater(len(models.data), 0, "No models found")
        except AuthenticationError:
            self.fail("OpenAI authentication failed")
        except Exception as e:
            self.fail(f"OpenAI connection failed: {str(e)}")

    def test_local_neo4j_connection(self):
        """Test local Neo4j database connectivity"""
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable

        uri = self.env_variable_exists('NEO4J_URI')
        username = self.env_variable_exists('NEO4J_USERNAME')
        password = self.env_variable_exists('NEO4J_PASSWORD')

        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
            self.assertTrue(True, "Local Neo4j connection successful")
            driver.close()
        except ServiceUnavailable:
            self.fail("Local Neo4j connection failed - database might not be running")
        except Exception as e:
            self.fail(f"Local Neo4j connection failed: {str(e)}")

    def test_remote_neo4j_connection(self):
        """Test remote Neo4j database connectivity"""
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable

        uri = self.env_variable_exists('REMOTE_NEO4J_URI')
        username = self.env_variable_exists('REMOTE_NEO4J_USERNAME')
        password = self.env_variable_exists('REMOTE_NEO4J_PASSWORD')

        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            driver.verify_connectivity()
            self.assertTrue(True, "Remote Neo4j connection successful")
            driver.close()
        except ServiceUnavailable:
            self.fail("Remote Neo4j connection failed - database might not be accessible")
        except Exception as e:
            self.fail(f"Remote Neo4j connection failed: {str(e)}")

class DetailedTestRunner(unittest.TextTestRunner):
    def run(self, test):
        print("\n=== Environment Test Results ===\n")
        result = super().run(test)
        
        # Get all test methods from TestEnvironment
        test_methods = [method for method in dir(TestEnvironment) if method.startswith('test_')]
        total_tests = len(test_methods)
        
        print("\nDetailed Test Results:")
        print("-" * 80)
        print(f"{'Test Name':<40} {'Status':<20} {'Details'}")
        print("-" * 80)
        
        for test_name in test_methods:
            test_method = getattr(TestEnvironment, test_name)
            doc = test_method.__doc__ or test_name.replace('_', ' ').title()
            
            # Check if test was skipped
            skipped_tests = [test[0].id().split('.')[-1] for test in result.skipped]
            failed_tests = [test[0].id().split('.')[-1] for test in result.failures]
            error_tests = [test[0].id().split('.')[-1] for test in result.errors]
            
            if test_name in skipped_tests:
                status = "SKIPPED"
                details = "Dependencies not met"
            elif test_name in failed_tests:
                status = "FAILED"
                details = "See error above"
            elif test_name in error_tests:
                status = "ERROR"
                details = "See error above"
            else:
                status = "PASSED"
                details = "All checks successful"
            
            print(f"{doc:<40} {status:<20} {details}")
        
        print("\nSummary:")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
        print(f"Failed: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        
        return result

def suite():
    """Create a test suite that includes all the test cases."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEnvironment))
    return suite

if __name__ == '__main__':
    runner = DetailedTestRunner(verbosity=2)
    runner.run(suite())