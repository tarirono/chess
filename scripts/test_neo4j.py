import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "chess123"

try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("Neo4j connection successful!")

    with driver.session() as session:
        result = session.run('RETURN "Chess Ecosystem connected!" AS message')
        print(result.single()["message"])

    driver.close()
except Exception as e:
    print(f"Connection failed: {e}")

