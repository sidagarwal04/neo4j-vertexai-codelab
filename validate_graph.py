import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# Neo4j credentials from .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_node_counts(tx):
    query = """
    CALL db.labels() YIELD label
    CALL {
        WITH label
        RETURN label, count(*) AS count
        CALL {
            WITH label
            RETURN count { MATCH (n:`${label}`) RETURN n } AS count
        }
    }
    RETURN label, count { MATCH (n:`${label}`) RETURN n } AS count
    """
    result = tx.run("CALL db.labels() YIELD label RETURN label")
    labels = [record["label"] for record in result]
    for label in labels:
        count_query = f"MATCH (n:`{label}`) RETURN count(n) AS count"
        count = tx.run(count_query).single()["count"]
        print(f"{label}: {count} nodes")

def fetch_relationship_counts(tx):
    result = tx.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    types = [record["relationshipType"] for record in result]
    for rel_type in types:
        count_query = f"MATCH ()-[:`{rel_type}`]->() RETURN count(*) AS count"
        count = tx.run(count_query).single()["count"]
        print(f"{rel_type}: {count} relationships")

with driver.session() as session:
    print("\n📦 Node Counts:")
    session.execute_read(fetch_node_counts)

    print("\n🔗 Relationship Counts:")
    session.execute_read(fetch_relationship_counts)

driver.close()
