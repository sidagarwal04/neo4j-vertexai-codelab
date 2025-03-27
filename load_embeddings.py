import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

class LoadEmbeddings:
    def __init__(self, uri, user, password, database='neo4j'):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)

    def close(self):
        self.driver.close()
    
    def load_embeddings(self, csv_file):
        query = """
        LOAD CSV WITH HEADERS FROM $csvFile AS row
        WITH row
        MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})
        SET m.embedding = apoc.convert.fromJsonList(row.embedding)
        RETURN count(m) AS count
        """
        with self.driver.session() as session:
            result = session.run(query, csvFile=f'{csv_file}')
            count = result.single()["count"]
            print(f"Embeddings loaded from {csv_file}, total embeddings stored: {count}")

def main():
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USER')
    password = os.getenv('NEO4J_PASSWORD')
    database = os.getenv('NEO4J_DATABASE')

    graph = LoadEmbeddings(uri, user, password, database)

    # Load embeddings
    graph.load_embeddings('https://storage.googleapis.com/neo4j-vertexai-codelab/movie_embeddings.csv')

    graph.close()

if __name__ == "__main__":
    main()