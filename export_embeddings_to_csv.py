import os
import csv
import neo4j
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection parameters
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def export_embeddings_to_csv(output_file='movie_embeddings.csv'):
    """
    Export movie embeddings from Neo4j to a CSV file.
    This script is useful for backing up or transferring embeddings.
    """
    # Create a Neo4j driver instance
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Cypher query to retrieve movie embeddings
            query = """
            MATCH (m:Movie)
            WHERE m.embedding IS NOT NULL
            RETURN m.tmdbId AS tmdbId, 
                   m.title AS title, 
                   toString(m.embedding) AS embedding
            """
            
            results = session.run(query)
            
            # Open CSV file for writing
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                # Write headers
                csvwriter.writerow(['tmdbId', 'title', 'embedding'])
                
                # Write data
                for record in results:
                    csvwriter.writerow([
                        record['tmdbId'], 
                        record['title'], 
                        record['embedding']
                    ])
            
            print(f"Embeddings exported to {output_file}")

    finally:
        driver.close()

if __name__ == '__main__':
    export_embeddings_to_csv()