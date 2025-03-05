import os
import csv
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

def export_movie_embeddings(output_file='movie_embeddings.csv'):
    # Initialize Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Query to fetch movies with embeddings
        query = """
        MATCH (m:Movie) 
        WHERE m.embedding IS NOT NULL 
        RETURN m.tmdbId AS tmdbId, 
            m.title AS title, 
            m.overview AS overview, 
            m.embedding AS embedding
        """
        
        # Open CSV file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            # CSV writer
            csvwriter = csv.writer(csvfile)
            
            # Write header
            csvwriter.writerow(['tmdbId', 'title', 'overview', 'embedding'])
            
            # Execute query and write results
            with driver.session() as session:
                results = session.run(query)
            
                # Track processing
                total_processed = 0
                
                # Iterate through results
                for record in results:
                    try:
                        # Safely extract data
                        tmdb_id = record['tmdbId']
                        title = record['title']
                        overview = record['overview']
                        
                        # Parse embedding (it's stored as a JSON string)
                        embedding = record['embedding']
                        
                        # Write row to CSV
                        csvwriter.writerow([
                            tmdb_id, 
                            title, 
                            overview, 
                            embedding  # This is already a JSON string
                        ])
                        
                        total_processed += 1
                        
                        # Periodic progress update
                        if total_processed % 1000 == 0:
                            print(f"Processed {total_processed} movies...")
                    
                    except Exception as e:
                        print(f"Error processing movie: {e}")
            
            print(f"\nTotal movies exported: {total_processed}")
    
    finally:
        # Ensure the driver is closed
        driver.close()
    
def main():
    export_movie_embeddings()

if __name__ == "__main__":
    main()