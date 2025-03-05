import os
import csv
import json
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Set GOOGLE_APPLICATION_CREDENTIALS
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account.json" #update this with actual .json file name linked to the service account

# Initialize Google Cloud AI Platform
vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))

# Vertex AI Embedding Model Endpoint
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

def retrieve_all_movies():
    query = """
    MATCH (m:Movie) 
    WHERE m.overview IS NOT NULL 
    AND m.overview <> ''
    RETURN m.tmdbId AS tmdbId, 
           m.title AS title, 
           m.overview AS overview
    """
    
    with driver.session() as session:
        results = session.run(query)
        movies = [
            {
                "tmdbId": row["tmdbId"], 
                "title": row["title"], 
                "overview": row["overview"]
            } for row in results
        ]
    return movies

def generate_embeddings_to_csv(output_file='movie_embeddings.csv'):
    # Retrieve all movies
    movies = retrieve_all_movies()
    print(f"Total movies to process: {len(movies)}")
    
    # Open file in write mode to overwrite any existing file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write headers
        csvwriter.writerow(['tmdbId', 'title', 'overview', 'embedding'])
        
        # Tracking variables
        processed_count = 0
        failed_count = 0
        
        # Process each movie
        for movie in movies:
            try:
                # Generate embedding
                embedding = embeddings.embed_query(movie['overview'])
                
                if embedding:
                    # Write to CSV
                    csvwriter.writerow([
                        movie['tmdbId'], 
                        movie['title'], 
                        movie['overview'], 
                        json.dumps(embedding)
                    ])
                    
                    processed_count += 1
                    
                    # Print progress
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} movies...")
                        csvfile.flush()  # Ensure data is written to disk
                else:
                    failed_count += 1
                    print(f"Failed to generate embedding for: {movie['title']}")
            
            except Exception as e:
                failed_count += 1
                print(f"Error processing {movie['title']}: {e}")
        
        # Final summary
        print("\nProcessing Complete:")
        print(f"Total movies processed: {processed_count}")
        print(f"Total movies failed: {failed_count}")

def main():
    generate_embeddings_to_csv()

if __name__ == "__main__":
    main()
