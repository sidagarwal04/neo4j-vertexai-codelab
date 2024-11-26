import os
import numpy as np
import json
import tempfile
import vertexai

from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from google.auth import credentials
from neo4j import GraphDatabase
from dotenv import load_dotenv
# from langchain_community.graphs import Neo4jGraph

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


# Retrieve movie plots and titles from Neo4j
def retrieve_movie_plots():
    query = "MATCH (m:Movie) WHERE m.embedding IS NULL RETURN m.tmdbId AS tmdbId, m.title AS title, m.overview AS overview"
    with driver.session() as session:
        results = session.run(query)
        movies = [{"tmdbId": row["tmdbId"], "title": row["title"], "overview": row["overview"]} for row in results]
    return movies

# Generate embeddings for movie plots using Haystack and store them immediately in Neo4j
def generate_and_store_embeddings(movies):
    for movie in movies:
        title = movie.get("title", "Unknown Title")  # Fetch the movie title
        overview = str(movie.get("overview", ""))  # Ensure the overview is a string, use empty string as default
        
        print(f"Generating embedding for movie: {title}")
        print(f"Overview for {title} movie: {overview}")

        # Check if the overview is not empty
        if overview.strip() == "":
            print(f"No overview available for movie: {title}. Skipping embedding generation.")
            continue
        
        try:
            # Generate embedding for the current overview (pass overview as a string to the embedder)
            embedding_result = embeddings.embed_query(overview)  # Pass overview as a string
            # print(str(embedding_result)[:100])
            # embedding = embedding_result.get("embedding", None)  # Safely access the embedding from the result
            
            if embedding_result:
                # Store the embedding in Neo4j immediately
                tmdbId = movie["tmdbId"]
                store_embedding_in_neo4j(tmdbId, embedding_result)
            else:
                print(f"Failed to generate embedding for movie: {title}")
        except Exception as e:
            print(f"Error generating embedding for movie {title}: {e}")


# Store the embedding in Neo4j
def store_embedding_in_neo4j(tmdbId, embedding):
    query = """
    MATCH (m:Movie {tmdbId: $tmdbId})
    SET m.embedding = $embedding
    """
    with driver.session() as session:
        session.run(query, tmdbId=tmdbId, embedding=embedding)
    print(f"Embedding for movie {tmdbId} successfully stored in Neo4j.")


# Verify embeddings stored in Neo4j
def verify_embeddings():
    query = "MATCH (m:Movie) WHERE m.embedding IS NOT NULL RETURN m.title, m.embedding LIMIT 10"
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            print(f"Movie: {record['m.title']}, Embedding: {np.array(record['m.embedding'])[:5]}...")  # Print first 5 values


# Main function to orchestrate the process
def main():
    # Step 1: Retrieve movie plots from Neo4j
    movies = retrieve_movie_plots()
    if not movies:
        print("No movies found in the Neo4j database.")
        return

    # Step 2: Generate embeddings for movie plots and store them immediately
    generate_and_store_embeddings(movies)

    # Step 3: Verify that embeddings are stored in Neo4j
    verify_embeddings()


if __name__ == "__main__":
    main()
