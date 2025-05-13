import os
import vertexai
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

# Google Cloud project ID
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')

class Neo4jDatabase:
    """Class to handle Neo4j database operations."""
    
    def __init__(self, uri, username, password, database="neo4j"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password), database=database)
        
    def close(self):
        """Close the driver connection."""
        self.driver.close()
    
    def setup_vector_index(self):
        """Set up a vector index in Neo4j for the movie embeddings."""
        with self.driver.session() as session:
            try:
                # Drop the existing vector index if it exists
                session.run("DROP INDEX overview_embeddings IF EXISTS")
                print("Old index dropped")
            except Exception as e:
                print(f"No index to drop: {e}")

            # Create a new vector index on the embedding property
            print("Creating new vector index")
            query_index = """
            CREATE VECTOR INDEX overview_embeddings IF NOT EXISTS
            FOR (m:Movie) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,  
                `vector.similarity_function`: 'cosine'}}
            """    
            session.run(query_index)
            print("Vector index created successfully")
    
    def get_movie_recommendations_by_vector(self, user_embedding, top_k=5):
        """
        Get movie recommendations from Neo4j using vector similarity search.
        
        Args:
            user_embedding: Vector representation of user query
            top_k: Number of recommendations to return
        """
        with self.driver.session() as session:
            # Vector similarity search query using the vector index
            query = """
            CALL db.index.vector.queryNodes(
              'overview_embeddings',
              $top_k,
              $embedding
            ) YIELD node as m, score
            RETURN m.title AS title, 
                   m.overview AS plot, 
                   m.release_date AS released, 
                   m.tagline AS tagline,
                   score
            """
            
            result = session.run(
                query, 
                embedding=user_embedding,
                top_k=top_k
            )
            
            recommendations = [
                {
                    "title": record["title"], 
                    "plot": record["plot"],
                    "released": record.get("released", "Unknown"),
                    "tagline": record.get("tagline", ""),
                    "similarity": record.get("score", 0)
                } 
                for record in result
            ]
            return recommendations

class VectorService:
    """Class to handle vector embeddings."""
    
    def __init__(self, project_id, location):
        """Initialize VertexAI."""
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
    def generate_embedding(self, text):
        """
        Generate embedding vector for the given text using Vertex AI text-embedding-005.
        """
        from vertexai.language_models import TextEmbeddingModel
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = embedding_model.get_embeddings([text])
        return embeddings[0].values

class GeminiService:
    """Class to handle Gemini API calls via Vertex AI."""
    
    def __init__(self, project_id, location):
        """Initialize Gemini service with Vertex AI."""
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Load the generative model
        from vertexai.generative_models import GenerativeModel
        self.model = GenerativeModel("gemini-2.0-flash-001")
    
    def generate_response(self, prompt):
        """Generate a response using Gemini."""
        response = self.model.generate_content(prompt)
        return response.text

class MovieRecommendationApp:
    """Main application class that combines Neo4j and Gemini for movie recommendations."""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, neo4j_database, project_id, location):
        """Initialize the application with Neo4j and Gemini services."""
        self.neo4j = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)
        self.gemini = GeminiService(project_id, location)
        self.vector_service = VectorService(project_id, location)
    
    def process_query(self, user_input):
        """Process a user query to get movie recommendations using vector search."""
        try:
            # Step 1: Generate embedding for user query - using the same model that was used for the movies
            query_embedding = self.vector_service.generate_embedding(user_input)
            
            # Step 2: Get recommendations using vector similarity search
            recommendations = self.neo4j.get_movie_recommendations_by_vector(query_embedding)
            
            # Step 3: Use Gemini to craft a personalized response
            if recommendations:
                movies_context = "\n".join([
                    f"Movie: {rec['title']}\n"
                    f"Plot: {rec['plot']}\n"
                    f"Released: {rec['released']}\n"
                    f"Tagline: {rec['tagline']}\n"
                    f"Similarity Score: {rec['similarity']:.4f}"
                    for rec in recommendations
                ])
                
                explanation_prompt = f"""
                The user asked: "{user_input}"
                
                Based on their query, I found these movies (with semantic similarity scores):
                {movies_context}
                
                Create a friendly and helpful response that:
                1. Acknowledges their request
                2. Explains why these recommendations match their request (referring to plot elements, themes, etc.)
                3. Presents the movies in a clear, readable format with titles, release years, and brief descriptions
                4. Asks if they'd like more specific recommendations
                
                Important note: Don't simply list out all the movies with bullet points or numbers. Format it as a conversational response while still highlighting the key information about each movie.
                """
                
                response = self.gemini.generate_response(explanation_prompt)
            else:
                response = f"I couldn't find any movies matching '{user_input}'. Our database might not have embeddings for all movies yet. Could you try a different query?"
            
            return response
        
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again."
    
    def close(self):
        """Close all connections."""
        self.neo4j.close()

def handle_user_input(user_input):
    """Gradio interface function to process user input and return recommendations."""
    app = MovieRecommendationApp(
        NEO4J_URI, 
        NEO4J_USER, 
        NEO4J_PASSWORD,
        NEO4J_DATABASE, 
        PROJECT_ID, 
        LOCATION
    )
    
    try:
        response = app.process_query(user_input)
        return response
    finally:
        app.close()

# Create Gradio interface
iface = gr.Interface(
    fn=handle_user_input, 
    inputs=gr.Textbox(
        placeholder="What kind of movie would you like to watch?",
        lines=3,
        label="Your movie preference"
    ),
    outputs=gr.Textbox(
        label="Recommendations",
        lines=12
    ),
    title="AI Movie Recommendation System",
    description="Get personalized movie recommendations using semantic search with Neo4j vector search and Google Vertex AI!",
    examples=[
        ["I want to watch a sci-fi movie with time travel"],
        ["Recommend me a romantic comedy with a happy ending"],
        ["I'm in the mood for something with superheroes but not too serious"],
        ["I want a thriller that keeps me on the edge of my seat"],
        ["Show me movies about artificial intelligence taking over the world"]
    ],
    allow_flagging="never"
)

# Initialize Neo4j and set up the vector index
neo4j_db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
neo4j_db.setup_vector_index()
neo4j_db.close()

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()