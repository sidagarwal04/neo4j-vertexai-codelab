from prompts import (
    cypher_generation_prompt,
    summarize_results_prompt
) 

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

# Google Cloud project ID
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')

class Neo4jDatabase:
    """Class to handle Neo4j database operations."""
    
    def __init__(self, uri, username, password):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the driver connection."""
        self.driver.close()
    
    def setup_vector_index(self):
        """Set up or load a vector index in Neo4j for the movie embeddings."""
        with self.driver.session() as session:
            try:
                # Check if the vector index already exists
                check_query = """
                SHOW VECTOR INDEXES YIELD name
                WHERE name = 'overview_embeddings'
                RETURN name
                """
                result = session.run(check_query)
                existing_index = result.single()

                if existing_index:
                    print("Vector index 'overview_embeddings' already exists. No need to create a new one.")
                else:
                    # Create a new vector index if it doesn't exist
                    print("Creating new vector index")
                    query_index = """
                    CREATE VECTOR INDEX overview_embeddings
                    FOR (m:Movie) ON (m.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 768,  
                        `vector.similarity_function`: 'cosine'}}
                    """
                    session.run(query_index)
                    print("Vector index created successfully")
            except Exception as e:
                print(f"Error while setting up vector index: {e}")
    
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
            ) YIELD node, score
            WITH node as m, score
            RETURN m.title AS title, 
                   m.overview AS plot, 
                   m.release_date AS released, 
                   m.tagline AS tagline,
                   score
            ORDER BY score DESC
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

def get_ontology_from_neo4j(driver):
    with driver.session() as session:
        result = session.run("CALL db.schema.nodeTypeProperties()")
        
        nodes = {}
        relationships = set()

        for record in result:
            node_labels = record["nodeLabels"]
            property_name = record["propertyName"]
            node_type = ":".join(node_labels)
            nodes.setdefault(node_type, set()).add(property_name)
        
        # Fetch relationships separately
        rel_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
        for record in rel_result:
            relationships.add(record["relationshipType"])

        # Construct ontology string
        ontology_str = ""

        for node, props in nodes.items():
            prop_list = ", ".join(props)
            ontology_str += f"({node}) has properties: {prop_list}\n"

        for rel in relationships:
            ontology_str += f"(:Node)-[:{rel}]->(:Node)\n"

        return ontology_str.strip()


class MovieRecommendationApp:
    """Main application class that combines Neo4j and Gemini for movie recommendations."""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, project_id, location):
        """Initialize the application with Neo4j and Gemini services."""
        self.neo4j = Neo4jDatabase(neo4j_uri, neo4j_user, neo4j_password)
        self.gemini = GeminiService(project_id, location)
        self.vector_service = VectorService(project_id, location)
    
    def process_query(self, user_input):
        try:
            # Step 1: Vector search
            query_embedding = self.vector_service.generate_embedding(user_input)
            vector_results = self.neo4j.get_movie_recommendations_by_vector(query_embedding, top_k=5)
            
            if not vector_results:
                return "Sorry, no relevant results found using vector search."

            # Step 2: Format the vector search results as context for the LLM
            context = "Information from vector search:\n"
            for i, result in enumerate(vector_results):
                context += f"[Result {i+1}] Title: {result['title']}\nPlot: {result['plot']}\n\n"

            # Step 3: Generate Cypher query using the context and Gemini
            ontology = get_ontology_from_neo4j(self.neo4j.driver)
            cypher_prompt = cypher_generation_prompt(user_input, context, ontology)
            generated_query = self.gemini.generate_response(cypher_prompt).strip()


            if generated_query.startswith("```"):
                lines = generated_query.splitlines()
                # Remove first line (e.g., ```cypher) and last line (```)
                lines = [line for line in lines if not line.strip().startswith("```")]
                generated_query = "\n".join(lines).strip()
            
            print("Generated Cypher:\n", generated_query)

            # Step 4: Run Cypher query
            with self.neo4j.driver.session() as session:
                result = session.run(generated_query)
                records = [record.data() for record in result]
            
            # Step 5: Summarize results
            summary_prompt = summarize_results_prompt(user_input, {"query": generated_query, "results": records}, len(records), str(records))
            summary = self.gemini.generate_response(summary_prompt)

            return summary

        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def close(self):
        """Close all connections."""
        self.neo4j.close()

def handle_user_input(user_input):
    """Gradio interface function to process user input and return recommendations."""
    app = MovieRecommendationApp(
        NEO4J_URI, 
        NEO4J_USER, 
        NEO4J_PASSWORD, 
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
    title="Smart Movie Recommender with GraphRAG",
    description=(
        "Discover movies you’ll love — powered by Neo4j and Vertex AI!\n"
        "This assistant combines semantic search with knowledge graph reasoning — using vector similarity for relevant matches and LLM-generated Cypher queries for deeper insights from movie plots, genres, and relationships."
    ),
    examples=[
        ["Which time travel movies star Bruce Willis?"],
        ["Find romantic comedies directed by female directors."],
        ["Recommend sci-fi movies featuring AI and starring Keanu Reeves."],
        ["Show me thrillers from the 2000s with mind-bending plots."],
        ["List superhero movies where the villain turns good."]
    ],
    flagging_mode="never"
)

# Initialize Neo4j and set up the vector index
neo4j_db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
neo4j_db.setup_vector_index()
neo4j_db.close()

# Launch the Gradio interface
iface.launch(server_name="0.0.0.0", server_port=8080)