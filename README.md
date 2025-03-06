
# Neo4j + Vertex AI Codelab

A movie recommendation application that combines Neo4j's graph database capabilities with Google Cloud's Vertex AI for semantic search and natural language movie recommendations.

## 📝 Blog Post

Check out the detailed explanation of this project in the blog post: [Building an Intelligent Movie Search with Neo4j and Vertex AI](https://sidagarwal04.medium.com/building-an-intelligent-movie-search-with-neo4j-and-vertex-ai-a38c75f79cf7)

## 🚀 Overview
This project demonstrates how to build an AI-powered movie recommendation engine using:

- **Neo4j**: Graph database for storing movie data and vector embeddings
- **Google Vertex AI**: For generating text embeddings and natural language processing
- **Gradio**: For creating a simple web interface

The system uses semantic search through vector embeddings generated by Vertex AI's text-embedding models to find movies based on natural language queries, and then leverages Gemini to produce conversational responses.

## 🧩 How It Works

1. **Data Processing**: Movie data is loaded into Neo4j graph database
2. **Vector Embeddings**: Text embeddings are generated for movie descriptions using Vertex AI
3. **Vector Search**: Semantic similarity is used to match user queries with relevant movies
4. **Natural Language Processing**: Gemini generates human-like responses based on the matches

## 🗂️ Repository Structure

- `chatbot.py`: Main application with Gradio interface for movie recommendations
- `generate_embeddings.py`: Script to generate vector embeddings for movie data
- `graph_build.py`: Script to populate Neo4j with movie data
- `example.env`: Template for environment variables
- `normalized_data/`: Directory containing movie data

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.7+
- Neo4j database (can be self-hosted or Aura DB)
- Google Cloud account with Vertex AI API enabled
- Service account with appropriate permissions for Vertex AI

### Environment Configuration

1. Clone this repository
2. Copy `example.env` to `.env` and fill in your configuration:
   ```
   NEO4J_URI = your-neo4j-connection-string
   NEO4J_USER = your-neo4j-username
   NEO4J_PASSWORD = your-neo4j-password
   PROJECT_ID = your-gcp-project-id
   LOCATION = your-gcp-location
   GOOGLE_CLOUD_PROJECT = your-gcp-project-id
   ```
3. Create a service account in Google Cloud and download the JSON key file
4. Place the service account key in the project directory (referenced in `generate_embeddings.py`)

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install neo4j vertexai gradio langchain_google_vertexai python-dotenv
```

## 🏃‍♀️ Running the Application

### 1. Build the Graph Database

First, load movie data into Neo4j:

```bash
python graph_build.py
```


### 2. Generate Embeddings

Generate vector embeddings for movie descriptions:

```bash
python generate_embeddings.py
```

**Embedding CSV Utilities**:
- `generate_embeddings_to_csv.py`: A one-time script used to generate `movie_embeddings.csv`, which contains pre-computed vector embeddings for movies.
- `export_embeddings_to_csv.py`: A utility script to export existing embeddings from Neo4j to a CSV file.

**Loading Embeddings Directly from CSV**:
If you want to skip generating embeddings, you can directly load the CSV into Neo4j using the following Cypher command:

```cypher
LOAD CSV WITH HEADERS FROM 'file:///movie_embeddings.csv' AS row
WITH row
MATCH (m:Movie {tmdbId: row.tmdbId})
CALL db.create.setNodeVectorProperty(m, 'embedding', apoc.convert.fromJsonList(row.embedding))
```

Note: Ensure you have the APOC library installed in your Neo4j database to use `apoc.convert.fromJsonList()`.

### 3. Start the Recommender Chatbot

Launch the Gradio web interface:

```bash
python chatbot.py
```

The application will be available at `http://localhost:7860` by default.

## 🚀 Deploying to Cloud Run
Before deploying to Cloud Run, ensure your `requirements.txt` file includes all necessary dependencies for Neo4j and Vertex AI integration. Additionally, you need a `Dockerfile` to containerize your application for deployment.

Both requirements.txt and Dockerfile are present in this repository:
- `requirements.txt`: Lists all the Python dependencies required to run the application.
- `Dockerfile`: Defines the container environment, including the base image, required packages, and how the application is executed.

If you want to deploy this application to Google Cloud Run for production use, follow these steps:

### 1. Set up Environment Variables

```bash
# Set your Google Cloud project ID
export GCP_PROJECT='your-project-id'  # Change this

# Set your preferred region
export GCP_REGION='us-central1'
```

### 2. Create the Repository and Build the Container Image

```bash
# Set the Artifact Registry repository name
export AR_REPO='your-repo-name'  # Change this

# Set your service name
export SERVICE_NAME='movies-chatbot'  # Change if needed

# Create the Artifact Registry repository
gcloud artifacts repositories create "$AR_REPO" \
  --location="$GCP_REGION" \
  --repository-format=Docker

# Configure Docker to use Google Cloud's Artifact Registry
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"

# Build and submit the container image
gcloud builds submit \
  --tag "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME"
```

### 3. Deploy to Cloud Run
Before deployment, ensure your requirements.txt file is properly configured with all necessary dependencies for your Neo4j and VertexAI integration.
```bash
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed \
  --project=$GCP_PROJECT \
  --set-env-vars=GCP_PROJECT=$GCP_PROJECT,GCP_REGION=$GCP_REGION
```

After deployment, your application will be accessible at a URL like:
`https://movies-chatbot-[unique-id].us-central1.run.app/`

Note: 
- Your `requirements.txt` should list all Python dependencies. 
- Make sure your application's `Dockerfile` is set up properly to run in a containerized environment. The `Dockerfile` should include a `pip install -r requirements.txt` command to ensure all dependencies are installed during the container build process.
- You'll need to include your service account credentials (unless running from Google Cloud Shell directly) and environment variables in the container.

## 🧪 Example Queries

- "I want to watch a sci-fi movie with time travel"
- "Recommend me a romantic comedy with a happy ending"
- "I'm in the mood for something with superheroes but not too serious"
- "I want a thriller that keeps me on the edge of my seat"
- "Show me movies about artificial intelligence taking over the world"

## 📚 Learning Resources

- [Neo4j Vector Search Documentation](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)
- [Vertex AI Embeddings](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
- [Gemini API](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)
- [Gradio Documentation](https://gradio.app/docs/)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
