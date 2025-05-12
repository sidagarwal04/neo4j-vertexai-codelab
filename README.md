
# Neo4j + Vertex AI Codelab

A movie recommendation application that combines Neo4j's graph database capabilities with Google Cloud's Vertex AI to deliver intelligent, natural language-based movie recommendations. The system performs semantic vector search using Vertex AI embeddings, then leverages large language models to generate and execute Cypher queries on the Neo4j knowledge graph, enabling multi-hop reasoning and contextual recommendations powered by the GraphRAG pattern.

## 📝 Blog Post

Check out the detailed explanation of this project in the blog post: [Building an Intelligent Movie Search with Neo4j and Vertex AI](https://sidagarwal04.medium.com/building-an-intelligent-movie-search-with-neo4j-and-vertex-ai-a38c75f79cf7)

## 🚀 Overview
This project demonstrates how to build an GenAI-powered movie recommendation engine using the GraphRAG pattern by integrating:

- **Neo4j**:  A graph database for storing movie data, knowledge graph relationships, and vector embeddings
- **Google Vertex AI**: For generating semantic embeddings (text-embedding-005) and leveraging Gemini for natural language understanding and Cypher generation
- **Gradio**: To build an intuitive web interface for interactive recommendations

The system performs semantic search using vector embeddings to retrieve relevant movie context, then dynamically generates Cypher queries using Gemini based on this context and the Neo4j knowledge graph schema. These Cypher queries are executed to fetch precise results, which are then summarized conversationally by Gemini — enabling a powerful, explainable, and context-aware movie recommendation experience.

## 🎬 [Live Demo](https://movies-reco-258362460261.us-central1.run.app/)

## 🧩 How It Works

1. **Data Ingestion**: Movie metadata (titles, plots, genres, actors, etc.) is loaded into a Neo4j graph database and modeled using nodes and relationships.
2. **Vector Embeddings**: Vertex AI's `text-embedding-005` model is used to generate semantic embeddings for movie descriptions, which are stored in Neo4j with a vector index.
3. **Vector Search**: When a user enters a query, the system computes its embedding and performs a vector similarity search in Neo4j to retrieve semantically relevant movies.
4. **Cypher Query Generation**: Using the vector search results and the graph schema (ontology), Gemini generates a Cypher query tailored to the user's intent.
5. **Graph Reasoning**: The generated Cypher query is executed on the Neo4j knowledge graph to perform multi-hop reasoning and extract deeper insights or related entities.
6. **Natural Language Summary**: Gemini then summarizes the Cypher query results in a human-friendly, conversational format.

## 🗂️ Repository Structure

- `example.env`: Template for required environment variables
- `.env.yaml` – Cloud Run deployment environment configuration
- `normalized_data/`: Contains normalized movie dataset
- `graph_build.py`: Loads movies, genres, actors, and relationships into Neo4j
- `generate_embeddings.py`: Generates semantic vector embeddings using Vertex AI
- `generate_embeddings_to_csv.py` / `export_embeddings_to_csv.py` – Scripts for exporting or generating embeddings into CSV
- `load_embeddings.py` – Loads generated embeddings into Neo4j with vector index
- `movie_embeddings.csv` – Precomputed embeddings file (used for faster load/testing)
- `prompts.py` – Prompt templates for Gemini (Cypher generation, summarization, query repair)
- `app.py`: Main application that powers the Gradio UI and implements the GraphRAG pipeline (vector search + LLM-based Cypher execution)
- `Dockerfile` – Used to containerize and deploy the application (e.g., to Cloud Run)
- `requirements.txt` – Python dependencies

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.7+
- Neo4j database (can be self-hosted or [Aura DB](https://console.neo4j.io/)(**recommended**))
- Google Cloud account with Vertex AI API enabled
- Service account with appropriate permissions for Vertex AI

### Environment Configuration
💡 Tip: Run these steps in [Google Cloud Shell](https://shell.cloud.google.com) for a pre-authenticated environment with gcloud, Vertex AI SDK, and permissions already set up — no need to manually manage service account keys.

1. Clone this repository
   ```bash
   git clone https://github.com/your-username/neo4j-vertexai-codelab.git
   cd neo4j-vertexai-codelab
   ```
2. Copy `example.env` to `.env` and fill in your configuration:
   ```bash
   NEO4J_URI=your-neo4j-connection-string
   NEO4J_USER=your-neo4j-username
   NEO4J_PASSWORD=your-neo4j-password
   PROJECT_ID=your-gcp-project-id
   LOCATION=your-gcp-location
   ```
3. (Optional) Create a service account in Google Cloud and download the JSON key file
    - Ensure it has access to Vertex AI and Cloud Storage.
    - Grant roles: Vertex AI User, Storage Object Viewer, etc.
4. (Optional) Place the service account key JSON file in the project directory (referenced in `generate_embeddings.py`)
    - Set the path using:
      ```bash
      export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"
      ```

### Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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

If you want to skip generating embeddings and load precomputed embeddings (in a CSV file) directly into Neo4j, you have two options:

**a. Running Cypher Directly in Neo4j Aura Console**

You can load the CSV file directly into Neo4j using the following Cypher query:

```cypher
LOAD CSV WITH HEADERS FROM 'https://storage.googleapis.com/neo4j-vertexai-codelab/movie_embeddings.csv' AS row
WITH row
MATCH (m:Movie {tmdbId: toInteger(row.tmdbId)})
SET m.embedding = apoc.convert.fromJsonList(row.embedding)
```

**b. Using the Python Script**

Alternatively, you can run the load_embeddings.py script, which automates this process via the Neo4j Python driver.
```bash
python load_embeddings.py
```

### 3. Start the Recommender Chatbot

Launch the Gradio web interface:

```bash
python app.py
```

The application will be available at `http://0.0.0.0:8080` by default.

## 🚀 Deploying to Cloud Run
Before deploying to Cloud Run, ensure your `requirements.txt` file includes all necessary dependencies for Neo4j and Vertex AI integration. Additionally, you need a `Dockerfile` to containerize your application for deployment.

Both requirements.txt and Dockerfile are present in this repository:
- `requirements.txt`: Lists all the Python dependencies required to run the application.
- `Dockerfile`: Defines the container environment, including the base image, required packages, and how the application is executed.

If you want to deploy this application to Google Cloud Run for production use, follow these steps:

### 1. Set up Environment Variables

```bash
# Set your Google Cloud project ID
export GCP_PROJECT='your-project-id'  # Change this as per your GCP Project ID

# Set your preferred region
export GCP_REGION='us-central1' # Change this as per your GCP region
```

### 2. Create the Repository and Build the Container Image

```bash
# Set the Artifact Registry repository name
export AR_REPO='movies-reco'  # Change this if needed

# Set your service name
export SERVICE_NAME='movies-reco'  # Change if needed

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

#### Setting Environment Variables from `.env.yaml` File
Before deploying the application to Cloud Run, configure `.env.yaml` for Cloud Run Deployment in your project root with the following structure:

```bash
NEO4J_URI: "bolt+s://<your-neo4j-uri>"
NEO4J_USER: "neo4j"
NEO4J_PASSWORD: "<your-neo4j-password>"
PROJECT_ID: "<your-gcp-project-id>"
LOCATION: "<your-gcp-region>"
```
✅ This YAML file is used during Cloud Run deployment to inject environment variables into your container runtime. Once set, you can proceed with the gcloud run deploy command.

The following command deploys your application to Cloud Run using environment variables defined in `.env.yaml`. It ensures proper formatting, removes any commented lines automatically handled by gcloud, and sets up the containerized app with public access:
```bash
gcloud run deploy "$SERVICE_NAME" \
  --port=8080 \
  --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$AR_REPO/$SERVICE_NAME" \
  --allow-unauthenticated \
  --region=$GCP_REGION \
  --platform=managed \
  --project=$GCP_PROJECT \
  --env-vars-file=.env.yaml
```

After deployment, your application will be accessible at a URL like:
`https://movies-reco-[unique-id].us-central1.run.app/`

Note: 
- Your `requirements.txt` should list all Python dependencies. 
- Make sure your application's `Dockerfile` is set up properly to run in a containerized environment. The `Dockerfile` should include a `pip install -r requirements.txt` command to ensure all dependencies are installed during the container build process.
- You'll need to include your service account credentials (unless running from Google Cloud Shell directly) and environment variables in the container.

## 🧪 Example Queries

- "Which time travel movies star Bruce Willis?"
- "Show me thrillers from the 2000s with mind-bending plots."
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
