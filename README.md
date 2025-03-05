
# Neo4j + Vertex AI Codelab

A movie recommendation application that combines Neo4j's graph database capabilities with Google Cloud's Vertex AI for semantic search and natural language movie recommendations.

## 📝 Blog Post

Check out the detailed explanation of this project in the blog post: [Building an Intelligent Movie Search with Neo4j and Vertex AI](https://sidagarwal04.medium.com/building-an-intelligent-movie-search-with-neo4j-and-vertex-ai-a38c75f79cf7)

## 🚀 Overview

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


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
