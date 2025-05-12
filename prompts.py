"""
LLM prompts module for movie knowledge graph interactions.

This module contains lambda functions that generate prompts for different
language model interactions needed throughout the application, including
fixing Cypher queries, generating Cypher from natural language, and
summarizing query results.
"""

# Prompt for generating Cypher queries from natural language questions
# Takes a user query, vector search context, and knowledge graph ontology
# Returns a prompt instructing the LLM how to generate an appropriate Cypher query
cypher_generation_prompt = lambda query, context, ontology:f"""
You are an assistant working with a Neo4j movie database.

Here is the ontology of the movie knowledge graph:

{ontology}

QUESTION: {query}

Here is some relevant context from a vector search:
{context}

Your task is to generate a Cypher query using the ontology and context above.

IMPORTANT GUIDELINES FOR CYPHER QUERIES:
1. Always start with a valid Cypher clause like MATCH, CREATE, MERGE, OPTIONAL, UNWIND, CALL, WITH, RETURN. 
2. DO NOT try to escape characters or produce special characters like new line, tab, etc. IT WILL result in a syntax error.
3. Use specific node labels like Movie, Genre, Person, etc., as per the ontology.
4. Use appropriate relationships between nodes like :ACTED_IN, :DIRECTED, :HAS_GENRE, etc.
5. Filter based on user intent using WHERE clauses.
6. Use a RETURN clause to specify what to return like movie title, overview, genre, release date, etc.
7. Use the provided context to understand entity types and relationships
8. Do not include triple backticks ``` or ```cypher or any additional text except the generated Cypher statement in your response.
9. Do not use any properties or relationships not included in the schema.

Based on this context and the question, generate an appropriate Cypher query to find the answer.
"""

# Prompt for summarizing Cypher query results in natural language
# Takes the original user query, Cypher results, result count, and formatted results
# Returns a prompt instructing the LLM to generate a human-readable summary
summarize_results_prompt = lambda query, cypher_results, result_count, formatted_cypher_results: f"""
You are a friendly movie assistant helping users find films that match their preferences.

The user asked: "{query}"

Here’s the Cypher query that was run on the Neo4j movie knowledge graph:
{cypher_results.get("query", "No query available")}

Results found: {result_count}

Results:
{formatted_cypher_results[:4000] if result_count > 0 else "No results found."}

Your task:
1. Provide a clear, engaging summary of the movies found — write as if you’re a movie enthusiast recommending films to a friend.
2. For each movie you mention, you MUST include:
   - The title
   - A brief but complete plot/overview
   - (Optional but helpful: release year, genre, or standout features if available)
3. Explain why each movie is a good match based on the user's request (themes, keywords, actors, etc.).
4. Do not list everything — focus on the most relevant results (top 3–5 is fine), but present them narratively with all required info.
5. If no results were found:
   - Suggest why (e.g., overly broad/specific query, data not available)
   - Offer tips to refine their query for better results next time

Keep the tone conversational and informative — like you're having a chat with someone at a movie club.
"""