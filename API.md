# API Documentation

This document describes the REST API endpoints provided by the Codebase Indexing Solution MCP Server.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, you should implement proper authentication and authorization.

## Endpoints

### Health Check

#### GET /health

Check if the server is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "MCP Server"
}
```

### System Status

#### GET /mcp/status

Get the current system status including database health and available models.

**Response:**
```json
{
  "qdrant_status": "healthy",
  "neo4j_status": "healthy", 
  "ollama_status": "available",
  "available_models": ["ollama", "openai"],
  "indexed_files": 42,
  "total_chunks": 1337
}
```

### Query Codebase

#### POST /mcp/query

Query the codebase using natural language or code snippets.

**Request Body:**
```json
{
  "query": "find authentication functions",
  "model": "ollama",
  "limit": 10,
  "include_context": true
}
```

**Parameters:**
- `query` (string, required): The search query
- `model` (string, optional): Embedding model to use ("ollama", "openai", "huggingface")
- `limit` (integer, optional): Maximum number of results (default: 10)
- `include_context` (boolean, optional): Include related chunks (default: true)

**Response:**
```json
{
  "query": "find authentication functions",
  "results": [
    {
      "chunk": {
        "id": "abc123",
        "content": "def authenticate_user(username, password):\n    ...",
        "file_path": "auth.py",
        "start_line": 10,
        "end_line": 25,
        "node_type": "function_definition",
        "name": "authenticate_user",
        "parent_id": null,
        "calls": ["hash_password", "get_user"],
        "called_by": ["login"],
        "imports": [],
        "metadata": {
          "language": "python"
        }
      },
      "score": 0.95,
      "context_chunks": [
        {
          "id": "def456",
          "content": "def hash_password(password):\n    ...",
          "file_path": "auth.py",
          "start_line": 5,
          "end_line": 8,
          "node_type": "function_definition",
          "name": "hash_password"
        }
      ]
    }
  ],
  "total_results": 1,
  "model_used": "ollama",
  "processing_time": 0.123
}
```

### Get Graph Data

#### GET /mcp/graph

Retrieve graph data for visualization.

**Query Parameters:**
- `file_path` (string, optional): Filter by specific file path
- `limit` (integer, optional): Maximum number of nodes (default: 1000)

**Example:**
```
GET /mcp/graph?file_path=auth.py&limit=500
```

**Response:**
```json
{
  "nodes": [
    {
      "id": "abc123",
      "label": "authenticate_user",
      "type": "function_definition",
      "file_path": "auth.py",
      "properties": {
        "start_line": 10,
        "end_line": 25,
        "content_preview": "def authenticate_user(username, password):\n    ..."
      }
    }
  ],
  "edges": [
    {
      "source": "abc123",
      "target": "def456", 
      "type": "calls",
      "weight": 1.0,
      "properties": {}
    }
  ],
  "metadata": {
    "total_nodes": 1,
    "total_edges": 1,
    "file_path": "auth.py"
  }
}
```

### Index Codebase

#### POST /mcp/index

Index a new codebase or reindex an existing one.

**Request Body:**
```json
{
  "path": "/path/to/codebase",
  "languages": ["python", "javascript"],
  "embedding_model": "ollama",
  "force_reindex": false
}
```

**Parameters:**
- `path` (string, required): Path to the codebase directory
- `languages` (array, optional): Programming languages to index (default: ["python"])
- `embedding_model` (string, optional): Embedding model to use
- `force_reindex` (boolean, optional): Force reindexing of existing files (default: false)

**Response:**
```json
{
  "message": "Indexing completed successfully",
  "total_files": 25,
  "total_chunks": 342,
  "embedding_model": "ollama"
}
```

## Data Models

### CodeChunk

Represents a chunk of code with metadata.

```json
{
  "id": "string",
  "content": "string",
  "file_path": "string", 
  "start_line": "integer",
  "end_line": "integer",
  "node_type": "function_definition|class_definition|method_definition|variable_definition|import_statement|module",
  "name": "string|null",
  "parent_id": "string|null",
  "calls": ["string"],
  "called_by": ["string"],
  "imports": ["string"],
  "metadata": {}
}
```

### QueryResult

Represents a search result with similarity score and context.

```json
{
  "chunk": "CodeChunk",
  "score": "float",
  "context_chunks": ["CodeChunk"]
}
```

### GraphNode

Represents a node in the code graph.

```json
{
  "id": "string",
  "label": "string",
  "type": "string",
  "file_path": "string",
  "properties": {}
}
```

### GraphEdge

Represents an edge in the code graph.

```json
{
  "source": "string",
  "target": "string", 
  "type": "parent_child|calls|called_by|imports|imported_by",
  "weight": "float",
  "properties": {}
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

### Example Error Response

```json
{
  "detail": "No response from server. Please check if the MCP server is running."
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, consider implementing rate limiting to prevent abuse.

## CORS

The server is configured to allow CORS requests from any origin during development. Configure this appropriately for production.

## OpenAPI Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Examples

### Python Client Example

```python
import requests

# Query the codebase
response = requests.post('http://localhost:8000/mcp/query', json={
    'query': 'find database connection functions',
    'limit': 5
})

results = response.json()
for result in results['results']:
    print(f"Found: {result['chunk']['name']} (score: {result['score']:.2f})")
```

### JavaScript Client Example

```javascript
// Query the codebase
const response = await fetch('http://localhost:8000/mcp/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'find error handling code',
    limit: 10
  })
});

const results = await response.json();
console.log(`Found ${results.total_results} results`);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/mcp/status

# Query codebase
curl -X POST http://localhost:8000/mcp/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find authentication functions", "limit": 5}'

# Get graph data
curl "http://localhost:8000/mcp/graph?limit=100"

# Index codebase
curl -X POST http://localhost:8000/mcp/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/codebase", "force_reindex": true}'
```
