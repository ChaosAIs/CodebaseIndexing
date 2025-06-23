# Codebase Indexing Solution

A comprehensive Python-based system to index codebases, parse them with Tree-sitter, chunk code while preserving hierarchical and relational context, embed chunks in a Qdrant vector database, and leverage Neo4j for Graph RAG with bidirectional pointers.

## Features

- **Tree-sitter Parsing**: Parse source code into ASTs, extracting functions, classes, and methods
- **Structured Chunking**: Create chunks with metadata preserving parent-child and caller-callee relationships
- **Embedding Generation**: Support both cloud-based AI models and local AI models (Ollama)
- **Qdrant Vector Database**: Store embeddings and metadata for fast similarity search
- **Neo4j Graph Database**: Store chunk relationships for Graph RAG and visualization
- **Model Context Protocol (MCP) Server**: Standardized API endpoint for LLMs
- **React Chat Frontend**: Interactive chat interface and graph visualization

## Architecture

```
[Codebase Files] → [Tree-sitter Parser] → [Chunking] → [Embeddings] 
    ↓
[Qdrant Vector DB] + [Neo4j Graph DB] → [MCP Server] → [React Frontend]
```

## Quick Start

1. **Setup Environment**:
   ```bash
   # Backend setup
   cd backend
   pip install -r requirements.txt
   
   # Frontend setup
   cd ../frontend
   npm install
   ```

2. **Start Services**:
   ```bash
   # Start databases
   docker-compose up -d
   
   # Start MCP server
   cd backend
   python -m uvicorn main:app --reload
   
   # Start frontend
   cd ../frontend
   npm start
   ```

3. **Index a Codebase**:
   ```bash
   cd backend
   python indexer.py --path /path/to/codebase
   ```

## Project Structure

```
├── backend/                 # Python backend
│   ├── src/
│   │   ├── parser/         # Tree-sitter parsing
│   │   ├── chunking/       # Code chunking logic
│   │   ├── embeddings/     # Embedding generation
│   │   ├── database/       # Qdrant & Neo4j clients
│   │   └── mcp_server/     # FastAPI MCP server
│   ├── requirements.txt
│   └── main.py
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   └── utils/          # Utilities
│   ├── package.json
│   └── public/
├── docker-compose.yml      # Database services
├── .env.example           # Environment variables template
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and configure:

- **Cloud AI Models**: API keys for embedding services
- **Local AI Models**: Ollama configuration
- **Database**: Qdrant and Neo4j connection settings

## Usage

1. **Index Codebase**: Use the indexer to parse and store code
2. **Query via MCP**: Use the MCP server endpoints for programmatic access
3. **Interactive Chat**: Use the React frontend for interactive queries and visualization

## API Endpoints

- `POST /mcp/query`: Query the codebase with natural language
- `GET /mcp/graph`: Retrieve graph data for visualization
- `GET /mcp/status`: Check system and model status

## License

MIT License
