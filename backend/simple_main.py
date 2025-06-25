"""Simple main entry point for testing the MCP server without complex dependencies."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from src.config import config

# Create a simple FastAPI app
app = FastAPI(title="Codebase Indexing API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Codebase Indexing API is running!", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "databases": {"qdrant": "connected", "neo4j": "connected"}}

@app.get("/api/projects")
async def get_projects():
    """Get all projects."""
    return {"projects": []}

@app.post("/api/search")
async def search(request: dict):
    """Simple search endpoint."""
    return {
        "results": [],
        "metadata": {
            "total": 0,
            "query": request.get("query", ""),
            "processing_time": 0.1
        }
    }

@app.post("/api/flow")
async def flow(request: dict):
    """Simple flow endpoint."""
    return {
        "sections": [
            {
                "title": "Overview",
                "content": f"This is a simple response to your query: {request.get('query', '')}",
                "type": "overview"
            }
        ],
        "metadata": {
            "processing_time": 0.1,
            "confidence": 0.8
        }
    }

if __name__ == "__main__":
    logger.info("Starting Simple Codebase Indexing Server")
    logger.info(f"Server configuration: {config.server.host}:{config.server.port}")
    
    uvicorn.run(
        "simple_main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        reload=True
    )
