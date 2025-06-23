"""Main entry point for the MCP server."""

import uvicorn
from loguru import logger
from src.config import config
from src.mcp_server.server import app

if __name__ == "__main__":
    logger.info("Starting Codebase Indexing MCP Server")
    logger.info(f"Server configuration: {config.server.host}:{config.server.port}")
    
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        reload=True
    )
