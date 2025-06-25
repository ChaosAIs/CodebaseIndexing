#!/usr/bin/env python3
"""Test script to start the MCP server."""

import sys
import os
import uvicorn

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting MCP server test...")
print(f"Python path: {sys.path[0]}")
print(f"Current directory: {os.getcwd()}")

try:
    from src.mcp_server.server import app
    print("Successfully imported app")
    
    print("Starting uvicorn server on 0.0.0.0:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
