#!/usr/bin/env python3
"""Minimal test server to isolate issues."""

import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Test Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test_endpoint():
    return {"status": "ok", "message": "Test server is working"}

@app.post("/test-query")
async def test_query(request: dict):
    print(f"Received query: {request}")
    
    # Test the intelligent query analyzer
    try:
        from src.query.intelligent_query_analyzer import IntelligentQueryAnalyzer
        analyzer = IntelligentQueryAnalyzer()
        result = await analyzer.analyze_query(request.get("query", "test"), 100)
        
        return {
            "query": request.get("query"),
            "analysis": {
                "complexity": result.complexity.value,
                "strategy": result.processing_strategy.value,
                "should_stream": result.should_stream,
                "estimated_time": result.estimated_processing_time,
                "agents_count": len(result.required_agents),
                "explanation": result.explanation
            }
        }
    except Exception as e:
        print(f"Error in query analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    print("Starting minimal test server on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
