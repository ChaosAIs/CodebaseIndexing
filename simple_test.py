#!/usr/bin/env python3
"""Simple test to verify the environment."""

print("🚀 Testing environment...")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import uvicorn
    print("✅ Uvicorn imported successfully")
except ImportError as e:
    print(f"❌ Uvicorn import failed: {e}")

try:
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    def read_root():
        return {"message": "Hello World"}
    
    print("✅ FastAPI app created successfully")
    print("🌐 Starting server on http://localhost:8000")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
