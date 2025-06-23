#!/bin/bash

# Codebase Indexing Solution - Start Script

set -e

echo "🚀 Starting Codebase Indexing Solution..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start databases
echo "📊 Starting databases..."
docker-compose up -d

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
sleep 10

# Check if databases are healthy
echo "🔍 Checking database health..."

# Check Qdrant
if curl -s http://localhost:6333/collections > /dev/null; then
    echo "✅ Qdrant is ready"
else
    echo "❌ Qdrant is not ready"
    exit 1
fi

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j is ready"
else
    echo "❌ Neo4j is not ready"
    exit 1
fi

# Start backend server
echo "🖥️  Starting MCP server..."
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
sleep 5

# Check backend health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend server is ready"
else
    echo "❌ Backend server is not ready"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend
echo "🌐 Starting frontend..."
cd ../frontend
npm start &
FRONTEND_PID=$!

echo "✅ All services started successfully!"
echo ""
echo "🔗 Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   Neo4j Browser: http://localhost:7474"
echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "📝 To index a codebase, run:"
echo "   cd backend && python indexer.py --path /path/to/your/codebase"
echo ""
echo "🛑 To stop all services, press Ctrl+C or run: ./scripts/stop.sh"

# Wait for user interrupt
trap 'echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; docker-compose down; exit 0' INT

wait
