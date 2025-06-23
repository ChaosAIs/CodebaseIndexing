#!/bin/bash

# Codebase Indexing Solution - Stop Script

echo "🛑 Stopping Codebase Indexing Solution..."

# Kill backend processes
echo "🖥️  Stopping backend server..."
pkill -f "uvicorn main:app" || true

# Kill frontend processes
echo "🌐 Stopping frontend..."
pkill -f "npm start" || true
pkill -f "react-scripts start" || true

# Stop Docker containers
echo "📊 Stopping databases..."
docker-compose down

echo "✅ All services stopped successfully!"
