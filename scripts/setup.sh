#!/bin/bash

# Codebase Indexing Solution - Setup Script

set -e

echo "🔧 Setting up Codebase Indexing Solution..."

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✅ Node.js $NODE_VERSION found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker."
    exit 1
fi

DOCKER_VERSION=$(docker --version)
echo "✅ Docker found: $DOCKER_VERSION"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

COMPOSE_VERSION=$(docker-compose --version)
echo "✅ Docker Compose found: $COMPOSE_VERSION"

# Setup backend
echo "🐍 Setting up Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete"

# Setup frontend
echo "🌐 Setting up React frontend..."
cd ../frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"

# Setup environment file
cd ..
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
else
    echo "✅ Environment file already exists"
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your configuration (API keys, etc.)"
echo "2. Start the services: ./scripts/start.sh"
echo "3. Index a codebase: cd backend && python indexer.py --path /path/to/codebase"
echo "4. Open http://localhost:3000 in your browser"
echo ""
echo "📚 For more information, see README.md"
