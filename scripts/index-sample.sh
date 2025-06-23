#!/bin/bash

# Sample indexing script for testing the solution

set -e

echo "📚 Indexing sample codebase..."

# Check if backend is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Backend server is not running. Please start it first with ./scripts/start.sh"
    exit 1
fi

# Create a sample Python project for testing
SAMPLE_DIR="sample_codebase"

if [ ! -d "$SAMPLE_DIR" ]; then
    echo "📁 Creating sample codebase..."
    mkdir -p $SAMPLE_DIR
    
    # Create sample Python files
    cat > $SAMPLE_DIR/main.py << 'EOF'
"""Main application module."""

from auth import authenticate_user
from database import DatabaseManager
from utils import log_message

class Application:
    """Main application class."""
    
    def __init__(self):
        """Initialize the application."""
        self.db = DatabaseManager()
        self.authenticated = False
    
    def start(self):
        """Start the application."""
        log_message("Starting application...")
        
        if self.login():
            self.run_main_loop()
        else:
            log_message("Authentication failed")
    
    def login(self):
        """Handle user login."""
        username = input("Username: ")
        password = input("Password: ")
        
        self.authenticated = authenticate_user(username, password)
        return self.authenticated
    
    def run_main_loop(self):
        """Run the main application loop."""
        while True:
            command = input("Enter command: ")
            if command == "quit":
                break
            self.process_command(command)
    
    def process_command(self, command):
        """Process user commands."""
        if command == "list":
            self.list_items()
        elif command == "add":
            self.add_item()
        else:
            log_message(f"Unknown command: {command}")
    
    def list_items(self):
        """List all items."""
        items = self.db.get_all_items()
        for item in items:
            print(f"- {item}")
    
    def add_item(self):
        """Add a new item."""
        name = input("Item name: ")
        self.db.add_item(name)
        log_message(f"Added item: {name}")

if __name__ == "__main__":
    app = Application()
    app.start()
EOF

    cat > $SAMPLE_DIR/auth.py << 'EOF'
"""Authentication module."""

import hashlib
from database import DatabaseManager

def authenticate_user(username, password):
    """Authenticate a user with username and password."""
    db = DatabaseManager()
    
    # Hash the password
    password_hash = hash_password(password)
    
    # Check against database
    user = db.get_user(username)
    if user and user['password_hash'] == password_hash:
        return True
    
    return False

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    """Create a new user."""
    db = DatabaseManager()
    password_hash = hash_password(password)
    
    return db.create_user(username, password_hash)

def change_password(username, old_password, new_password):
    """Change user password."""
    if authenticate_user(username, old_password):
        db = DatabaseManager()
        new_hash = hash_password(new_password)
        return db.update_user_password(username, new_hash)
    
    return False
EOF

    cat > $SAMPLE_DIR/database.py << 'EOF'
"""Database management module."""

import sqlite3
from typing import List, Dict, Optional
from utils import log_message

class DatabaseManager:
    """Manages database operations."""
    
    def __init__(self, db_path="app.db"):
        """Initialize database manager."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            log_message("Database initialized")
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'password_hash': row[2],
                    'created_at': row[3]
                }
            return None
    
    def create_user(self, username: str, password_hash: str) -> bool:
        """Create a new user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, password_hash)
                )
                conn.commit()
                log_message(f"User created: {username}")
                return True
        except sqlite3.IntegrityError:
            log_message(f"User already exists: {username}")
            return False
    
    def update_user_password(self, username: str, new_password_hash: str) -> bool:
        """Update user password."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (new_password_hash, username)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def get_all_items(self) -> List[str]:
        """Get all items."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM items ORDER BY created_at")
            return [row[0] for row in cursor.fetchall()]
    
    def add_item(self, name: str) -> bool:
        """Add a new item."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO items (name) VALUES (?)", (name,))
            conn.commit()
            return True
    
    def delete_item(self, name: str) -> bool:
        """Delete an item."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM items WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
EOF

    cat > $SAMPLE_DIR/utils.py << 'EOF'
"""Utility functions."""

import datetime
from typing import Any

def log_message(message: str, level: str = "INFO"):
    """Log a message with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def format_error(error: Exception) -> str:
    """Format an error message."""
    return f"Error: {type(error).__name__}: {str(error)}"

def validate_input(value: Any, expected_type: type) -> bool:
    """Validate input type."""
    return isinstance(value, expected_type)

def safe_divide(a: float, b: float) -> float:
    """Safely divide two numbers."""
    if b == 0:
        log_message("Division by zero attempted", "WARNING")
        return 0.0
    return a / b

def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate a string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
EOF

    echo "✅ Sample codebase created in $SAMPLE_DIR/"
fi

# Index the sample codebase
echo "🔍 Indexing sample codebase..."
cd backend

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the indexer
python indexer.py --path "../$SAMPLE_DIR" --model local --force

echo ""
echo "🎉 Sample codebase indexed successfully!"
echo ""
echo "💡 Try these sample queries in the chat interface:"
echo "   - 'find authentication functions'"
echo "   - 'show me database operations'"
echo "   - 'find error handling code'"
echo "   - 'show me the main application class'"
echo ""
echo "🌐 Open http://localhost:3000 to start querying!"
