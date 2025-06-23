#!/usr/bin/env python3
"""Script to check project status in the database."""

import sqlite3
import sys
from pathlib import Path

def check_projects():
    """Check the status of projects in the database."""
    db_path = Path("data/projects.db")
    
    if not db_path.exists():
        print("Database file not found!")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, name, status, created_at, indexed_at, 
                       total_files, total_chunks, indexing_error 
                FROM projects 
                ORDER BY created_at DESC
            """)
            
            projects = cursor.fetchall()
            
            if not projects:
                print("No projects found in database.")
                return
            
            print(f"Found {len(projects)} projects:")
            print("-" * 80)
            
            for project in projects:
                print(f"ID: {project['id']}")
                print(f"Name: {project['name']}")
                print(f"Status: {project['status']}")
                print(f"Created: {project['created_at']}")
                print(f"Indexed: {project['indexed_at']}")
                print(f"Files: {project['total_files']}")
                print(f"Chunks: {project['total_chunks']}")
                if project['indexing_error']:
                    print(f"Error: {project['indexing_error']}")
                print("-" * 80)
                
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_projects()
