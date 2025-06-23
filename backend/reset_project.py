#!/usr/bin/env python3
"""Script to reset a project's indexing status."""

import sqlite3
import sys
from pathlib import Path

def reset_project_status(project_id, new_status="not_indexed"):
    """Reset a project's indexing status."""
    db_path = Path("data/projects.db")
    
    if not db_path.exists():
        print("Database file not found!")
        return False
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            # First check if project exists
            cursor = conn.execute("SELECT name FROM projects WHERE id = ?", (project_id,))
            project = cursor.fetchone()
            
            if not project:
                print(f"Project with ID {project_id} not found!")
                return False
            
            print(f"Found project: {project[0]}")
            
            # Reset the status
            cursor = conn.execute("""
                UPDATE projects 
                SET status = ?, 
                    indexed_at = NULL, 
                    total_files = 0, 
                    total_chunks = 0, 
                    indexing_error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_status, project_id))
            
            conn.commit()
            
            if cursor.rowcount > 0:
                print(f"Successfully reset project status to '{new_status}'")
                return True
            else:
                print("Failed to update project status")
                return False
                
    except Exception as e:
        print(f"Error resetting project status: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reset_project.py <project_id>")
        print("Example: python reset_project.py 8c97311d-5b1e-428d-a7fa-7086f69735ca")
        sys.exit(1)
    
    project_id = sys.argv[1]
    reset_project_status(project_id)
