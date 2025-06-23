#!/usr/bin/env python3
"""Script to check detailed project information."""

import sqlite3
import sys
import os
from pathlib import Path

def check_project_details(project_id):
    """Check detailed information about a project."""
    db_path = Path("data/projects.db")
    
    if not db_path.exists():
        print("Database file not found!")
        return
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM projects WHERE id = ?
            """, (project_id,))
            
            project = cursor.fetchone()
            
            if not project:
                print(f"Project with ID {project_id} not found!")
                return
            
            print("Project Details:")
            print("-" * 50)
            for key in project.keys():
                print(f"{key}: {project[key]}")
            
            print("\nSource Path Analysis:")
            print("-" * 50)
            source_path = project['source_path']
            print(f"Source Path: {source_path}")
            
            if os.path.exists(source_path):
                print("✅ Path exists")
                if os.path.isdir(source_path):
                    print("✅ Path is a directory")
                    
                    # Count files in directory
                    try:
                        files = list(Path(source_path).rglob("*"))
                        total_files = len([f for f in files if f.is_file()])
                        print(f"📁 Total files in directory: {total_files}")
                        
                        # Check for common code file extensions
                        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb'}
                        code_files = [f for f in files if f.is_file() and f.suffix.lower() in code_extensions]
                        print(f"💻 Code files found: {len(code_files)}")
                        
                        if code_files:
                            print("Sample code files:")
                            for f in code_files[:5]:  # Show first 5
                                print(f"  - {f.relative_to(source_path)}")
                            if len(code_files) > 5:
                                print(f"  ... and {len(code_files) - 5} more")
                        
                    except Exception as e:
                        print(f"❌ Error analyzing directory: {e}")
                else:
                    print("❌ Path is not a directory")
            else:
                print("❌ Path does not exist")
                
    except Exception as e:
        print(f"Error checking project details: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_project_details.py <project_id>")
        print("Example: python check_project_details.py 8c97311d-5b1e-428d-a7fa-7086f69735ca")
        sys.exit(1)
    
    project_id = sys.argv[1]
    check_project_details(project_id)
