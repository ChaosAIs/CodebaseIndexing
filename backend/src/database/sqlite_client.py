"""SQLite database client for project management."""

import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from ..models import Project, ProjectCreate, ProjectUpdate, ProjectStatus
from ..config import config


class ProjectManager:
    """SQLite database client for project management."""
    
    def __init__(self, db_path: str = None):
        """Initialize SQLite client."""
        self.db_path = db_path or config.database.sqlite_path
        self.ensure_db_directory()
        self.initialize_database()
    
    def ensure_db_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def initialize_database(self):
        """Initialize database schema."""
        try:
            with self.get_connection() as conn:
                # Create projects table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS projects (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        description TEXT,
                        source_path TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        indexed_at TIMESTAMP,
                        status TEXT DEFAULT 'not_indexed',
                        total_files INTEGER DEFAULT 0,
                        total_chunks INTEGER DEFAULT 0,
                        embedding_model TEXT,
                        indexing_error TEXT
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at)")
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    async def create_project(self, project_data: ProjectCreate) -> Project:
        """Create a new project."""
        try:
            project_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO projects (id, name, description, source_path, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    project_data.name,
                    project_data.description,
                    project_data.source_path,
                    now,
                    now
                ))
                conn.commit()
            
            return await self.get_project(project_id)
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Project with name '{project_data.name}' already exists")
            raise
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise
    
    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
                row = cursor.fetchone()
                
                if row:
                    return Project(**dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            raise
    
    async def get_project_by_name(self, name: str) -> Optional[Project]:
        """Get project by name."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT * FROM projects WHERE name = ?", (name,))
                row = cursor.fetchone()
                
                if row:
                    return Project(**dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Failed to get project by name {name}: {e}")
            raise
    
    async def list_projects(self, skip: int = 0, limit: int = 100) -> List[Project]:
        """List all projects with pagination."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM projects 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, skip))
                rows = cursor.fetchall()
                
                return [Project(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise
    
    async def update_project(self, project_id: str, project_data: ProjectUpdate) -> Optional[Project]:
        """Update project."""
        try:
            now = datetime.utcnow()
            update_fields = []
            values = []
            
            if project_data.name is not None:
                update_fields.append("name = ?")
                values.append(project_data.name)
            
            if project_data.description is not None:
                update_fields.append("description = ?")
                values.append(project_data.description)
            
            if project_data.source_path is not None:
                update_fields.append("source_path = ?")
                values.append(project_data.source_path)
            
            if not update_fields:
                return await self.get_project(project_id)
            
            update_fields.append("updated_at = ?")
            values.append(now)
            values.append(project_id)
            
            with self.get_connection() as conn:
                cursor = conn.execute(f"""
                    UPDATE projects 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """, values)
                
                if cursor.rowcount == 0:
                    return None
                
                conn.commit()
            
            return await self.get_project(project_id)
            
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise ValueError(f"Project with name '{project_data.name}' already exists")
            raise
        except Exception as e:
            logger.error(f"Failed to update project {project_id}: {e}")
            raise
    
    async def delete_project(self, project_id: str) -> bool:
        """Delete project."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            raise
    
    async def update_project_indexing_status(
        self, 
        project_id: str, 
        status: ProjectStatus, 
        total_files: int = None,
        total_chunks: int = None,
        embedding_model: str = None,
        error: str = None
    ) -> bool:
        """Update project indexing status."""
        try:
            now = datetime.utcnow()
            update_fields = ["status = ?", "updated_at = ?"]
            values = [status.value, now]
            
            if status == ProjectStatus.INDEXED:
                update_fields.append("indexed_at = ?")
                values.append(now)
            
            if total_files is not None:
                update_fields.append("total_files = ?")
                values.append(total_files)
            
            if total_chunks is not None:
                update_fields.append("total_chunks = ?")
                values.append(total_chunks)
            
            if embedding_model is not None:
                update_fields.append("embedding_model = ?")
                values.append(embedding_model)
            
            if error is not None:
                update_fields.append("indexing_error = ?")
                values.append(error)
            elif status != ProjectStatus.ERROR:
                update_fields.append("indexing_error = ?")
                values.append(None)
            
            values.append(project_id)
            
            with self.get_connection() as conn:
                cursor = conn.execute(f"""
                    UPDATE projects 
                    SET {', '.join(update_fields)}
                    WHERE id = ?
                """, values)
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update project indexing status {project_id}: {e}")
            raise
    
    async def get_projects_by_status(self, status: ProjectStatus) -> List[Project]:
        """Get projects by status."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC", 
                    (status.value,)
                )
                rows = cursor.fetchall()
                
                return [Project(**dict(row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get projects by status {status}: {e}")
            raise
    
    async def get_project_count(self) -> int:
        """Get total project count."""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM projects")
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get project count: {e}")
            raise
