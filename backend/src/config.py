"""Configuration management for the codebase indexing solution."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    # Qdrant settings
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # Neo4j settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")

    # SQLite settings
    sqlite_path: str = Field(default="data/projects.db", env="SQLITE_PATH")


class AIModelConfig(BaseSettings):
    """AI model configuration settings."""

    # Cloud AI models
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    xai_api_key: Optional[str] = Field(default=None, env="XAI_API_KEY")

    # Local AI models (Ollama) - General settings
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="codegemma", env="OLLAMA_MODEL")

    # Ollama Embedding-specific settings (fallback to general settings if not specified)
    ollama_embedding_host: Optional[str] = Field(default=None, env="OLLAMA_EMBEDDING_HOST")
    ollama_embedding_model: Optional[str] = Field(default=None, env="OLLAMA_EMBEDDING_MODEL")
    ollama_embedding_api_key: Optional[str] = Field(default=None, env="OLLAMA_EMBEDDING_API_KEY")

    # Default model configuration
    default_embedding_model: str = Field(default="local", env="DEFAULT_EMBEDDING_MODEL")
    default_cloud_model: str = Field(default="openai", env="DEFAULT_CLOUD_MODEL")
    default_local_model: str = Field(default="codegemma", env="DEFAULT_LOCAL_MODEL")

    @property
    def effective_ollama_embedding_host(self) -> str:
        """Get the effective Ollama embedding host (embedding-specific or fallback to general)."""
        return self.ollama_embedding_host or self.ollama_host

    @property
    def effective_ollama_embedding_model(self) -> str:
        """Get the effective Ollama embedding model (embedding-specific or fallback to general)."""
        return self.ollama_embedding_model or self.ollama_model


class ServerConfig(BaseSettings):
    """Server configuration settings."""
    
    host: str = Field(default="0.0.0.0", env="MCP_SERVER_HOST")
    port: int = Field(default=8000, env="MCP_SERVER_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class IndexingConfig(BaseSettings):
    """Indexing configuration settings."""

    max_chunk_size: int = Field(default=1000, env="MAX_CHUNK_SIZE")
    overlap_size: int = Field(default=100, env="OVERLAP_SIZE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")

    # Default directories to exclude from indexing
    excluded_dirs: set = Field(default_factory=lambda: {
        'node_modules',      # JavaScript/TypeScript dependencies
        '__pycache__',       # Python bytecode cache
        '.git',              # Git repository data
        '.svn',              # SVN repository data
        '.hg',               # Mercurial repository data
        'venv',              # Python virtual environment
        '.venv',             # Python virtual environment
        'env',               # Python virtual environment
        '.env',              # Environment files directory
        'build',             # Build output
        'dist',              # Distribution files
        '.idea',             # JetBrains IDE files
        '.vscode',           # VS Code settings
        '.pytest_cache',     # Pytest cache
        '.mypy_cache',       # MyPy cache
        '.tox',              # Tox testing
        'coverage',          # Coverage reports
        '.coverage',         # Coverage data
        'htmlcov',           # Coverage HTML reports
        '.DS_Store',         # macOS system files
        'Thumbs.db',         # Windows system files
    })


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.ai_models = AIModelConfig()
        self.server = ServerConfig()
        self.indexing = IndexingConfig()
    
    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.database.qdrant_host}:{self.database.qdrant_port}"
    
    @property
    def neo4j_config(self) -> dict:
        """Get Neo4j connection configuration."""
        return {
            "uri": self.database.neo4j_uri,
            "auth": (self.database.neo4j_user, self.database.neo4j_password)
        }


# Global configuration instance
config = Config()
