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


class AIModelConfig(BaseSettings):
    """AI model configuration settings."""
    
    # Cloud AI models
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    huggingface_api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    xai_api_key: Optional[str] = Field(default=None, env="XAI_API_KEY")
    
    # Local AI models (Ollama)
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="codegemma", env="OLLAMA_MODEL")
    
    # Default model configuration
    default_embedding_model: str = Field(default="local", env="DEFAULT_EMBEDDING_MODEL")
    default_cloud_model: str = Field(default="openai", env="DEFAULT_CLOUD_MODEL")
    default_local_model: str = Field(default="codegemma", env="DEFAULT_LOCAL_MODEL")


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
