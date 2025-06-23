"""Data models for the codebase indexing solution."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """Types of code nodes."""
    FUNCTION = "function_definition"
    CLASS = "class_definition"
    METHOD = "method_definition"
    VARIABLE = "variable_definition"
    IMPORT = "import_statement"
    MODULE = "module"


class RelationshipType(str, Enum):
    """Types of relationships between code chunks."""
    PARENT_CHILD = "parent_child"
    CALLS = "calls"
    CALLED_BY = "called_by"
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"


class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="Raw code content")
    file_path: str = Field(..., description="Path to the source file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    node_type: NodeType = Field(..., description="Type of code node")
    name: Optional[str] = Field(None, description="Name of the function/class/method")
    parent_id: Optional[str] = Field(None, description="ID of parent chunk")
    calls: List[str] = Field(default_factory=list, description="IDs of chunks this chunk calls")
    called_by: List[str] = Field(default_factory=list, description="IDs of chunks that call this chunk")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingModel(str, Enum):
    """Available embedding models."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    XAI = "xai"
    OLLAMA_LOCAL = "ollama_local"


class QueryRequest(BaseModel):
    """Request model for querying the codebase."""
    
    query: str = Field(..., description="Natural language or code query")
    model: Optional[EmbeddingModel] = Field(None, description="Embedding model to use")
    limit: int = Field(default=10, description="Maximum number of results")
    include_context: bool = Field(default=True, description="Include related chunks in results")


class QueryResult(BaseModel):
    """Result model for codebase queries."""
    
    chunk: CodeChunk = Field(..., description="Primary matching chunk")
    score: float = Field(..., description="Similarity score")
    context_chunks: List[CodeChunk] = Field(default_factory=list, description="Related chunks")


class CodeAnalysis(BaseModel):
    """Code analysis with explanations."""

    summary: str = Field(..., description="Brief summary of the code analysis")
    detailed_explanation: str = Field(..., description="Detailed explanation of the code")
    code_flow: List[str] = Field(default_factory=list, description="Step-by-step code flow")
    key_components: List[Dict[str, str]] = Field(default_factory=list, description="Key code components")
    relationships: List[Dict[str, str]] = Field(default_factory=list, description="Component relationships")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations and insights")


class QueryResponse(BaseModel):
    """Response model for codebase queries."""

    query: str = Field(..., description="Original query")
    results: List[QueryResult] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")
    model_used: str = Field(..., description="Embedding model used")
    processing_time: float = Field(..., description="Query processing time in seconds")
    analysis: Optional[CodeAnalysis] = Field(None, description="Intelligent code analysis")


class GraphNode(BaseModel):
    """Graph node for visualization."""
    
    id: str = Field(..., description="Node ID")
    label: str = Field(..., description="Node label")
    type: NodeType = Field(..., description="Node type")
    file_path: str = Field(..., description="Source file path")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class GraphEdge(BaseModel):
    """Graph edge for visualization."""
    
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: RelationshipType = Field(..., description="Relationship type")
    weight: float = Field(default=1.0, description="Edge weight")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class GraphData(BaseModel):
    """Graph data for visualization."""
    
    nodes: List[GraphNode] = Field(..., description="Graph nodes")
    edges: List[GraphEdge] = Field(..., description="Graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")


class SystemStatus(BaseModel):
    """System status information."""
    
    qdrant_status: str = Field(..., description="Qdrant database status")
    neo4j_status: str = Field(..., description="Neo4j database status")
    ollama_status: str = Field(..., description="Ollama service status")
    available_models: List[str] = Field(..., description="Available embedding models")
    indexed_files: int = Field(..., description="Number of indexed files")
    total_chunks: int = Field(..., description="Total number of chunks")


class IndexingRequest(BaseModel):
    """Request model for indexing a codebase."""
    
    path: str = Field(..., description="Path to the codebase")
    languages: List[str] = Field(default=["python"], description="Programming languages to index")
    embedding_model: Optional[EmbeddingModel] = Field(None, description="Embedding model to use")
    force_reindex: bool = Field(default=False, description="Force reindexing of existing files")
