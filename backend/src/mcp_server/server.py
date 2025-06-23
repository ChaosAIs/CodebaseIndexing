"""FastAPI-based Model Context Protocol (MCP) server."""

import time
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..models import (
    QueryRequest, QueryResponse, QueryResult, GraphData, SystemStatus,
    IndexingRequest, EmbeddingModel, CodeAnalysis, Project, ProjectCreate,
    ProjectUpdate, ProjectIndexRequest, ProjectStatus
)
from ..database.qdrant_client import QdrantVectorStore
from ..database.neo4j_client import Neo4jGraphStore
from ..database.sqlite_client import ProjectManager
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..chunking.chunk_processor import ChunkProcessor
from ..analysis.code_analyzer import CodeAnalyzer
from ..config import config


class MCPServer:
    """Model Context Protocol server for codebase querying."""
    
    def __init__(self):
        """Initialize MCP server."""
        self.app = FastAPI(
            title="Codebase Indexing MCP Server",
            description="Model Context Protocol server for codebase knowledge retrieval",
            version="1.0.0"
        )
        
        # Initialize components
        self.vector_store = QdrantVectorStore()
        self.graph_store = Neo4jGraphStore()
        self.project_manager = ProjectManager()
        self.embedding_generator = EmbeddingGenerator()
        self.chunk_processor = ChunkProcessor()
        self.code_analyzer = CodeAnalyzer()
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/mcp/query", response_model=QueryResponse)
        async def query_codebase(request: QueryRequest):
            """Query the codebase with natural language or code snippets."""
            start_time = time.time()
            
            try:
                logger.info(f"Processing query: {request.query}")
                
                # Generate query embedding
                query_embedding = await self._generate_query_embedding(request.query, request.model)
                
                # Search similar chunks in Qdrant with project filtering
                similar_chunks = await self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    limit=request.limit,
                    project_ids=request.project_ids
                )

                # Get comprehensive graph context for analysis
                all_graph_context = {}
                if request.include_context and similar_chunks:
                    # Get context for top chunks
                    for chunk, _ in similar_chunks[:3]:  # Top 3 for context
                        context = await self.graph_store.get_chunk_context(chunk.id)
                        for context_type, chunks in context.items():
                            if context_type not in all_graph_context:
                                all_graph_context[context_type] = []
                            all_graph_context[context_type].extend(chunks)

                # Generate intelligent analysis
                analysis = None
                if similar_chunks:
                    analysis_data = await self.code_analyzer.analyze_query_results(
                        query=request.query,
                        vector_results=similar_chunks,
                        graph_context=all_graph_context
                    )

                    analysis = CodeAnalysis(
                        summary=analysis_data.get("summary", ""),
                        detailed_explanation=analysis_data.get("detailed_explanation", ""),
                        code_flow=analysis_data.get("code_flow", []),
                        key_components=analysis_data.get("key_components", []),
                        relationships=analysis_data.get("relationships", []),
                        recommendations=analysis_data.get("recommendations", [])
                    )

                # Build results with context
                results = []
                for chunk, score in similar_chunks:
                    context_chunks = []

                    if request.include_context:
                        # Get contextual chunks from Neo4j for this specific chunk
                        context = await self.graph_store.get_chunk_context(chunk.id)

                        # Combine all context types
                        for context_type, chunks in context.items():
                            context_chunks.extend(chunks)

                    result = QueryResult(
                        chunk=chunk,
                        score=score,
                        context_chunks=context_chunks
                    )
                    results.append(result)
                
                processing_time = time.time() - start_time
                
                response = QueryResponse(
                    query=request.query,
                    results=results,
                    total_results=len(results),
                    model_used=request.model or config.ai_models.default_embedding_model,
                    processing_time=processing_time,
                    analysis=analysis
                )
                
                logger.info(f"Query processed in {processing_time:.2f}s, found {len(results)} results")
                return response
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/graph", response_model=GraphData)
        async def get_graph_data(
            file_path: Optional[str] = Query(None, description="Filter by file path"),
            project_ids: Optional[str] = Query(None, description="Comma-separated project IDs to filter by"),
            limit: int = Query(1000, description="Maximum number of nodes")
        ):
            """Get graph data for visualization."""
            try:
                # Parse project IDs if provided
                project_id_list = None
                if project_ids:
                    project_id_list = [pid.strip() for pid in project_ids.split(',') if pid.strip()]

                logger.info(f"Getting graph data for file: {file_path}, projects: {project_id_list}")

                graph_data = await self.graph_store.get_graph_data(
                    file_path=file_path,
                    project_ids=project_id_list,
                    limit=limit
                )

                logger.info(f"Retrieved graph with {len(graph_data.nodes)} nodes and {len(graph_data.edges)} edges")
                return graph_data

            except Exception as e:
                logger.error(f"Error getting graph data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/status", response_model=SystemStatus)
        async def get_system_status():
            """Get system status and health information."""
            try:
                # Check database health
                qdrant_healthy = await self.vector_store.health_check()
                neo4j_healthy = await self.graph_store.health_check()
                
                # Get available models
                available_providers = await self.embedding_generator.get_available_providers()
                available_models = [name for name, available in available_providers.items() if available]
                
                # Get statistics
                qdrant_info = await self.vector_store.get_collection_info()
                neo4j_stats = await self.graph_store.get_statistics()
                
                status = SystemStatus(
                    qdrant_status="healthy" if qdrant_healthy else "unhealthy",
                    neo4j_status="healthy" if neo4j_healthy else "unhealthy",
                    ollama_status="available" if "ollama" in available_models else "unavailable",
                    available_models=available_models,
                    indexed_files=neo4j_stats.get("total_files", 0),
                    total_chunks=neo4j_stats.get("total_chunks", 0)
                )
                
                return status
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/index")
        async def index_codebase(request: IndexingRequest):
            """Index a codebase."""
            try:
                logger.info(f"Starting indexing of: {request.path}")
                
                # Process codebase
                all_chunks = self.chunk_processor.process_codebase(request.path)
                
                # Flatten chunks
                chunks = []
                for file_chunks in all_chunks.values():
                    chunks.extend(file_chunks)
                
                if not chunks:
                    raise HTTPException(status_code=400, detail="No code chunks found")
                
                # Generate embeddings
                provider_name = request.embedding_model.value if request.embedding_model else None
                embeddings = await self.embedding_generator.generate_chunk_embeddings(
                    chunks, provider_name
                )
                
                # Get embedding dimension
                dimension = self.embedding_generator.get_embedding_dimension(provider_name)
                
                # Initialize databases
                await self.vector_store.initialize_collection(dimension, request.force_reindex)
                await self.graph_store.initialize_schema()
                
                # Store in databases
                await self.vector_store.store_chunks(chunks, embeddings)
                await self.graph_store.store_chunks(chunks)
                await self.graph_store.create_relationships(chunks)
                
                logger.info(f"Successfully indexed {len(chunks)} chunks from {len(all_chunks)} files")
                
                return {
                    "message": "Indexing completed successfully",
                    "total_files": len(all_chunks),
                    "total_chunks": len(chunks),
                    "embedding_model": provider_name or config.ai_models.default_embedding_model
                }
                
            except Exception as e:
                logger.error(f"Error indexing codebase: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "healthy", "service": "MCP Server"}

        # File System Endpoints
        @self.app.get("/filesystem/browse")
        async def browse_directory(
            path: str = Query("", description="Directory path to browse"),
            show_hidden: bool = Query(False, description="Show hidden files and directories")
        ):
            """Browse directory contents for folder picker."""
            import os
            import platform
            from pathlib import Path

            try:
                # If no path provided, return system drives/root directories
                if not path:
                    if platform.system() == "Windows":
                        # Return available drives on Windows
                        drives = []
                        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            drive_path = f"{letter}:\\"
                            if os.path.exists(drive_path):
                                drives.append({
                                    "name": f"{letter}:",
                                    "path": drive_path,
                                    "type": "drive",
                                    "is_directory": True
                                })
                        return {"items": drives, "current_path": ""}
                    else:
                        # Return root directory on Unix-like systems
                        path = "/"

                # Validate and normalize path
                try:
                    path_obj = Path(path).resolve()
                    if not path_obj.exists():
                        raise HTTPException(status_code=404, detail="Directory not found")
                    if not path_obj.is_dir():
                        raise HTTPException(status_code=400, detail="Path is not a directory")
                except (OSError, ValueError) as e:
                    raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")

                # Get directory contents
                items = []
                try:
                    for item in path_obj.iterdir():
                        # Skip hidden files unless requested
                        if not show_hidden and item.name.startswith('.'):
                            continue

                        # Skip system files on Windows
                        if platform.system() == "Windows" and item.name.lower() in ['$recycle.bin', 'system volume information']:
                            continue

                        try:
                            is_dir = item.is_dir()
                            items.append({
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if is_dir else "file",
                                "is_directory": is_dir
                            })
                        except (OSError, PermissionError):
                            # Skip items we can't access
                            continue

                    # Sort items: directories first, then files, both alphabetically
                    items.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))

                    # Add parent directory option if not at root
                    parent_path = path_obj.parent
                    if str(parent_path) != str(path_obj):
                        items.insert(0, {
                            "name": "..",
                            "path": str(parent_path),
                            "type": "parent",
                            "is_directory": True
                        })

                    return {
                        "items": items,
                        "current_path": str(path_obj)
                    }

                except PermissionError:
                    raise HTTPException(status_code=403, detail="Permission denied")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading directory: {str(e)}")

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error browsing directory: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Project Management Endpoints
        @self.app.post("/projects", response_model=Project)
        async def create_project(project_data: ProjectCreate):
            """Create a new project."""
            try:
                logger.info(f"Creating project: {project_data.name}")
                project = await self.project_manager.create_project(project_data)
                logger.info(f"Created project: {project.id}")
                return project
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Error creating project: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/projects", response_model=List[Project])
        async def list_projects(
            skip: int = Query(0, description="Number of projects to skip"),
            limit: int = Query(100, description="Maximum number of projects to return")
        ):
            """List all projects."""
            try:
                projects = await self.project_manager.list_projects(skip=skip, limit=limit)
                return projects
            except Exception as e:
                logger.error(f"Error listing projects: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/projects/{project_id}", response_model=Project)
        async def get_project(project_id: str):
            """Get project by ID."""
            try:
                project = await self.project_manager.get_project(project_id)
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")
                return project
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting project {project_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.put("/projects/{project_id}", response_model=Project)
        async def update_project(project_id: str, project_data: ProjectUpdate):
            """Update project."""
            try:
                project = await self.project_manager.update_project(project_id, project_data)
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")
                logger.info(f"Updated project: {project_id}")
                return project
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating project {project_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/projects/{project_id}")
        async def delete_project(project_id: str):
            """Delete project."""
            try:
                success = await self.project_manager.delete_project(project_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Project not found")
                logger.info(f"Deleted project: {project_id}")
                return {"message": "Project deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting project {project_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/projects/{project_id}/index")
        async def index_project(project_id: str, request: ProjectIndexRequest):
            """Index a project's codebase."""
            try:
                # Get project
                project = await self.project_manager.get_project(project_id)
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")

                logger.info(f"Starting indexing of project: {project.name} at {project.source_path}")

                # Update project status to indexing
                await self.project_manager.update_project_indexing_status(
                    project_id, ProjectStatus.INDEXING
                )

                try:
                    # Process codebase
                    all_chunks = self.chunk_processor.process_codebase(project.source_path)

                    # Flatten chunks and add project_id
                    chunks = []
                    for file_chunks in all_chunks.values():
                        for chunk in file_chunks:
                            chunk.project_id = project_id
                            chunks.append(chunk)

                    if not chunks:
                        await self.project_manager.update_project_indexing_status(
                            project_id, ProjectStatus.ERROR, error="No code chunks found"
                        )
                        raise HTTPException(status_code=400, detail="No code chunks found")

                    # Generate embeddings
                    provider_name = request.embedding_model.value if request.embedding_model else None
                    embeddings = await self.embedding_generator.generate_chunk_embeddings(
                        chunks, provider_name
                    )

                    # Get embedding dimension
                    dimension = self.embedding_generator.get_embedding_dimension(provider_name)

                    # Initialize databases
                    await self.vector_store.initialize_collection(dimension, request.force_reindex)
                    await self.graph_store.initialize_schema()

                    # Store in databases
                    await self.vector_store.store_chunks(chunks, embeddings)
                    await self.graph_store.store_chunks(chunks)
                    await self.graph_store.create_relationships(chunks)

                    # Update project status to indexed
                    await self.project_manager.update_project_indexing_status(
                        project_id,
                        ProjectStatus.INDEXED,
                        total_files=len(all_chunks),
                        total_chunks=len(chunks),
                        embedding_model=provider_name or config.ai_models.default_embedding_model
                    )

                    logger.info(f"Successfully indexed project {project.name}: {len(chunks)} chunks from {len(all_chunks)} files")

                    return {
                        "message": "Project indexing completed successfully",
                        "project_id": project_id,
                        "project_name": project.name,
                        "total_files": len(all_chunks),
                        "total_chunks": len(chunks),
                        "embedding_model": provider_name or config.ai_models.default_embedding_model
                    }

                except Exception as indexing_error:
                    # Update project status to error
                    await self.project_manager.update_project_indexing_status(
                        project_id, ProjectStatus.ERROR, error=str(indexing_error)
                    )
                    raise indexing_error

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error indexing project {project_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_query_embedding(self, query: str, model: Optional[EmbeddingModel] = None) -> List[float]:
        """Generate embedding for a query."""
        provider_name = model.value if model else None
        
        # Create a dummy chunk for the query
        from ..models import CodeChunk, NodeType
        query_chunk = CodeChunk(
            id="query",
            content=query,
            file_path="query",
            start_line=1,
            end_line=1,
            node_type=NodeType.FUNCTION,
            name="query"
        )
        
        embeddings = await self.embedding_generator.generate_chunk_embeddings(
            [query_chunk], provider_name
        )
        
        return embeddings["query"]
    
    async def startup(self):
        """Startup tasks."""
        try:
            # Initialize database connections
            await self.graph_store.initialize_schema()
            logger.info("MCP Server started successfully")
        except Exception as e:
            logger.error(f"Error during startup: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown tasks."""
        try:
            self.graph_store.close()
            logger.info("MCP Server shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global server instance
mcp_server = MCPServer()
app = mcp_server.app


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event."""
    await mcp_server.startup()


@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event."""
    await mcp_server.shutdown()
