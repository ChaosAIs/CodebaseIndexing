"""FastAPI-based Model Context Protocol (MCP) server."""

import time
from typing import List, Dict, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..models import (
    QueryRequest, QueryResponse, QueryResult, GraphData, SystemStatus,
    IndexingRequest, EmbeddingModel, CodeAnalysis, Project, ProjectCreate,
    ProjectUpdate, ProjectIndexRequest, ProjectStatus, CodeChunk
)
from ..database.qdrant_client import QdrantVectorStore
from ..database.neo4j_client import Neo4jGraphStore
from ..database.sqlite_client import ProjectManager
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..chunking.chunk_processor import ChunkProcessor
from ..analysis.code_analyzer import CodeAnalyzer
from ..query.query_processor import QueryProcessor, QueryIntent
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
        self.query_processor = QueryProcessor()
        
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

                # Gather project context for enhanced processing
                project_context = await self._gather_project_context(request.project_ids)

                # Classify query intent and enhance for better embedding
                intent, confidence = self.query_processor.classify_query_intent(request.query)
                logger.info(f"Query intent: {intent.value} (confidence: {confidence:.2f})")

                # Extract code entities from query
                entities = self.query_processor.extract_code_entities(request.query)

                # Check if query is abstract and needs intelligent expansion
                is_abstract = self.query_processor.is_abstract_query(request.query)
                logger.info(f"Query is abstract: {is_abstract}")

                # Handle abstract queries with intelligent expansion
                search_queries = [request.query]
                if is_abstract:
                    logger.info("Expanding abstract query for better search results")
                    expanded_terms = await self.query_processor.expand_abstract_query(
                        request.query, project_context
                    )
                    search_queries.extend(expanded_terms[:6])  # Use top 6 expanded terms
                    logger.info(f"Expanded query into {len(search_queries)} search terms")

                # Perform multi-query search for comprehensive coverage
                all_similar_chunks = []
                for i, search_query in enumerate(search_queries):
                    # Enhance each search query for better embedding generation
                    enhanced_query = self.query_processor.enhance_query_for_embedding(
                        search_query, intent, project_context
                    )

                    # Generate search filters (use original query for consistency)
                    search_filters = self.query_processor.generate_search_filters(
                        request.query, intent, entities
                    )

                    # Generate query embedding
                    query_embedding = await self._generate_query_embedding(enhanced_query, request.model)

                    # Convert search filters to Qdrant format
                    qdrant_filters = self._convert_search_filters_to_qdrant(search_filters)

                    # Adjust limit per query to get comprehensive results
                    query_limit = request.limit if i == 0 else max(5, request.limit // len(search_queries))

                    # Search similar chunks
                    chunks = await self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        limit=query_limit,
                        project_ids=request.project_ids,
                        filters=qdrant_filters
                    )

                    # Add query source info for scoring
                    chunks_with_source = [(chunk, score, i) for chunk, score in chunks]
                    all_similar_chunks.extend(chunks_with_source)

                # Deduplicate and re-rank results with intelligent scoring
                similar_chunks = self._deduplicate_and_rerank_chunks(all_similar_chunks, request.limit, is_abstract)

                # Apply post-processing based on intent and entities
                similar_chunks = self._post_process_search_results(
                    similar_chunks, intent, entities, confidence
                )

                # Get comprehensive graph context for enhanced understanding
                all_graph_context = {}
                comprehensive_context = {}

                if request.include_context and similar_chunks:
                    # Get enhanced context for top chunks (increased from 3 to 5)
                    top_chunk_ids = [chunk.id for chunk, _ in similar_chunks[:5]]

                    # Get individual chunk contexts
                    for chunk, _ in similar_chunks[:5]:
                        context = await self.graph_store.get_chunk_context(chunk.id)
                        for context_type, chunks in context.items():
                            if context_type not in all_graph_context:
                                all_graph_context[context_type] = []
                            all_graph_context[context_type].extend(chunks)

                    # Get comprehensive architectural context
                    comprehensive_context = await self.graph_store.get_comprehensive_context(
                        top_chunk_ids, request.query
                    )

                # Gather project context for analysis
                project_context = await self._gather_project_context(request.project_ids)

                # Generate intelligent analysis with comprehensive context
                analysis = None
                if similar_chunks:
                    analysis_data = await self.code_analyzer.analyze_query_results(
                        query=request.query,
                        vector_results=similar_chunks,
                        graph_context=all_graph_context,
                        project_context=project_context,
                        comprehensive_context=comprehensive_context
                    )

                    analysis = CodeAnalysis(
                        summary=analysis_data.get("summary", ""),
                        detailed_explanation=analysis_data.get("detailed_explanation", ""),
                        code_flow=analysis_data.get("code_flow", []),
                        key_components=analysis_data.get("key_components", []),
                        relationships=analysis_data.get("relationships", []),
                        recommendations=analysis_data.get("recommendations", [])
                    )
                elif is_abstract:
                    # For abstract queries with no results, provide intelligent fallback analysis
                    analysis = await self._generate_abstract_fallback_analysis(
                        request.query, project_context, intent
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
                        # Return available drives on Windows, with common folders
                        drives = []

                        # Add user home directory as first option
                        user_home = os.path.expanduser("~")
                        if os.path.exists(user_home):
                            drives.append({
                                "name": "Home",
                                "path": user_home,
                                "type": "home",
                                "is_directory": True
                            })

                        # Add Desktop if it exists
                        desktop_path = os.path.join(user_home, "Desktop")
                        if os.path.exists(desktop_path):
                            drives.append({
                                "name": "Desktop",
                                "path": desktop_path,
                                "type": "desktop",
                                "is_directory": True
                            })

                        # Add Documents if it exists
                        documents_path = os.path.join(user_home, "Documents")
                        if os.path.exists(documents_path):
                            drives.append({
                                "name": "Documents",
                                "path": documents_path,
                                "type": "documents",
                                "is_directory": True
                            })

                        # Add available drives
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

    def _deduplicate_and_rerank_chunks(self, chunks_with_source: List[Tuple], limit: int, is_abstract: bool = False) -> List[Tuple]:
        """
        Deduplicate and re-rank chunks from multiple search queries.

        Args:
            chunks_with_source: List of (chunk, score, query_index) tuples
            limit: Maximum number of results to return
            is_abstract: Whether the original query was abstract

        Returns:
            List of (chunk, final_score) tuples
        """
        chunk_scores = {}

        for chunk, score, query_index in chunks_with_source:
            chunk_id = chunk.id

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "scores": [],
                    "query_indices": []
                }

            chunk_scores[chunk_id]["scores"].append(score)
            chunk_scores[chunk_id]["query_indices"].append(query_index)

        # Calculate final scores with intelligent weighting
        final_results = []
        for chunk_id, data in chunk_scores.items():
            chunk = data["chunk"]
            scores = data["scores"]
            query_indices = data["query_indices"]

            # Base score: maximum score across all queries
            base_score = max(scores)

            # Boost for appearing in multiple queries (indicates relevance)
            multi_query_boost = len(scores) * 0.1 if len(scores) > 1 else 0

            # Boost for original query (index 0) vs expanded queries
            original_query_boost = 0.2 if 0 in query_indices else 0

            # For abstract queries, give more weight to expanded terms that found results
            abstract_boost = 0.15 if is_abstract and len(scores) > 1 else 0

            # Calculate final score
            final_score = base_score + multi_query_boost + original_query_boost + abstract_boost

            final_results.append((chunk, final_score))

        # Sort by final score and return top results
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:limit]

    async def _generate_abstract_fallback_analysis(self, query: str, project_context: Dict[str, Any], intent: QueryIntent) -> CodeAnalysis:
        """
        Generate intelligent fallback analysis for abstract queries when no specific code is found.

        This provides architectural insights and recommendations even when embedding search
        doesn't find specific code matches.
        """
        try:
            if not self.code_analyzer.client:
                return self._basic_abstract_fallback(query, intent)

            # Build comprehensive system context
            system_context = """
You are analyzing a codebase indexing and knowledge retrieval system with the following architecture:

CORE COMPONENTS:
- Tree-sitter Parser: Extracts AST from source code files
- Chunk Processor: Breaks code into meaningful chunks with hierarchical relationships
- Embedding Generator: Creates vector embeddings using OpenAI/HuggingFace/Ollama
- Qdrant Vector Store: Stores and searches embeddings for semantic similarity
- Neo4j Graph Store: Models code relationships and dependencies
- FastAPI MCP Server: Provides API endpoints for LLM integration
- React Frontend: Chat interface for natural language code queries

ARCHITECTURAL PATTERNS:
- Microservices architecture with clear separation of concerns
- Event-driven processing for indexing pipeline
- Graph + Vector hybrid search for comprehensive code understanding
- RESTful API design with proper error handling
- Modular design with dependency injection
"""

            project_info = ""
            if project_context:
                project_name = project_context.get("name", "Unknown")
                technologies = project_context.get("technologies", [])
                project_info = f"\nPROJECT CONTEXT:\n- Name: {project_name}\n- Technologies: {', '.join(technologies)}"

            prompt = f"""
{system_context}
{project_info}

USER QUERY: "{query}"

The user is asking about this codebase indexing system, but no specific code components were found that directly match their query. However, you can still provide valuable architectural insights and analysis.

Please provide a comprehensive response that:

1. **Addresses the Query Directly**: Answer what the user is asking about in the context of this codebase indexing system
2. **System Architecture Analysis**: Explain relevant architectural aspects that relate to their question
3. **Design Patterns & Principles**: Discuss design decisions and patterns used in the system
4. **Technical Implementation**: Describe how the system would handle the aspects they're asking about
5. **Recommendations**: Provide specific suggestions for improvements or considerations

Focus on the codebase indexing domain and provide insights about:
- Scalability considerations for large codebases
- Maintainability through modular design
- Performance optimization strategies
- Reliability and error handling approaches
- Security considerations for code analysis systems

Be specific and technical, as if you're a senior architect explaining the system design.
"""

            response = self.code_analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior software architect specializing in codebase indexing and knowledge retrieval systems."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            analysis_text = response.choices[0].message.content.strip()

            # Extract summary (first paragraph)
            lines = analysis_text.split('\n')
            summary = lines[0] if lines else "Architectural analysis of the codebase indexing system."

            return CodeAnalysis(
                summary=summary,
                detailed_explanation=analysis_text,
                code_flow=[
                    "Parse source code using Tree-sitter",
                    "Generate embeddings and store in Qdrant",
                    "Model relationships in Neo4j graph",
                    "Process queries through MCP server"
                ],
                key_components=[
                    {"name": "Tree-sitter Parser", "purpose": "AST extraction and code parsing", "location": "parsing layer"},
                    {"name": "Qdrant Vector Store", "purpose": "Semantic search and similarity matching", "location": "storage layer"},
                    {"name": "Neo4j Graph Store", "purpose": "Relationship modeling and graph traversal", "location": "storage layer"},
                    {"name": "MCP Server", "purpose": "API endpoints and LLM integration", "location": "service layer"}
                ],
                relationships=[
                    {"from": "Parser", "to": "Chunk Processor", "relationship": "feeds_data", "context": "AST nodes become code chunks"},
                    {"from": "Chunk Processor", "to": "Vector Store", "relationship": "stores_embeddings", "context": "Chunks converted to vectors"},
                    {"from": "Vector Store", "to": "Graph Store", "relationship": "complements", "context": "Hybrid search approach"}
                ],
                recommendations=self._generate_abstract_recommendations(query, intent)
            )

        except Exception as e:
            logger.error(f"Error generating abstract fallback analysis: {e}")
            return self._basic_abstract_fallback(query, intent)

    def _basic_abstract_fallback(self, query: str, intent: QueryIntent) -> CodeAnalysis:
        """Basic fallback analysis when LLM is not available."""
        query_lower = query.lower()

        if "scalability" in query_lower or "scale" in query_lower:
            summary = "This codebase indexing system is designed with scalability in mind through distributed storage and async processing."
            explanation = """The system addresses scalability through several key architectural decisions:

1. **Distributed Storage**: Uses Qdrant for vector storage and Neo4j for graph relationships, both designed for horizontal scaling
2. **Async Processing**: Implements asynchronous operations throughout the indexing pipeline
3. **Modular Architecture**: Separates concerns between parsing, embedding, storage, and querying
4. **Batch Processing**: Processes multiple code chunks efficiently in batches
5. **Caching Strategies**: Implements intelligent caching for frequently accessed code patterns"""

        elif "maintainability" in query_lower or "maintain" in query_lower:
            summary = "The system emphasizes maintainability through modular design, clear separation of concerns, and comprehensive error handling."
            explanation = """Maintainability is achieved through:

1. **Modular Design**: Clear separation between parsing, embedding, storage, and API layers
2. **Dependency Injection**: Loose coupling between components for easier testing and modification
3. **Error Handling**: Comprehensive exception handling and graceful degradation
4. **Configuration Management**: Centralized configuration for easy deployment and updates
5. **Logging and Monitoring**: Detailed logging for debugging and system health monitoring"""

        else:
            summary = f"Analysis of the codebase indexing system architecture related to: {query}"
            explanation = """This codebase indexing system provides a comprehensive solution for code analysis and retrieval:

**Core Architecture**: The system combines vector embeddings for semantic search with graph relationships for structural understanding.

**Key Components**: Tree-sitter parsing, Qdrant vector storage, Neo4j graph modeling, FastAPI MCP server, and React frontend.

**Design Principles**: Modular architecture, separation of concerns, scalable storage, and intelligent search capabilities."""

        return CodeAnalysis(
            summary=summary,
            detailed_explanation=explanation,
            code_flow=[
                "Parse source code with Tree-sitter",
                "Generate embeddings for semantic search",
                "Store relationships in Neo4j graph",
                "Serve results through MCP API"
            ],
            key_components=[
                {"name": "Indexing Pipeline", "purpose": "Process and index source code", "location": "core system"},
                {"name": "Search Engine", "purpose": "Semantic and graph-based search", "location": "query layer"},
                {"name": "API Server", "purpose": "LLM integration and endpoints", "location": "service layer"}
            ],
            relationships=[],
            recommendations=self._generate_abstract_recommendations(query, intent)
        )

    def _generate_abstract_recommendations(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate recommendations for abstract queries."""
        query_lower = query.lower()
        recommendations = []

        if "scalability" in query_lower:
            recommendations.extend([
                "Implement horizontal scaling for Qdrant and Neo4j clusters",
                "Add caching layers for frequently accessed code patterns",
                "Consider implementing distributed processing for large codebases",
                "Monitor and optimize embedding generation performance"
            ])

        if "maintainability" in query_lower:
            recommendations.extend([
                "Implement comprehensive unit and integration tests",
                "Add automated code quality checks and linting",
                "Create detailed API documentation and system architecture docs",
                "Implement health checks and monitoring dashboards"
            ])

        if "performance" in query_lower:
            recommendations.extend([
                "Optimize embedding generation with batch processing",
                "Implement query result caching for common searches",
                "Add database indexing strategies for faster retrieval",
                "Profile and optimize graph traversal queries"
            ])

        if not recommendations:
            recommendations = [
                "Review system architecture for optimization opportunities",
                "Implement comprehensive monitoring and alerting",
                "Consider adding automated testing and CI/CD pipelines"
            ]

        return recommendations[:3]  # Limit to top 3

    async def _gather_project_context(self, project_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Gather project context for analysis."""
        if not project_ids:
            return {
                "name": "Codebase Indexing Solution",
                "description": "A comprehensive codebase indexing and knowledge retrieval system",
                "technologies": ["Python", "FastAPI", "React", "Qdrant", "Neo4j", "Tree-sitter", "OpenAI"]
            }

        try:
            # Get project information from database
            projects = []
            for project_id in project_ids:
                project = await self.project_manager.get_project(project_id)
                if project:
                    projects.append(project)

            if not projects:
                return {
                    "name": "Unknown Project",
                    "description": "Project information not available",
                    "technologies": []
                }

            # Combine project information
            if len(projects) == 1:
                project = projects[0]
                return {
                    "name": project.name,
                    "description": project.description or "No description available",
                    "technologies": project.technologies or []
                }
            else:
                # Multiple projects
                project_names = [p.name for p in projects]
                all_technologies = set()
                for p in projects:
                    if p.technologies:
                        all_technologies.update(p.technologies)

                return {
                    "name": f"Multi-project analysis ({', '.join(project_names)})",
                    "description": f"Analysis across {len(projects)} projects: {', '.join(project_names)}",
                    "technologies": list(all_technologies)
                }

        except Exception as e:
            logger.warning(f"Error gathering project context: {e}")
            return {
                "name": "Project Context Error",
                "description": "Unable to retrieve project information",
                "technologies": []
            }

    def _convert_search_filters_to_qdrant(self, search_filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert search filters to Qdrant-compatible format."""
        qdrant_filters = {}

        # Handle function names
        if "function_names" in search_filters and search_filters["function_names"]:
            # Use the first function name for exact matching
            qdrant_filters["name"] = search_filters["function_names"][0]

        # Handle class names
        if "class_names" in search_filters and search_filters["class_names"]:
            # Use the first class name for exact matching
            qdrant_filters["name"] = search_filters["class_names"][0]

        # Handle file patterns
        if "file_patterns" in search_filters and search_filters["file_patterns"]:
            # Use the first file pattern for exact matching
            qdrant_filters["file_path"] = search_filters["file_patterns"][0]

        # Handle preferred node types
        if "preferred_node_types" in search_filters and search_filters["preferred_node_types"]:
            # Use the first preferred node type
            qdrant_filters["node_type"] = search_filters["preferred_node_types"][0].value

        return qdrant_filters

    def _post_process_search_results(
        self,
        results: List[Tuple[CodeChunk, float]],
        intent: QueryIntent,
        entities: Dict[str, List[str]],
        confidence: float
    ) -> List[Tuple[CodeChunk, float]]:
        """Post-process search results based on query analysis."""
        if not results:
            return results

        # Apply intent-based scoring adjustments
        adjusted_results = []
        for chunk, score in results:
            adjusted_score = score

            # Boost scores based on intent matching
            if intent == QueryIntent.ARCHITECTURE:
                if chunk.node_type.value in ["class", "module"]:
                    adjusted_score *= 1.2
            elif intent == QueryIntent.IMPLEMENTATION:
                if chunk.node_type.value in ["function", "method"]:
                    adjusted_score *= 1.2
            elif intent == QueryIntent.FUNCTIONALITY:
                if chunk.node_type.value in ["function", "class", "method"]:
                    adjusted_score *= 1.1

            # Boost scores for entity matches
            if entities["functions"] and chunk.name:
                for func_name in entities["functions"]:
                    if func_name.lower() in chunk.name.lower():
                        adjusted_score *= 1.3
                        break

            if entities["classes"] and chunk.name:
                for class_name in entities["classes"]:
                    if class_name.lower() in chunk.name.lower():
                        adjusted_score *= 1.3
                        break

            # Apply confidence-based adjustment
            if confidence > 0.7:
                adjusted_score *= (1 + (confidence - 0.7) * 0.5)

            adjusted_results.append((chunk, adjusted_score))

        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x[1], reverse=True)

        return adjusted_results

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
