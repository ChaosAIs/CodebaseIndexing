"""FastAPI-based Model Context Protocol (MCP) server."""

import asyncio
import json
import time
import uuid
from collections import Counter
from typing import List, Dict, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from ..models import (
    QueryRequest, QueryResponse, QueryResult, GraphData, SystemStatus,
    IndexingRequest, EmbeddingModel, CodeAnalysis, Project, ProjectCreate,
    ProjectUpdate, ProjectIndexRequest, ProjectStatus, CodeChunk,
    FlowAnalysis, AgentPerspectiveModel
)
from ..database.qdrant_client import QdrantVectorStore
from ..database.neo4j_client import Neo4jGraphStore
from ..database.sqlite_client import ProjectManager
from ..embeddings.embedding_generator import EmbeddingGenerator
from ..chunking.chunk_processor import ChunkProcessor
from ..analysis.code_analyzer import CodeAnalyzer
from ..query.query_processor import QueryProcessor, QueryIntent
from ..query.intelligent_query_analyzer import IntelligentQueryAnalyzer, ProcessingStrategy
from ..agents.agent_orchestrator import AgentOrchestrator
from ..orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
from ..streaming.stream_processor import stream_processor, StreamEventType
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

        # Initialize agent orchestrator with LLM client if available
        llm_client = None
        try:
            if config.ai_models.openai_api_key:
                from openai import OpenAI
                llm_client = OpenAI(api_key=config.ai_models.openai_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client for agents: {e}")

        self.agent_orchestrator = AgentOrchestrator(llm_client)

        # Initialize intelligent query analyzer
        self.query_analyzer = IntelligentQueryAnalyzer(openai_client=llm_client)

        # Initialize enhanced agent orchestrator with base orchestrator
        self.enhanced_orchestrator = EnhancedAgentOrchestrator(
            openai_client=llm_client,
            base_orchestrator=self.agent_orchestrator
        )

        # Rate limiting
        self.active_requests = {}
        self.max_concurrent_requests = 3
        self.request_timeout = 300  # 5 minutes

        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3001",  # Frontend development server
                "http://127.0.0.1:3001",  # Alternative localhost
                "*"  # Allow all for development - configure appropriately for production
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()

    def _check_rate_limit(self, request_id: str = None) -> bool:
        """Check if request can proceed based on rate limiting."""
        current_time = time.time()

        # Clean up expired requests
        expired_requests = [
            req_id for req_id, start_time in self.active_requests.items()
            if current_time - start_time > self.request_timeout
        ]
        for req_id in expired_requests:
            del self.active_requests[req_id]

        # Check if we're at the limit
        if len(self.active_requests) >= self.max_concurrent_requests:
            return False

        # Add this request
        if request_id:
            self.active_requests[request_id] = current_time

        return True

    def _release_request(self, request_id: str):
        """Release a request from rate limiting."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]

    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.post("/mcp/query", response_model=QueryResponse)
        async def query_codebase(request: QueryRequest):
            """Query the codebase with natural language or code snippets."""
            start_time = time.time()
            
            try:
                logger.info(f"Processing query: {request.query}")

                # Step 1: Intelligent query analysis for optimization
                available_chunks = await self._estimate_available_chunks(request.project_ids)
                analysis_result = await self.query_analyzer.analyze_query(request.query, available_chunks)
                logger.info(f"Query analysis: {analysis_result.explanation}")

                # Step 2: Check if streaming is recommended
                if analysis_result.should_stream and analysis_result.estimated_processing_time > 30:
                    # For very complex queries, suggest streaming
                    logger.info("Query complexity suggests using streaming endpoint")
                    return JSONResponse(
                        status_code=202,
                        content={
                            "message": "Query complexity suggests using streaming endpoint for better user experience",
                            "stream_endpoint": "/mcp/query/stream",
                            "estimated_time": analysis_result.estimated_processing_time,
                            "complexity": analysis_result.complexity.value,
                            "use_streaming": True
                        }
                    )

                # Step 3: Continue with optimized processing
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

                # Generate multi-agent analysis for comprehensive insights
                analysis = None
                if similar_chunks or is_abstract:
                    # Prepare context for agents with intelligent analysis
                    agent_context = {
                        "project_context": project_context,
                        "graph_context": all_graph_context,
                        "comprehensive_context": comprehensive_context,
                        "intent": intent,
                        "confidence": confidence,
                        "is_abstract": is_abstract,
                        "analysis_result": analysis_result,
                        "selected_agents": analysis_result.required_agents
                    }

                    # Get chunks for analysis with optimized targeting
                    if similar_chunks:
                        initial_chunks = [chunk for chunk, _ in similar_chunks]

                        # Use intelligent analysis to determine chunk requirements
                        total_chunks_needed = sum(task.estimated_chunks_needed for task in analysis_result.required_agents)
                        target_chunks_for_agents = min(total_chunks_needed, 150)  # Cap at 150 for performance

                        chunks_for_analysis = await self._enhance_chunks_for_architectural_coverage(
                            initial_chunks,
                            request.query,
                            request.project_ids,
                            max(target_chunks_for_agents, request.limit)
                        )
                    else:
                        chunks_for_analysis = []

                    # Run optimized multi-agent analysis
                    logger.info(f"Running optimized multi-agent analysis with {len(analysis_result.required_agents)} selected agents")
                    flow_response = await self._run_optimized_agent_analysis(
                        request.query,
                        chunks_for_analysis,
                        agent_context,
                        analysis_result.required_agents
                    )

                    # Convert FlowResponse to CodeAnalysis for compatibility
                    analysis = self._convert_flow_response_to_analysis(flow_response)
                    logger.info(f"Multi-agent analysis completed with {len(flow_response.agent_perspectives)} perspectives")

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

                # Debug log regular query final result
                self._log_regular_query_result_debug(request.query, response, processing_time)

                logger.info(f"Query processed in {processing_time:.2f}s, found {len(results)} results")
                return response
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/mcp/query/stream")
        async def query_codebase_stream(request: QueryRequest):
            """Stream query processing with real-time updates."""
            stream_id = str(uuid.uuid4())

            # Check rate limiting
            if not self._check_rate_limit(stream_id):
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many concurrent requests. Maximum {self.max_concurrent_requests} allowed."
                )

            logger.info(f"Starting streaming query processing: {stream_id}")

            # Create the stream generator first
            stream_generator = stream_processor.create_stream(stream_id)

            async def process_stream():
                try:
                    logger.info(f"Background processing started for stream {stream_id}")
                    # Small delay to ensure stream is ready
                    await asyncio.sleep(0.2)
                    # Process query with streaming updates
                    await self._process_query_with_streaming(request, stream_id)
                    logger.info(f"Background processing completed for stream {stream_id}")
                except Exception as e:
                    logger.error(f"Error in streaming query: {e}")
                    await stream_processor.emit_event(
                        stream_id,
                        StreamEventType.ERROR,
                        {"error": str(e)},
                        f"Processing error: {e}"
                    )
                finally:
                    # Release rate limit
                    self._release_request(stream_id)
                    logger.info(f"Background processing cleanup completed for stream {stream_id}")

            # Start processing in background
            task = asyncio.create_task(process_stream())
            logger.info(f"Background task created for stream {stream_id}")

            # Return streaming response
            return StreamingResponse(
                stream_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "X-Stream-ID": stream_id
                }
            )

        @self.app.post("/mcp/test/stream")
        async def test_streaming():
            """Test streaming endpoint to verify SSE functionality."""
            stream_id = f"test_{uuid.uuid4()}"
            logger.info(f"Starting test stream: {stream_id}")

            async def test_stream_generator():
                try:
                    # Send initial event
                    yield f"data: {json.dumps({'event_type': 'test_start', 'message': 'Test stream started', 'stream_id': stream_id})}\n\n"

                    # Send a few test events with delays
                    for i in range(5):
                        await asyncio.sleep(1)
                        event_data = {
                            'event_type': 'test_progress',
                            'message': f'Test event {i+1}/5',
                            'progress': (i+1) * 20,
                            'stream_id': stream_id
                        }
                        yield f"data: {json.dumps(event_data)}\n\n"
                        logger.info(f"Sent test event {i+1} for stream {stream_id}")

                    # Send completion event
                    yield f"data: {json.dumps({'event_type': 'test_complete', 'message': 'Test stream completed', 'stream_id': stream_id})}\n\n"
                    logger.info(f"Test stream {stream_id} completed")

                except Exception as e:
                    logger.error(f"Error in test stream {stream_id}: {e}")
                    yield f"data: {json.dumps({'event_type': 'error', 'message': f'Test error: {e}'})}\n\n"

            return StreamingResponse(
                test_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "X-Stream-ID": stream_id
                }
            )

        @self.app.post("/mcp/query/flow", response_model=FlowAnalysis)
        async def query_codebase_flow(request: QueryRequest):
            """Query the codebase with enhanced multi-agent flow analysis using entity-first approach."""
            start_time = time.time()

            try:
                logger.info(f"Processing flow query with entity-first approach: {request.query}")

                # Step 1: Extract entities from query first
                entities = self.query_processor.extract_code_entities(request.query)
                logger.info(f"Extracted entities: {entities}")

                # Step 2: Expand entities using graph relationships
                expanded_entities = self.query_processor.expand_entities_with_graph(entities, self.graph_store)
                logger.info(f"Expanded entities with graph: {len(expanded_entities.get('related_components', []))} related components found")

                # Step 3: Generate comprehensive search terms from expanded entities
                comprehensive_search_terms = self.query_processor.generate_comprehensive_search_terms(
                    request.query, expanded_entities
                )
                logger.info(f"Generated {len(comprehensive_search_terms)} comprehensive search terms")

                # Step 4: Gather project context for enhanced processing
                project_context = await self._gather_project_context(request.project_ids)

                # Step 5: Classify query intent
                intent, confidence = self.query_processor.classify_query_intent(request.query)
                logger.info(f"Query intent: {intent.value} (confidence: {confidence:.2f})")

                # Step 6: Perform comprehensive search using expanded context
                all_similar_chunks = []
                search_weights = [1.0, 0.8, 0.6, 0.4, 0.3]  # Decreasing weights for search terms

                for i, search_term in enumerate(comprehensive_search_terms[:5]):  # Limit to top 5 terms
                    # Enhance search term for better embedding generation
                    enhanced_query = self.query_processor.enhance_query_for_embedding(
                        search_term, intent, project_context
                    )

                    # Generate search filters based on expanded entities
                    search_filters = self.query_processor.generate_search_filters(
                        search_term, intent, expanded_entities
                    )

                    # Generate query embedding
                    query_embedding = await self._generate_query_embedding(enhanced_query, request.model)

                    # Convert search filters to Qdrant format
                    qdrant_filters = self._convert_search_filters_to_qdrant(search_filters)

                    # Adjust limit per search term
                    term_limit = max(5, request.limit // len(comprehensive_search_terms[:5]))

                    # Search similar chunks
                    chunks = await self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        limit=term_limit,
                        project_ids=request.project_ids,
                        filters=qdrant_filters
                    )

                    # Add query source info for scoring
                    chunks_with_source = [(chunk, score, i) for chunk, score in chunks]
                    all_similar_chunks.extend(chunks_with_source)

                # Deduplicate and re-rank results with intelligent scoring
                similar_chunks = self._deduplicate_and_rerank_chunks(all_similar_chunks, request.limit)

                # Apply post-processing based on intent and expanded entities
                similar_chunks = self._post_process_search_results(
                    similar_chunks, intent, expanded_entities, confidence
                )

                # Get comprehensive graph context for enhanced understanding
                all_graph_context = {}
                comprehensive_context = {}

                if request.include_context and similar_chunks:
                    # Get enhanced context for top chunks
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

                # Check if query is abstract for context
                is_abstract = self.query_processor.is_abstract_query(request.query)

                # Prepare context for agents
                agent_context = {
                    "project_context": project_context,
                    "graph_context": all_graph_context,
                    "comprehensive_context": comprehensive_context,
                    "intent": intent,
                    "confidence": confidence,
                    "is_abstract": is_abstract,
                    "expanded_entities": expanded_entities,
                    "processing_time": time.time() - start_time
                }

                # Get chunks for analysis with enhanced coverage for agents
                if similar_chunks:
                    initial_chunks = [chunk for chunk, _ in similar_chunks]
                    target_chunks_for_agents = 120  # 5-8 agents × 15 chunks each

                    chunks_for_analysis = await self._enhance_chunks_for_architectural_coverage(
                        initial_chunks,
                        request.query,
                        request.project_ids,
                        max(target_chunks_for_agents, request.limit)
                    )
                else:
                    chunks_for_analysis = []

                # Analyze query for orchestration
                available_chunks = await self._estimate_available_chunks(request.project_ids)
                analysis_result = await self.query_analyzer.analyze_query(request.query, available_chunks)

                # Run multi-agent analysis
                logger.info("Running multi-agent flow analysis for comprehensive insights")
                orchestration_result = await self.enhanced_orchestrator.orchestrate_agents(
                    query=request.query,
                    analysis_result=analysis_result,
                    context=agent_context
                )

                # Extract FlowResponse from OrchestrationResult
                flow_response = orchestration_result.final_response if orchestration_result else None

                if not flow_response:
                    raise HTTPException(status_code=500, detail="Failed to generate flow response from agents")

                # Convert to response model
                flow_analysis = FlowAnalysis(
                    executive_summary=flow_response.executive_summary,
                    detailed_analysis=flow_response.detailed_analysis,
                    agent_perspectives=[
                        AgentPerspectiveModel(
                            role=p.role.value,
                            analysis=p.analysis,
                            key_insights=p.key_insights,
                            recommendations=p.recommendations,
                            confidence=p.confidence,
                            focus_areas=p.focus_areas
                        ) for p in flow_response.agent_perspectives
                    ],
                    synthesis=flow_response.synthesis,
                    action_items=flow_response.action_items,
                    follow_up_questions=flow_response.follow_up_questions
                )

                processing_time = time.time() - start_time
                logger.info(f"Flow query processed in {processing_time:.2f}s with {len(flow_response.agent_perspectives)} agent perspectives")

                return flow_analysis

            except Exception as e:
                logger.error(f"Error processing flow query: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/mcp/query/enhanced", response_model=FlowAnalysis)
        async def query_codebase_enhanced(request: QueryRequest):
            """
            Enhanced query processing with entity-first approach and distributed agent analysis.
            This endpoint provides the most comprehensive analysis by:
            1. Extracting entities from query first
            2. Expanding entities using graph relationships
            3. Distributing different chunks to different agents
            4. Providing broader architectural understanding
            """
            start_time = time.time()

            try:
                logger.info(f"Processing enhanced query: {request.query}")

                # Step 1: Extract entities and expand with graph
                entities = self.query_processor.extract_code_entities(request.query)
                expanded_entities = self.query_processor.expand_entities_with_graph(entities, self.graph_store)

                # Step 2: Generate comprehensive search terms
                search_terms = self.query_processor.generate_comprehensive_search_terms(
                    request.query, expanded_entities
                )

                # Step 3: Perform distributed search across all terms
                all_chunks = []
                for term in search_terms[:10]:  # Use top 10 terms for comprehensive coverage
                    # Generate embedding for each term
                    query_embedding = await self._generate_query_embedding(term, request.model)

                    # Search with smaller limit per term to get diverse results
                    chunks = await self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        limit=max(3, request.limit // len(search_terms[:10])),
                        project_ids=request.project_ids
                    )
                    all_chunks.extend(chunks)

                # Step 4: Deduplicate and select diverse chunks
                unique_chunks = self._deduplicate_chunks_by_content(all_chunks)
                selected_chunks = unique_chunks[:request.limit]

                # Step 5: Prepare enhanced context
                project_context = await self._gather_project_context(request.project_ids)
                intent, confidence = self.query_processor.classify_query_intent(request.query)

                agent_context = {
                    "project_context": project_context,
                    "project_ids": request.project_ids,
                    "expanded_entities": expanded_entities,
                    "search_terms": search_terms,
                    "intent": intent,
                    "confidence": confidence,
                    "entity_first_approach": True
                }

                # Step 6: Analyze query for orchestration
                available_chunks = await self._estimate_available_chunks(request.project_ids)
                analysis_result = await self.query_analyzer.analyze_query(request.query, available_chunks)

                # Step 6: Run distributed agent analysis
                orchestration_result = await self.enhanced_orchestrator.orchestrate_agents(
                    query=request.query,
                    analysis_result=analysis_result,
                    context=agent_context
                )

                # Extract FlowResponse from OrchestrationResult
                flow_response = orchestration_result.final_response if orchestration_result else None

                if not flow_response:
                    raise HTTPException(status_code=500, detail="Failed to generate flow response from agents")

                # Step 7: Create comprehensive response
                response = FlowAnalysis(
                    executive_summary=flow_response.executive_summary,
                    detailed_analysis=flow_response.detailed_analysis,
                    agent_perspectives=[
                        AgentPerspectiveModel(
                            role=p.role.value,
                            analysis=p.analysis,
                            key_insights=p.key_insights,
                            recommendations=p.recommendations,
                            confidence=p.confidence,
                            focus_areas=p.focus_areas
                        ) for p in flow_response.agent_perspectives
                    ],
                    synthesis=flow_response.synthesis,
                    action_items=flow_response.action_items,
                    follow_up_questions=flow_response.follow_up_questions
                )

                logger.info(f"Enhanced query completed in {time.time() - start_time:.2f}s")
                return response

            except Exception as e:
                logger.error(f"Error processing enhanced query: {e}")
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
                logger.info(f"Starting embedding generation for {len(chunks)} chunks with provider: {provider_name}")
                logger.debug(f"Chunk types distribution: {dict(Counter(chunk.node_type.value for chunk in chunks))}")

                embeddings = await self.embedding_generator.generate_chunk_embeddings(
                    chunks, provider_name
                )

                logger.info(f"Embedding generation completed. Generated {len(embeddings)} embeddings")
                
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
                    logger.info(f"Starting embedding generation for project {project.name}: {len(chunks)} chunks with provider: {provider_name}")
                    logger.debug(f"Project chunk types distribution: {dict(Counter(chunk.node_type.value for chunk in chunks))}")

                    embeddings = await self.embedding_generator.generate_chunk_embeddings(
                        chunks, provider_name
                    )

                    logger.info(f"Project embedding generation completed. Generated {len(embeddings)} embeddings")

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

    def _deduplicate_chunks_by_content(self, chunks_with_scores: List[Tuple]) -> List[Tuple]:
        """
        Deduplicate chunks by content similarity to ensure diverse results.

        Args:
            chunks_with_scores: List of (chunk, score) tuples

        Returns:
            List of unique (chunk, score) tuples
        """
        if not chunks_with_scores:
            return []

        unique_chunks = []
        seen_content_hashes = set()

        for chunk, score in chunks_with_scores:
            # Create a content hash for deduplication
            content_hash = hash(chunk.content[:200])  # Use first 200 chars for hash

            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_chunks.append((chunk, score))

        # Sort by score (descending)
        unique_chunks.sort(key=lambda x: x[1], reverse=True)
        return unique_chunks

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
                    "System architecture analysis based on discovered components",
                    "Dynamic component categorization and relationship mapping",
                    "Intelligent query processing and response generation"
                ],
                key_components=[
                    {"name": "Dynamic Architecture Discovery", "purpose": "Automatically discovers system architecture patterns", "location": "analysis layer"},
                    {"name": "Intelligent Query Processor", "purpose": "Processes abstract and technical queries", "location": "query layer"},
                    {"name": "Adaptive Response Generator", "purpose": "Generates contextual responses based on discovered patterns", "location": "response layer"}
                ],
                relationships=[
                    {"from": "Query Processor", "to": "Architecture Discovery", "relationship": "triggers_analysis", "context": "Query analysis drives architecture discovery"},
                    {"from": "Architecture Discovery", "to": "Response Generator", "relationship": "provides_context", "context": "Discovered patterns inform response generation"}
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

    def _convert_flow_response_to_analysis(self, flow_response) -> CodeAnalysis:
        """Convert FlowResponse from agent orchestrator to CodeAnalysis for compatibility."""

        # Extract key components with more specific details from agent perspectives
        key_components = []
        for perspective in flow_response.agent_perspectives:
            # Create more detailed component entries
            for i, insight in enumerate(perspective.key_insights[:2]):  # Top 2 insights per agent
                component_name = f"{perspective.role.value.title()} Analysis"
                if "file" in insight.lower() or "function" in insight.lower() or "class" in insight.lower():
                    # Extract specific code element if mentioned
                    words = insight.split()
                    for word in words:
                        if word.endswith('.py') or word.endswith('.js') or word.endswith('.ts'):
                            component_name = f"File: {word}"
                            break
                        elif word.startswith('def ') or word.endswith('()'):
                            component_name = f"Function: {word.replace('def ', '').replace('()', '')}"
                            break
                        elif word.startswith('class '):
                            component_name = f"Class: {word.replace('class ', '')}"
                            break

                key_components.append({
                    "name": component_name,
                    "purpose": insight,
                    "location": f"{perspective.role.value} perspective",
                    "confidence": f"{perspective.confidence:.1f}",
                    "type": perspective.role.value
                })

        # Extract more meaningful relationships from agent perspectives
        relationships = []
        for i, perspective in enumerate(flow_response.agent_perspectives):
            # Create relationships based on focus areas and insights
            for focus_area in perspective.focus_areas[:1]:  # Top focus area
                relationships.append({
                    "from": perspective.role.value.title(),
                    "to": focus_area,
                    "relationship": "analyzes",
                    "context": f"{perspective.role.value} expert focuses on {focus_area}",
                    "strength": f"{perspective.confidence:.1f}"
                })

        # Create more detailed code flow from agent analysis
        code_flow = []
        for perspective in flow_response.agent_perspectives:
            if perspective.focus_areas:
                # Create more specific flow steps
                flow_step = f"🔍 {perspective.role.value.title()}: Analyze {perspective.focus_areas[0]}"
                if len(perspective.key_insights) > 0:
                    # Add specific insight to flow
                    insight_preview = perspective.key_insights[0][:50] + "..." if len(perspective.key_insights[0]) > 50 else perspective.key_insights[0]
                    flow_step += f" → {insight_preview}"
                code_flow.append(flow_step)

        # Add synthesis as final flow step with more detail
        if flow_response.synthesis:
            code_flow.append("🔗 Synthesis: Integrate multi-perspective insights for comprehensive understanding")

        return CodeAnalysis(
            summary=flow_response.executive_summary,
            detailed_explanation=flow_response.detailed_analysis,
            code_flow=code_flow[:5],  # Limit to 5 steps
            key_components=key_components[:8],  # Increased to 8 components
            relationships=relationships[:6],  # Increased to 6 relationships
            recommendations=flow_response.action_items[:6]  # Increased to 6 recommendations
        )

    async def _enhance_chunks_for_architectural_coverage(
        self,
        initial_chunks: List[CodeChunk],
        query: str,
        project_ids: List[str],
        target_limit: int
    ) -> List[CodeChunk]:
        """
        Enhance the initial chunk set with additional chunks for better architectural coverage.
        This ensures agents get diverse chunks from different parts of the codebase.
        """
        try:
            # If we already have enough diverse chunks, diversify and return them
            if len(initial_chunks) >= target_limit:
                return self._diversify_chunks_by_file_type(initial_chunks, target_limit)

            logger.info(f"Expanding chunk search from {len(initial_chunks)} to {target_limit} chunks for comprehensive agent analysis")

            # Analyze what architectural areas are missing
            covered_areas = self._analyze_architectural_coverage(initial_chunks)
            missing_areas = self._identify_missing_architectural_areas(covered_areas)

            # Search for chunks in missing architectural areas
            additional_chunks = []
            chunks_needed = target_limit - len(initial_chunks)

            # If we need many more chunks, do broader searches
            if chunks_needed > 20:
                # Do a broad architectural search
                broad_searches = [
                    "class function method implementation",
                    "service business logic processing",
                    "controller handler endpoint API",
                    "model schema data structure",
                    "configuration settings environment",
                    "test testing unit integration",
                    "database query SQL migration",
                    "authentication security authorization",
                    "utility helper common function",
                    "component interface frontend backend"
                ]

                chunks_per_search = max(3, chunks_needed // len(broad_searches))

                for search_term in broad_searches:
                    if len(additional_chunks) >= chunks_needed:
                        break

                    query_embedding = await self._generate_query_embedding(f"{query} {search_term}")
                    area_chunks = await self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        limit=chunks_per_search,
                        project_ids=project_ids
                    )

                    for chunk, score in area_chunks:
                        if chunk not in initial_chunks and chunk not in additional_chunks:
                            additional_chunks.append(chunk)
                            if len(additional_chunks) >= chunks_needed:
                                break
            else:
                # Do targeted searches for missing areas
                for area in missing_areas:
                    area_query = self._build_architectural_query(area, query)

                    # Generate embedding for architectural area
                    query_embedding = await self._generate_query_embedding(area_query)

                    # Search for chunks in this architectural area
                    area_chunks = await self.vector_store.search_similar(
                        query_embedding=query_embedding,
                        limit=max(3, chunks_needed // len(missing_areas)),
                        project_ids=project_ids
                    )

                    # Add chunks that aren't already included
                    for chunk, score in area_chunks:
                        if chunk not in initial_chunks and chunk not in additional_chunks:
                            additional_chunks.append(chunk)

            # Combine and diversify
            all_chunks = initial_chunks + additional_chunks
            return self._diversify_chunks_by_file_type(all_chunks, target_limit)

        except Exception as e:
            logger.error(f"Error enhancing chunks for architectural coverage: {e}")
            return initial_chunks[:target_limit]

    def _analyze_architectural_coverage(self, chunks: List[CodeChunk]) -> Dict[str, int]:
        """Analyze what architectural areas are covered by the current chunks."""
        coverage = {
            'models': 0, 'services': 0, 'controllers': 0, 'views': 0,
            'utils': 0, 'config': 0, 'tests': 0, 'database': 0,
            'auth': 0, 'api': 0, 'frontend': 0, 'backend': 0
        }

        for chunk in chunks:
            file_path = chunk.file_path.lower()
            content = chunk.content.lower()

            # Count coverage by file patterns
            if any(pattern in file_path for pattern in ['model', 'entity', 'schema']):
                coverage['models'] += 1
            elif any(pattern in file_path for pattern in ['service', 'business']):
                coverage['services'] += 1
            elif any(pattern in file_path for pattern in ['controller', 'handler', 'route']):
                coverage['controllers'] += 1
            elif any(pattern in file_path for pattern in ['view', 'component', 'template']):
                coverage['views'] += 1
            elif any(pattern in file_path for pattern in ['util', 'helper']):
                coverage['utils'] += 1
            elif any(pattern in file_path for pattern in ['config', 'setting']):
                coverage['config'] += 1
            elif any(pattern in file_path for pattern in ['test', 'spec']):
                coverage['tests'] += 1
            elif any(pattern in file_path for pattern in ['db', 'database']):
                coverage['database'] += 1
            elif any(pattern in file_path for pattern in ['auth', 'login', 'security']):
                coverage['auth'] += 1
            elif any(pattern in file_path for pattern in ['api', 'rest']):
                coverage['api'] += 1
            elif any(pattern in file_path for pattern in ['frontend', 'client', 'web']):
                coverage['frontend'] += 1
            elif any(pattern in file_path for pattern in ['backend', 'server']):
                coverage['backend'] += 1

        return coverage

    def _identify_missing_architectural_areas(self, coverage: Dict[str, int]) -> List[str]:
        """Identify architectural areas that need more coverage."""
        # Areas with 0 or very low coverage should be prioritized
        missing_areas = []
        for area, count in coverage.items():
            if count == 0:
                missing_areas.append(area)

        # If no completely missing areas, add areas with low coverage
        if not missing_areas:
            low_coverage_areas = [area for area, count in coverage.items() if count <= 1]
            missing_areas.extend(low_coverage_areas[:3])  # Add top 3 low coverage areas

        return missing_areas[:5]  # Limit to 5 areas to avoid too many searches

    def _build_architectural_query(self, area: str, original_query: str) -> str:
        """Build a search query focused on a specific architectural area."""
        area_keywords = {
            'models': 'data model class entity schema structure',
            'services': 'service business logic process function method',
            'controllers': 'controller handler endpoint route API',
            'views': 'view component template UI interface',
            'utils': 'utility helper function tool common',
            'config': 'configuration settings environment setup',
            'tests': 'test testing spec unit integration',
            'database': 'database query SQL migration schema',
            'auth': 'authentication authorization security login',
            'api': 'API REST endpoint interface service',
            'frontend': 'frontend client UI component interface',
            'backend': 'backend server service logic processing'
        }

        area_terms = area_keywords.get(area, area)
        return f"{original_query} {area_terms}"

    def _diversify_chunks_by_file_type(self, chunks: List[CodeChunk], limit: int) -> List[CodeChunk]:
        """Diversify chunks by ensuring different file types and paths are represented."""
        if len(chunks) <= limit:
            return chunks

        # Group chunks by file extension and directory
        file_groups = {}
        for chunk in chunks:
            file_ext = chunk.file_path.split('.')[-1] if '.' in chunk.file_path else 'no_ext'
            file_dir = '/'.join(chunk.file_path.split('/')[:-1]) if '/' in chunk.file_path else 'root'
            group_key = f"{file_ext}_{file_dir}"

            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(chunk)

        # Select chunks to maximize diversity
        selected_chunks = []
        group_keys = list(file_groups.keys())

        # Round-robin selection from different groups
        while len(selected_chunks) < limit and group_keys:
            for group_key in group_keys[:]:
                if len(selected_chunks) >= limit:
                    break
                if file_groups[group_key]:
                    selected_chunks.append(file_groups[group_key].pop(0))
                else:
                    group_keys.remove(group_key)

        return selected_chunks

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
                # Infer technologies from project structure/files
                technologies = self._infer_project_technologies(project)
                return {
                    "name": project.name,
                    "description": project.description or "No description available",
                    "technologies": technologies
                }
            else:
                # Multiple projects
                project_names = [p.name for p in projects]
                all_technologies = set()
                for p in projects:
                    technologies = self._infer_project_technologies(p)
                    all_technologies.update(technologies)

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

    def _infer_project_technologies(self, project) -> List[str]:
        """Infer technologies used in a project based on file extensions and patterns."""
        technologies = []

        try:
            # Basic technology inference based on common patterns
            source_path = getattr(project, 'source_path', '')

            # Check for common technology indicators
            if 'python' in source_path.lower() or any(ext in source_path.lower() for ext in ['.py', 'requirements.txt', 'setup.py']):
                technologies.append("Python")

            if any(ext in source_path.lower() for ext in ['.js', '.jsx', '.ts', '.tsx', 'package.json']):
                technologies.append("JavaScript/TypeScript")

            if 'react' in source_path.lower() or 'node_modules' in source_path.lower():
                technologies.append("React")

            if 'fastapi' in source_path.lower() or 'uvicorn' in source_path.lower():
                technologies.append("FastAPI")

            # Default fallback
            if not technologies:
                technologies = ["General"]

        except Exception as e:
            logger.warning(f"Error inferring technologies for project: {e}")
            technologies = ["Unknown"]

        return technologies

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

    async def _estimate_available_chunks(self, project_ids: Optional[List[str]] = None) -> int:
        """Estimate the number of available chunks for processing."""
        try:
            # Get collection info from Qdrant
            collection_info = await self.vector_store.get_collection_info()
            if collection_info and 'points_count' in collection_info:
                return collection_info['points_count']
            return 100  # Default estimate
        except Exception as e:
            logger.warning(f"Could not estimate available chunks: {e}")
            return 100

    async def _process_query_with_streaming(self, request: QueryRequest, stream_id: str):
        """Process query with streaming updates."""
        start_time = time.time()

        try:
            # Step 1: Query analysis with user-friendly streaming
            await stream_processor.emit_processing_start(stream_id, request.query)
            await stream_processor.emit_query_analysis_start(stream_id, request.query)

            available_chunks = await self._estimate_available_chunks(request.project_ids)
            analysis_result = await self.query_analyzer.analyze_query(request.query, available_chunks)

            await stream_processor.emit_query_analysis_complete(stream_id, analysis_result)

            # Step 2: Search phase
            search_terms = [request.query]
            if analysis_result.search_optimization_hints.get("expand_query", False):
                project_context = await self._gather_project_context(request.project_ids)
                expanded_terms = await self.query_processor.expand_abstract_query(request.query, project_context)
                search_terms.extend(expanded_terms[:5])

            await stream_processor.emit_search_start(stream_id, search_terms)

            # Perform search with progress updates
            all_chunks = []
            for i, search_term in enumerate(search_terms):
                await stream_processor.emit_search_progress(stream_id, i, len(search_terms), len(all_chunks))

                # Generate embedding for the search term
                query_embedding = await self._generate_query_embedding(search_term, request.model)

                chunks = await self.vector_store.search_similar(
                    query_embedding=query_embedding,
                    limit=max(10, request.limit // len(search_terms)),
                    project_ids=request.project_ids
                )
                all_chunks.extend(chunks)

            # Deduplicate chunks
            unique_chunks = self._deduplicate_chunks_by_content(all_chunks)
            selected_chunks = unique_chunks[:request.limit * 2]  # Get more for agent analysis

            await stream_processor.emit_search_complete(stream_id, len(selected_chunks))

            # Step 3: Agent analysis with streaming
            agent_context = {
                "project_context": await self._gather_project_context(request.project_ids),
                "project_ids": request.project_ids,
                "search_terms": search_terms,
                "analysis_result": analysis_result
            }

            chunks_for_analysis = [chunk for chunk, _ in selected_chunks]

            # Run enhanced agent orchestration with streaming updates
            orchestration_result = await self.enhanced_orchestrator.orchestrate_agents(
                query=request.query,
                analysis_result=analysis_result,
                context=agent_context,
                stream_id=stream_id
            )

            flow_response = orchestration_result.final_response

            # Debug log orchestration result details
            self._log_orchestration_result_debug(request.query, orchestration_result, stream_id)

            # Emit orchestration logs for monitoring
            await stream_processor.emit_orchestration_log(stream_id, orchestration_result.orchestration_logs)

            # Step 4: Complete processing with flow response
            total_time = time.time() - start_time

            if flow_response:
                # We have a comprehensive flow response from agents
                agent_count = len(flow_response.agent_perspectives) if flow_response.agent_perspectives else 0

                await stream_processor.emit_user_message(
                    stream_id,
                    f"🎉 Analysis complete! Generated comprehensive response with insights from {agent_count} expert perspectives in {total_time:.1f} seconds.",
                    100.0
                )

                # Convert FlowResponse dataclass to dictionary for JSON serialization
                flow_response_dict = {
                    "executive_summary": flow_response.executive_summary,
                    "detailed_analysis": flow_response.detailed_analysis,
                    "agent_perspectives": [
                        {
                            "role": perspective.role.value if hasattr(perspective.role, 'value') else str(perspective.role),
                            "analysis": perspective.analysis,
                            "key_insights": perspective.key_insights,
                            "recommendations": perspective.recommendations,
                            "confidence": perspective.confidence,
                            "focus_areas": perspective.focus_areas
                        }
                        for perspective in (flow_response.agent_perspectives or [])
                    ],
                    "synthesis": flow_response.synthesis,
                    "action_items": flow_response.action_items or [],
                    "follow_up_questions": flow_response.follow_up_questions or []
                }

                # Debug log final response dictionary before sending to client
                self._log_final_response_dict_debug(request.query, flow_response_dict, total_time, agent_count, stream_id)

                await stream_processor.emit_processing_complete(stream_id, flow_response_dict, total_time, agent_count)
            else:
                # Fallback to search results
                results = []
                for chunk, score in selected_chunks[:request.limit]:
                    result = QueryResult(
                        chunk=chunk,
                        score=score,
                        context_chunks=[]
                    )
                    results.append(result)

                await stream_processor.emit_synthesis_complete(stream_id, results)

                await stream_processor.emit_user_message(
                    stream_id,
                    f"🎉 Analysis complete! Found {len(results)} relevant code sections in {total_time:.1f} seconds.",
                    100.0
                )

                await stream_processor.emit_processing_complete(stream_id, results, total_time, len(results))

        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
            await stream_processor.emit_event(
                stream_id,
                StreamEventType.ERROR,
                {"error": str(e)},
                f"Processing error: {e}"
            )

    async def _run_agents_with_streaming(self, stream_id: str, agent_tasks: List, query: str,
                                       chunks: List[CodeChunk], context: Dict[str, Any]):
        """Run agents with streaming progress updates."""
        try:
            # Convert agent tasks to agent roles for compatibility
            agent_roles = [task.agent_role for task in agent_tasks]

            # Emit start events for each agent
            for i, task in enumerate(agent_tasks):
                await stream_processor.emit_agent_start(
                    stream_id, task.agent_role, task.task_description, i, len(agent_tasks)
                )

            # Run agents (this will use the enhanced orchestrator)
            # Note: analysis_result should be passed from the calling function
            available_chunks = len(chunks) if chunks else 0
            analysis_result = await self.query_analyzer.analyze_query(query, available_chunks)

            orchestration_result = await self.enhanced_orchestrator.orchestrate_agents(
                query=query,
                analysis_result=analysis_result,
                context={}  # Empty context for this call
            )

            # Extract FlowResponse from OrchestrationResult
            flow_response = orchestration_result.final_response if orchestration_result else None

            # Emit completion events
            if flow_response and hasattr(flow_response, 'agent_perspectives'):
                for i, perspective in enumerate(flow_response.agent_perspectives):
                    await stream_processor.emit_agent_complete(
                        stream_id, perspective.role, perspective.confidence, i, len(agent_tasks)
                    )

            return flow_response

        except Exception as e:
            logger.error(f"Error in agent streaming: {e}")
            await stream_processor.emit_event(
                stream_id,
                StreamEventType.ERROR,
                {"error": str(e)},
                f"Agent processing error: {e}"
            )
            return None

    async def _run_optimized_agent_analysis(self, query: str, chunks: List[CodeChunk],
                                          context: Dict[str, Any], agent_tasks: List) -> Any:
        """Run optimized agent analysis using intelligent task assignments."""
        try:
            # Create a custom context with specific agent tasks
            optimized_context = context.copy()
            optimized_context["agent_tasks"] = agent_tasks
            optimized_context["optimization_mode"] = True

            # Log the optimization
            agent_names = [task.agent_role.value for task in agent_tasks]
            logger.info(f"Running optimized analysis with agents: {', '.join(agent_names)}")

            # Analyze query for orchestration
            available_chunks = len(chunks) if chunks else 0
            analysis_result = await self.query_analyzer.analyze_query(query, available_chunks)

            # Use the enhanced agent orchestrator with optimized context
            orchestration_result = await self.enhanced_orchestrator.orchestrate_agents(
                query=query,
                analysis_result=analysis_result,
                context=optimized_context
            )

            # Extract FlowResponse from OrchestrationResult
            return orchestration_result.final_response if orchestration_result else None

        except Exception as e:
            logger.error(f"Error in optimized agent analysis: {e}")
            # Fallback to enhanced analysis
            available_chunks = len(chunks) if chunks else 0
            analysis_result = await self.query_analyzer.analyze_query(query, available_chunks)
            fallback_result = await self.enhanced_orchestrator.orchestrate_agents(
                query=query,
                analysis_result=analysis_result,
                context=context
            )
            # Extract FlowResponse from fallback OrchestrationResult
            return fallback_result.final_response if fallback_result else None

    def _log_orchestration_result_debug(self, query: str, orchestration_result, stream_id: str):
        """Log comprehensive debug information about the orchestration result."""
        logger.info("🔍 === SERVER ORCHESTRATION RESULT DEBUG ===")
        logger.info(f"📝 Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        logger.info(f"🆔 Stream ID: {stream_id}")
        logger.info(f"⏱️ Total Processing Time: {orchestration_result.total_processing_time:.3f}s")
        logger.info(f"👥 Agent Summary: {orchestration_result.successful_agents}/{orchestration_result.total_agents} successful")

        # Log agent results summary
        for i, agent_result in enumerate(orchestration_result.agent_results):
            status = "✅" if agent_result.success else "❌"
            logger.info(f"  {status} Agent {i+1}: {agent_result.agent_role.value} - {agent_result.processing_time:.2f}s")
            if agent_result.success and agent_result.perspective:
                insights = len(agent_result.perspective.key_insights) if agent_result.perspective.key_insights else 0
                recommendations = len(agent_result.perspective.recommendations) if agent_result.perspective.recommendations else 0
                logger.info(f"    📊 Output: {insights} insights, {recommendations} recommendations, confidence {agent_result.perspective.confidence:.2f}")

        # Log final response structure
        if orchestration_result.final_response:
            fr = orchestration_result.final_response
            logger.info("📋 Final Response Structure:")
            logger.info(f"  📄 Executive Summary: {len(fr.executive_summary)} chars")
            logger.info(f"  📖 Detailed Analysis: {len(fr.detailed_analysis)} chars")
            logger.info(f"  🔄 Synthesis: {len(fr.synthesis)} chars")
            logger.info(f"  👁️ Agent Perspectives: {len(fr.agent_perspectives)} perspectives")
            logger.info(f"  ✅ Action Items: {len(fr.action_items)} items")
            logger.info(f"  ❓ Follow-up Questions: {len(fr.follow_up_questions)} questions")

            # Log content quality metrics
            total_content_length = len(fr.executive_summary) + len(fr.detailed_analysis) + len(fr.synthesis)
            avg_perspective_length = sum(len(p.analysis) for p in fr.agent_perspectives) / len(fr.agent_perspectives) if fr.agent_perspectives else 0
            logger.info(f"📏 Content Metrics: {total_content_length} total chars, {avg_perspective_length:.0f} avg perspective length")
        else:
            logger.warning("❌ Final Response: None - orchestration failed to generate response")

        # Log orchestration logs count
        logger.info(f"📝 Orchestration Logs: {len(orchestration_result.orchestration_logs)} entries")
        logger.info("🔍 === END SERVER ORCHESTRATION RESULT DEBUG ===")

    def _log_final_response_dict_debug(self, query: str, flow_response_dict: dict, total_time: float, agent_count: int, stream_id: str):
        """Log comprehensive debug information about the final response dictionary being sent to client."""
        logger.info("🚀 === FINAL RESPONSE DICT DEBUG (TO CLIENT) ===")
        logger.info(f"📝 Query: {query[:150]}{'...' if len(query) > 150 else ''}")
        logger.info(f"🆔 Stream ID: {stream_id}")
        logger.info(f"⏱️ Total Time: {total_time:.3f}s")
        logger.info(f"👥 Agent Count: {agent_count}")

        # Log response structure details
        logger.info("📋 Response Dictionary Structure:")
        logger.info(f"  📄 Executive Summary: {len(flow_response_dict.get('executive_summary', ''))} chars")
        logger.info(f"  📖 Detailed Analysis: {len(flow_response_dict.get('detailed_analysis', ''))} chars")
        logger.info(f"  🔄 Synthesis: {len(flow_response_dict.get('synthesis', ''))} chars")
        logger.info(f"  ✅ Action Items: {len(flow_response_dict.get('action_items', []))} items")
        logger.info(f"  ❓ Follow-up Questions: {len(flow_response_dict.get('follow_up_questions', []))} questions")

        # Log agent perspectives details
        perspectives = flow_response_dict.get('agent_perspectives', [])
        logger.info(f"  👁️ Agent Perspectives: {len(perspectives)} perspectives")
        for i, perspective in enumerate(perspectives):
            role = perspective.get('role', 'unknown')
            analysis_len = len(perspective.get('analysis', ''))
            insights_count = len(perspective.get('key_insights', []))
            recommendations_count = len(perspective.get('recommendations', []))
            confidence = perspective.get('confidence', 0)
            focus_areas_count = len(perspective.get('focus_areas', []))
            logger.info(f"    Perspective {i+1} ({role}): {analysis_len} chars, {insights_count} insights, {recommendations_count} recs, conf {confidence:.2f}, {focus_areas_count} focus areas")

        # Log content quality metrics
        total_text_content = (
            len(flow_response_dict.get('executive_summary', '')) +
            len(flow_response_dict.get('detailed_analysis', '')) +
            len(flow_response_dict.get('synthesis', '')) +
            sum(len(p.get('analysis', '')) for p in perspectives)
        )
        total_structured_content = (
            len(flow_response_dict.get('action_items', [])) +
            len(flow_response_dict.get('follow_up_questions', [])) +
            sum(len(p.get('key_insights', [])) + len(p.get('recommendations', [])) for p in perspectives)
        )

        logger.info(f"📏 Content Quality Metrics:")
        logger.info(f"  📝 Total Text Content: {total_text_content} characters")
        logger.info(f"  📊 Total Structured Items: {total_structured_content} items")
        logger.info(f"  ⚡ Content per Second: {total_text_content / total_time:.0f} chars/sec")

        # Log sample content (first 100 chars of each section)
        logger.info("📖 Content Samples:")
        exec_summary = flow_response_dict.get('executive_summary', '')
        if exec_summary:
            logger.info(f"  📄 Executive Summary: {exec_summary[:100]}{'...' if len(exec_summary) > 100 else ''}")

        synthesis = flow_response_dict.get('synthesis', '')
        if synthesis:
            logger.info(f"  🔄 Synthesis: {synthesis[:100]}{'...' if len(synthesis) > 100 else ''}")

        logger.info("🚀 === END FINAL RESPONSE DICT DEBUG ===")

    def _log_regular_query_result_debug(self, query: str, response: QueryResponse, processing_time: float):
        """Log comprehensive debug information about regular query results."""
        logger.info("🔍 === REGULAR QUERY RESULT DEBUG ===")
        logger.info(f"📝 Query: {query[:150]}{'...' if len(query) > 150 else ''}")
        logger.info(f"⏱️ Processing Time: {processing_time:.3f}s")
        logger.info(f"📊 Total Results: {response.total_results}")
        logger.info(f"🤖 Model Used: {response.model_used}")

        # Log results details
        logger.info("📋 Results Structure:")
        for i, result in enumerate(response.results[:5]):  # Log first 5 results
            chunk = result.chunk
            logger.info(f"  Result {i+1}: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line}) - Score: {result.score:.3f}")
            logger.info(f"    Content: {len(chunk.content)} chars, Type: {chunk.node_type}, Name: {chunk.name}")
            if result.context_chunks:
                logger.info(f"    Context: {len(result.context_chunks)} context chunks")

        if len(response.results) > 5:
            logger.info(f"  ... and {len(response.results) - 5} more results")

        # Log analysis details if present
        if response.analysis:
            logger.info("🔬 Analysis Details:")
            analysis = response.analysis
            if hasattr(analysis, 'complexity'):
                logger.info(f"  🎯 Complexity: {analysis.complexity}")
            if hasattr(analysis, 'agents_used'):
                logger.info(f"  👥 Agents Used: {analysis.agents_used}")
            if hasattr(analysis, 'agents_skipped'):
                logger.info(f"  ⏭️ Agents Skipped: {analysis.agents_skipped}")

        # Log performance metrics
        if processing_time > 0:
            results_per_second = response.total_results / processing_time
            logger.info(f"⚡ Performance: {results_per_second:.1f} results/second")

        # Log content metrics
        total_content_chars = sum(len(result.chunk.content) for result in response.results)
        avg_content_length = total_content_chars / len(response.results) if response.results else 0
        logger.info(f"📏 Content Metrics: {total_content_chars} total chars, {avg_content_length:.0f} avg chars per result")

        logger.info("🔍 === END REGULAR QUERY RESULT DEBUG ===")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
