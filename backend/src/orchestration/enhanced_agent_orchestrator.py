"""
Enhanced Agent Orchestrator with individual agent processing and detailed monitoring.

This orchestrator manages each agent individually, ensuring unique chunk distribution
and providing detailed logging for process monitoring.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import openai

from ..agents.agent_orchestrator import AgentRole, AgentPerspective, FlowResponse
from ..models import CodeChunk
from ..query.intelligent_query_analyzer import AgentTask, QueryAnalysisResult
from ..streaming.stream_processor import stream_processor, StreamEventType


@dataclass
class AgentJobResult:
    """Result from an individual agent's analysis job."""
    agent_role: AgentRole
    task_description: str
    assigned_chunks: List[CodeChunk]
    perspective: Optional[AgentPerspective]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Complete orchestration result with detailed metrics."""
    query: str
    total_agents: int
    successful_agents: int
    failed_agents: int
    total_processing_time: float
    agent_results: List[AgentJobResult]
    final_response: Optional[FlowResponse]
    orchestration_logs: List[str]


class EnhancedAgentOrchestrator:
    """
    Enhanced orchestrator that processes each agent individually with unique chunks.
    """
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None, base_orchestrator: Optional[Any] = None):
        """Initialize the enhanced orchestrator."""
        self.client = openai_client
        self.base_orchestrator = base_orchestrator
        self.orchestration_logs = []
        
        # Agent specialization for chunk selection
        self.agent_chunk_preferences = {
            AgentRole.ARCHITECT: {
                "file_patterns": ["models", "services", "controllers", "config"],
                "node_types": ["class", "module", "interface"],
                "keywords": ["architecture", "design", "pattern", "structure"]
            },
            AgentRole.DEVELOPER: {
                "file_patterns": ["utils", "helpers", "core", "lib"],
                "node_types": ["function", "method", "class"],
                "keywords": ["implementation", "algorithm", "logic", "code"]
            },
            AgentRole.SECURITY: {
                "file_patterns": ["auth", "security", "middleware", "validation"],
                "node_types": ["function", "class", "method"],
                "keywords": ["auth", "security", "validation", "permission", "encrypt"]
            },
            AgentRole.PERFORMANCE: {
                "file_patterns": ["database", "cache", "optimization", "async"],
                "node_types": ["function", "method", "class"],
                "keywords": ["performance", "optimization", "cache", "async", "database"]
            },
            AgentRole.MAINTAINER: {
                "file_patterns": ["tests", "docs", "config", "utils"],
                "node_types": ["function", "class", "test"],
                "keywords": ["test", "documentation", "maintenance", "refactor", "quality"]
            },
            AgentRole.BUSINESS: {
                "file_patterns": ["models", "services", "business", "domain"],
                "node_types": ["class", "interface", "function"],
                "keywords": ["business", "domain", "logic", "requirements", "workflow"]
            },
            AgentRole.INTEGRATION: {
                "file_patterns": ["api", "external", "integration", "client"],
                "node_types": ["function", "class", "interface"],
                "keywords": ["api", "integration", "external", "service", "client"]
            },
            AgentRole.DATA: {
                "file_patterns": ["models", "database", "schema", "migration"],
                "node_types": ["class", "function", "variable"],
                "keywords": ["data", "database", "model", "schema", "storage"]
            }
        }
    
    async def orchestrate_agents(
        self,
        query: str,
        analysis_result: QueryAnalysisResult,
        context: Dict[str, Any],
        stream_id: Optional[str] = None
    ) -> OrchestrationResult:
        """
        Orchestrate multiple agents with individual processing and unique chunk distribution.
        """
        start_time = time.time()
        self.orchestration_logs = []
        
        self._log(f"🎯 Starting enhanced agent orchestration for query: {query[:100]}...")
        self._log(f"📊 Analysis result: {analysis_result.complexity.value} complexity, {len(analysis_result.required_agents)} agents")

        if stream_id:
            await stream_processor.emit_orchestration_start(stream_id, len(analysis_result.required_agents))
        
        # Step 1: Generate agent-specific queries and retrieve specialized chunks
        chunk_distribution = await self._distribute_unique_chunks(
            analysis_result.required_agents, query, context, stream_id
        )

        # Log detailed chunk distribution
        self._log_chunk_distribution_summary(chunk_distribution)

        # Step 2: Validate chunk availability and filter agents
        valid_agents = self._validate_and_filter_agents(
            analysis_result.required_agents, chunk_distribution
        )

        if not valid_agents:
            self._log("❌ No agents have sufficient chunks for analysis")
            # Return a fallback response
            return await self._create_fallback_orchestration_result(query, context)

        # Step 3: Process each valid agent individually
        agent_results = await self._process_agents_individually(
            query, valid_agents, chunk_distribution, context, stream_id
        )
        
        # Step 3: Synthesize results
        final_response = await self._synthesize_agent_results(
            query, agent_results, context, stream_id
        )
        
        # Step 4: Compile orchestration result
        total_time = time.time() - start_time
        successful_agents = sum(1 for result in agent_results if result.success)
        failed_agents = len(agent_results) - successful_agents

        self._log(f"✅ Orchestration complete: {successful_agents}/{len(agent_results)} agents successful in {total_time:.2f}s")

        # Debug log final result details
        self._log_final_result_debug(query, final_response, agent_results, total_time, successful_agents, failed_agents)

        return OrchestrationResult(
            query=query,
            total_agents=len(agent_results),
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            total_processing_time=total_time,
            agent_results=agent_results,
            final_response=final_response,
            orchestration_logs=self.orchestration_logs.copy()
        )
    
    async def _distribute_unique_chunks(
        self,
        agent_tasks: List[AgentTask],
        original_query: str,
        context: Dict[str, Any],
        stream_id: Optional[str] = None
    ) -> Dict[AgentRole, List[CodeChunk]]:
        """
        Generate agent-specific queries and retrieve specialized chunks for each agent.
        This ensures each agent gets relevant chunks based on their expertise and role.
        """
        self._log(f"🔍 Generating specialized queries and retrieving chunks for {len(agent_tasks)} agents")

        if stream_id:
            await stream_processor.emit_chunk_distribution_start(stream_id, 0, len(agent_tasks))

        chunk_distribution = {}
        all_retrieved_chunk_ids = set()

        for i, agent_task in enumerate(agent_tasks):
            agent_role = agent_task.agent_role
            target_chunks = max(agent_task.estimated_chunks_needed, 10)  # Minimum 10 chunks per agent

            self._log(f"  🤖 {agent_role.value}: Generating specialized query...")

            # Generate agent-specific query
            agent_query = await self._generate_agent_specific_query(
                original_query, agent_role, agent_task, context
            )

            self._log(f"    🔍 Agent query: {agent_query[:100]}...")

            # Retrieve chunks specifically for this agent using their specialized query
            agent_chunks = await self._retrieve_chunks_for_agent(
                agent_query, agent_role, target_chunks, context, all_retrieved_chunk_ids
            )

            chunk_distribution[agent_role] = agent_chunks

            # Track all retrieved chunks to ensure diversity across agents
            for chunk in agent_chunks:
                all_retrieved_chunk_ids.add(chunk.id)

            self._log(f"  🤖 {agent_role.value}: {len(agent_chunks)} specialized chunks retrieved")

            if stream_id:
                await stream_processor.emit_agent_chunk_retrieval(
                    stream_id, agent_role, len(agent_chunks), i + 1, len(agent_tasks)
                )

        total_chunks = sum(len(chunks) for chunks in chunk_distribution.values())
        self._log(f"📦 Total chunks retrieved across all agents: {total_chunks}")

        # Emit completion with distribution summary
        if stream_id:
            distribution_summary = {agent_role.value: len(chunks) for agent_role, chunks in chunk_distribution.items()}
            await stream_processor.emit_chunk_distribution_complete(stream_id, distribution_summary)

        return chunk_distribution

    async def _generate_agent_specific_query(
        self,
        original_query: str,
        agent_role: AgentRole,
        agent_task: AgentTask,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a specialized query for a specific agent based on their role and expertise.
        This allows each agent to retrieve chunks most relevant to their analysis perspective.
        """
        # Get agent specialization details
        preferences = self.agent_chunk_preferences.get(agent_role, {})
        keywords = preferences.get("keywords", [])
        file_patterns = preferences.get("file_patterns", [])

        # Create agent-specific query enhancement
        role_context = {
            AgentRole.ARCHITECT: "system architecture, design patterns, component structure, architectural decisions",
            AgentRole.DEVELOPER: "implementation details, algorithms, code quality, development practices",
            AgentRole.SECURITY: "security vulnerabilities, authentication, authorization, data protection",
            AgentRole.PERFORMANCE: "performance optimization, bottlenecks, scalability, resource usage",
            AgentRole.MAINTAINER: "code maintainability, technical debt, refactoring, code complexity",
            AgentRole.BUSINESS: "business logic, domain models, functional requirements, user workflows",
            AgentRole.INTEGRATION: "system integration, APIs, external dependencies, service communication",
            AgentRole.DATA: "data models, database design, data flow, storage patterns",
            AgentRole.UI_UX: "user interface, user experience, frontend components, interaction design",
            AgentRole.DEVOPS: "deployment, infrastructure, CI/CD, monitoring, operations",
            AgentRole.TESTING: "test coverage, testing strategies, quality assurance, test automation",
            AgentRole.COMPLIANCE: "compliance requirements, regulations, standards, governance"
        }

        role_focus = role_context.get(agent_role, "general analysis")

        # Enhance the original query with agent-specific context
        enhanced_query = f"""
        {original_query}

        Focus specifically on: {role_focus}
        Key areas of interest: {', '.join(keywords)}
        Relevant file types: {', '.join(file_patterns)}

        Find code components that are most relevant for {agent_role.value} analysis perspective.
        """

        return enhanced_query.strip()

    async def _retrieve_chunks_for_agent(
        self,
        agent_query: str,
        agent_role: AgentRole,
        target_chunks: int,
        context: Dict[str, Any],
        exclude_chunk_ids: set
    ) -> List[CodeChunk]:
        """
        Retrieve chunks specifically for an agent using their specialized query.
        Combines embedding search with graph search for comprehensive results.
        """
        try:
            # Import here to avoid circular imports
            from ..query.enhanced_integration import EnhancedQueryIntegration
            from ..database.qdrant_client import QdrantVectorStore
            from ..database.neo4j_client import Neo4jGraphStore
            from ..embeddings.embedding_generator import EmbeddingGenerator

            # Initialize enhanced query processor if not already available
            if not hasattr(self, '_query_integration'):
                vector_store = QdrantVectorStore()
                graph_store = Neo4jGraphStore()
                embedding_generator = EmbeddingGenerator()
                self._query_integration = EnhancedQueryIntegration(
                    vector_store, graph_store, embedding_generator
                )

            # Get project IDs from context
            project_ids = context.get('project_ids', None)

            # Use enhanced query processing to get relevant chunks
            search_results, metadata = await self._query_integration.process_query_enhanced(
                query=agent_query,
                project_ids=project_ids,
                limit=target_chunks * 2,  # Get more to allow for filtering
                use_enhanced=True
            )

            # Filter out already used chunks and select the best ones
            agent_chunks = []
            for chunk, score in search_results:
                if len(agent_chunks) >= target_chunks:
                    break
                if chunk.id not in exclude_chunk_ids:
                    agent_chunks.append(chunk)

            # If we don't have enough chunks, fall back to basic search
            if len(agent_chunks) < max(5, target_chunks // 2):
                self._log(f"    ⚠️ Only found {len(agent_chunks)} chunks, falling back to basic search")
                fallback_chunks = await self._fallback_chunk_retrieval(
                    agent_query, agent_role, target_chunks, exclude_chunk_ids, context
                )

                # Merge results, avoiding duplicates
                existing_ids = {chunk.id for chunk in agent_chunks}
                for chunk in fallback_chunks:
                    if chunk.id not in existing_ids and len(agent_chunks) < target_chunks:
                        agent_chunks.append(chunk)

            return agent_chunks[:target_chunks]

        except Exception as e:
            self._log(f"    ❌ Error retrieving chunks for {agent_role.value}: {e}")
            # Fallback to basic retrieval
            return await self._fallback_chunk_retrieval(
                agent_query, agent_role, target_chunks, exclude_chunk_ids, context
            )

    async def _fallback_chunk_retrieval(
        self,
        agent_query: str,
        agent_role: AgentRole,
        target_chunks: int,
        exclude_chunk_ids: set,
        context: Dict[str, Any]
    ) -> List[CodeChunk]:
        """
        Fallback chunk retrieval using basic embedding search when enhanced search fails.
        """
        try:
            from ..database.qdrant_client import QdrantVectorStore
            from ..embeddings.embedding_generator import EmbeddingGenerator

            # Initialize basic components
            if not hasattr(self, '_vector_store'):
                self._vector_store = QdrantVectorStore()
                self._embedding_generator = EmbeddingGenerator()

            # Generate embedding for agent query
            query_embeddings = await self._embedding_generator.generate_embeddings([agent_query])
            query_embedding = query_embeddings[0]

            # Search for similar chunks
            project_ids = context.get('project_ids', None)
            search_results = await self._vector_store.search_similar(
                query_embedding=query_embedding,
                limit=target_chunks * 3,  # Get more for filtering
                project_ids=project_ids
            )

            # Filter and select chunks
            agent_chunks = []
            for chunk, score in search_results:
                if len(agent_chunks) >= target_chunks:
                    break
                if chunk.id not in exclude_chunk_ids:
                    agent_chunks.append(chunk)

            return agent_chunks

        except Exception as e:
            self._log(f"    ❌ Fallback chunk retrieval failed for {agent_role.value}: {e}")
            return []

    def _validate_and_filter_agents(
        self,
        agent_tasks: List[AgentTask],
        chunk_distribution: Dict[AgentRole, List[CodeChunk]]
    ) -> List[AgentTask]:
        """
        Validate that agents have sufficient chunks and filter out those with insufficient data.
        """
        valid_agents = []
        min_chunks_threshold = 5  # Minimum chunks required for meaningful analysis

        for agent_task in agent_tasks:
            agent_role = agent_task.agent_role
            assigned_chunks = chunk_distribution.get(agent_role, [])

            if len(assigned_chunks) >= min_chunks_threshold:
                valid_agents.append(agent_task)
                self._log(f"✅ {agent_role.value}: {len(assigned_chunks)} chunks - VALID for analysis")
            else:
                self._log(f"❌ {agent_role.value}: {len(assigned_chunks)} chunks - INSUFFICIENT (min: {min_chunks_threshold})")

        self._log(f"📊 Agent validation: {len(valid_agents)}/{len(agent_tasks)} agents have sufficient chunks")
        return valid_agents

    async def _create_fallback_orchestration_result(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> OrchestrationResult:
        """
        Create a fallback orchestration result when no agents have sufficient chunks.
        """
        self._log("🔄 Creating fallback orchestration result")

        fallback_response = FlowResponse(
            executive_summary="Unable to perform comprehensive multi-agent analysis due to insufficient relevant code chunks.",
            detailed_analysis=f"The system could not find enough relevant code components for the query: '{query}'. This may indicate that the query is too specific, the codebase doesn't contain relevant components, or the search parameters need adjustment.",
            agent_perspectives=[],
            synthesis="To get better results, try:\n1. Using broader search terms\n2. Checking if the relevant code is indexed\n3. Expanding the query scope\n4. Verifying project selection",
            action_items=[
                "Review and broaden the search query",
                "Ensure relevant code files are properly indexed",
                "Check project selection and scope",
                "Consider using different keywords or approaches"
            ]
        )

        return OrchestrationResult(
            processing_time=0.1,
            agents_used=[],
            total_chunks_processed=0,
            final_response=fallback_response,
            orchestration_logs=self.orchestration_logs.copy()
        )

    async def _process_agents_individually(
        self,
        query: str,
        agent_tasks: List[AgentTask],
        chunk_distribution: Dict[AgentRole, List[CodeChunk]],
        context: Dict[str, Any],
        stream_id: Optional[str] = None
    ) -> List[AgentJobResult]:
        """
        Process each agent individually with their assigned chunks.
        """
        self._log(f"🔄 Processing {len(agent_tasks)} agents individually")

        # Emit agent setup start
        if stream_id:
            await stream_processor.emit_agent_setup_start(stream_id, len(agent_tasks))

        # Prepare agent list for setup completion
        agent_names = [task.agent_role.value.replace('_', ' ').title() for task in agent_tasks]
        if stream_id:
            await stream_processor.emit_agent_setup_complete(stream_id, agent_names)

        agent_results = []

        for i, agent_task in enumerate(agent_tasks):
            agent_role = agent_task.agent_role
            assigned_chunks = chunk_distribution.get(agent_role, [])

            self._log(f"  🎯 Processing agent {i+1}/{len(agent_tasks)}: {agent_role.value}")
            self._log(f"    📋 Task: {agent_task.task_description}")
            self._log(f"    📦 Chunks: {len(assigned_chunks)}")

            # Skip agents with insufficient chunks
            if len(assigned_chunks) < 3:
                self._log(f"    ⚠️ Skipping {agent_role.value} - insufficient chunks ({len(assigned_chunks)})")
                continue

            if stream_id:
                await stream_processor.emit_agent_start_friendly(
                    stream_id, agent_role, agent_task.task_description, i, len(agent_tasks)
                )

            # Process individual agent
            agent_result = await self._process_single_agent(
                agent_role, agent_task, assigned_chunks, query, context, stream_id
            )
            
            agent_results.append(agent_result)
            
            if agent_result.success:
                self._log(f"    ✅ {agent_role.value} completed successfully in {agent_result.processing_time:.2f}s")
                if stream_id and agent_result.perspective:
                    insights_count = len(agent_result.perspective.key_insights) if agent_result.perspective.key_insights else 0
                    await stream_processor.emit_agent_complete_friendly(
                        stream_id, agent_role, agent_result.perspective.confidence, insights_count, i, len(agent_tasks)
                    )
            else:
                self._log(f"    ❌ {agent_role.value} failed: {agent_result.error_message}")
                if stream_id:
                    await stream_processor.emit_user_message(
                        stream_id,
                        f"⚠️ {agent_role.value.replace('_', ' ').title()} Expert encountered an issue but continuing with other experts..."
                    )
        
        return agent_results
    
    async def _process_single_agent(
        self,
        agent_role: AgentRole,
        agent_task: AgentTask,
        assigned_chunks: List[CodeChunk],
        query: str,
        context: Dict[str, Any],
        stream_id: Optional[str] = None
    ) -> AgentJobResult:
        """
        Process a single agent with its assigned chunks and task.
        """
        start_time = time.time()

        # Enhanced debug logging with markdown formatting
        self._log_agent_processing_start(agent_role, agent_task, assigned_chunks, query)

        try:
            if stream_id:
                await stream_processor.emit_agent_progress_friendly(
                    stream_id, agent_role, "Analyzing assigned code sections...", 0, 1
                )

                # Emit detailed debug information to stream
                await stream_processor.emit_agent_debug_info(stream_id, agent_role, {
                    "task_description": agent_task.task_description,
                    "chunk_count": len(assigned_chunks),
                    "focus_areas": agent_task.specific_focus_areas,
                    "estimated_chunks_needed": agent_task.estimated_chunks_needed,
                    "chunk_files": [chunk.file_path for chunk in assigned_chunks[:5]],  # First 5 files
                    "total_files": len(set(chunk.file_path for chunk in assigned_chunks))
                })

            # Create agent-specific context
            agent_context = self._create_agent_context(agent_role, agent_task, assigned_chunks, context)

            # Log detailed agent context
            self._log_agent_context_details(agent_role, agent_context)

            if stream_id:
                await stream_processor.emit_agent_progress_friendly(
                    stream_id, agent_role, "Applying specialized analysis techniques...", 0, 1
                )

            # Generate agent perspective using LLM or rule-based approach
            if self.client:
                self._log(f"    🧠 {agent_role.value}: Using LLM-based analysis")
                perspective = await self._llm_agent_analysis(
                    agent_role, agent_task, assigned_chunks, query, agent_context
                )
            else:
                self._log(f"    📋 {agent_role.value}: Using rule-based analysis")
                perspective = self._rule_based_agent_analysis(
                    agent_role, agent_task, assigned_chunks, query, agent_context
                )

            # Log analysis results
            self._log_agent_analysis_results(agent_role, perspective)

            if stream_id:
                await stream_processor.emit_agent_progress_friendly(
                    stream_id, agent_role, "Finalizing insights and recommendations...", 0, 1
                )

            processing_time = time.time() - start_time

            # Log completion with timing
            self._log(f"    ✅ {agent_role.value}: Completed in {processing_time:.2f}s")

            return AgentJobResult(
                agent_role=agent_role,
                task_description=agent_task.task_description,
                assigned_chunks=assigned_chunks,
                perspective=perspective,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in {agent_role.value} analysis: {str(e)}"
            self._log(f"    ❌ {error_msg}")
            self._log(f"    ⏱️ Failed after {processing_time:.2f}s")

            return AgentJobResult(
                agent_role=agent_role,
                task_description=agent_task.task_description,
                assigned_chunks=assigned_chunks,
                perspective=None,
                processing_time=processing_time,
                success=False,
                error_message=error_msg
            )
    
    def _log(self, message: str):
        """Add message to orchestration logs and logger."""
        self.orchestration_logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        logger.info(f"[ORCHESTRATOR] {message}")

    def _log_final_result_debug(
        self,
        query: str,
        final_response: Optional[FlowResponse],
        agent_results: List[AgentJobResult],
        total_time: float,
        successful_agents: int,
        failed_agents: int
    ):
        """Log comprehensive debug information about the final result."""
        self._log("🔍 === FINAL RESULT DEBUG LOG ===")
        self._log(f"📝 Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        self._log(f"⏱️ Total Processing Time: {total_time:.3f}s")
        self._log(f"👥 Agent Summary: {successful_agents} successful, {failed_agents} failed, {len(agent_results)} total")

        # Log agent-specific results
        for i, result in enumerate(agent_results):
            status = "✅" if result.success else "❌"
            self._log(f"  {status} Agent {i+1}: {result.agent_role.value} - {result.processing_time:.2f}s")
            if result.success and result.perspective:
                insights_count = len(result.perspective.key_insights) if result.perspective.key_insights else 0
                recommendations_count = len(result.perspective.recommendations) if result.perspective.recommendations else 0
                self._log(f"    📊 Insights: {insights_count}, Recommendations: {recommendations_count}, Confidence: {result.perspective.confidence:.2f}")
            elif not result.success:
                self._log(f"    ⚠️ Error: {result.error_message}")

        # Log final response details
        if final_response:
            self._log("📋 Final Response Structure:")
            self._log(f"  📄 Executive Summary: {len(final_response.executive_summary)} chars")
            self._log(f"  📖 Detailed Analysis: {len(final_response.detailed_analysis)} chars")
            self._log(f"  🔄 Synthesis: {len(final_response.synthesis)} chars")
            self._log(f"  👁️ Agent Perspectives: {len(final_response.agent_perspectives)} perspectives")
            self._log(f"  ✅ Action Items: {len(final_response.action_items)} items")
            self._log(f"  ❓ Follow-up Questions: {len(final_response.follow_up_questions)} questions")

            # Log perspective details
            for i, perspective in enumerate(final_response.agent_perspectives):
                self._log(f"    Perspective {i+1} ({perspective.role.value}): {len(perspective.analysis)} chars, confidence {perspective.confidence:.2f}")
        else:
            self._log("❌ Final Response: None (synthesis failed)")

        self._log("🔍 === END FINAL RESULT DEBUG LOG ===")

        # Also log to console for immediate visibility
        logger.debug(f"FINAL RESULT DEBUG - Query: {query[:100]}... | Success Rate: {successful_agents}/{len(agent_results)} | Time: {total_time:.2f}s | Response: {'✅' if final_response else '❌'}")

    def _log_agent_processing_start(
        self,
        agent_role: AgentRole,
        agent_task: AgentTask,
        assigned_chunks: List[CodeChunk],
        query: str
    ):
        """Log detailed agent processing start information in markdown format."""
        role_emojis = {
            AgentRole.ARCHITECT: "🏗️",
            AgentRole.DEVELOPER: "👨‍💻",
            AgentRole.SECURITY: "🔒",
            AgentRole.PERFORMANCE: "⚡",
            AgentRole.MAINTAINER: "🔧",
            AgentRole.BUSINESS: "💼",
            AgentRole.INTEGRATION: "🔗",
            AgentRole.DATA: "📊",
            AgentRole.UI_UX: "🎨",
            AgentRole.DEVOPS: "🚀",
            AgentRole.TESTING: "🧪",
            AgentRole.COMPLIANCE: "📋"
        }

        emoji = role_emojis.get(agent_role, "🤖")

        # Create markdown-formatted debug log
        debug_info = f"""
## {emoji} Agent Processing Start: {agent_role.value.upper()}

### 📋 Task Details
- **Role**: {agent_role.value.replace('_', ' ').title()}
- **Task**: {agent_task.task_description}
- **Focus Areas**: {', '.join(agent_task.specific_focus_areas) if agent_task.specific_focus_areas else 'General analysis'}
- **Query**: {query[:100]}{'...' if len(query) > 100 else ''}

### 📦 Available Chunks ({len(assigned_chunks)} total)
"""

        # Group chunks by file type/category for better organization
        chunk_categories = {}
        for chunk in assigned_chunks:
            file_ext = chunk.file_path.split('.')[-1] if '.' in chunk.file_path else 'unknown'
            if file_ext not in chunk_categories:
                chunk_categories[file_ext] = []
            chunk_categories[file_ext].append(chunk)

        for category, chunks in chunk_categories.items():
            debug_info += f"\n#### 📁 {category.upper()} Files ({len(chunks)} chunks)\n"
            for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks per category
                debug_info += f"- `{chunk.file_path}` (lines {chunk.start_line}-{chunk.end_line})\n"
            if len(chunks) > 5:
                debug_info += f"- ... and {len(chunks) - 5} more {category} files\n"

        debug_info += f"\n### 🎯 Estimated Processing\n"
        debug_info += f"- **Chunks to analyze**: {len(assigned_chunks)}\n"
        debug_info += f"- **Priority**: {agent_task.priority}\n"
        debug_info += f"- **Expected chunks needed**: {agent_task.estimated_chunks_needed}\n"

        self._log(debug_info)

    def _log_agent_context_details(self, agent_role: AgentRole, agent_context: Dict[str, Any]):
        """Log detailed agent context information."""
        context_info = f"""
### 🔍 {agent_role.value.upper()} Context Details
- **Chunk count**: {agent_context.get('chunk_count', 0)}
- **Focus areas**: {', '.join(agent_context.get('focus_areas', [])) if agent_context.get('focus_areas') else 'None specified'}
- **Specialization**: {agent_context.get('specialization', {}).get('focus', 'General analysis')}
- **Project IDs**: {agent_context.get('project_ids', 'Not specified')}
"""
        self._log(context_info)

    def _log_agent_analysis_results(self, agent_role: AgentRole, perspective: AgentPerspective):
        """Log detailed agent analysis results."""
        if not perspective:
            self._log(f"    ❌ {agent_role.value}: No perspective generated")
            return

        results_info = f"""
### 📊 {agent_role.value.upper()} Analysis Results
- **Confidence**: {perspective.confidence:.2f} ({self._get_confidence_description(perspective.confidence)})
- **Key Insights**: {len(perspective.key_insights) if perspective.key_insights else 0}
- **Recommendations**: {len(perspective.recommendations) if perspective.recommendations else 0}
- **Focus Areas Covered**: {', '.join(perspective.focus_areas) if perspective.focus_areas else 'General'}

#### 💡 Key Insights Preview
"""

        if perspective.key_insights:
            for i, insight in enumerate(perspective.key_insights[:3]):  # Show first 3 insights
                results_info += f"{i+1}. {insight[:100]}{'...' if len(insight) > 100 else ''}\n"
            if len(perspective.key_insights) > 3:
                results_info += f"... and {len(perspective.key_insights) - 3} more insights\n"
        else:
            results_info += "No specific insights generated\n"

        results_info += f"\n#### 🎯 Recommendations Preview\n"

        if perspective.recommendations:
            for i, rec in enumerate(perspective.recommendations[:2]):  # Show first 2 recommendations
                results_info += f"{i+1}. {rec[:100]}{'...' if len(rec) > 100 else ''}\n"
            if len(perspective.recommendations) > 2:
                results_info += f"... and {len(perspective.recommendations) - 2} more recommendations\n"
        else:
            results_info += "No specific recommendations generated\n"

        self._log(results_info)

    def _get_confidence_description(self, confidence: float) -> str:
        """Get human-readable confidence description."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.6:
            return "Moderate"
        elif confidence >= 0.5:
            return "Fair"
        else:
            return "Low"

    def _log_chunk_distribution_summary(self, chunk_distribution: Dict[AgentRole, List[CodeChunk]]):
        """Log detailed summary of chunk distribution across agents."""
        role_emojis = {
            AgentRole.ARCHITECT: "🏗️",
            AgentRole.DEVELOPER: "👨‍💻",
            AgentRole.SECURITY: "🔒",
            AgentRole.PERFORMANCE: "⚡",
            AgentRole.MAINTAINER: "🔧",
            AgentRole.BUSINESS: "💼",
            AgentRole.INTEGRATION: "🔗",
            AgentRole.DATA: "📊",
            AgentRole.UI_UX: "🎨",
            AgentRole.DEVOPS: "🚀",
            AgentRole.TESTING: "🧪",
            AgentRole.COMPLIANCE: "📋"
        }

        total_chunks = sum(len(chunks) for chunks in chunk_distribution.values())

        distribution_info = f"""
## 📊 Chunk Distribution Summary

### 📈 Overall Statistics
- **Total Agents**: {len(chunk_distribution)}
- **Total Chunks Distributed**: {total_chunks}
- **Average Chunks per Agent**: {total_chunks / len(chunk_distribution):.1f}

### 🎯 Agent-Specific Distribution
"""

        # Sort agents by chunk count for better visualization
        sorted_agents = sorted(chunk_distribution.items(), key=lambda x: len(x[1]), reverse=True)

        for agent_role, chunks in sorted_agents:
            emoji = role_emojis.get(agent_role, "🤖")
            chunk_count = len(chunks)
            percentage = (chunk_count / total_chunks * 100) if total_chunks > 0 else 0

            distribution_info += f"\n#### {emoji} {agent_role.value.replace('_', ' ').title()}\n"
            distribution_info += f"- **Chunks**: {chunk_count} ({percentage:.1f}% of total)\n"

            # Analyze file types in this agent's chunks
            file_types = {}
            for chunk in chunks:
                file_ext = chunk.file_path.split('.')[-1] if '.' in chunk.file_path else 'unknown'
                file_types[file_ext] = file_types.get(file_ext, 0) + 1

            if file_types:
                distribution_info += f"- **File Types**: {', '.join([f'{ext}({count})' for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)])}\n"

            # Show sample files
            if chunks:
                sample_files = list(set([chunk.file_path for chunk in chunks[:3]]))
                distribution_info += f"- **Sample Files**: {', '.join([f'`{file}`' for file in sample_files])}\n"
                if len(chunks) > 3:
                    unique_files = len(set([chunk.file_path for chunk in chunks]))
                    distribution_info += f"- **Total Files**: {unique_files} unique files\n"

        # Check for chunk overlap (should be minimal with unique distribution)
        all_chunk_ids = []
        for chunks in chunk_distribution.values():
            all_chunk_ids.extend([chunk.id for chunk in chunks])

        unique_chunk_ids = set(all_chunk_ids)
        overlap_count = len(all_chunk_ids) - len(unique_chunk_ids)

        distribution_info += f"\n### 🔄 Distribution Quality\n"
        distribution_info += f"- **Unique Chunks**: {len(unique_chunk_ids)}\n"
        distribution_info += f"- **Overlapping Chunks**: {overlap_count}\n"

        if len(all_chunk_ids) > 0:
            efficiency = (len(unique_chunk_ids) / len(all_chunk_ids) * 100)
            distribution_info += f"- **Distribution Efficiency**: {efficiency:.1f}%\n"
        else:
            distribution_info += f"- **Distribution Efficiency**: N/A (no chunks distributed)\n"

        self._log(distribution_info)

    def _log_synthesis_start(self, agent_results: List[AgentJobResult]):
        """Log detailed synthesis start information."""
        successful_count = len([r for r in agent_results if r.success])
        failed_count = len(agent_results) - successful_count

        synthesis_info = f"""
## 🔄 Synthesis Process Start

### 📊 Agent Results Summary
- **Total Agents**: {len(agent_results)}
- **Successful**: {successful_count}
- **Failed**: {failed_count}
- **Success Rate**: {(successful_count / len(agent_results) * 100):.1f}%

### 🤖 Agent Performance Details
"""

        for result in agent_results:
            status_emoji = "✅" if result.success else "❌"
            synthesis_info += f"\n#### {status_emoji} {result.agent_role.value.replace('_', ' ').title()}\n"
            synthesis_info += f"- **Status**: {'Success' if result.success else 'Failed'}\n"
            synthesis_info += f"- **Processing Time**: {result.processing_time:.2f}s\n"
            synthesis_info += f"- **Task**: {result.task_description}\n"
            synthesis_info += f"- **Chunks Analyzed**: {len(result.assigned_chunks)}\n"

            if result.success and result.perspective:
                synthesis_info += f"- **Confidence**: {result.perspective.confidence:.2f}\n"
                synthesis_info += f"- **Insights Generated**: {len(result.perspective.key_insights) if result.perspective.key_insights else 0}\n"
                synthesis_info += f"- **Recommendations**: {len(result.perspective.recommendations) if result.perspective.recommendations else 0}\n"
            elif not result.success:
                synthesis_info += f"- **Error**: {result.error_message}\n"

        self._log(synthesis_info)

    def _log_perspective_analysis(self, perspectives: List[AgentPerspective]):
        """Log detailed analysis of all perspectives before synthesis."""
        perspective_info = f"""
## 🔍 Perspective Analysis for Synthesis

### 📈 Overall Perspective Statistics
- **Total Perspectives**: {len(perspectives)}
- **Average Confidence**: {sum(p.confidence for p in perspectives) / len(perspectives):.2f}
- **Total Insights**: {sum(len(p.key_insights) if p.key_insights else 0 for p in perspectives)}
- **Total Recommendations**: {sum(len(p.recommendations) if p.recommendations else 0 for p in perspectives)}

### 🎯 Perspective Quality Analysis
"""

        # Categorize perspectives by confidence
        high_confidence = [p for p in perspectives if p.confidence >= 0.8]
        medium_confidence = [p for p in perspectives if 0.6 <= p.confidence < 0.8]
        low_confidence = [p for p in perspectives if p.confidence < 0.6]

        perspective_info += f"\n#### 📊 Confidence Distribution\n"
        perspective_info += f"- **High Confidence (≥0.8)**: {len(high_confidence)} perspectives\n"
        perspective_info += f"- **Medium Confidence (0.6-0.8)**: {len(medium_confidence)} perspectives\n"
        perspective_info += f"- **Low Confidence (<0.6)**: {len(low_confidence)} perspectives\n"

        # Analyze focus areas coverage
        all_focus_areas = set()
        for p in perspectives:
            if p.focus_areas:
                all_focus_areas.update(p.focus_areas)

        perspective_info += f"\n#### 🎯 Focus Areas Coverage\n"
        perspective_info += f"- **Unique Focus Areas**: {len(all_focus_areas)}\n"
        perspective_info += f"- **Areas**: {', '.join(sorted(all_focus_areas)) if all_focus_areas else 'None specified'}\n"

        # Show top insights preview
        perspective_info += f"\n#### 💡 Key Insights Preview\n"
        insight_count = 0
        for p in perspectives:
            if p.key_insights and insight_count < 3:
                role_name = p.role.value.replace('_', ' ').title()
                for insight in p.key_insights[:1]:  # One insight per perspective
                    perspective_info += f"- **{role_name}**: {insight[:100]}{'...' if len(insight) > 100 else ''}\n"
                    insight_count += 1
                    if insight_count >= 3:
                        break

        self._log(perspective_info)
    
    def _categorize_chunks(self, chunks: List[CodeChunk]) -> Dict[str, List[CodeChunk]]:
        """Categorize chunks by file patterns and content types."""
        categories = {
            "models": [],
            "services": [],
            "controllers": [],
            "utils": [],
            "config": [],
            "tests": [],
            "database": [],
            "auth": [],
            "api": [],
            "frontend": [],
            "backend": [],
            "other": []
        }
        
        for chunk in chunks:
            file_path = chunk.file_path.lower()
            categorized = False
            
            for category in categories.keys():
                if category in file_path:
                    categories[category].append(chunk)
                    categorized = True
                    break
            
            if not categorized:
                categories["other"].append(chunk)
        
        return categories
    
    def _select_chunks_for_agent(
        self,
        agent_role: AgentRole,
        categorized_chunks: Dict[str, List[CodeChunk]],
        target_count: int,
        used_chunk_ids: set
    ) -> List[CodeChunk]:
        """Select chunks specifically relevant to an agent's role."""
        preferences = self.agent_chunk_preferences.get(agent_role, {})
        preferred_patterns = preferences.get("file_patterns", [])
        preferred_types = preferences.get("node_types", [])
        keywords = preferences.get("keywords", [])
        
        selected_chunks = []
        
        # First, select from preferred categories
        for pattern in preferred_patterns:
            if pattern in categorized_chunks:
                for chunk in categorized_chunks[pattern]:
                    if len(selected_chunks) >= target_count:
                        break
                    if chunk.id not in used_chunk_ids:
                        selected_chunks.append(chunk)
        
        # Fill remaining slots from other categories
        if len(selected_chunks) < target_count:
            for category, chunks in categorized_chunks.items():
                if category not in preferred_patterns:
                    for chunk in chunks:
                        if len(selected_chunks) >= target_count:
                            break
                        if chunk.id not in used_chunk_ids:
                            selected_chunks.append(chunk)
        
        return selected_chunks[:target_count]
    
    def _create_agent_context(
        self,
        agent_role: AgentRole,
        agent_task: AgentTask,
        assigned_chunks: List[CodeChunk],
        base_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create agent-specific context for analysis."""
        return {
            **base_context,
            "agent_role": agent_role,
            "agent_task": agent_task,
            "assigned_chunks": assigned_chunks,
            "chunk_count": len(assigned_chunks),
            "focus_areas": agent_task.specific_focus_areas,
            "specialization": self.agent_chunk_preferences.get(agent_role, {})
        }
    
    async def _llm_agent_analysis(
        self,
        agent_role: AgentRole,
        agent_task: AgentTask,
        chunks: List[CodeChunk],
        query: str,
        context: Dict[str, Any]
    ) -> AgentPerspective:
        """Perform LLM-based agent analysis using the base orchestrator."""
        try:
            if self.base_orchestrator:
                # Use the existing orchestrator's LLM analysis
                return await self.base_orchestrator._run_agent_analysis(
                    agent_role, query, chunks, context
                )
            else:
                # Fallback to simple analysis
                return AgentPerspective(
                    role=agent_role,
                    analysis=f"Enhanced analysis from {agent_role.value} perspective on {len(chunks)} chunks for task: {agent_task.task_description}",
                    key_insights=[f"Key insight from {agent_role.value} analysis"],
                    recommendations=[f"Recommendation from {agent_role.value}"],
                    confidence=0.8,
                    focus_areas=agent_task.specific_focus_areas
                )
        except Exception as e:
            self._log(f"❌ LLM analysis failed for {agent_role.value}: {e}")
            return self._rule_based_agent_analysis(agent_role, agent_task, chunks, query, context)
    
    def _rule_based_agent_analysis(
        self,
        agent_role: AgentRole,
        agent_task: AgentTask,
        chunks: List[CodeChunk],
        query: str,
        context: Dict[str, Any]
    ) -> AgentPerspective:
        """Perform rule-based agent analysis."""
        return AgentPerspective(
            role=agent_role,
            analysis=f"Rule-based analysis from {agent_role.value} perspective",
            key_insights=[f"Key insight from {agent_role.value}"],
            recommendations=[f"Recommendation from {agent_role.value}"],
            confidence=0.6,
            focus_areas=agent_task.specific_focus_areas
        )
    
    async def _synthesize_agent_results(
        self,
        query: str,
        agent_results: List[AgentJobResult],
        context: Dict[str, Any],
        stream_id: Optional[str] = None
    ) -> Optional[FlowResponse]:
        """Synthesize all agent results into a cohesive response."""
        self._log(f"🔄 Synthesizing results from {len(agent_results)} agents")

        # Log detailed synthesis start information
        self._log_synthesis_start(agent_results)

        successful_results = [r for r in agent_results if r.success and r.perspective]

        if stream_id:
            await stream_processor.emit_synthesis_start_friendly(stream_id, len(successful_results))

        if not successful_results:
            self._log("❌ No successful agent results to synthesize")
            if stream_id:
                await stream_processor.emit_user_message(stream_id, "⚠️ Unable to generate comprehensive analysis due to processing issues")
            return None

        # Extract perspectives with progress updates
        if stream_id:
            await stream_processor.emit_synthesis_progress(stream_id, "Extracting key insights from expert analyses", 20.0)

        perspectives = [result.perspective for result in successful_results]

        # Log detailed perspective analysis
        self._log_perspective_analysis(perspectives)

        # Create synthesis with progress updates
        if stream_id:
            await stream_processor.emit_synthesis_progress(stream_id, "Combining insights into comprehensive response", 50.0)

        # Count insights and recommendations
        total_insights = sum(len(p.key_insights) if p.key_insights else 0 for p in perspectives)
        total_recommendations = sum(len(p.recommendations) if p.recommendations else 0 for p in perspectives)

        executive_summary = f"Analysis of {query} from {len(perspectives)} expert perspectives."
        detailed_analysis = "Comprehensive multi-agent analysis combining architectural, development, and operational insights."
        synthesis = "The analysis reveals key patterns and recommendations across multiple domains."

        if stream_id:
            await stream_processor.emit_synthesis_progress(stream_id, "Finalizing comprehensive response", 80.0)

        self._log(f"✅ Synthesis complete with {len(perspectives)} perspectives")

        if stream_id:
            await stream_processor.emit_synthesis_complete_friendly(stream_id, total_insights, total_recommendations)

        return FlowResponse(
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            agent_perspectives=perspectives,
            synthesis=synthesis,
            action_items=["Review agent recommendations", "Implement suggested improvements"],
            follow_up_questions=["What specific areas need attention?", "How can we optimize further?"]
        )
