"""
Multi-Agent Orchestrator for Comprehensive Code Analysis

This module implements a sophisticated agent-based system that analyzes codebases
from multiple perspectives to provide rich, flowing responses with diverse viewpoints.

Performance Optimizations:
- Smart agent selection based on relevance scoring
- Parallel processing with controlled concurrency
- Early termination for low-relevance agents
- Caching for similar queries
- Resource management for LLM calls
"""

import asyncio
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from ..models import CodeChunk
from ..config import config

# Set up logging
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Different agent roles for multi-perspective analysis."""
    ARCHITECT = "architect"           # System architecture and design patterns
    DEVELOPER = "developer"          # Implementation details and code quality
    SECURITY = "security"            # Security analysis and vulnerabilities
    PERFORMANCE = "performance"      # Performance optimization and bottlenecks
    MAINTAINER = "maintainer"        # Maintainability and technical debt
    BUSINESS = "business"            # Business logic and domain understanding
    INTEGRATION = "integration"      # System integration and dependencies
    DATA = "data"                    # Data modeling and database design
    UI_UX = "ui_ux"                 # User interface and experience
    DEVOPS = "devops"               # Deployment and infrastructure
    TESTING = "testing"             # Testing strategy and quality assurance
    COMPLIANCE = "compliance"        # Regulatory and compliance requirements


@dataclass
class AgentPerspective:
    """Represents an agent's perspective on the codebase."""
    role: AgentRole
    analysis: str
    key_insights: List[str]
    recommendations: List[str]
    confidence: float
    focus_areas: List[str]


@dataclass
class FlowResponse:
    """Structured response with flowing narrative and multiple perspectives."""
    executive_summary: str
    detailed_analysis: str
    agent_perspectives: List[AgentPerspective]
    synthesis: str
    action_items: List[str]
    follow_up_questions: List[str]


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents to provide comprehensive code analysis
    from different perspectives, creating rich, flowing responses.

    Performance Features:
    - Smart agent selection with relevance thresholds
    - Parallel processing with controlled concurrency
    - Query result caching
    - Resource management for LLM calls
    """

    def __init__(self, llm_client=None, max_concurrent_agents=5, cache_size=100):
        self.client = llm_client
        self.agents = self._initialize_agents()
        self.max_concurrent_agents = max_concurrent_agents
        self.cache_size = cache_size
        self.query_cache = {}
        self.cache_access_times = {}

        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'agents_skipped': 0,
            'avg_response_time': 0.0
        }
        
    def _initialize_agents(self) -> Dict[AgentRole, Dict[str, Any]]:
        """Initialize agent configurations with their specializations."""
        return {
            AgentRole.ARCHITECT: {
                "focus": "System architecture, design patterns, component relationships",
                "keywords": ["architecture", "design", "pattern", "structure", "component", "module", "layer"],
                "perspective": "high-level system design and architectural decisions",
                "priority": 1  # Always include for comprehensive analysis
            },
            AgentRole.DEVELOPER: {
                "focus": "Code implementation, algorithms, data structures, best practices",
                "keywords": ["implementation", "algorithm", "function", "class", "method", "code", "logic"],
                "perspective": "technical implementation and coding practices",
                "priority": 1  # Always include for comprehensive analysis
            },
            AgentRole.SECURITY: {
                "focus": "Security vulnerabilities, authentication, authorization, data protection",
                "keywords": ["security", "auth", "permission", "validation", "encryption", "vulnerability", "token"],
                "perspective": "security implications and risk assessment",
                "priority": 2
            },
            AgentRole.PERFORMANCE: {
                "focus": "Performance optimization, scalability, resource utilization",
                "keywords": ["performance", "optimization", "scalability", "bottleneck", "efficiency", "cache", "async"],
                "perspective": "performance characteristics and optimization opportunities",
                "priority": 2
            },
            AgentRole.MAINTAINER: {
                "focus": "Code maintainability, technical debt, refactoring opportunities",
                "keywords": ["maintainability", "refactor", "debt", "complexity", "documentation", "test"],
                "perspective": "long-term maintainability and code health",
                "priority": 1  # Always include for comprehensive analysis
            },
            AgentRole.BUSINESS: {
                "focus": "Business logic, domain modeling, functional requirements",
                "keywords": ["business", "domain", "logic", "requirement", "functionality", "feature", "workflow"],
                "perspective": "business value and domain understanding",
                "priority": 3
            },
            AgentRole.INTEGRATION: {
                "focus": "System integration, dependencies, external services, APIs",
                "keywords": ["integration", "dependency", "api", "service", "external", "interface", "client"],
                "perspective": "system integration and external dependencies",
                "priority": 2
            },
            AgentRole.DATA: {
                "focus": "Data modeling, database design, data flow, storage optimization",
                "keywords": ["database", "data", "model", "schema", "query", "storage", "persistence"],
                "perspective": "data architecture and information management",
                "priority": 2
            },
            AgentRole.UI_UX: {
                "focus": "User interface design, user experience, frontend architecture",
                "keywords": ["ui", "ux", "frontend", "interface", "user", "component", "react", "vue"],
                "perspective": "user experience and interface design",
                "priority": 3
            },
            AgentRole.DEVOPS: {
                "focus": "Deployment, infrastructure, CI/CD, monitoring, scalability",
                "keywords": ["deploy", "infrastructure", "docker", "kubernetes", "ci", "cd", "monitoring"],
                "perspective": "deployment and operational considerations",
                "priority": 3
            },
            AgentRole.TESTING: {
                "focus": "Testing strategy, test coverage, quality assurance, automation",
                "keywords": ["test", "testing", "coverage", "quality", "automation", "mock", "unit"],
                "perspective": "testing strategy and quality assurance",
                "priority": 2
            },
            AgentRole.COMPLIANCE: {
                "focus": "Regulatory compliance, standards adherence, audit requirements",
                "keywords": ["compliance", "regulation", "standard", "audit", "policy", "governance"],
                "perspective": "compliance and regulatory requirements",
                "priority": 4
            }
        }
    
    async def analyze_with_agents(
        self,
        query: str,
        chunks: List[CodeChunk],
        context: Dict[str, Any] = None
    ) -> FlowResponse:
        """
        Orchestrate multiple agents to analyze the codebase from different perspectives.

        Performance optimizations:
        - Smart agent selection with relevance thresholds
        - Parallel processing with controlled concurrency
        - Query caching for similar requests
        - Early termination for low-relevance agents

        Args:
            query: User's question or analysis request
            chunks: Relevant code chunks found by search
            context: Additional context (project info, graph data, etc.)

        Returns:
            FlowResponse with multi-perspective analysis
        """
        start_time = time.time()
        self.performance_stats['total_queries'] += 1

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, chunks)
            if cache_key in self.query_cache:
                self.performance_stats['cache_hits'] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                return self.query_cache[cache_key]

            # Check if we have intelligent agent selection from query analyzer
            if context.get("optimization_mode") and context.get("agent_tasks"):
                # Use intelligent agent selection
                agent_tasks = context["agent_tasks"]
                relevant_agents = [task.agent_role for task in agent_tasks]
                logger.info(f"Using intelligent agent selection: {[a.value for a in relevant_agents]}")
            else:
                # Smart agent selection with performance considerations
                relevant_agents = self._select_relevant_agents_optimized(query, chunks)
                logger.info(f"Selected {len(relevant_agents)} agents: {[a.value for a in relevant_agents]}")

            if not relevant_agents:
                logger.warning("No relevant agents selected for query")
                return self._fallback_response(query, chunks)

            # Run agents with controlled concurrency
            agent_perspectives = await self._run_agents_with_concurrency_control(
                relevant_agents, query, chunks, context
            )

            # Filter out any failed analyses
            valid_perspectives = [
                p for p in agent_perspectives
                if isinstance(p, AgentPerspective) and p.confidence > 0.3
            ]

            if not valid_perspectives:
                logger.warning("No valid agent perspectives generated")
                return self._fallback_response(query, chunks)

            # Synthesize all perspectives into a flowing response
            flow_response = await self._synthesize_perspectives(
                query, valid_perspectives, chunks, context
            )

            # Cache the result
            self._cache_result(cache_key, flow_response)

            # Update performance stats
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)

            return flow_response

        except Exception as e:
            logger.error(f"Error in agent orchestration: {e}")
            return self._fallback_response(query, chunks)

    def _generate_cache_key(self, query: str, chunks: List[CodeChunk]) -> str:
        """Generate a cache key for the query and code chunks."""
        # Create a hash based on query and chunk content
        content_hash = hashlib.md5()
        content_hash.update(query.encode('utf-8'))

        # Include relevant chunk information
        for chunk in chunks[:5]:  # Limit to first 5 chunks for cache key
            content_hash.update(f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}".encode('utf-8'))
            content_hash.update(chunk.content[:500].encode('utf-8'))  # First 500 chars

        return content_hash.hexdigest()

    def _cache_result(self, cache_key: str, result: FlowResponse):
        """Cache the analysis result with LRU eviction."""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache_access_times.keys(), key=self.cache_access_times.get)
            del self.query_cache[oldest_key]
            del self.cache_access_times[oldest_key]

        self.query_cache[cache_key] = result
        self.cache_access_times[cache_key] = time.time()

    def _update_performance_stats(self, response_time: float):
        """Update performance statistics."""
        total_queries = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_response_time']

        # Calculate new average response time
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_stats['avg_response_time'] = new_avg

        logger.info(f"Query completed in {response_time:.2f}s (avg: {new_avg:.2f}s)")

    async def _run_agents_with_concurrency_control(
        self,
        agent_roles: List[AgentRole],
        query: str,
        chunks: List[CodeChunk],
        context: Dict[str, Any]
    ) -> List[AgentPerspective]:
        """Run agents with controlled concurrency and distributed chunk analysis."""
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)

        # Distribute chunks among agents for diverse perspectives
        agent_chunk_assignments = self._distribute_chunks_to_agents(agent_roles, chunks)

        async def run_single_agent(agent_role: AgentRole) -> AgentPerspective:
            async with semaphore:
                # Get the specific chunks assigned to this agent
                assigned_chunks = agent_chunk_assignments.get(agent_role)

                # If no chunks assigned, create a unique fallback for this agent
                if not assigned_chunks:
                    agent_index = list(agent_roles).index(agent_role)
                    chunks_per_fallback_agent = max(15, len(chunks) // len(agent_roles))  # At least 15 chunks per agent
                    start_idx = agent_index * chunks_per_fallback_agent
                    end_idx = min(start_idx + chunks_per_fallback_agent, len(chunks))
                    assigned_chunks = chunks[start_idx:end_idx]

                    # If we run out of chunks, use round-robin distribution
                    if not assigned_chunks and chunks:
                        # Distribute remaining chunks in round-robin fashion
                        assigned_chunks = []
                        for i in range(agent_index, len(chunks), len(agent_roles)):
                            assigned_chunks.append(chunks[i])
                            if len(assigned_chunks) >= 15:  # Cap at 15 chunks
                                break

                        # Ensure minimum chunks
                        if len(assigned_chunks) < 3 and chunks:
                            assigned_chunks = chunks[:3]

                return await self._run_agent_analysis(agent_role, query, assigned_chunks, context)

        # Create tasks for all agents
        tasks = [run_single_agent(agent_role) for agent_role in agent_roles]

        # Run with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid perspectives
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, AgentPerspective):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Agent {agent_roles[i].value} failed: {result}")
                self.performance_stats['agents_skipped'] += 1

        return valid_results

    def _distribute_chunks_to_agents(
        self,
        agent_roles: List[AgentRole],
        chunks: List[CodeChunk]
    ) -> Dict[AgentRole, List[CodeChunk]]:
        """
        Distribute chunks among agents to ensure diverse analysis perspectives.
        Each agent gets completely different chunks based on their specialization and architectural layers.
        """
        if not chunks or not agent_roles:
            return {}

        # Categorize chunks by architectural layers and file types
        chunk_categories = self._categorize_chunks_by_architecture(chunks)

        # Ensure we have enough diverse chunks
        total_chunks = len(chunks)
        num_agents = len(agent_roles)

        # Target at least 15 chunks per agent for comprehensive analysis
        min_chunks_per_agent = 15
        chunks_per_agent = max(min_chunks_per_agent, total_chunks // num_agents)

        # If we don't have enough chunks, we'll need to expand the search
        if total_chunks < num_agents * min_chunks_per_agent:
            logger.warning(f"Only {total_chunks} chunks available for {num_agents} agents. "
                         f"Each agent needs at least {min_chunks_per_agent} chunks for comprehensive analysis.")
            chunks_per_agent = max(3, total_chunks // num_agents)  # Minimum 3 chunks per agent

        agent_assignments = {}
        used_chunk_ids = set()

        # Distribute chunks ensuring each agent gets unique architectural perspectives
        for i, agent_role in enumerate(agent_roles):
            # Get chunks specifically relevant to this agent's domain
            agent_chunks = self._select_unique_chunks_for_agent(
                agent_role, chunk_categories, chunks_per_agent, used_chunk_ids
            )

            agent_assignments[agent_role] = agent_chunks

            # Track used chunks to prevent overlap using chunk IDs
            for chunk in agent_chunks:
                chunk_id = getattr(chunk, 'id', chunk.file_path)
                used_chunk_ids.add(chunk_id)

        # Log distribution for debugging with more detail
        logger.info(f"Distributed {total_chunks} chunks among {num_agents} agents with architectural diversity:")
        for agent_role, assigned_chunks in agent_assignments.items():
            file_types = set(chunk.file_path.split('.')[-1] for chunk in assigned_chunks if '.' in chunk.file_path)
            logger.info(f"  {agent_role.value}: {len(assigned_chunks)} chunks from {len(file_types)} file types")

        return agent_assignments

    def _categorize_chunks_by_architecture(self, chunks: List[CodeChunk]) -> Dict[str, List[CodeChunk]]:
        """Categorize chunks by architectural layers and concerns."""
        categories = {
            'models': [],           # Data models, schemas, entities
            'services': [],         # Business logic, services
            'controllers': [],      # API controllers, handlers
            'views': [],           # UI components, templates
            'utils': [],           # Utilities, helpers
            'config': [],          # Configuration files
            'tests': [],           # Test files
            'database': [],        # Database related
            'auth': [],            # Authentication/authorization
            'api': [],             # API definitions
            'frontend': [],        # Frontend specific
            'backend': [],         # Backend specific
            'infrastructure': [],  # DevOps, deployment
            'other': []            # Uncategorized
        }

        for chunk in chunks:
            file_path = chunk.file_path.lower()
            content = chunk.content.lower()

            # Categorize by file path patterns
            if any(pattern in file_path for pattern in ['model', 'entity', 'schema', 'dto']):
                categories['models'].append(chunk)
            elif any(pattern in file_path for pattern in ['service', 'business', 'logic']):
                categories['services'].append(chunk)
            elif any(pattern in file_path for pattern in ['controller', 'handler', 'endpoint', 'route']):
                categories['controllers'].append(chunk)
            elif any(pattern in file_path for pattern in ['view', 'component', 'template', 'ui']):
                categories['views'].append(chunk)
            elif any(pattern in file_path for pattern in ['util', 'helper', 'tool']):
                categories['utils'].append(chunk)
            elif any(pattern in file_path for pattern in ['config', 'setting', 'env']):
                categories['config'].append(chunk)
            elif any(pattern in file_path for pattern in ['test', 'spec']):
                categories['tests'].append(chunk)
            elif any(pattern in file_path for pattern in ['db', 'database', 'migration', 'sql']):
                categories['database'].append(chunk)
            elif any(pattern in file_path for pattern in ['auth', 'login', 'security', 'permission']):
                categories['auth'].append(chunk)
            elif any(pattern in file_path for pattern in ['api', 'rest', 'graphql']):
                categories['api'].append(chunk)
            elif any(pattern in file_path for pattern in ['frontend', 'client', 'web', 'react', 'vue', 'angular']):
                categories['frontend'].append(chunk)
            elif any(pattern in file_path for pattern in ['backend', 'server', 'src']):
                categories['backend'].append(chunk)
            elif any(pattern in file_path for pattern in ['docker', 'deploy', 'infra', 'k8s', 'terraform']):
                categories['infrastructure'].append(chunk)
            else:
                # Categorize by content patterns if file path doesn't match
                if any(pattern in content for pattern in ['class ', 'def __init__', 'model', 'schema']):
                    categories['models'].append(chunk)
                elif any(pattern in content for pattern in ['service', 'business', 'process']):
                    categories['services'].append(chunk)
                elif any(pattern in content for pattern in ['@app.route', '@router', 'fastapi', 'flask']):
                    categories['controllers'].append(chunk)
                elif any(pattern in content for pattern in ['render', 'template', 'component']):
                    categories['views'].append(chunk)
                else:
                    categories['other'].append(chunk)

        return categories

    def _select_unique_chunks_for_agent(
        self,
        agent_role: AgentRole,
        chunk_categories: Dict[str, List[CodeChunk]],
        target_count: int,
        used_indices: set
    ) -> List[CodeChunk]:
        """Select unique chunks for an agent based on their specialization and architectural focus."""
        # Define which architectural categories each agent should focus on
        agent_category_preferences = {
            AgentRole.ARCHITECT: ['models', 'services', 'controllers', 'config'],
            AgentRole.DEVELOPER: ['utils', 'backend', 'other', 'services'],
            AgentRole.SECURITY: ['auth', 'api', 'config', 'controllers'],
            AgentRole.PERFORMANCE: ['database', 'backend', 'services', 'utils'],
            AgentRole.MAINTAINER: ['tests', 'utils', 'config', 'other'],
            AgentRole.BUSINESS: ['models', 'services', 'controllers', 'api'],
            AgentRole.INTEGRATION: ['api', 'services', 'config', 'infrastructure'],
            AgentRole.DATA: ['models', 'database', 'services', 'backend'],
            AgentRole.UI_UX: ['views', 'frontend', 'other', 'utils'],
            AgentRole.DEVOPS: ['infrastructure', 'config', 'database', 'backend'],
            AgentRole.TESTING: ['tests', 'services', 'controllers', 'utils'],
            AgentRole.COMPLIANCE: ['config', 'auth', 'models', 'api']
        }

        preferred_categories = agent_category_preferences.get(agent_role, ['other'])
        selected_chunks = []

        # Create a flat list of all available chunks with their IDs
        all_available_chunks = []
        seen_chunk_ids = set()

        for category in preferred_categories:
            if category in chunk_categories:
                for chunk in chunk_categories[category]:
                    chunk_id = getattr(chunk, 'id', chunk.file_path)
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_available_chunks.append(chunk)

        # Add chunks from other categories if needed for diversity
        for category, chunks in chunk_categories.items():
            if category not in preferred_categories:
                for chunk in chunks:
                    chunk_id = getattr(chunk, 'id', chunk.file_path)
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_available_chunks.append(chunk)

        # Select chunks that haven't been used by other agents
        for chunk in all_available_chunks:
            if len(selected_chunks) >= target_count:
                break

            chunk_id = getattr(chunk, 'id', chunk.file_path)

            # Check if this chunk has already been assigned to another agent
            if chunk_id not in used_indices:
                selected_chunks.append(chunk)

        return selected_chunks[:target_count]

    def _select_chunks_for_agent(
        self,
        agent_role: AgentRole,
        all_chunks: List[CodeChunk],
        start_idx: int,
        target_count: int,
        used_indices: set
    ) -> List[CodeChunk]:
        """
        Select chunks most relevant to the agent's specialization.
        """
        agent_config = self.agents.get(agent_role, {})
        agent_keywords = agent_config.get("keywords", [])

        # Score chunks based on relevance to agent
        chunk_scores = []
        for i, chunk in enumerate(all_chunks):
            score = self._calculate_chunk_relevance_for_agent(chunk, agent_keywords)
            # Prefer unused chunks but allow some overlap
            if i not in used_indices:
                score += 0.2  # Bonus for unused chunks
            chunk_scores.append((i, chunk, score))

        # Sort by relevance score
        chunk_scores.sort(key=lambda x: x[2], reverse=True)

        # Select top chunks, ensuring we get the target count
        selected_chunks = []
        selected_count = 0

        # First, try to get chunks starting from the calculated start position
        for i in range(start_idx, min(start_idx + target_count, len(all_chunks))):
            if selected_count < target_count:
                selected_chunks.append(all_chunks[i])
                selected_count += 1

        # If we need more chunks, select from the highest scoring remaining chunks
        if selected_count < target_count:
            for idx, chunk, score in chunk_scores:
                if chunk not in selected_chunks and selected_count < target_count:
                    selected_chunks.append(chunk)
                    selected_count += 1

        return selected_chunks[:target_count]

    def _calculate_chunk_relevance_for_agent(
        self,
        chunk: CodeChunk,
        agent_keywords: List[str]
    ) -> float:
        """Calculate how relevant a chunk is to a specific agent."""
        if not agent_keywords:
            return 0.5  # Neutral score

        content_lower = chunk.content.lower()
        file_path_lower = chunk.file_path.lower()

        score = 0.0
        keyword_matches = 0

        for keyword in agent_keywords:
            keyword_lower = keyword.lower()

            # Check content
            if keyword_lower in content_lower:
                score += 1.0
                keyword_matches += 1

            # Check file path
            if keyword_lower in file_path_lower:
                score += 0.5
                keyword_matches += 1

        # Normalize score
        if keyword_matches > 0:
            score = score / len(agent_keywords)

        # Add bonus for certain node types based on agent specialization
        node_type_bonus = {
            AgentRole.ARCHITECT: {"class": 0.3, "module": 0.3, "interface": 0.2},
            AgentRole.DEVELOPER: {"function": 0.3, "method": 0.3, "class": 0.2},
            AgentRole.SECURITY: {"function": 0.2, "method": 0.2, "class": 0.1},
            AgentRole.PERFORMANCE: {"function": 0.3, "method": 0.3, "loop": 0.2},
        }

        if hasattr(chunk, 'node_type') and chunk.node_type:
            bonus_map = node_type_bonus.get(AgentRole.ARCHITECT, {})  # Default fallback
            score += bonus_map.get(chunk.node_type.lower(), 0.0)

        return min(score, 2.0)  # Cap at 2.0

    def _select_relevant_agents_optimized(self, query: str, chunks: List[CodeChunk]) -> List[AgentRole]:
        """Optimized agent selection with dynamic thresholds and early termination."""
        query_lower = query.lower()
        code_content = " ".join([chunk.content.lower() for chunk in chunks[:10]])
        file_paths = [chunk.file_path.lower() for chunk in chunks[:10]]

        # Score each agent based on relevance
        agent_scores = {}

        # Determine query complexity to adjust thresholds
        query_complexity = self._assess_query_complexity(query_lower, code_content)

        for agent_role, config in self.agents.items():
            score = self._calculate_agent_relevance_score(
                agent_role, config, query_lower, code_content, file_paths
            )
            agent_scores[agent_role] = score

        # Dynamic threshold based on query complexity
        if query_complexity == "simple":
            min_score_threshold = 80
            max_agents = 4
        elif query_complexity == "moderate":
            min_score_threshold = 50
            max_agents = 6
        else:  # complex
            min_score_threshold = 30
            max_agents = 8

        # Sort agents by score and select top performers
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        selected_agents = []

        # Always include priority 1 agents if they meet minimum threshold
        priority_1_agents = [role for role, config in self.agents.items() if config.get("priority") == 1]
        for agent_role in priority_1_agents:
            if agent_scores[agent_role] >= min_score_threshold:
                selected_agents.append(agent_role)

        # Add other high-scoring agents up to max limit
        for agent_role, score in sorted_agents:
            if len(selected_agents) >= max_agents:
                break
            if agent_role not in selected_agents and score >= min_score_threshold:
                selected_agents.append(agent_role)

        # Log skipped agents for performance tracking
        skipped_count = len(self.agents) - len(selected_agents)
        self.performance_stats['agents_skipped'] += skipped_count

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} low-relevance agents for performance")

        return selected_agents

    def _assess_query_complexity(self, query: str, code_content: str) -> str:
        """Assess query complexity to determine appropriate agent selection strategy."""
        # Simple heuristics for query complexity
        complexity_indicators = {
            'simple': ['what', 'how', 'where', 'when', 'show', 'list'],
            'moderate': ['explain', 'analyze', 'compare', 'review', 'optimize'],
            'complex': ['architecture', 'design', 'refactor', 'security', 'performance', 'comprehensive']
        }

        query_words = query.lower().split()

        # Check for complex indicators first
        if any(indicator in query for indicator in complexity_indicators['complex']):
            return "complex"
        elif any(indicator in query for indicator in complexity_indicators['moderate']):
            return "moderate"
        elif len(query_words) > 10 or len(code_content) > 5000:
            return "moderate"
        else:
            return "simple"

    def _calculate_agent_relevance_score(
        self,
        agent_role: AgentRole,
        config: Dict[str, Any],
        query: str,
        code_content: str,
        file_paths: List[str]
    ) -> int:
        """Calculate relevance score for an agent with optimized scoring."""
        score = 0

        # Base priority score
        priority = config.get("priority", 3)
        if priority == 1:
            score += 100  # Always include high priority agents
        elif priority == 2:
            score += 50   # Include if relevant
        elif priority == 3:
            score += 20   # Include if highly relevant
        else:
            score += 5    # Include only if very specific match

        # Query keyword matching with weighted scoring
        keywords = config["keywords"]
        query_matches = sum(1 for keyword in keywords if keyword in query)
        score += query_matches * 30

        # Code content matching
        content_matches = sum(1 for keyword in keywords if keyword in code_content)
        score += content_matches * 20

        # File path matching
        path_matches = sum(1 for keyword in keywords if any(keyword in path for path in file_paths))
        score += path_matches * 15

        # Special relevance boosts
        score += self._calculate_special_relevance(agent_role, query, code_content, file_paths)

        return score

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['total_queries'], 1) * 100
        )
        stats['cache_size'] = len(self.query_cache)
        return stats

    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.cache_access_times.clear()
        logger.info("Query cache cleared")

    def _calculate_special_relevance(self, agent_role: AgentRole, query: str, code_content: str, file_paths: List[str]) -> int:
        """Calculate special relevance bonuses for specific contexts."""
        bonus = 0

        if agent_role == AgentRole.ARCHITECT:
            if any(word in query for word in ["overview", "architecture", "design", "system", "structure"]):
                bonus += 50
            if any(word in code_content for word in ["class", "module", "component", "service"]):
                bonus += 30

        elif agent_role == AgentRole.SECURITY:
            if any(word in query for word in ["security", "auth", "login", "permission", "vulnerability"]):
                bonus += 60
            if any(word in code_content for word in ["password", "token", "jwt", "auth", "encrypt"]):
                bonus += 40

        elif agent_role == AgentRole.PERFORMANCE:
            if any(word in query for word in ["performance", "speed", "optimization", "slow", "bottleneck"]):
                bonus += 60
            if any(word in code_content for word in ["async", "cache", "optimize", "performance"]):
                bonus += 40

        elif agent_role == AgentRole.DATA:
            if any(word in query for word in ["database", "data", "storage", "query", "model"]):
                bonus += 60
            if any(word in code_content for word in ["database", "db", "query", "model", "schema"]):
                bonus += 40
            if any("database" in path or "model" in path for path in file_paths):
                bonus += 30

        elif agent_role == AgentRole.UI_UX:
            if any(word in query for word in ["ui", "interface", "frontend", "user", "component"]):
                bonus += 60
            if any("frontend" in path or "ui" in path or "component" in path for path in file_paths):
                bonus += 40

        elif agent_role == AgentRole.TESTING:
            if any(word in query for word in ["test", "testing", "quality", "coverage"]):
                bonus += 60
            if any("test" in path for path in file_paths):
                bonus += 50
            if any(word in code_content for word in ["test", "mock", "assert", "coverage"]):
                bonus += 30

        elif agent_role == AgentRole.DEVOPS:
            if any(word in query for word in ["deploy", "infrastructure", "docker", "kubernetes", "ci"]):
                bonus += 60
            if any(word in code_content for word in ["docker", "deploy", "config", "env"]):
                bonus += 40

        return bonus
    
    async def _run_agent_analysis(
        self, 
        agent_role: AgentRole, 
        query: str, 
        chunks: List[CodeChunk], 
        context: Dict[str, Any]
    ) -> AgentPerspective:
        """Run analysis from a specific agent's perspective."""
        try:
            if self.client:
                return await self._llm_agent_analysis(agent_role, query, chunks, context)
            else:
                return self._rule_based_agent_analysis(agent_role, query, chunks, context)
        except Exception as e:
            logger.error(f"Error in {agent_role.value} agent analysis: {e}")
            return self._fallback_agent_perspective(agent_role, query, chunks)
    
    async def _llm_agent_analysis(
        self, 
        agent_role: AgentRole, 
        query: str, 
        chunks: List[CodeChunk], 
        context: Dict[str, Any]
    ) -> AgentPerspective:
        """Use LLM to generate agent-specific analysis."""
        agent_config = self.agents[agent_role]
        
        # Prepare code context
        code_context = self._prepare_code_context_for_agent(chunks, agent_role)
        
        # Build agent-specific prompt
        prompt = self._build_agent_prompt(agent_role, agent_config, query, code_context, context, chunks)
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a senior {agent_role.value} analyzing a codebase from your specialized perspective."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )

        analysis_text = response.choices[0].message.content.strip()

        # Log the raw LLM response for debugging
        logger.debug(f"Raw LLM response for {agent_role.value}:")
        logger.debug(f"Response length: {len(analysis_text)} characters")
        logger.debug(f"First 500 chars: {analysis_text[:500]}")
        logger.debug(f"Last 200 chars: {analysis_text[-200:]}")

        # Parse the structured response
        return self._parse_agent_response(agent_role, analysis_text)
    
    def _build_agent_prompt(
        self,
        agent_role: AgentRole,
        agent_config: Dict[str, Any],
        query: str,
        code_context: str,
        context: Dict[str, Any],
        chunks: List[CodeChunk] = None
    ) -> str:
        """Build a specialized prompt for each agent role."""

        # Extract specific code elements for more detailed analysis
        code_files = self._extract_code_files(code_context)
        functions = self._extract_functions(code_context)
        classes = self._extract_classes(code_context)

        # Count chunks for analysis context
        chunk_count = len(chunks) if chunks else 0

        base_context = f"""
AGENT ROLE: {agent_role.value.title()} Expert
SPECIALIZATION: {agent_config['focus']}
PERSPECTIVE: {agent_config['perspective']} analysis

USER QUERY: "{query}"

CODE ANALYSIS CONTEXT:
Code chunks analyzed: {chunk_count} unique chunks
Files analyzed: {len(code_files)} files
Functions found: {len(functions)} functions
Classes found: {len(classes)} classes

DETAILED CODE CONTEXT:
{code_context}

SPECIFIC CODE ELEMENTS IDENTIFIED:
Files: {', '.join(code_files[:5])}{'...' if len(code_files) > 5 else ''}
Key Functions: {', '.join(functions[:8])}{'...' if len(functions) > 8 else ''}
Key Classes: {', '.join(classes[:5])}{'...' if len(classes) > 5 else ''}
"""
        
        role_specific_instructions = {
            AgentRole.ARCHITECT: """
As a Senior Software Architect, conduct a comprehensive architectural analysis:

CORE ANALYSIS AREAS:
- System Architecture: Analyze overall system structure, layering, component organization and project purpose (solve what problems)
- Design Patterns: Identify architectural patterns (such as MVC, microservices, event-driven, etc.)
- Component Relationships: Map dependencies, interfaces, and communication patterns, further offer ASCII Flow diagram to explain the relationships.
- Scalability Design: Evaluate horizontal/vertical scaling capabilities and bottlenecks
- Extensibility: Assess how easily new features or components can be added
- Technology Stack: Analyze technology choices and their architectural implications

ADVANCED CONSIDERATIONS:
- Cross-cutting concerns (logging, security, caching, error handling)
- Data flow architecture and information management
- Service boundaries and domain separation
- Integration patterns and external system interactions
- Deployment architecture and infrastructure requirements
- Performance implications of architectural decisions

Provide strategic insights on architectural strengths, weaknesses, and evolution paths.
""",
            AgentRole.DEVELOPER: """
As a Senior Software Developer, conduct a deep technical implementation analysis:

CODE QUALITY ANALYSIS:
- Implementation Patterns: Analyze coding patterns, algorithms, and data structures, further offer ASCII Flow diagram for data flows.
- Code Organization: Evaluate modularity, separation of concerns, and code structure
- Best Practices: Assess adherence to SOLID principles, DRY, KISS, and language-specific conventions
- Error Handling: Analyze exception handling, validation, and edge case management
- Resource Management: Evaluate memory usage, connection handling, and cleanup patterns

TECHNICAL IMPLEMENTATION:
- Algorithm Efficiency: Analyze time/space complexity and optimization opportunities
- Concurrency Handling: Evaluate thread safety, async patterns, and parallel processing
- API Design: Assess interface design, parameter validation, and return value handling
- Code Reusability: Identify opportunities for abstraction and component reuse
- Technical Debt: Spot code smells, anti-patterns, and refactoring opportunities

Focus on actionable improvements that enhance code quality, maintainability, and performance.
""",
            AgentRole.SECURITY: """
As a Security Engineer, conduct a comprehensive security analysis:

SECURITY FUNDAMENTALS:
- Authentication & Authorization: Analyze user verification and access control mechanisms
- Input Validation: Evaluate data sanitization, injection prevention, and boundary checks, further offer ASCII Flow diagram for authentication flows.
- Data Protection: Assess encryption, hashing, and sensitive data handling
- Session Management: Analyze session security, token handling, and state management
- Access Control: Evaluate permission systems and privilege escalation prevention

VULNERABILITY ASSESSMENT:
- Common Vulnerabilities: Check for OWASP Top 10 and language-specific security issues
- Configuration Security: Analyze security configurations and hardening measures
- Dependency Security: Assess third-party library vulnerabilities and update practices
- Communication Security: Evaluate TLS/SSL usage, certificate management, and secure protocols
- Audit & Logging: Analyze security event logging and monitoring capabilities

Provide specific security recommendations with risk assessments and mitigation strategies.
""",
            AgentRole.PERFORMANCE: """
As a Performance Engineer, conduct a comprehensive performance analysis:

PERFORMANCE CHARACTERISTICS:
- Bottleneck Identification: Analyze CPU, memory, I/O, and network performance constraints
- Algorithm Efficiency: Evaluate computational complexity and optimization opportunities
- Resource Utilization: Assess memory usage patterns, garbage collection, and resource leaks
- Concurrency Performance: Analyze parallel processing, thread contention, and async patterns
- Database Performance: Evaluate query efficiency, indexing strategies, and connection pooling

OPTIMIZATION OPPORTUNITIES:
- Caching Strategies: Analyze caching layers, cache hit rates, and invalidation patterns
- Load Balancing: Evaluate distribution strategies and scaling approaches
- Code Optimization: Identify hot paths, inefficient loops, and optimization opportunities
- Infrastructure Performance: Assess deployment architecture and infrastructure bottlenecks
- Monitoring & Profiling: Evaluate performance measurement and monitoring capabilities

Provide data-driven performance improvement recommendations with expected impact.
""",
            AgentRole.MAINTAINER: """
As a Technical Lead focused on maintainability, conduct a comprehensive code health analysis:

MAINTAINABILITY ASSESSMENT:
- Code Complexity: Analyze cyclomatic complexity, nesting levels, and cognitive load
- Technical Debt: Identify code smells, anti-patterns, and areas requiring refactoring
- Documentation Quality: Evaluate code comments, API documentation, and knowledge transfer
- Testing Coverage: Assess test quality, coverage metrics, and testing strategies
- Code Organization: Analyze module structure, dependency management, and architectural clarity

LONG-TERM HEALTH:
- Evolution Capability: Assess how easily the codebase can adapt to changing requirements
- Knowledge Distribution: Evaluate bus factor and knowledge concentration risks
- Refactoring Opportunities: Identify areas for improvement and modernization
- Dependency Management: Analyze third-party dependencies and update strategies
- Development Workflow: Evaluate development practices and team collaboration patterns

Focus on sustainable development practices and long-term codebase health.
""",
            AgentRole.BUSINESS: """
As a Business Analyst with technical expertise, conduct a comprehensive business logic analysis:

BUSINESS LOGIC EVALUATION:
- Domain Modeling: Analyze how business concepts are represented in code
- Business Rules: Evaluate implementation of business rules and validation logic
- Workflow Implementation: Assess business process automation and workflow management
- Data Integrity: Analyze business data validation and consistency enforcement
- User Experience: Evaluate how technical implementation supports user needs

BUSINESS VALUE ASSESSMENT:
- Requirement Fulfillment: Assess how well the code meets functional requirements
- Business Process Support: Evaluate automation of business processes and efficiency gains
- Stakeholder Value: Analyze value delivery to different user groups and stakeholders
- Compliance & Governance: Assess adherence to business policies and regulatory requirements
- ROI Considerations: Evaluate technical decisions from a business value perspective

Provide insights on business-technical alignment and value optimization opportunities.
""",
            AgentRole.INTEGRATION: """
As an Integration Architect, conduct a comprehensive system integration analysis:

INTEGRATION ARCHITECTURE:
- External Dependencies: Analyze third-party service integrations and API usage patterns
- Communication Patterns: Evaluate synchronous/asynchronous communication and messaging
- Data Exchange: Assess data transformation, serialization, and protocol handling
- Service Boundaries: Analyze microservice boundaries and inter-service communication
- API Design: Evaluate REST/GraphQL/gRPC implementations and versioning strategies

INTEGRATION QUALITY:
- Error Handling: Analyze failure scenarios, retry logic, and circuit breaker patterns
- Monitoring & Observability: Evaluate integration monitoring and distributed tracing
- Performance: Assess integration performance, latency, and throughput characteristics
- Security: Analyze authentication, authorization, and secure communication patterns
- Resilience: Evaluate fault tolerance, graceful degradation, and recovery mechanisms

Focus on integration reliability, maintainability, and operational excellence.
""",
            AgentRole.DATA: """
As a Data Architect, conduct a comprehensive data architecture analysis:

DATA ARCHITECTURE:
- Data Modeling: Analyze entity relationships, schema design, and normalization strategies
- Database Design: Evaluate table structures, indexing strategies, and query optimization
- Data Flow: Assess data movement, transformation pipelines, and ETL processes
- Storage Strategy: Analyze data storage patterns, partitioning, and archival strategies
- Data Consistency: Evaluate ACID properties, transaction management, and consistency models

DATA MANAGEMENT:
- Performance Optimization: Analyze query performance, indexing effectiveness, and caching
- Scalability: Evaluate data scaling strategies and distributed data management
- Data Quality: Assess validation, cleansing, and data integrity mechanisms
- Backup & Recovery: Analyze data protection, backup strategies, and disaster recovery
- Compliance: Evaluate data privacy, retention policies, and regulatory compliance

Provide insights on data architecture optimization and management best practices.
""",
            AgentRole.UI_UX: """
As a Frontend Architect with UX expertise, conduct a comprehensive UI/UX analysis:

FRONTEND ARCHITECTURE:
- Component Design: Analyze component architecture, reusability, and composition patterns, further offer ASCII Flow diagram for core components' interactions.
- State Management: Evaluate state management patterns and data flow architecture
- Performance: Assess frontend performance, bundle optimization, and loading strategies
- Accessibility: Analyze WCAG compliance and inclusive design implementation
- Responsive Design: Evaluate multi-device support and adaptive layouts

USER EXPERIENCE:
- Interaction Patterns: Analyze user workflows, navigation, and interaction design
- Usability: Evaluate ease of use, error prevention, and user feedback mechanisms
- Performance UX: Assess perceived performance, loading states, and user feedback
- Design System: Analyze consistency, design tokens, and component standardization
- User Journey: Evaluate end-to-end user experience and conversion optimization

Focus on technical implementation that enhances user experience and frontend maintainability.
""",
            AgentRole.DEVOPS: """
As a DevOps Engineer, conduct a comprehensive operational analysis:

DEPLOYMENT & INFRASTRUCTURE:
- Infrastructure as Code: Analyze infrastructure automation and configuration management
- Containerization: Evaluate Docker, Kubernetes, and container orchestration patterns
- CI/CD Pipeline: Assess build, test, and deployment automation and pipeline efficiency
- Environment Management: Analyze environment consistency and configuration management
- Scalability: Evaluate auto-scaling, load balancing, and capacity management

OPERATIONAL EXCELLENCE:
- Monitoring & Observability: Analyze logging, metrics, tracing, and alerting systems
- Reliability: Evaluate SLA/SLO compliance, error budgets, and incident response
- Security Operations: Assess security scanning, vulnerability management, and compliance
- Performance Monitoring: Analyze application and infrastructure performance monitoring
- Disaster Recovery: Evaluate backup strategies, failover mechanisms, and recovery procedures

Provide insights on operational efficiency, reliability, and automation opportunities.
""",
            AgentRole.TESTING: """
As a Quality Assurance Engineer, conduct a comprehensive testing analysis:

TESTING STRATEGY:
- Test Coverage: Analyze unit, integration, and end-to-end test coverage and effectiveness
- Test Architecture: Evaluate test organization, test data management, and test environments
- Automation: Assess test automation strategies, frameworks, and CI/CD integration
- Quality Gates: Analyze quality metrics, code coverage thresholds, and release criteria
- Testing Pyramid: Evaluate balance between unit, integration, and UI tests

QUALITY ASSURANCE:
- Bug Prevention: Analyze static analysis, linting, and code quality tools
- Performance Testing: Evaluate load testing, stress testing, and performance benchmarks
- Security Testing: Assess security testing practices and vulnerability scanning
- Usability Testing: Analyze user acceptance testing and usability validation
- Regression Testing: Evaluate regression test strategies and change impact analysis

Focus on comprehensive quality assurance and testing optimization strategies.
""",
            AgentRole.COMPLIANCE: """
As a Compliance Officer with technical expertise, conduct a comprehensive compliance analysis:

REGULATORY COMPLIANCE:
- Data Privacy: Analyze GDPR, CCPA, and other privacy regulation compliance
- Industry Standards: Evaluate adherence to industry-specific regulations (HIPAA, PCI-DSS, SOX)
- Security Compliance: Assess security frameworks (ISO 27001, NIST, SOC 2) implementation
- Audit Requirements: Analyze audit trail capabilities and compliance reporting
- Documentation: Evaluate compliance documentation and policy enforcement

GOVERNANCE & RISK:
- Risk Assessment: Analyze technical risks and mitigation strategies
- Policy Enforcement: Evaluate automated policy enforcement and compliance monitoring
- Access Control: Assess role-based access control and privilege management
- Data Governance: Analyze data classification, retention, and lifecycle management
- Change Management: Evaluate change control processes and compliance impact assessment

Provide insights on compliance gaps, risk mitigation, and governance improvements.
"""
        }
        
        specific_instructions = role_specific_instructions.get(
            agent_role, 
            "Analyze the code from your specialized perspective."
        )
        
        return f"""
{base_context}

{specific_instructions}

ANALYSIS METHODOLOGY FOR {agent_role.value.upper()} PERSPECTIVE:

You are analyzing {chunk_count} unique code chunks that have been specifically selected for your expertise area. Each chunk represents a different aspect of the codebase to ensure comprehensive coverage.

ANALYSIS REQUIREMENTS:
- Conduct deep, technical analysis beyond surface-level observations
- Synthesize insights across all provided code chunks to identify patterns and relationships
- Focus on systemic issues and opportunities rather than isolated code snippets
- Provide evidence-based conclusions with specific code references
- Consider the broader architectural context and implications
- Identify both immediate issues and strategic considerations

RESPONSE STRUCTURE:

EXECUTIVE SUMMARY:
[2-3 sentences summarizing your key findings and overall assessment from your specialized perspective]

DETAILED ANALYSIS:
[Comprehensive analysis covering:]
- Primary findings from your specialized domain
- Cross-cutting patterns observed across the code chunks
- Systemic strengths and weaknesses identified
- Technical implications and architectural considerations
- Integration with broader system context

KEY INSIGHTS:
- [Strategic insight with architectural/business implications]
- [Technical pattern or anti-pattern with system-wide impact]
- [Critical finding that affects multiple components or layers]
- [Opportunity for significant improvement or optimization]

RECOMMENDATIONS:
- [High-impact recommendation with implementation approach and expected benefits]
- [Strategic improvement with timeline and resource considerations]
- [Technical enhancement with specific steps and success metrics]

RISK ASSESSMENT:
- [Critical risks identified in your domain area]
- [Potential impact and likelihood assessment]
- [Mitigation strategies and preventive measures]

CONFIDENCE: [High/Medium/Low] - [Detailed reasoning based on code coverage, complexity, and domain expertise]

FOCUS AREAS FOR IMMEDIATE ATTENTION:
- [Priority 1: Critical issue requiring immediate action]
- [Priority 2: Important improvement with significant impact]
- [Priority 3: Enhancement opportunity for future consideration]

Remember: Your analysis should demonstrate deep domain expertise and provide actionable insights that go beyond obvious observations. Focus on value-driven recommendations that align with business and technical objectives.
"""

    def _prepare_code_context_for_agent(self, chunks: List[CodeChunk], agent_role: AgentRole) -> str:
        """Prepare code context tailored for specific agent perspective with comprehensive chunk analysis."""
        context_parts = []

        # Use all assigned chunks for this agent (should be 10+ unique chunks)
        # Increase content limit for better analysis
        max_content_per_chunk = 1500

        # Add architectural context summary first
        architectural_summary = self._generate_architectural_summary(chunks, agent_role)
        if architectural_summary:
            context_parts.append(f"ARCHITECTURAL CONTEXT:\n{architectural_summary}\n")

        for i, chunk in enumerate(chunks):
            # Extract more detailed metadata for better analysis
            chunk_metadata = self._extract_chunk_metadata(chunk)

            chunk_info = f"""
CODE CHUNK {i+1}:
File: {chunk.file_path}
Lines: {chunk.start_line}-{chunk.end_line}
Type: {chunk.node_type.value if chunk.node_type else 'unknown'}
Name: {chunk.name or 'unnamed'}
Metadata: {chunk_metadata}

Content:
{chunk.content[:max_content_per_chunk]}{'...' if len(chunk.content) > max_content_per_chunk else ''}
"""
            context_parts.append(chunk_info)

        return "\n".join(context_parts)

    def _generate_architectural_summary(self, chunks: List[CodeChunk], agent_role: AgentRole) -> str:
        """Generate architectural summary relevant to the agent's perspective."""
        if not chunks:
            return ""

        # Analyze file patterns and architectural layers
        file_patterns = {}
        for chunk in chunks:
            file_path = chunk.file_path.lower()

            # Categorize by architectural patterns
            if any(pattern in file_path for pattern in ['model', 'entity', 'schema']):
                file_patterns['data_layer'] = file_patterns.get('data_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['service', 'business', 'logic']):
                file_patterns['business_layer'] = file_patterns.get('business_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['controller', 'handler', 'api', 'route']):
                file_patterns['api_layer'] = file_patterns.get('api_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['view', 'component', 'ui', 'frontend']):
                file_patterns['presentation_layer'] = file_patterns.get('presentation_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['util', 'helper', 'common']):
                file_patterns['utility_layer'] = file_patterns.get('utility_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['config', 'setting', 'env']):
                file_patterns['config_layer'] = file_patterns.get('config_layer', 0) + 1
            elif any(pattern in file_path for pattern in ['test', 'spec']):
                file_patterns['test_layer'] = file_patterns.get('test_layer', 0) + 1

        # Generate role-specific architectural insights
        summary_parts = []

        if file_patterns:
            summary_parts.append(f"Analyzing {len(chunks)} code chunks across {len(file_patterns)} architectural layers:")
            for layer, count in file_patterns.items():
                summary_parts.append(f"- {layer.replace('_', ' ').title()}: {count} chunks")

        # Add role-specific context
        role_context = {
            AgentRole.ARCHITECT: "Focus on system structure, component relationships, and design patterns",
            AgentRole.DEVELOPER: "Focus on implementation quality, algorithms, and coding practices",
            AgentRole.SECURITY: "Focus on authentication, authorization, and vulnerability patterns",
            AgentRole.PERFORMANCE: "Focus on optimization opportunities, bottlenecks, and scalability",
            AgentRole.MAINTAINER: "Focus on code complexity, technical debt, and maintainability",
            AgentRole.BUSINESS: "Focus on business logic, domain modeling, and functional requirements",
            AgentRole.INTEGRATION: "Focus on external dependencies, APIs, and system integration",
            AgentRole.DATA: "Focus on data models, database design, and information architecture",
            AgentRole.UI_UX: "Focus on user interface components, interactions, and frontend architecture",
            AgentRole.DEVOPS: "Focus on deployment, infrastructure, and operational considerations",
            AgentRole.TESTING: "Focus on test coverage, quality assurance, and testing strategies",
            AgentRole.COMPLIANCE: "Focus on regulatory requirements, security compliance, and governance"
        }

        if agent_role in role_context:
            summary_parts.append(f"\nYour {agent_role.value} perspective: {role_context[agent_role]}")

        return "\n".join(summary_parts)

    def _extract_chunk_metadata(self, chunk: CodeChunk) -> str:
        """Extract detailed metadata from a code chunk for better analysis."""
        metadata_parts = []

        # Analyze content patterns
        content = chunk.content.lower()

        # Detect programming constructs
        constructs = []
        if 'class ' in content:
            constructs.append('classes')
        if 'def ' in content or 'function ' in content:
            constructs.append('functions')
        if 'import ' in content or 'from ' in content:
            constructs.append('imports')
        if 'async ' in content:
            constructs.append('async_code')
        if 'try:' in content or 'except' in content:
            constructs.append('error_handling')
        if 'test' in content or 'assert' in content:
            constructs.append('testing')
        if any(db_term in content for db_term in ['select', 'insert', 'update', 'delete', 'query']):
            constructs.append('database_operations')
        if any(api_term in content for api_term in ['@app.route', '@router', 'fastapi', 'flask', 'request', 'response']):
            constructs.append('api_endpoints')

        if constructs:
            metadata_parts.append(f"Contains: {', '.join(constructs)}")

        # Estimate complexity
        lines = len(chunk.content.split('\n'))
        if lines > 100:
            metadata_parts.append("High complexity (100+ lines)")
        elif lines > 50:
            metadata_parts.append("Medium complexity (50+ lines)")
        else:
            metadata_parts.append("Low complexity (<50 lines)")

        return "; ".join(metadata_parts) if metadata_parts else "Basic code structure"

    def _extract_code_files(self, code_context: str) -> List[str]:
        """Extract file names from code context."""
        files = []
        for line in code_context.split('\n'):
            if line.strip().startswith('File:'):
                file_path = line.split('File:', 1)[1].strip()
                if file_path:
                    # Extract just the filename for readability
                    filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
                    files.append(filename)
        return list(set(files))  # Remove duplicates

    def _extract_functions(self, code_context: str) -> List[str]:
        """Extract function names from code context."""
        functions = []
        for line in code_context.split('\n'):
            # Look for function definitions
            if 'def ' in line:
                # Extract function name
                try:
                    func_part = line.split('def ')[1].split('(')[0].strip()
                    if func_part and func_part.isidentifier():
                        functions.append(func_part)
                except:
                    pass
            # Look for async function definitions
            if 'async def ' in line:
                try:
                    func_part = line.split('async def ')[1].split('(')[0].strip()
                    if func_part and func_part.isidentifier():
                        functions.append(f"async {func_part}")
                except:
                    pass
        return list(set(functions))  # Remove duplicates

    def _extract_classes(self, code_context: str) -> List[str]:
        """Extract class names from code context."""
        classes = []
        for line in code_context.split('\n'):
            if 'class ' in line:
                try:
                    class_part = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    if class_part and class_part.isidentifier():
                        classes.append(class_part)
                except:
                    pass
        return list(set(classes))  # Remove duplicates

    def _parse_agent_response(self, agent_role: AgentRole, response_text: str) -> AgentPerspective:
        """Parse structured agent response into AgentPerspective object."""
        try:
            # Log the raw response for debugging
            logger.debug(f"Parsing {agent_role.value} response: {response_text[:200]}...")
            logger.debug(f"Full response text: {response_text}")

            # Extract sections using improved parsing
            sections = {}
            current_section = None
            current_content = []

            for line in response_text.split('\n'):
                line = line.strip()
                # Match the actual section headers from the LLM response
                section_headers = [
                    '### EXECUTIVE SUMMARY:', '### DETAILED ANALYSIS:', '### ANALYSIS:',
                    '### KEY INSIGHTS:', '### RECOMMENDATIONS:', '### RISK ASSESSMENT:',
                    '### CONFIDENCE:', '### FOCUS AREAS FOR IMMEDIATE ATTENTION:', '### FOCUS AREAS:'
                ]

                if any(line.startswith(header) for header in section_headers):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    # Extract section name without the ### prefix
                    current_section = line.replace('###', '').split(':')[0].strip().upper()
                    current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
                    logger.debug(f"Found section header: {current_section}")
                elif current_section and line:
                    current_content.append(line)

            # Add the last section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()

            logger.debug(f"Parsed sections: {list(sections.keys())}")
            for section_name, content in sections.items():
                logger.debug(f"Section '{section_name}': {content[:100]}...")

            # Extract structured data with fallbacks
            analysis = (sections.get('DETAILED ANALYSIS') or
                       sections.get('ANALYSIS') or
                       sections.get('EXECUTIVE SUMMARY') or
                       'Analysis not provided')

            key_insights = []
            if 'KEY INSIGHTS' in sections:
                insights_text = sections['KEY INSIGHTS']
                logger.debug(f"Processing KEY INSIGHTS section: {insights_text}")
                # Look for bullet points or numbered items
                for line in insights_text.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                               line.startswith('**') and line.endswith('**:')):
                        # Extract the insight text
                        insight = line.lstrip('-•*').strip()
                        if insight.endswith(':'):
                            insight = insight[:-1].strip()
                        if insight:
                            key_insights.append(insight)
                            logger.debug(f"Found insight: {insight}")

            recommendations = []
            if 'RECOMMENDATIONS' in sections:
                rec_text = sections['RECOMMENDATIONS']
                logger.debug(f"Processing RECOMMENDATIONS section: {rec_text}")
                # Look for numbered items or bullet points
                for line in rec_text.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or
                               line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or
                               line.startswith('**') and line.endswith('**:')):
                        # Extract the recommendation text
                        rec = line.lstrip('-•*123456789.').strip()
                        if rec.endswith(':'):
                            rec = rec[:-1].strip()
                        if rec:
                            recommendations.append(rec)
                            logger.debug(f"Found recommendation: {rec}")
                logger.debug(f"Total recommendations found: {len(recommendations)}")

            confidence_text = sections.get('CONFIDENCE', 'medium').lower()
            confidence = 0.8 if 'high' in confidence_text else 0.6 if 'medium' in confidence_text else 0.4

            focus_areas = []
            focus_section = (sections.get('FOCUS AREAS FOR IMMEDIATE ATTENTION') or
                           sections.get('FOCUS AREAS'))
            if focus_section:
                focus_areas = [
                    line.strip('- ').strip()
                    for line in focus_section.split('\n')
                    if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•'))
                ]

            # Log parsed results for debugging
            logger.debug(f"{agent_role.value} parsed: {len(key_insights)} insights, {len(recommendations)} recommendations, confidence: {confidence}")
            logger.debug(f"Key insights: {key_insights}")
            logger.debug(f"Recommendations: {recommendations}")

            return AgentPerspective(
                role=agent_role,
                analysis=analysis,
                key_insights=key_insights[:3],  # Limit to top 3
                recommendations=recommendations[:3],  # Limit to top 3
                confidence=confidence,
                focus_areas=focus_areas[:2]  # Limit to top 2
            )

        except Exception as e:
            logger.error(f"Error parsing {agent_role.value} response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return self._fallback_agent_perspective(agent_role, "", [])

    def _rule_based_agent_analysis(
        self,
        agent_role: AgentRole,
        query: str,
        chunks: List[CodeChunk],
        context: Dict[str, Any]
    ) -> AgentPerspective:
        """Provide rule-based analysis when LLM is not available."""

        # Analyze code content for patterns
        code_content = " ".join([chunk.content.lower() for chunk in chunks[:3]])
        file_paths = [chunk.file_path.lower() for chunk in chunks[:3]]

        # Route to specific rule-based analysis methods
        analysis_methods = {
            AgentRole.ARCHITECT: self._architect_rule_based_analysis,
            AgentRole.DEVELOPER: self._developer_rule_based_analysis,
            AgentRole.SECURITY: self._security_rule_based_analysis,
            AgentRole.PERFORMANCE: self._performance_rule_based_analysis,
            AgentRole.MAINTAINER: self._maintainer_rule_based_analysis,
            AgentRole.BUSINESS: self._business_rule_based_analysis,
            AgentRole.INTEGRATION: self._integration_rule_based_analysis,
            AgentRole.DATA: self._data_rule_based_analysis,
            AgentRole.UI_UX: self._ui_ux_rule_based_analysis,
            AgentRole.DEVOPS: self._devops_rule_based_analysis,
            AgentRole.TESTING: self._testing_rule_based_analysis,
            AgentRole.COMPLIANCE: self._compliance_rule_based_analysis,
        }

        analysis_method = analysis_methods.get(agent_role)
        if analysis_method:
            return analysis_method(query, chunks, code_content, file_paths)
        else:
            return self._fallback_agent_perspective(agent_role, query, chunks)

    def _architect_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based architectural analysis."""
        analysis = "From an architectural perspective, this system demonstrates a layered architecture with clear separation of concerns."

        key_insights = []
        recommendations = []
        focus_areas = ["System Architecture", "Design Patterns"]

        # Detect architectural patterns
        if any("fastapi" in path or "api" in path for path in file_paths):
            key_insights.append("RESTful API architecture with FastAPI framework")
            recommendations.append("Consider implementing API versioning and rate limiting")

        if any("database" in code_content or "db" in code_content):
            key_insights.append("Data persistence layer with database integration")
            recommendations.append("Implement database connection pooling for better performance")

        if "async" in code_content or "await" in code_content:
            key_insights.append("Asynchronous architecture for improved concurrency")
            recommendations.append("Ensure proper async/await usage throughout the system")

        return AgentPerspective(
            role=AgentRole.ARCHITECT,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.7,
            focus_areas=focus_areas
        )

    def _developer_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based developer analysis."""
        analysis = "From a development perspective, the code shows good structure with clear function definitions and proper error handling patterns."

        key_insights = []
        recommendations = []
        focus_areas = ["Code Quality", "Implementation"]

        # Analyze code patterns
        if "try:" in code_content and "except" in code_content:
            key_insights.append("Proper exception handling implemented")
        else:
            recommendations.append("Add comprehensive error handling and exception management")

        if "class " in code_content:
            key_insights.append("Object-oriented design with class-based architecture")
            recommendations.append("Consider implementing design patterns like Factory or Strategy")

        if "def " in code_content:
            key_insights.append("Modular design with well-defined functions")
            recommendations.append("Ensure functions follow single responsibility principle")

        return AgentPerspective(
            role=AgentRole.DEVELOPER,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.7,
            focus_areas=focus_areas
        )

    def _security_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based security analysis."""
        analysis = "From a security perspective, the system requires attention to authentication, authorization, and input validation mechanisms."

        key_insights = []
        recommendations = []
        focus_areas = ["Security", "Access Control"]

        # Check for security patterns
        if any(word in code_content for word in ["auth", "token", "jwt", "password"]):
            key_insights.append("Authentication mechanisms are implemented")
            recommendations.append("Ensure secure token storage and transmission")
        else:
            recommendations.append("Implement robust authentication and authorization")

        if "validate" in code_content or "sanitize" in code_content:
            key_insights.append("Input validation patterns detected")
        else:
            recommendations.append("Add comprehensive input validation and sanitization")

        recommendations.append("Implement security headers and HTTPS enforcement")

        return AgentPerspective(
            role=AgentRole.SECURITY,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _performance_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based performance analysis."""
        analysis = "From a performance perspective, the system shows potential for optimization through caching, async operations, and efficient data handling."

        key_insights = []
        recommendations = []
        focus_areas = ["Performance", "Optimization"]

        # Check for performance patterns
        if "async" in code_content:
            key_insights.append("Asynchronous operations for better concurrency")
        else:
            recommendations.append("Consider implementing async operations for I/O bound tasks")

        if "cache" in code_content:
            key_insights.append("Caching mechanisms implemented")
        else:
            recommendations.append("Implement caching for frequently accessed data")

        if any(word in code_content for word in ["index", "query", "search"]):
            key_insights.append("Search and indexing capabilities present")
            recommendations.append("Optimize database queries and add proper indexing")

        return AgentPerspective(
            role=AgentRole.PERFORMANCE,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _maintainer_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based maintainability analysis."""
        analysis = "From a maintainability perspective, the code structure supports long-term maintenance through modular design and clear organization."

        key_insights = []
        recommendations = []
        focus_areas = ["Maintainability", "Code Health"]

        # Check for maintainability patterns
        if len(chunks) > 1:
            key_insights.append("Modular code organization with multiple components")

        if any("test" in path for path in file_paths):
            key_insights.append("Testing infrastructure in place")
        else:
            recommendations.append("Implement comprehensive unit and integration tests")

        if any("config" in path for path in file_paths):
            key_insights.append("Configuration management implemented")
        else:
            recommendations.append("Centralize configuration management")

        recommendations.append("Add comprehensive documentation and code comments")

        return AgentPerspective(
            role=AgentRole.MAINTAINER,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.7,
            focus_areas=focus_areas
        )

    def _business_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based business analysis."""
        analysis = "From a business perspective, the system implements domain logic that supports core business processes and user workflows."

        key_insights = []
        recommendations = []
        focus_areas = ["Business Logic", "Domain Modeling"]

        # Analyze business patterns
        if any(word in code_content for word in ["user", "customer", "order", "payment", "workflow"]):
            key_insights.append("Core business entities and workflows are implemented")
            recommendations.append("Ensure business rules are clearly documented and testable")

        if "validate" in code_content or "business" in code_content:
            key_insights.append("Business validation logic is present")
        else:
            recommendations.append("Implement comprehensive business rule validation")

        if any("model" in path or "domain" in path for path in file_paths):
            key_insights.append("Domain modeling structure is organized")
        else:
            recommendations.append("Consider implementing domain-driven design patterns")

        return AgentPerspective(
            role=AgentRole.BUSINESS,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _integration_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based integration analysis."""
        analysis = "From an integration perspective, the system demonstrates patterns for external service communication and dependency management."

        key_insights = []
        recommendations = []
        focus_areas = ["System Integration", "API Design"]

        # Analyze integration patterns
        if any(word in code_content for word in ["api", "client", "request", "response", "http"]):
            key_insights.append("HTTP API integration patterns are implemented")
            recommendations.append("Implement proper error handling and retry mechanisms for external calls")

        if "async" in code_content and ("client" in code_content or "api" in code_content):
            key_insights.append("Asynchronous integration patterns for better performance")
        else:
            recommendations.append("Consider async patterns for external service calls")

        if any(word in code_content for word in ["config", "env", "settings"]):
            key_insights.append("Configuration management for external dependencies")
        else:
            recommendations.append("Implement centralized configuration for external services")

        return AgentPerspective(
            role=AgentRole.INTEGRATION,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _data_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based data analysis."""
        analysis = "From a data perspective, the system implements data management patterns with consideration for storage, retrieval, and data integrity."

        key_insights = []
        recommendations = []
        focus_areas = ["Data Architecture", "Database Design"]

        # Analyze data patterns
        if any(word in code_content for word in ["database", "db", "query", "sql"]):
            key_insights.append("Database integration and query patterns are implemented")
            recommendations.append("Optimize database queries and implement proper indexing")

        if any(word in code_content for word in ["model", "schema", "table", "collection"]):
            key_insights.append("Data modeling structures are defined")
        else:
            recommendations.append("Implement comprehensive data models and schemas")

        if "async" in code_content and any(word in code_content for word in ["db", "database", "query"]):
            key_insights.append("Asynchronous database operations for better performance")
        else:
            recommendations.append("Consider async database operations for scalability")

        return AgentPerspective(
            role=AgentRole.DATA,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _ui_ux_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based UI/UX analysis."""
        analysis = "From a UI/UX perspective, the system implements user interface patterns that support user interaction and experience."

        key_insights = []
        recommendations = []
        focus_areas = ["User Interface", "Frontend Architecture"]

        # Analyze UI patterns
        if any(word in code_content for word in ["component", "react", "vue", "angular", "ui"]):
            key_insights.append("Component-based UI architecture is implemented")
            recommendations.append("Ensure component reusability and consistent design patterns")

        if any("frontend" in path or "ui" in path or "component" in path for path in file_paths):
            key_insights.append("Organized frontend code structure")
        else:
            recommendations.append("Implement proper frontend code organization")

        if any(word in code_content for word in ["state", "store", "context", "redux"]):
            key_insights.append("State management patterns are implemented")
        else:
            recommendations.append("Implement proper state management for complex UIs")

        return AgentPerspective(
            role=AgentRole.UI_UX,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _devops_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based DevOps analysis."""
        analysis = "From a DevOps perspective, the system shows consideration for deployment, infrastructure, and operational concerns."

        key_insights = []
        recommendations = []
        focus_areas = ["Deployment", "Infrastructure"]

        # Analyze DevOps patterns
        if any(word in code_content for word in ["docker", "container", "kubernetes", "k8s"]):
            key_insights.append("Containerization and orchestration patterns are implemented")
            recommendations.append("Optimize container images and resource allocation")

        if any(word in code_content for word in ["config", "env", "environment", "settings"]):
            key_insights.append("Environment configuration management is present")
        else:
            recommendations.append("Implement proper environment configuration management")

        if any("docker" in path or "k8s" in path or "deploy" in path for path in file_paths):
            key_insights.append("Infrastructure as code patterns are organized")
        else:
            recommendations.append("Implement infrastructure as code practices")

        return AgentPerspective(
            role=AgentRole.DEVOPS,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.6,
            focus_areas=focus_areas
        )

    def _testing_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based testing analysis."""
        analysis = "From a testing perspective, the system demonstrates patterns for quality assurance and test automation."

        key_insights = []
        recommendations = []
        focus_areas = ["Testing Strategy", "Quality Assurance"]

        # Analyze testing patterns
        if any("test" in path for path in file_paths):
            key_insights.append("Dedicated testing infrastructure is organized")
            recommendations.append("Ensure comprehensive test coverage across all components")
        else:
            recommendations.append("Implement comprehensive testing infrastructure")

        if any(word in code_content for word in ["test", "mock", "assert", "expect"]):
            key_insights.append("Testing patterns and assertions are implemented")
        else:
            recommendations.append("Add unit and integration tests for all critical components")

        if "async" in code_content and "test" in code_content:
            key_insights.append("Asynchronous testing patterns for concurrent code")
        else:
            recommendations.append("Implement proper testing for asynchronous operations")

        return AgentPerspective(
            role=AgentRole.TESTING,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.7,
            focus_areas=focus_areas
        )

    def _compliance_rule_based_analysis(self, query: str, chunks: List[CodeChunk], code_content: str, file_paths: List[str]) -> AgentPerspective:
        """Rule-based compliance analysis."""
        analysis = "From a compliance perspective, the system requires attention to regulatory requirements, data protection, and audit capabilities."

        key_insights = []
        recommendations = []
        focus_areas = ["Compliance", "Risk Management"]

        # Analyze compliance patterns
        if any(word in code_content for word in ["audit", "log", "logging", "track"]):
            key_insights.append("Audit trail and logging mechanisms are implemented")
            recommendations.append("Ensure comprehensive audit logging for all critical operations")
        else:
            recommendations.append("Implement comprehensive audit logging and tracking")

        if any(word in code_content for word in ["privacy", "gdpr", "data", "personal"]):
            key_insights.append("Data privacy considerations are present")
        else:
            recommendations.append("Implement data privacy and protection measures")

        recommendations.append("Establish compliance monitoring and reporting mechanisms")

        return AgentPerspective(
            role=AgentRole.COMPLIANCE,
            analysis=analysis,
            key_insights=key_insights[:3],
            recommendations=recommendations[:3],
            confidence=0.5,
            focus_areas=focus_areas
        )

    def _fallback_agent_perspective(self, agent_role: AgentRole, query: str, chunks: List[CodeChunk]) -> AgentPerspective:
        """Fallback perspective when analysis fails."""
        return AgentPerspective(
            role=agent_role,
            analysis=f"Analysis from {agent_role.value} perspective is currently unavailable.",
            key_insights=[f"Unable to provide {agent_role.value} insights at this time"],
            recommendations=[f"Consider manual {agent_role.value} review"],
            confidence=0.3,
            focus_areas=[agent_role.value.title()]
        )

    async def _synthesize_perspectives(
        self,
        query: str,
        perspectives: List[AgentPerspective],
        chunks: List[CodeChunk],
        context: Dict[str, Any]
    ) -> FlowResponse:
        """Synthesize multiple agent perspectives into a flowing, comprehensive response."""

        if self.client:
            return await self._llm_synthesize_perspectives(query, perspectives, chunks, context)
        else:
            return self._rule_based_synthesize_perspectives(query, perspectives, chunks, context)

    async def _llm_synthesize_perspectives(
        self,
        query: str,
        perspectives: List[AgentPerspective],
        chunks: List[CodeChunk],
        context: Dict[str, Any]
    ) -> FlowResponse:
        """Use LLM to create a flowing synthesis of all perspectives."""

        # Prepare perspectives summary for LLM
        perspectives_summary = self._format_perspectives_for_synthesis(perspectives)

        synthesis_prompt = f"""
You are a senior technical architect synthesizing multiple expert perspectives on a detailed codebase analysis.

USER QUERY: "{query}"

EXPERT PERSPECTIVES FROM SPECIALIZED AGENTS:
{perspectives_summary}

Your task is to create a comprehensive, detailed technical response that:

1. **EXECUTIVE SUMMARY**: Provide a specific technical overview (3-4 sentences) that mentions actual code components, files, or architectural patterns identified

2. **DETAILED ANALYSIS**: Create a flowing technical narrative that:
   - References specific files, functions, classes, and code patterns mentioned by agents
   - Explains the actual implementation approach and architectural decisions
   - Discusses concrete technical trade-offs and design choices
   - Integrates insights from multiple perspectives into a cohesive technical story
   - Includes specific examples of code patterns, design decisions, or implementation details

3. **SYNTHESIS**: Connect different technical viewpoints by:
   - Highlighting how different aspects (architecture, performance, security, etc.) interact
   - Identifying common themes across multiple expert perspectives
   - Explaining the technical relationships between different system components

4. **ACTION ITEMS**: Provide specific, implementable technical recommendations:
   - Include concrete steps with technical details
   - Reference specific files, functions, or components that need attention
   - Prioritize based on impact and technical complexity

5. **FOLLOW-UP QUESTIONS**: Suggest 2-3 technical follow-up questions that would provide deeper insights

IMPORTANT: Be highly specific and technical. Reference actual code elements, implementation patterns, and architectural decisions. Avoid generic statements. Focus on the actual codebase being analyzed.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior technical architect who synthesizes multiple expert perspectives into comprehensive, flowing analyses."},
                {"role": "user", "content": synthesis_prompt}
            ],
            temperature=0.5,
            max_tokens=1200
        )

        synthesis_text = response.choices[0].message.content.strip()

        # Parse the synthesis response
        return self._parse_synthesis_response(synthesis_text, perspectives, query)

    def _format_perspectives_for_synthesis(self, perspectives: List[AgentPerspective]) -> str:
        """Format agent perspectives for LLM synthesis."""
        formatted_perspectives = []

        for perspective in perspectives:
            formatted = f"""
{perspective.role.value.upper()} PERSPECTIVE (Confidence: {perspective.confidence:.1f}):
Analysis: {perspective.analysis}

Key Insights:
{chr(10).join(f"• {insight}" for insight in perspective.key_insights)}

Recommendations:
{chr(10).join(f"• {rec}" for rec in perspective.recommendations)}

Focus Areas: {', '.join(perspective.focus_areas)}
"""
            formatted_perspectives.append(formatted)

        return "\n" + "="*50 + "\n".join(formatted_perspectives)

    def _parse_synthesis_response(self, synthesis_text: str, perspectives: List[AgentPerspective], query: str) -> FlowResponse:
        """Parse LLM synthesis response into structured FlowResponse."""

        # Simple parsing - in production, you might want more sophisticated parsing
        sections = synthesis_text.split('\n\n')

        executive_summary = ""
        detailed_analysis = ""
        synthesis = ""
        action_items = []
        follow_up_questions = []

        current_section = "summary"

        for section in sections:
            section = section.strip()
            if not section:
                continue

            section_lower = section.lower()

            if "executive summary" in section_lower or current_section == "summary":
                executive_summary += section + "\n\n"
                current_section = "analysis"
            elif "detailed analysis" in section_lower or "analysis" in section_lower:
                detailed_analysis += section + "\n\n"
                current_section = "synthesis"
            elif "synthesis" in section_lower or current_section == "synthesis":
                synthesis += section + "\n\n"
                current_section = "actions"
            elif "action" in section_lower or "next steps" in section_lower:
                # Extract action items
                lines = section.split('\n')
                for line in lines:
                    if line.strip().startswith(('•', '-', '1.', '2.', '3.')):
                        action_items.append(line.strip().lstrip('•-123. '))
                current_section = "questions"
            elif "follow" in section_lower or "question" in section_lower:
                # Extract questions
                lines = section.split('\n')
                for line in lines:
                    if line.strip().startswith(('•', '-', '1.', '2.', '3.')) or '?' in line:
                        follow_up_questions.append(line.strip().lstrip('•-123. '))

        # Fallback if parsing didn't work well
        if not executive_summary:
            executive_summary = synthesis_text[:200] + "..."
        if not detailed_analysis:
            detailed_analysis = synthesis_text

        return FlowResponse(
            executive_summary=executive_summary.strip(),
            detailed_analysis=detailed_analysis.strip(),
            agent_perspectives=perspectives,
            synthesis=synthesis.strip(),
            action_items=action_items[:5],  # Limit to 5 action items
            follow_up_questions=follow_up_questions[:3]  # Limit to 3 questions
        )

    def _rule_based_synthesize_perspectives(
        self,
        query: str,
        perspectives: List[AgentPerspective],
        chunks: List[CodeChunk],
        context: Dict[str, Any]
    ) -> FlowResponse:
        """Rule-based synthesis when LLM is not available."""

        # Create executive summary with more detail
        agent_roles = [p.role.value.replace('_', ' ').title() for p in perspectives]
        focus_areas = list(set(area for p in perspectives for area in p.focus_areas))

        executive_summary = f"""Comprehensive analysis from {len(perspectives)} specialized perspectives ({', '.join(agent_roles[:3])}{'...' if len(agent_roles) > 3 else ''}) reveals a complex system with significant opportunities for enhancement across {len(focus_areas)} key areas: {', '.join(focus_areas[:5])}{'...' if len(focus_areas) > 5 else ''}."""

        # Create detailed analysis with better organization
        detailed_analysis = f"This multi-dimensional analysis examines {len(chunks)} code components through {len(perspectives)} expert lenses, providing comprehensive insights into system architecture, implementation quality, and improvement opportunities.\n\n"

        # Group perspectives by priority/importance
        high_priority_perspectives = [p for p in perspectives if self.agents.get(p.role, {}).get('priority', 3) <= 2]
        other_perspectives = [p for p in perspectives if self.agents.get(p.role, {}).get('priority', 3) > 2]

        # Add high priority perspectives first
        if high_priority_perspectives:
            detailed_analysis += "**Core System Analysis:**\n\n"
            for perspective in high_priority_perspectives:
                detailed_analysis += f"• **{perspective.role.value.replace('_', ' ').title()}**: {perspective.analysis}\n\n"

        # Add other perspectives
        if other_perspectives:
            detailed_analysis += "**Specialized Analysis:**\n\n"
            for perspective in other_perspectives:
                detailed_analysis += f"• **{perspective.role.value.replace('_', ' ').title()}**: {perspective.analysis}\n\n"

        # Create enhanced synthesis
        all_insights = []
        all_recommendations = []
        confidence_scores = []

        for perspective in perspectives:
            all_insights.extend(perspective.key_insights)
            all_recommendations.extend(perspective.recommendations)
            confidence_scores.append(perspective.confidence)

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        unique_insights = list(set(all_insights))
        unique_recommendations = list(set(all_recommendations))

        synthesis = f"""The convergence of {len(perspectives)} expert perspectives (average confidence: {avg_confidence:.1f}) reveals {len(unique_insights)} distinct insights and {len(unique_recommendations)} actionable recommendations. Key themes emerge around system architecture, code quality, and operational excellence, with particular emphasis on {focus_areas[0] if focus_areas else 'system improvement'}."""

        # Prioritize action items by frequency and importance
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        # Sort by frequency (most mentioned first)
        sorted_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        action_items = [rec for rec, count in sorted_recommendations[:7]]

        # Create more insightful follow-up questions
        follow_up_questions = [
            f"Which of the {len(unique_recommendations)} recommendations should be prioritized based on business impact?",
            f"How do the {perspective.role.value.replace('_', ' ')} insights align with current development priorities?" if perspectives else "What are the current development priorities?",
            f"What would be the estimated effort to address the top {min(3, len(action_items))} action items?",
            "Are there any dependencies between the recommended improvements that should influence implementation order?"
        ]

        return FlowResponse(
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            agent_perspectives=perspectives,
            synthesis=synthesis,
            action_items=action_items,
            follow_up_questions=follow_up_questions[:3]  # Limit to 3 most relevant
        )

    def _fallback_response(self, query: str, chunks: List[CodeChunk]) -> FlowResponse:
        """Fallback response when agent orchestration fails."""
        return FlowResponse(
            executive_summary=f"Analysis of {len(chunks)} code components related to your query.",
            detailed_analysis=f"The system contains {len(chunks)} relevant components that address your query about the codebase. While detailed multi-perspective analysis is currently unavailable, the identified components provide insight into the system's structure and functionality.",
            agent_perspectives=[],
            synthesis="A comprehensive multi-agent analysis would provide deeper insights into this system's architecture, implementation, and optimization opportunities.",
            action_items=[
                "Review the identified code components",
                "Consider implementing comprehensive code analysis",
                "Evaluate system architecture and design patterns"
            ],
            follow_up_questions=[
                "What specific aspects of these components would you like to explore further?",
                "Are there particular architectural concerns you'd like to address?",
                "Would you like to focus on any specific quality attributes?"
            ]
        )
