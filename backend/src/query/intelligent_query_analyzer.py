"""
Intelligent Query Analyzer for optimizing query processing strategy.

This module analyzes incoming queries to determine:
1. Query complexity and processing requirements
2. Required agent roles and their specific tasks
3. Processing strategy (streaming vs batch)
4. Resource allocation and optimization hints
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import openai

from ..agents.agent_orchestrator import AgentRole
from ..models import CodeChunk


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"           # Single concept, specific search
    MODERATE = "moderate"       # Multiple concepts, focused analysis
    COMPLEX = "complex"         # Broad analysis, multiple perspectives
    ARCHITECTURAL = "architectural"  # System-wide analysis


class ProcessingStrategy(Enum):
    """Processing strategy options."""
    DIRECT_SEARCH = "direct_search"     # Simple embedding search
    FOCUSED_ANALYSIS = "focused_analysis"  # 1-3 agents
    MULTI_PERSPECTIVE = "multi_perspective"  # 4-6 agents
    COMPREHENSIVE = "comprehensive"     # 7+ agents with streaming


@dataclass
class AgentTask:
    """Specific task assignment for an agent."""
    agent_role: AgentRole
    task_description: str
    priority: int
    estimated_chunks_needed: int
    specific_focus_areas: List[str]


@dataclass
class QueryAnalysisResult:
    """Result of intelligent query analysis."""
    complexity: QueryComplexity
    processing_strategy: ProcessingStrategy
    required_agents: List[AgentTask]
    estimated_processing_time: float
    should_stream: bool
    search_optimization_hints: Dict[str, Any]
    explanation: str


class IntelligentQueryAnalyzer:
    """
    Analyzes queries to optimize processing strategy and agent selection.
    """
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None):
        """Initialize the intelligent query analyzer."""
        self.client = openai_client
        
        # Query pattern analysis
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r"what is \w+", r"how does \w+ work", r"find \w+",
                r"show me \w+", r"where is \w+", r"explain \w+"
            ],
            QueryComplexity.MODERATE: [
                r"analyze \w+", r"compare \w+ and \w+", r"how to \w+",
                r"best practices for \w+", r"optimize \w+"
            ],
            QueryComplexity.COMPLEX: [
                r"review \w+", r"assessment of \w+", r"comprehensive \w+",
                r"evaluate \w+", r"audit \w+"
            ],
            QueryComplexity.ARCHITECTURAL: [
                r"architecture", r"system design", r"overall structure",
                r"solution architecture", r"system overview", r"design patterns"
            ]
        }
        
        # Agent specialization mapping
        self.agent_specializations = {
            AgentRole.ARCHITECT: {
                "triggers": ["architecture", "design", "structure", "system", "overview"],
                "tasks": ["system design analysis", "architectural pattern identification", "component relationship mapping"],
                "base_priority": 1
            },
            AgentRole.DEVELOPER: {
                "triggers": ["code", "implementation", "function", "class", "algorithm"],
                "tasks": ["code quality assessment", "implementation analysis", "best practices review"],
                "base_priority": 1
            },
            AgentRole.SECURITY: {
                "triggers": ["security", "auth", "vulnerability", "permission", "encryption"],
                "tasks": ["security vulnerability assessment", "authentication analysis", "data protection review"],
                "base_priority": 2
            },
            AgentRole.PERFORMANCE: {
                "triggers": ["performance", "optimization", "scalability", "bottleneck", "efficiency"],
                "tasks": ["performance bottleneck identification", "scalability assessment", "optimization recommendations"],
                "base_priority": 2
            },
            AgentRole.MAINTAINER: {
                "triggers": ["maintainability", "refactor", "debt", "complexity", "quality"],
                "tasks": ["technical debt analysis", "maintainability assessment", "refactoring recommendations"],
                "base_priority": 1
            },
            AgentRole.BUSINESS: {
                "triggers": ["business", "domain", "logic", "requirements", "functional"],
                "tasks": ["business logic analysis", "domain model review", "functional requirements assessment"],
                "base_priority": 3
            },
            AgentRole.INTEGRATION: {
                "triggers": ["integration", "api", "dependency", "external", "service"],
                "tasks": ["integration point analysis", "dependency review", "API design assessment"],
                "base_priority": 3
            },
            AgentRole.DATA: {
                "triggers": ["data", "database", "model", "schema", "storage"],
                "tasks": ["data model analysis", "database design review", "data flow assessment"],
                "base_priority": 3
            }
        }
    
    async def analyze_query(self, query: str, available_chunks: int = 0) -> QueryAnalysisResult:
        """
        Perform intelligent analysis of the query to determine optimal processing strategy.
        
        Args:
            query: User's natural language query
            available_chunks: Number of code chunks available for analysis
            
        Returns:
            QueryAnalysisResult with processing recommendations
        """
        logger.info(f"Analyzing query for optimal processing: {query}")
        
        # Step 1: Determine query complexity
        complexity = self._assess_query_complexity(query)
        logger.info(f"Query complexity assessed as: {complexity.value}")
        
        # Step 2: Select required agents and their tasks
        required_agents = await self._select_agents_with_tasks(query, complexity)
        logger.info(f"Selected {len(required_agents)} agents for processing")
        
        # Step 3: Determine processing strategy
        processing_strategy = self._determine_processing_strategy(complexity, len(required_agents), available_chunks)
        
        # Step 4: Estimate processing time and streaming decision
        estimated_time, should_stream = self._estimate_processing_requirements(
            complexity, len(required_agents), available_chunks
        )
        
        # Step 5: Generate search optimization hints
        search_hints = self._generate_search_optimization_hints(query, complexity, required_agents)
        
        # Step 6: Generate explanation
        explanation = self._generate_processing_explanation(
            complexity, processing_strategy, required_agents, should_stream
        )
        
        result = QueryAnalysisResult(
            complexity=complexity,
            processing_strategy=processing_strategy,
            required_agents=required_agents,
            estimated_processing_time=estimated_time,
            should_stream=should_stream,
            search_optimization_hints=search_hints,
            explanation=explanation
        )
        
        logger.info(f"Query analysis complete: {result.explanation}")
        return result
    
    def _assess_query_complexity(self, query: str) -> QueryComplexity:
        """Assess the complexity of the query based on patterns and keywords."""
        query_lower = query.lower()
        
        # Check for architectural patterns first (highest complexity)
        for pattern in self.complexity_patterns[QueryComplexity.ARCHITECTURAL]:
            if pattern in query_lower:
                return QueryComplexity.ARCHITECTURAL
        
        # Check for complex analysis patterns
        for pattern in self.complexity_patterns[QueryComplexity.COMPLEX]:
            if pattern in query_lower:
                return QueryComplexity.COMPLEX
        
        # Check for moderate complexity patterns
        for pattern in self.complexity_patterns[QueryComplexity.MODERATE]:
            if pattern in query_lower:
                return QueryComplexity.MODERATE
        
        # Default to simple
        return QueryComplexity.SIMPLE
    
    async def _select_agents_with_tasks(self, query: str, complexity: QueryComplexity) -> List[AgentTask]:
        """Select agents and assign specific tasks based on query analysis."""
        query_lower = query.lower()
        selected_agents = []
        
        # Score agents based on query relevance
        agent_scores = {}
        for agent_role, config in self.agent_specializations.items():
            score = 0
            
            # Base priority score
            priority = config["base_priority"]
            if priority == 1:
                score += 100
            elif priority == 2:
                score += 50
            else:
                score += 20
            
            # Trigger word matching
            trigger_matches = sum(1 for trigger in config["triggers"] if trigger in query_lower)
            score += trigger_matches * 30
            
            agent_scores[agent_role] = score
        
        # Determine how many agents to select based on complexity
        max_agents = {
            QueryComplexity.SIMPLE: 2,
            QueryComplexity.MODERATE: 4,
            QueryComplexity.COMPLEX: 6,
            QueryComplexity.ARCHITECTURAL: 8
        }[complexity]
        
        # Select top scoring agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        for agent_role, score in sorted_agents[:max_agents]:
            if score >= 50:  # Minimum relevance threshold
                config = self.agent_specializations[agent_role]
                
                # Assign specific task based on query and agent specialization
                task_description = await self._generate_agent_task(agent_role, query, complexity)
                
                # Estimate chunks needed
                chunks_needed = self._estimate_chunks_for_agent(agent_role, complexity)
                
                # Extract focus areas from query
                focus_areas = [trigger for trigger in config["triggers"] if trigger in query_lower]
                
                agent_task = AgentTask(
                    agent_role=agent_role,
                    task_description=task_description,
                    priority=config["base_priority"],
                    estimated_chunks_needed=chunks_needed,
                    specific_focus_areas=focus_areas
                )
                
                selected_agents.append(agent_task)
        
        return selected_agents
    
    async def _generate_agent_task(self, agent_role: AgentRole, query: str, complexity: QueryComplexity) -> str:
        """Generate specific task description for an agent based on the query."""
        if self.client:
            try:
                # Use LLM to generate specific task
                prompt = f"""
                Given this user query: "{query}"
                And this agent role: {agent_role.value}
                
                Generate a specific, actionable task for this agent that directly addresses the user's question.
                The task should be focused and achievable within the agent's expertise.
                
                Respond with just the task description (1-2 sentences).
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Failed to generate LLM task for {agent_role.value}: {e}")
        
        # Fallback to predefined tasks
        config = self.agent_specializations[agent_role]
        return config["tasks"][0]  # Use first default task
    
    def _estimate_chunks_for_agent(self, agent_role: AgentRole, complexity: QueryComplexity) -> int:
        """Estimate how many chunks an agent needs for effective analysis."""
        base_chunks = {
            QueryComplexity.SIMPLE: 5,
            QueryComplexity.MODERATE: 10,
            QueryComplexity.COMPLEX: 15,
            QueryComplexity.ARCHITECTURAL: 20
        }[complexity]
        
        # Some agents need more chunks for comprehensive analysis
        if agent_role in [AgentRole.ARCHITECT, AgentRole.DEVELOPER]:
            return base_chunks + 5
        
        return base_chunks
    
    def _determine_processing_strategy(self, complexity: QueryComplexity, num_agents: int, available_chunks: int) -> ProcessingStrategy:
        """Determine the optimal processing strategy."""
        if complexity == QueryComplexity.SIMPLE and num_agents <= 2:
            return ProcessingStrategy.DIRECT_SEARCH
        elif complexity == QueryComplexity.MODERATE and num_agents <= 3:
            return ProcessingStrategy.FOCUSED_ANALYSIS
        elif num_agents <= 6:
            return ProcessingStrategy.MULTI_PERSPECTIVE
        else:
            return ProcessingStrategy.COMPREHENSIVE
    
    def _estimate_processing_requirements(self, complexity: QueryComplexity, num_agents: int, available_chunks: int) -> Tuple[float, bool]:
        """Estimate processing time and determine if streaming is needed."""
        # Base time estimates (in seconds)
        base_times = {
            QueryComplexity.SIMPLE: 5,
            QueryComplexity.MODERATE: 15,
            QueryComplexity.COMPLEX: 30,
            QueryComplexity.ARCHITECTURAL: 45
        }
        
        base_time = base_times[complexity]
        
        # Add time for each agent (assuming some parallelization)
        agent_time = (num_agents * 8) / 3  # Assuming 3-way parallelization
        
        total_time = base_time + agent_time
        
        # Stream if processing will take more than 20 seconds
        should_stream = total_time > 20
        
        return total_time, should_stream
    
    def _generate_search_optimization_hints(self, query: str, complexity: QueryComplexity, agents: List[AgentTask]) -> Dict[str, Any]:
        """Generate hints for optimizing the search process."""
        return {
            "expand_query": complexity in [QueryComplexity.COMPLEX, QueryComplexity.ARCHITECTURAL],
            "use_graph_search": len(agents) > 3,
            "chunk_limit_per_agent": max(10, 100 // len(agents)) if agents else 10,
            "prioritize_architectural_files": complexity == QueryComplexity.ARCHITECTURAL,
            "enable_entity_extraction": complexity != QueryComplexity.SIMPLE
        }
    
    def _generate_processing_explanation(self, complexity: QueryComplexity, strategy: ProcessingStrategy, 
                                       agents: List[AgentTask], should_stream: bool) -> str:
        """Generate human-friendly explanation of the processing plan."""
        agent_names = [agent.agent_role.value for agent in agents]
        
        explanation = f"Query complexity: {complexity.value}. "
        explanation += f"Processing strategy: {strategy.value}. "
        explanation += f"Selected {len(agents)} agents: {', '.join(agent_names)}. "
        
        if should_stream:
            explanation += "Using streaming response for real-time updates."
        else:
            explanation += "Processing synchronously for quick response."
        
        return explanation
