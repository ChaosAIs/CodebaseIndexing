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

            # Smart agent selection with performance considerations
            relevant_agents = self._select_relevant_agents_optimized(query, chunks)

            if not relevant_agents:
                logger.warning("No relevant agents selected for query")
                return self._fallback_response(query, chunks)

            logger.info(f"Selected {len(relevant_agents)} agents: {[a.value for a in relevant_agents]}")

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
        """Run agents with controlled concurrency to avoid overwhelming the system."""
        semaphore = asyncio.Semaphore(self.max_concurrent_agents)

        async def run_single_agent(agent_role: AgentRole) -> AgentPerspective:
            async with semaphore:
                return await self._run_agent_analysis(agent_role, query, chunks, context)

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
        prompt = self._build_agent_prompt(agent_role, agent_config, query, code_context, context)
        
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
        
        # Parse the structured response
        return self._parse_agent_response(agent_role, analysis_text)
    
    def _build_agent_prompt(
        self, 
        agent_role: AgentRole, 
        agent_config: Dict[str, Any], 
        query: str, 
        code_context: str, 
        context: Dict[str, Any]
    ) -> str:
        """Build a specialized prompt for each agent role."""
        
        base_context = f"""
AGENT ROLE: {agent_role.value.title()}
FOCUS AREA: {agent_config['focus']}
PERSPECTIVE: Analyze from the {agent_config['perspective']} viewpoint

USER QUERY: "{query}"

CODE CONTEXT:
{code_context}
"""
        
        role_specific_instructions = {
            AgentRole.ARCHITECT: """
Analyze the system architecture and design patterns. Focus on:
- Overall system structure and component organization
- Architectural patterns and design principles used
- Component relationships and dependencies
- Scalability and extensibility considerations
- Design trade-offs and architectural decisions

Provide insights on how the architecture supports the system's goals.
""",
            AgentRole.DEVELOPER: """
Analyze the code implementation and development practices. Focus on:
- Code quality and implementation patterns
- Algorithm efficiency and data structure choices
- Coding standards and best practices adherence
- Error handling and edge case management
- Code organization and modularity

Provide insights on the technical implementation quality.
""",
            AgentRole.SECURITY: """
Analyze security aspects and potential vulnerabilities. Focus on:
- Authentication and authorization mechanisms
- Input validation and sanitization
- Data protection and encryption usage
- Access control and permission management
- Potential security vulnerabilities and risks

Provide insights on security posture and recommendations.
""",
            AgentRole.PERFORMANCE: """
Analyze performance characteristics and optimization opportunities. Focus on:
- Performance bottlenecks and optimization opportunities
- Resource utilization (CPU, memory, I/O)
- Scalability limitations and solutions
- Caching strategies and efficiency improvements
- Database query optimization and indexing

Provide insights on performance optimization potential.
""",
            AgentRole.MAINTAINER: """
Analyze maintainability and long-term code health. Focus on:
- Code complexity and maintainability metrics
- Technical debt and refactoring opportunities
- Documentation quality and completeness
- Testing coverage and quality
- Code organization and modularity for maintenance

Provide insights on long-term maintainability and health.
""",
            AgentRole.BUSINESS: """
Analyze business logic and domain modeling. Focus on:
- Business rule implementation and clarity
- Domain model accuracy and completeness
- Functional requirement fulfillment
- Business process automation and efficiency
- User experience and business value delivery

Provide insights on business logic implementation and domain understanding.
""",
            AgentRole.INTEGRATION: """
Analyze system integration and external dependencies. Focus on:
- External service integrations and API usage
- Dependency management and version control
- Inter-service communication patterns
- Data flow between system components
- Integration testing and error handling

Provide insights on integration architecture and dependency management.
""",
            AgentRole.DATA: """
Analyze data architecture and information management. Focus on:
- Database design and schema optimization
- Data modeling and relationships
- Query performance and indexing strategies
- Data flow and transformation processes
- Storage optimization and data lifecycle management

Provide insights on data architecture and information management strategies.
""",
            AgentRole.UI_UX: """
Analyze user interface and experience design. Focus on:
- Component architecture and reusability
- User interaction patterns and workflows
- Frontend performance and optimization
- Accessibility and usability considerations
- State management and data binding

Provide insights on user interface design and frontend architecture.
""",
            AgentRole.DEVOPS: """
Analyze deployment and operational considerations. Focus on:
- Infrastructure as code and deployment strategies
- Containerization and orchestration patterns
- CI/CD pipeline design and automation
- Monitoring, logging, and observability
- Scalability and reliability considerations

Provide insights on deployment and operational excellence.
""",
            AgentRole.TESTING: """
Analyze testing strategy and quality assurance. Focus on:
- Test coverage and testing pyramid implementation
- Unit, integration, and end-to-end testing strategies
- Test automation and continuous testing
- Quality gates and code quality metrics
- Performance and load testing considerations

Provide insights on testing strategy and quality assurance practices.
""",
            AgentRole.COMPLIANCE: """
Analyze compliance and regulatory requirements. Focus on:
- Regulatory compliance and standards adherence
- Data privacy and protection requirements
- Audit trails and documentation standards
- Security compliance and risk management
- Policy enforcement and governance frameworks

Provide insights on compliance requirements and risk mitigation.
"""
        }
        
        specific_instructions = role_specific_instructions.get(
            agent_role, 
            "Analyze the code from your specialized perspective."
        )
        
        return f"""
{base_context}

{specific_instructions}

Please provide your analysis in the following format:

ANALYSIS:
[Your detailed analysis from your role's perspective - 2-3 paragraphs]

KEY INSIGHTS:
- [Insight 1]
- [Insight 2] 
- [Insight 3]

RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]

CONFIDENCE: [High/Medium/Low]

FOCUS AREAS:
- [Area 1]
- [Area 2]
"""

    def _prepare_code_context_for_agent(self, chunks: List[CodeChunk], agent_role: AgentRole) -> str:
        """Prepare code context tailored for specific agent perspective."""
        context_parts = []

        # Limit chunks to avoid token overflow
        relevant_chunks = chunks[:3]

        for i, chunk in enumerate(relevant_chunks):
            chunk_info = f"""
CODE CHUNK {i+1}:
File: {chunk.file_path}
Lines: {chunk.start_line}-{chunk.end_line}
Type: {chunk.node_type.value if chunk.node_type else 'unknown'}
Name: {chunk.name or 'unnamed'}

Content:
{chunk.content[:1000]}{'...' if len(chunk.content) > 1000 else ''}
"""
            context_parts.append(chunk_info)

        return "\n".join(context_parts)

    def _parse_agent_response(self, agent_role: AgentRole, response_text: str) -> AgentPerspective:
        """Parse structured agent response into AgentPerspective object."""
        try:
            # Extract sections using simple parsing
            sections = {}
            current_section = None
            current_content = []

            for line in response_text.split('\n'):
                line = line.strip()
                if line.upper().startswith(('ANALYSIS:', 'KEY INSIGHTS:', 'RECOMMENDATIONS:', 'CONFIDENCE:', 'FOCUS AREAS:')):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = line.split(':')[0].upper()
                    current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
                elif current_section and line:
                    current_content.append(line)

            # Add the last section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()

            # Extract structured data
            analysis = sections.get('ANALYSIS', 'Analysis not provided')

            key_insights = []
            if 'KEY INSIGHTS' in sections:
                insights_text = sections['KEY INSIGHTS']
                key_insights = [
                    line.strip('- ').strip()
                    for line in insights_text.split('\n')
                    if line.strip().startswith('-')
                ]

            recommendations = []
            if 'RECOMMENDATIONS' in sections:
                rec_text = sections['RECOMMENDATIONS']
                recommendations = [
                    line.strip('- ').strip()
                    for line in rec_text.split('\n')
                    if line.strip().startswith('-')
                ]

            confidence_text = sections.get('CONFIDENCE', 'medium').lower()
            confidence = 0.8 if 'high' in confidence_text else 0.6 if 'medium' in confidence_text else 0.4

            focus_areas = []
            if 'FOCUS AREAS' in sections:
                focus_text = sections['FOCUS AREAS']
                focus_areas = [
                    line.strip('- ').strip()
                    for line in focus_text.split('\n')
                    if line.strip().startswith('-')
                ]

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
You are a senior technical lead synthesizing multiple expert perspectives on a codebase analysis.

USER QUERY: "{query}"

EXPERT PERSPECTIVES:
{perspectives_summary}

Your task is to create a comprehensive, flowing response that:

1. **EXECUTIVE SUMMARY**: Start with a clear, concise overview (2-3 sentences)
2. **DETAILED ANALYSIS**: Provide a flowing narrative that weaves together insights from all perspectives
3. **SYNTHESIS**: Connect the different viewpoints and highlight key themes
4. **ACTION ITEMS**: Provide concrete, prioritized next steps
5. **FOLLOW-UP QUESTIONS**: Suggest 2-3 insightful follow-up questions

Make the response feel like a natural conversation with a senior architect who has consulted with multiple experts. Use a flowing, narrative style rather than bullet points where possible.

Focus on creating connections between different perspectives and providing a holistic view of the system.
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
