"""Enhanced query processing for better semantic understanding and intent classification."""

import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from loguru import logger
from openai import OpenAI

from ..models import CodeChunk, NodeType
from ..config import config


class QueryIntent(Enum):
    """Types of query intents."""
    ARCHITECTURE = "architecture"  # Questions about system architecture, design patterns
    IMPLEMENTATION = "implementation"  # Questions about specific implementations
    FUNCTIONALITY = "functionality"  # Questions about what code does
    DEBUGGING = "debugging"  # Questions about errors, issues, troubleshooting
    USAGE = "usage"  # Questions about how to use something
    SEARCH = "search"  # General search queries
    RELATIONSHIP = "relationship"  # Questions about how components relate


class QueryProcessor:
    """Enhanced query processor for better semantic understanding and abstract query handling."""

    def __init__(self):
        """Initialize query processor with LLM capabilities."""
        self.client = None
        try:
            if config.ai_models.openai_api_key:
                self.client = OpenAI(api_key=config.ai_models.openai_api_key)
                logger.info("OpenAI client initialized for intelligent query processing")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")

        self.intent_patterns = self._build_intent_patterns()
        self.architecture_keywords = {
            "architecture", "design", "pattern", "structure", "overview", "system",
            "components", "modules", "organization", "layout", "framework", "scalability",
            "maintainability", "performance", "reliability", "security", "modularity"
        }
        self.implementation_keywords = {
            "implement", "implementation", "code", "function", "method", "class",
            "algorithm", "logic", "how does", "works", "written"
        }
        self.functionality_keywords = {
            "what", "purpose", "does", "functionality", "feature", "capability",
            "behavior", "action", "operation", "task"
        }
        self.debugging_keywords = {
            "error", "bug", "issue", "problem", "fix", "debug", "troubleshoot",
            "exception", "fail", "broken", "wrong"
        }
        self.usage_keywords = {
            "how to", "usage", "use", "example", "tutorial", "guide", "help",
            "documentation", "api", "interface"
        }
        self.relationship_keywords = {
            "relationship", "connect", "related", "dependency", "depends", "calls",
            "uses", "inherits", "extends", "imports", "references"
        }

        # Abstract concept mappings for codebase indexing domain
        self.abstract_concept_mappings = {
            "scalability": [
                "performance optimization", "concurrent processing", "async operations",
                "database connection pooling", "caching mechanisms", "load balancing",
                "batch processing", "memory management", "resource utilization"
            ],
            "maintainability": [
                "code organization", "modular design", "separation of concerns",
                "error handling", "logging", "configuration management",
                "documentation", "testing", "code quality", "refactoring"
            ],
            "reliability": [
                "error handling", "exception management", "retry mechanisms",
                "health checks", "monitoring", "graceful degradation",
                "fault tolerance", "backup strategies", "data validation"
            ],
            "security": [
                "authentication", "authorization", "input validation",
                "data encryption", "secure communication", "access control",
                "api security", "vulnerability management", "secure storage"
            ],
            "performance": [
                "optimization", "caching", "indexing", "query performance",
                "memory usage", "cpu utilization", "response time",
                "throughput", "bottlenecks", "profiling"
            ]
        }
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build regex patterns for intent classification."""
        return {
            QueryIntent.ARCHITECTURE: [
                r"\b(architecture|design|structure|overview|system)\b",
                r"\b(components?|modules?|organization)\b",
                r"\bhow\s+is\s+.+\s+(structured|organized|designed)\b",
                r"\bwhat\s+is\s+the\s+(architecture|design|structure)\b"
            ],
            QueryIntent.IMPLEMENTATION: [
                r"\b(implement|implementation|algorithm|logic)\b",
                r"\bhow\s+(does|is)\s+.+\s+(implemented|coded|written)\b",
                r"\bshow\s+me\s+the\s+(code|implementation)\b",
                r"\b(function|method|class)\s+.+\s+(works|implementation)\b"
            ],
            QueryIntent.FUNCTIONALITY: [
                r"\bwhat\s+(does|is)\s+.+\s+(do|for)\b",
                r"\b(purpose|functionality|feature|capability)\b",
                r"\bwhat\s+is\s+the\s+(purpose|function)\s+of\b",
                r"\b(behavior|action|operation|task)\b"
            ],
            QueryIntent.DEBUGGING: [
                r"\b(error|bug|issue|problem|fix|debug)\b",
                r"\b(exception|fail|broken|wrong)\b",
                r"\bwhy\s+(is|does)\s+.+\s+(not\s+work|fail|error)\b",
                r"\bhow\s+to\s+(fix|debug|solve)\b"
            ],
            QueryIntent.USAGE: [
                r"\bhow\s+to\s+(use|call|invoke)\b",
                r"\b(usage|example|tutorial|guide|help)\b",
                r"\b(api|interface|documentation)\b",
                r"\bshow\s+me\s+(how|example|usage)\b"
            ],
            QueryIntent.RELATIONSHIP: [
                r"\b(relationship|connect|related|dependency)\b",
                r"\b(depends|calls|uses|inherits|extends)\b",
                r"\bhow\s+(does|is)\s+.+\s+(connected|related)\b",
                r"\bwhat\s+(calls|uses|depends)\s+on\b"
            ]
        }
    
    def classify_query_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify the intent of a query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        query_lower = query.lower()
        intent_scores = {}
        
        # Score each intent based on pattern matches
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches * 2  # Weight pattern matches highly
            
            # Add keyword-based scoring
            keywords = getattr(self, f"{intent.value}_keywords", set())
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            intent_scores[intent] = score
        
        # Find the highest scoring intent
        if not intent_scores or max(intent_scores.values()) == 0:
            return QueryIntent.SEARCH, 0.5  # Default to search with low confidence
        
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        
        # Calculate confidence (normalize to 0-1 range)
        confidence = min(max_score / 5.0, 1.0)  # Cap at 1.0
        
        return best_intent, confidence
    
    def enhance_query_for_embedding(self, query: str, intent: QueryIntent, project_context: Dict[str, Any] = None) -> str:
        """
        Enhance query for better embedding generation based on intent and context.
        
        Args:
            query: Original user query
            intent: Classified query intent
            project_context: Project-specific context
            
        Returns:
            Enhanced query string for embedding
        """
        enhanced_parts = [query]
        
        # Add intent-specific context
        if intent == QueryIntent.ARCHITECTURE:
            enhanced_parts.extend([
                "system architecture", "design patterns", "component structure",
                "module organization", "framework design"
            ])
        elif intent == QueryIntent.IMPLEMENTATION:
            enhanced_parts.extend([
                "code implementation", "algorithm logic", "function method",
                "programming solution", "technical implementation"
            ])
        elif intent == QueryIntent.FUNCTIONALITY:
            enhanced_parts.extend([
                "functionality purpose", "feature capability", "behavior operation",
                "what does this do", "code function"
            ])
        elif intent == QueryIntent.DEBUGGING:
            enhanced_parts.extend([
                "error handling", "bug fix", "troubleshooting", "exception handling",
                "debugging solution", "problem resolution"
            ])
        elif intent == QueryIntent.USAGE:
            enhanced_parts.extend([
                "usage example", "how to use", "api interface", "documentation",
                "tutorial guide", "code example"
            ])
        elif intent == QueryIntent.RELATIONSHIP:
            enhanced_parts.extend([
                "component relationship", "dependency connection", "code interaction",
                "module communication", "system integration"
            ])
        
        # Add project-specific context
        if project_context:
            technologies = project_context.get("technologies", [])
            if technologies:
                enhanced_parts.extend(technologies)
            
            # Add codebase indexing specific terms
            enhanced_parts.extend([
                "codebase indexing", "code search", "semantic search",
                "vector embeddings", "graph relationships"
            ])
        
        return " ".join(enhanced_parts)
    
    def extract_code_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract potential code entities from the query.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "functions": [],
            "classes": [],
            "files": [],
            "variables": [],
            "keywords": []
        }
        
        # Function patterns
        function_patterns = [
            r'\b([a-z_][a-z0-9_]*)\s*\(',  # function_name(
            r'\bfunction\s+([a-z_][a-z0-9_]*)\b',  # function function_name
            r'\bdef\s+([a-z_][a-z0-9_]*)\b',  # def function_name
        ]
        
        # Class patterns
        class_patterns = [
            r'\bclass\s+([A-Z][a-zA-Z0-9_]*)\b',  # class ClassName
            r'\b([A-Z][a-zA-Z0-9_]*)\s+class\b',  # ClassName class
        ]
        
        # File patterns
        file_patterns = [
            r'\b([a-z_][a-z0-9_]*\.py)\b',  # filename.py
            r'\b([a-z_][a-z0-9_]*\.js)\b',  # filename.js
            r'\b([a-z_][a-z0-9_]*\.ts)\b',  # filename.ts
        ]
        
        # Extract entities
        for pattern in function_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["functions"].extend(matches)
        
        for pattern in class_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["classes"].extend(matches)
        
        for pattern in file_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["files"].extend(matches)
        
        # Extract potential variable names (camelCase or snake_case)
        variable_pattern = r'\b([a-z][a-zA-Z0-9_]*)\b'
        potential_vars = re.findall(variable_pattern, query)
        # Filter out common English words
        common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "among", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "must",
            "can", "this", "that", "these", "those", "what", "which", "who",
            "when", "where", "why", "how", "all", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "now"
        }
        entities["variables"] = [var for var in potential_vars if var.lower() not in common_words]
        
        return entities

    def expand_entities_with_graph(self, entities: Dict[str, List[str]], graph_service) -> Dict[str, List[str]]:
        """
        Expand entities using graph relationships to get broader context.

        Args:
            entities: Initially extracted entities
            graph_service: Graph service for relationship queries

        Returns:
            Expanded entities with related components
        """
        expanded_entities = {
            "functions": list(entities.get("functions", [])),
            "classes": list(entities.get("classes", [])),
            "files": list(entities.get("files", [])),
            "variables": list(entities.get("variables", [])),
            "keywords": list(entities.get("keywords", [])),
            "related_components": []
        }

        try:
            # For each extracted entity, find related components through graph
            all_entity_names = []
            for entity_type, entity_list in entities.items():
                all_entity_names.extend(entity_list)

            if all_entity_names and graph_service:
                # Query graph for related entities
                related_entities = graph_service.find_related_entities(all_entity_names)

                # Categorize related entities
                for related in related_entities:
                    entity_name = related.get("name", "")
                    entity_type = related.get("type", "")

                    if entity_type == "function" and entity_name not in expanded_entities["functions"]:
                        expanded_entities["functions"].append(entity_name)
                    elif entity_type == "class" and entity_name not in expanded_entities["classes"]:
                        expanded_entities["classes"].append(entity_name)
                    elif entity_type == "file" and entity_name not in expanded_entities["files"]:
                        expanded_entities["files"].append(entity_name)
                    else:
                        expanded_entities["related_components"].append(entity_name)

        except Exception as e:
            logger.warning(f"Failed to expand entities with graph: {e}")

        return expanded_entities

    def generate_comprehensive_search_terms(self, query: str, expanded_entities: Dict[str, List[str]]) -> List[str]:
        """
        Generate comprehensive search terms from query and expanded entities.

        Args:
            query: Original query
            expanded_entities: Entities expanded with graph relationships

        Returns:
            List of search terms for embedding search
        """
        search_terms = [query]  # Start with original query

        # Add entity-based search terms
        for entity_type, entity_list in expanded_entities.items():
            if entity_list:
                # Create specific search terms for each entity type
                if entity_type == "functions":
                    search_terms.extend([f"function {entity}" for entity in entity_list[:5]])
                    search_terms.extend([f"def {entity}" for entity in entity_list[:5]])
                elif entity_type == "classes":
                    search_terms.extend([f"class {entity}" for entity in entity_list[:5]])
                    search_terms.extend([f"{entity} implementation" for entity in entity_list[:5]])
                elif entity_type == "files":
                    search_terms.extend([f"file {entity}" for entity in entity_list[:5]])
                elif entity_type == "related_components":
                    search_terms.extend(entity_list[:10])  # Add related components directly

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms[:20]  # Limit to top 20 search terms

    def generate_search_filters(self, query: str, intent: QueryIntent, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate search filters based on query analysis.
        
        Args:
            query: Original query
            intent: Classified intent
            entities: Extracted code entities
            
        Returns:
            Dictionary of search filters
        """
        filters = {}
        
        # Intent-based node type filtering
        if intent == QueryIntent.ARCHITECTURE:
            # Focus on classes and modules for architecture queries
            filters["preferred_node_types"] = [NodeType.CLASS, NodeType.MODULE]
        elif intent == QueryIntent.IMPLEMENTATION:
            # Focus on functions and methods for implementation queries
            filters["preferred_node_types"] = [NodeType.FUNCTION, NodeType.METHOD]
        elif intent == QueryIntent.FUNCTIONALITY:
            # Include all types but prefer functions and classes
            filters["preferred_node_types"] = [NodeType.FUNCTION, NodeType.CLASS, NodeType.METHOD]
        
        # Entity-based filtering
        if entities["files"]:
            filters["file_patterns"] = entities["files"]
        
        if entities["functions"]:
            filters["function_names"] = entities["functions"]
        
        if entities["classes"]:
            filters["class_names"] = entities["classes"]
        
        return filters

    async def expand_abstract_query(self, query: str, project_context: Dict[str, Any] = None) -> List[str]:
        """
        Expand abstract queries into concrete technical search terms.

        This is crucial for handling questions like "scalability assessment" that don't
        contain specific technical keywords but need to search for relevant code patterns.
        """
        try:
            if self.client:
                return await self._llm_expand_query(query, project_context)
            else:
                return self._rule_based_expand_query(query)
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return self._rule_based_expand_query(query)

    async def _llm_expand_query(self, query: str, project_context: Dict[str, Any] = None) -> List[str]:
        """Use LLM to intelligently expand abstract queries."""
        try:
            # Build context about the codebase indexing system
            system_context = """
You are analyzing a codebase indexing and knowledge retrieval system that includes:
- Tree-sitter parsing for AST extraction
- Code chunking with hierarchical relationships
- Vector embeddings using OpenAI/HuggingFace/Ollama
- Qdrant vector database for semantic search
- Neo4j graph database for relationship modeling
- FastAPI MCP server for LLM integration
- React frontend with chat interface

Your task is to expand abstract queries into specific technical search terms that would help find relevant code components.
"""

            project_info = ""
            if project_context:
                project_name = project_context.get("name", "Unknown")
                technologies = project_context.get("technologies", [])
                project_info = f"Project: {project_name}, Technologies: {', '.join(technologies)}"

            prompt = f"""
{system_context}
{project_info}

User Query: "{query}"

Please expand this query into 8-12 specific technical search terms that would help find relevant code components. Focus on:
1. Technical implementation patterns
2. Code structure and organization
3. Specific functions, classes, or modules
4. Architecture and design patterns
5. Performance and optimization aspects
6. Error handling and reliability
7. Configuration and setup
8. API and interface design

Return only the search terms, one per line, without explanations.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert software architect helping to expand abstract queries into concrete technical search terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            expanded_terms = response.choices[0].message.content.strip().split('\n')
            return [term.strip() for term in expanded_terms if term.strip()]

        except Exception as e:
            logger.error(f"Error in LLM query expansion: {e}")
            return self._rule_based_expand_query(query)

    def _rule_based_expand_query(self, query: str) -> List[str]:
        """Rule-based query expansion for abstract concepts."""
        query_lower = query.lower()
        expanded_terms = [query]  # Always include original query

        # Check for abstract concepts and expand them
        for concept, technical_terms in self.abstract_concept_mappings.items():
            if concept in query_lower:
                expanded_terms.extend(technical_terms)

        # Add domain-specific expansions for codebase indexing
        if any(word in query_lower for word in ["assessment", "analysis", "evaluation", "review"]):
            expanded_terms.extend([
                "code quality", "architecture patterns", "design principles",
                "error handling", "performance metrics", "resource management",
                "configuration", "logging", "monitoring", "testing"
            ])

        if any(word in query_lower for word in ["scalability", "scale", "scaling"]):
            expanded_terms.extend([
                "async", "concurrent", "parallel", "threading", "multiprocessing",
                "connection pool", "cache", "optimization", "batch", "queue",
                "load balancing", "distributed", "clustering"
            ])

        if any(word in query_lower for word in ["maintainability", "maintenance", "maintain"]):
            expanded_terms.extend([
                "modular", "separation", "abstraction", "interface", "dependency injection",
                "configuration", "logging", "error handling", "documentation",
                "testing", "refactoring", "code organization"
            ])

        if any(word in query_lower for word in ["performance", "optimization", "efficiency"]):
            expanded_terms.extend([
                "cache", "index", "query optimization", "memory", "cpu",
                "profiling", "benchmark", "bottleneck", "latency", "throughput",
                "algorithm complexity", "data structure"
            ])

        if any(word in query_lower for word in ["reliability", "robust", "fault tolerance"]):
            expanded_terms.extend([
                "error handling", "exception", "retry", "circuit breaker",
                "health check", "monitoring", "graceful shutdown", "backup",
                "recovery", "validation", "defensive programming"
            ])

        if any(word in query_lower for word in ["security", "secure", "protection"]):
            expanded_terms.extend([
                "authentication", "authorization", "validation", "sanitization",
                "encryption", "hashing", "token", "session", "access control",
                "input validation", "sql injection", "xss protection"
            ])

        # Add codebase indexing specific terms
        expanded_terms.extend([
            "tree-sitter", "ast parsing", "code chunking", "embedding generation",
            "vector search", "graph relationships", "neo4j", "qdrant",
            "semantic search", "code analysis", "mcp server", "fastapi"
        ])

        return list(set(expanded_terms))  # Remove duplicates

    def is_abstract_query(self, query: str) -> bool:
        """
        Determine if a query is abstract and needs expansion.

        Abstract queries typically:
        1. Don't contain specific technical terms
        2. Use high-level concepts like "scalability", "maintainability"
        3. Ask for assessments, evaluations, or analyses
        4. Don't mention specific code elements
        """
        query_lower = query.lower()

        # Check for abstract concept keywords
        abstract_indicators = [
            "assessment", "analysis", "evaluation", "review", "overview",
            "scalability", "maintainability", "reliability", "security",
            "performance", "optimization", "quality", "best practices",
            "recommendations", "improvements", "architecture", "design",
            "patterns", "principles", "strategy", "approach"
        ]

        # Check for specific technical terms (if present, query is less abstract)
        technical_terms = [
            "function", "class", "method", "variable", "import", "export",
            "api", "endpoint", "database", "query", "table", "index",
            "async", "await", "promise", "callback", "event", "listener"
        ]

        has_abstract = any(indicator in query_lower for indicator in abstract_indicators)
        has_technical = any(term in query_lower for term in technical_terms)

        # Query is abstract if it has abstract indicators but few technical terms
        return has_abstract and not has_technical
