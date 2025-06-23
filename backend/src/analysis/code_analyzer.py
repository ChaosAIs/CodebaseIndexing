"""Intelligent code analysis service that combines RAG search with LLM-powered explanations."""

import json
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import openai
from openai import OpenAI

from ..models import CodeChunk, QueryResult
from ..config import config


class CodeAnalyzer:
    """Intelligent code analyzer that provides comprehensive explanations."""
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.client = None
        if config.ai_models.openai_api_key:
            self.client = OpenAI(api_key=config.ai_models.openai_api_key)
    
    async def analyze_query_results(
        self,
        query: str,
        vector_results: List[Tuple[CodeChunk, float]],
        graph_context: Dict[str, List[CodeChunk]],
        project_context: Dict[str, Any] = None,
        comprehensive_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze query results and provide comprehensive explanations.

        Args:
            query: Original user query
            vector_results: Results from vector similarity search
            graph_context: Additional context from graph relationships
            project_context: Project-specific context (name, description, technologies)

        Returns:
            Comprehensive analysis with explanations
        """
        try:
            # Combine and rank all relevant code chunks
            all_chunks = self._combine_and_rank_chunks(vector_results, graph_context)
            
            # Generate comprehensive analysis with project and architectural context
            analysis = await self._generate_analysis(query, all_chunks, project_context, comprehensive_context)
            
            # Structure the response
            return {
                "summary": analysis.get("summary", ""),
                "detailed_explanation": analysis.get("detailed_explanation", ""),
                "code_flow": analysis.get("code_flow", []),
                "key_components": analysis.get("key_components", []),
                "relationships": analysis.get("relationships", []),
                "recommendations": analysis.get("recommendations", []),
                "relevant_chunks": all_chunks[:10],  # Top 10 most relevant
                "total_chunks_analyzed": len(all_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return self._fallback_analysis(query, vector_results)
    
    def _combine_and_rank_chunks(
        self,
        vector_results: List[Tuple[CodeChunk, float]],
        graph_context: Dict[str, List[CodeChunk]]
    ) -> List[Dict[str, Any]]:
        """Combine vector and graph results with enhanced relationship-based ranking."""
        chunk_scores = {}

        # Add vector search results with their scores
        for chunk, score in vector_results:
            chunk_scores[chunk.id] = {
                "chunk": chunk,
                "vector_score": score,
                "graph_score": 0.0,
                "context_type": "primary",
                "architectural_relevance": 0.0
            }

        # Enhanced graph context with more sophisticated scoring
        context_weights = {
            "parents": 0.9,           # Higher weight for hierarchical context
            "children": 0.8,          # Important for understanding implementation
            "calls": 0.7,             # Function call relationships
            "called_by": 0.6,         # Reverse call relationships
            "imports": 0.8,           # Import dependencies are crucial
            "imported_by": 0.7,       # Reverse import relationships
            "siblings": 0.5,          # Same-level components
            "architectural_context": 0.9  # High-level architectural context
        }

        for context_type, chunks in graph_context.items():
            weight = context_weights.get(context_type, 0.3)
            for chunk in chunks:
                if chunk.id in chunk_scores:
                    chunk_scores[chunk.id]["graph_score"] += weight
                    # Boost architectural relevance for certain types
                    if context_type in ["parents", "architectural_context", "imports"]:
                        chunk_scores[chunk.id]["architectural_relevance"] += 0.3
                else:
                    arch_relevance = 0.3 if context_type in ["parents", "architectural_context", "imports"] else 0.0
                    chunk_scores[chunk.id] = {
                        "chunk": chunk,
                        "vector_score": 0.0,
                        "graph_score": weight,
                        "context_type": context_type,
                        "architectural_relevance": arch_relevance
                    }

        # Calculate enhanced combined scores
        for chunk_id, data in chunk_scores.items():
            # Base score from vector + graph
            base_score = data["vector_score"] + data["graph_score"]

            # Architectural bonus for system understanding
            arch_bonus = data["architectural_relevance"] * 0.2

            # File-level importance bonus
            file_bonus = self._calculate_file_importance_bonus(data["chunk"])

            data["combined_score"] = base_score + arch_bonus + file_bonus

        # Sort by combined score with architectural preference
        sorted_chunks = sorted(
            chunk_scores.values(),
            key=lambda x: (x["combined_score"], x["architectural_relevance"]),
            reverse=True
        )

        return sorted_chunks

    def _calculate_file_importance_bonus(self, chunk: CodeChunk) -> float:
        """Calculate importance bonus based on file characteristics."""
        file_path = chunk.file_path.lower()
        bonus = 0.0

        # Core system files get higher priority
        if any(keyword in file_path for keyword in ['main', 'app', 'server', 'client', 'core']):
            bonus += 0.3

        # API and interface files are important for understanding
        if any(keyword in file_path for keyword in ['api', 'interface', 'endpoint', 'route']):
            bonus += 0.2

        # Configuration and setup files provide system context
        if any(keyword in file_path for keyword in ['config', 'setup', 'init']):
            bonus += 0.15

        # Model and schema files are crucial for data understanding
        if any(keyword in file_path for keyword in ['model', 'schema', 'entity']):
            bonus += 0.1

        return bonus
    
    async def _generate_analysis(self, query: str, ranked_chunks: List[Dict[str, Any]], project_context: Dict[str, Any] = None, comprehensive_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis using LLM with enhanced context."""
        if not self.client:
            return self._fallback_analysis_dict(query, ranked_chunks)

        try:
            # Prepare enhanced code context for LLM (increased from 8 to 12 chunks)
            code_context = self._prepare_code_context(ranked_chunks[:12])

            # Create analysis prompt with comprehensive context
            prompt = self._create_analysis_prompt(query, code_context, project_context, comprehensive_context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cost-effective model
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return self._fallback_analysis_dict(query, ranked_chunks)
    
    def _prepare_code_context(self, ranked_chunks: List[Dict[str, Any]]) -> str:
        """Prepare code context for LLM analysis."""
        context_parts = []
        
        for i, chunk_data in enumerate(ranked_chunks):
            chunk = chunk_data["chunk"]
            score = chunk_data["combined_score"]
            context_type = chunk_data["context_type"]
            
            context_parts.append(f"""
## Code Chunk {i+1} (Score: {score:.3f}, Type: {context_type})
**File:** {chunk.file_path}
**Function/Class:** {chunk.name or 'unnamed'}
**Type:** {chunk.node_type.value}
**Lines:** {chunk.start_line}-{chunk.end_line}

```python
{chunk.content}
```
""")
        
        return "\n".join(context_parts)
    
    def _create_analysis_prompt(self, query: str, code_context: str, project_context: Dict[str, Any] = None, comprehensive_context: Dict[str, Any] = None) -> str:
        """Create enhanced analysis prompt with comprehensive architectural context."""

        # Build project-specific context
        project_info = ""
        if project_context:
            project_name = project_context.get("name", "Unknown Project")
            project_description = project_context.get("description", "")
            technologies = project_context.get("technologies", [])

            project_info = f"""
PROJECT CONTEXT:
- Project Name: {project_name}
- Description: {project_description}
- Technologies: {', '.join(technologies) if technologies else 'Not specified'}
"""

        # Build comprehensive architectural context
        architectural_info = ""
        if comprehensive_context:
            arch_patterns = comprehensive_context.get("architectural_patterns", [])
            data_flows = comprehensive_context.get("data_flow_patterns", [])
            dependencies = comprehensive_context.get("dependency_patterns", {})
            usage_patterns = comprehensive_context.get("usage_patterns", [])
            system_overview = comprehensive_context.get("system_overview", {})

            if arch_patterns:
                layers = [p.get("layer", "Unknown") for p in arch_patterns[:5]]
                architectural_info += f"\nARCHITECTURAL LAYERS: {', '.join(set(layers))}"

            if data_flows:
                flow_count = len(data_flows)
                architectural_info += f"\nDATA FLOWS: {flow_count} component interactions identified"

            if dependencies.get("file_dependencies"):
                dep_count = len(dependencies["file_dependencies"])
                architectural_info += f"\nDEPENDENCIES: {dep_count} file-level dependencies"

            if usage_patterns:
                high_usage = [p for p in usage_patterns if "High Usage" in p.get("usage_category", "")]
                if high_usage:
                    core_components = [p.get("component", "Unknown") for p in high_usage[:3]]
                    architectural_info += f"\nCORE COMPONENTS: {', '.join(core_components)}"

            if system_overview:
                complexity = system_overview.get("system_complexity", {})
                if complexity:
                    level = complexity.get("complexity_level", "Unknown")
                    architectural_info += f"\nSYSTEM COMPLEXITY: {level}"

        return f"""
SYSTEM CONTEXT: You are analyzing a sophisticated codebase indexing and knowledge retrieval system with full architectural understanding. This system:
- Parses codebases using Tree-sitter to extract ASTs
- Chunks code while preserving hierarchical relationships
- Generates embeddings using OpenAI/HuggingFace/Ollama models
- Stores embeddings in Qdrant vector database for semantic search
- Uses Neo4j graph database to model code relationships
- Provides a FastAPI MCP server for LLM integration
- Has a React frontend with chat interface for codebase queries
{project_info}
{architectural_info}

USER QUERY: "{query}"

RELEVANT CODE WITH ARCHITECTURAL CONTEXT:
{code_context}

ENHANCED INSTRUCTIONS:
You have access to comprehensive architectural context including system layers, data flows, dependencies, and usage patterns. Use this information to provide deep, contextual understanding.

Provide a natural, conversational response that:
1. Leverages the architectural context to explain how components fit into the broader system
2. Addresses the user's question with full system understanding (top-to-ground perspective)
3. Explains relationships between components and their roles in the indexing pipeline
4. Provides insights about system design patterns and architectural decisions
5. Uses the dependency and usage information to explain component importance and interactions

Focus on:
- **System Architecture**: How components work together across different layers
- **Data Flow**: How information moves through the indexing pipeline
- **Component Relationships**: Dependencies, usage patterns, and interactions
- **Design Patterns**: Architectural decisions and their implications
- **System Context**: How the queried components fit into the overall solution

Respond as a senior architect explaining the system design and implementation details with full contextual understanding.
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for code analysis."""
        return """You are an expert code analyst specializing in codebase indexing and knowledge retrieval systems. You are analyzing a sophisticated codebase indexing solution that combines:

- Tree-sitter parsing for AST extraction
- Vector embeddings with Qdrant for semantic search
- Neo4j graph database for relationship modeling
- FastAPI MCP (Model Context Protocol) server
- React frontend with chat interface
- Multi-project support with intelligent search

Your role is to provide natural, conversational analysis that helps developers understand this specific codebase indexing system. When analyzing code:

1. **Be Conversational**: Respond naturally as if explaining to a colleague, not in rigid structured formats
2. **Answer Directly**: Address the specific question being asked without forcing unnecessary structure
3. **Focus on the Indexing Domain**: Always relate your analysis to the codebase indexing and search pipeline
4. **Be Technical but Clear**: Use appropriate technical language for developers working on code analysis systems
5. **Provide Context**: Explain how components fit into the parsing → chunking → embedding → storage → search pipeline

Respond naturally and conversationally. Do not use rigid sections like "Analysis Summary" or "How It Works" unless specifically requested. Focus on answering the user's actual question about this codebase indexing system."""
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis."""
        # For natural language responses, we'll use the response as the main explanation
        # and extract key information dynamically

        # Clean up the response
        cleaned_response = response_text.strip()

        # Try to extract a brief summary (first paragraph or first 2 sentences)
        lines = cleaned_response.split('\n')
        first_paragraph = lines[0] if lines else cleaned_response

        # Extract summary (first 2 sentences or first 200 chars)
        sentences = first_paragraph.split('. ')
        if len(sentences) >= 2:
            summary = '. '.join(sentences[:2]) + '.'
        else:
            summary = first_paragraph[:200] + "..." if len(first_paragraph) > 200 else first_paragraph

        # Extract key components by looking for technical terms
        key_components = self._extract_key_components_from_text(cleaned_response)

        # Extract relationships by looking for connection words
        relationships = self._extract_relationships_from_text(cleaned_response)

        # Extract recommendations by looking for suggestion patterns
        recommendations = self._extract_recommendations_from_text(cleaned_response)

        return {
            "summary": summary,
            "detailed_explanation": cleaned_response,
            "code_flow": [],  # Will be populated by extracting step-by-step information
            "key_components": key_components,
            "relationships": relationships,
            "recommendations": recommendations
        }

    def _extract_key_components_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract key components mentioned in the text."""
        components = []

        # Look for common patterns that indicate components
        import re

        # Pattern for class/function mentions
        class_pattern = r'\b([A-Z][a-zA-Z0-9_]*(?:Client|Server|Manager|Processor|Generator|Analyzer))\b'
        function_pattern = r'\b([a-z_][a-z0-9_]*(?:_[a-z0-9_]+)*)\(\)'

        classes = re.findall(class_pattern, text)
        functions = re.findall(function_pattern, text)

        for class_name in set(classes):
            components.append({
                "name": class_name,
                "purpose": f"Core component in the codebase indexing system",
                "location": "system component"
            })

        for func_name in set(functions):
            components.append({
                "name": func_name,
                "purpose": f"Function in the indexing pipeline",
                "location": "code function"
            })

        return components[:5]  # Limit to top 5

    def _extract_relationships_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships mentioned in the text."""
        relationships = []

        # Look for relationship indicators
        relationship_patterns = [
            (r'(\w+)\s+(?:calls|invokes|uses)\s+(\w+)', 'calls'),
            (r'(\w+)\s+(?:inherits from|extends)\s+(\w+)', 'inherits'),
            (r'(\w+)\s+(?:connects to|communicates with)\s+(\w+)', 'communicates_with'),
            (r'(\w+)\s+(?:depends on|relies on)\s+(\w+)', 'depends_on')
        ]

        import re
        for pattern, rel_type in relationship_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for from_comp, to_comp in matches:
                relationships.append({
                    "from": from_comp,
                    "to": to_comp,
                    "relationship": rel_type,
                    "context": "Identified from code analysis"
                })

        return relationships[:3]  # Limit to top 3

    def _extract_recommendations_from_text(self, text: str) -> List[str]:
        """Extract recommendations from the text."""
        recommendations = []

        # Look for recommendation patterns
        import re

        # Patterns that indicate recommendations
        rec_patterns = [
            r'(?:consider|recommend|suggest|should|could|might want to)\s+([^.]+)',
            r'(?:to improve|for better|to enhance)\s+([^.]+)',
            r'(?:optimization|improvement):\s*([^.]+)'
        ]

        for pattern in rec_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 10:  # Only meaningful recommendations
                    recommendations.append(match.strip())

        return recommendations[:3]  # Limit to top 3
    
    def _fallback_analysis(self, query: str, vector_results: List[Tuple[CodeChunk, float]]) -> Dict[str, Any]:
        """Provide fallback analysis when LLM is not available."""
        chunks = [chunk for chunk, _ in vector_results[:5]]
        return self._fallback_analysis_dict(query, [{"chunk": chunk, "combined_score": score} for chunk, score in vector_results])
    
    def _fallback_analysis_dict(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate contextually relevant analysis without LLM."""
        chunks = [chunk_data["chunk"] for chunk_data in ranked_chunks[:5]]

        # Analyze components in context of codebase indexing system
        file_paths = list(set(chunk.file_path for chunk in chunks))
        node_types = list(set(chunk.node_type.value for chunk in chunks))

        # Categorize components by their role in the indexing system
        system_components = self._categorize_system_components(chunks)

        # Generate contextually aware summary
        summary = self._generate_contextual_summary(query, chunks, system_components)

        return {
            "summary": summary,
            "detailed_explanation": self._generate_detailed_explanation(query, chunks, system_components, file_paths),
            "code_flow": self._generate_code_flow(chunks, system_components),
            "key_components": self._discover_key_components(chunks, system_components),
            "relationships": self._infer_relationships(chunks),
            "recommendations": self._generate_contextual_recommendations(query, system_components)
        }

    def _categorize_system_components(self, chunks: List[CodeChunk]) -> Dict[str, List[CodeChunk]]:
        """Dynamically categorize code chunks by analyzing their actual content and structure."""
        # Start with discovered categories instead of predefined ones
        categories = {}

        # Analyze each chunk to understand its role
        for chunk in chunks:
            category = self._discover_component_category(chunk)
            if category not in categories:
                categories[category] = []
            categories[category].append(chunk)

        return categories

    def _discover_component_category(self, chunk: CodeChunk) -> str:
        """
        Intelligently discover what category a component belongs to based on:
        1. File path patterns
        2. Code content analysis
        3. Function/class names
        4. Import statements
        5. Code patterns
        """
        file_path = chunk.file_path.lower()
        content = chunk.content.lower()
        name = (chunk.name or "").lower()

        # Analyze imports to understand dependencies
        imports = self._extract_imports_from_content(chunk.content)

        # File path analysis - more flexible pattern matching
        path_indicators = self._analyze_file_path_patterns(file_path)

        # Content analysis - look for specific patterns
        content_indicators = self._analyze_content_patterns(content, name)

        # Import analysis - understand external dependencies
        import_indicators = self._analyze_import_patterns(imports)

        # Combine all indicators to determine category
        all_indicators = {**path_indicators, **content_indicators, **import_indicators}

        # Find the category with highest confidence
        if all_indicators:
            best_category = max(all_indicators, key=all_indicators.get)
            confidence = all_indicators[best_category]

            # Only return if confidence is reasonable
            if confidence >= 0.3:
                return best_category

        # Fallback to generic categorization
        return self._generic_categorization(file_path, content, name)

    def _analyze_file_path_patterns(self, file_path: str) -> Dict[str, float]:
        """Analyze file path to determine component category."""
        indicators = {}

        # Common architectural patterns
        patterns = {
            "data_access": ["repository", "dao", "database", "db", "storage", "store"],
            "business_logic": ["service", "business", "logic", "domain", "core"],
            "api_layer": ["api", "controller", "endpoint", "route", "handler"],
            "ui_layer": ["ui", "frontend", "component", "view", "page", "interface"],
            "infrastructure": ["config", "util", "helper", "common", "shared"],
            "data_processing": ["processor", "parser", "transformer", "converter"],
            "communication": ["client", "server", "gateway", "proxy"],
            "security": ["auth", "security", "permission", "access"],
            "monitoring": ["log", "monitor", "metric", "health", "diagnostic"]
        }

        for category, keywords in patterns.items():
            score = sum(1.0 for keyword in keywords if keyword in file_path)
            if score > 0:
                indicators[category] = score / len(keywords)  # Normalize

        return indicators

    def _analyze_content_patterns(self, content: str, name: str) -> Dict[str, float]:
        """Analyze code content to determine component category."""
        indicators = {}

        # Look for specific code patterns
        patterns = {
            "data_access": [
                "select", "insert", "update", "delete", "query", "connection",
                "database", "table", "collection", "repository"
            ],
            "api_layer": [
                "request", "response", "endpoint", "route", "handler",
                "get", "post", "put", "delete", "api"
            ],
            "business_logic": [
                "calculate", "process", "validate", "transform", "business",
                "rule", "logic", "algorithm"
            ],
            "ui_layer": [
                "render", "component", "props", "state", "event",
                "click", "submit", "form", "button"
            ],
            "data_processing": [
                "parse", "transform", "convert", "process", "extract",
                "chunk", "embed", "index"
            ],
            "communication": [
                "client", "server", "request", "response", "http",
                "websocket", "grpc", "rest"
            ],
            "security": [
                "authenticate", "authorize", "permission", "token",
                "encrypt", "decrypt", "hash", "validate"
            ]
        }

        for category, keywords in patterns.items():
            score = sum(0.5 for keyword in keywords if keyword in content)
            score += sum(1.0 for keyword in keywords if keyword in name)
            if score > 0:
                indicators[category] = min(score / len(keywords), 1.0)  # Normalize and cap

        return indicators

    def _analyze_import_patterns(self, imports: List[str]) -> Dict[str, float]:
        """Analyze import statements to determine component category."""
        indicators = {}

        import_patterns = {
            "data_access": ["sqlite", "postgres", "mysql", "mongodb", "redis", "qdrant", "neo4j"],
            "api_layer": ["fastapi", "flask", "django", "express", "axios"],
            "ui_layer": ["react", "vue", "angular", "svelte", "jquery"],
            "data_processing": ["pandas", "numpy", "sklearn", "torch", "transformers"],
            "communication": ["requests", "httpx", "aiohttp", "websockets"],
            "security": ["jwt", "bcrypt", "cryptography", "oauth"],
            "infrastructure": ["logging", "config", "os", "sys", "pathlib"]
        }

        for category, packages in import_patterns.items():
            score = sum(1.0 for imp in imports for pkg in packages if pkg in imp.lower())
            if score > 0:
                indicators[category] = min(score / len(packages), 1.0)

        return indicators

    def _extract_imports_from_content(self, content: str) -> List[str]:
        """Extract import statements from code content."""
        import re
        imports = []

        # Python imports
        python_imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', content)
        for match in python_imports:
            imports.extend([imp for imp in match if imp])

        # JavaScript/TypeScript imports
        js_imports = re.findall(r'import.*?from\s+[\'"]([^\'"]+)[\'"]', content)
        imports.extend(js_imports)

        return imports

    def _generic_categorization(self, file_path: str, content: str, name: str) -> str:
        """Fallback categorization when specific patterns aren't found."""
        # Use file extension and basic patterns
        if any(ext in file_path for ext in [".py", ".js", ".ts"]):
            if "test" in file_path or "spec" in file_path:
                return "testing"
            elif "config" in file_path or "setting" in file_path:
                return "configuration"
            elif "util" in file_path or "helper" in file_path:
                return "utilities"
            elif "model" in file_path or "schema" in file_path:
                return "data_models"
            else:
                return "application_logic"

        return "miscellaneous"

    def _generate_contextual_summary(self, query: str, chunks: List[CodeChunk], system_components: Dict[str, List[CodeChunk]]) -> str:
        """Generate a natural, question-specific response."""
        primary_categories = list(system_components.keys())[:2]  # Top 2 categories

        # Generate natural responses based on the question type
        if "architecture" in query.lower() or "overview" in query.lower():
            response = f"This codebase indexing system has a multi-stage architecture with {len(chunks)} key components. "
            if primary_categories:
                response += f"The main subsystems I found are focused on {' and '.join(primary_categories)}, "
                response += "which work together in the indexing pipeline: source code gets parsed using Tree-sitter, "
                response += "chunked while preserving relationships, embedded using AI models, stored in Qdrant for vector search "
                response += "and Neo4j for graph relationships, then served through a FastAPI MCP server to the React frontend."
            return response

        elif "how" in query.lower() and any(word in query.lower() for word in ["work", "works", "working"]):
            if any(cat in primary_categories for cat in ["parsing", "chunking"]):
                return f"The code processing works by first parsing source code with Tree-sitter to extract AST nodes, then chunking the code while preserving hierarchical relationships. I found {len(chunks)} components handling this part of the pipeline."
            elif any(cat in primary_categories for cat in ["embedding", "storage"]):
                return f"The embedding and storage system works by generating vector representations of code chunks using AI models (OpenAI/HuggingFace/Ollama), then storing them in Qdrant for semantic search and Neo4j for relationship modeling. The {len(chunks)} components I found handle this critical part of the indexing pipeline."
            elif any(cat in primary_categories for cat in ["search", "api"]):
                return f"The search system works by processing user queries, generating embeddings, performing similarity search in Qdrant, retrieving related chunks from Neo4j, and serving results through the FastAPI MCP server. I found {len(chunks)} components managing this query processing pipeline."

        elif "what" in query.lower():
            if any(cat in primary_categories for cat in ["api", "server"]):
                return f"This is the MCP (Model Context Protocol) server component of the codebase indexing system. It provides API endpoints for LLM integration and serves as the bridge between the frontend and the indexing/search backend. The {len(chunks)} components handle query processing, project management, and system coordination."
            elif any(cat in primary_categories for cat in ["frontend"]):
                return f"This is the React-based frontend interface for the codebase indexing system. It provides a chat interface where users can query their indexed codebases using natural language. The {len(chunks)} components handle user interactions, project selection, and result visualization."

        # Default natural response
        return f"I found {len(chunks)} relevant components in this codebase indexing system. These components are part of the {', '.join(primary_categories)} subsystems that work together to parse, index, and search through codebases using semantic search and graph relationships."

    def _generate_detailed_explanation(self, query: str, chunks: List[CodeChunk], system_components: Dict[str, List[CodeChunk]], file_paths: List[str]) -> str:
        """Generate detailed explanation based on system context."""
        explanations = []

        for category, category_chunks in system_components.items():
            if category == "parsing":
                explanations.append(f"The parsing components ({len(category_chunks)} found) handle Tree-sitter AST extraction and code structure analysis.")
            elif category == "chunking":
                explanations.append(f"The chunking components ({len(category_chunks)} found) process parsed code into meaningful chunks while preserving relationships.")
            elif category == "embedding":
                explanations.append(f"The embedding components ({len(category_chunks)} found) generate vector representations using AI models for semantic search.")
            elif category == "storage":
                explanations.append(f"The storage components ({len(category_chunks)} found) manage Qdrant vector database and Neo4j graph database operations.")
            elif category == "search":
                explanations.append(f"The search components ({len(category_chunks)} found) handle query processing and similarity search across indexed code.")
            elif category == "api":
                explanations.append(f"The API components ({len(category_chunks)} found) provide MCP server endpoints for LLM integration and frontend communication.")
            elif category == "frontend":
                explanations.append(f"The frontend components ({len(category_chunks)} found) implement the React-based chat interface and visualization features.")

        return " ".join(explanations) if explanations else f"The identified components span {len(file_paths)} files in the codebase indexing system."

    def _generate_code_flow(self, chunks: List[CodeChunk], system_components: Dict[str, List[CodeChunk]]) -> List[str]:
        """Dynamically generate code flow based on discovered system architecture."""
        flow_steps = []

        # Analyze the actual components to understand the flow
        component_flow = self._discover_system_flow(system_components, chunks)

        if component_flow:
            return component_flow

        # Fallback: generate flow from individual chunks
        return self._generate_chunk_based_flow(chunks)

    def _discover_system_flow(self, system_components: Dict[str, List[CodeChunk]], chunks: List[CodeChunk]) -> List[str]:
        """Discover the actual system flow by analyzing component relationships."""
        flow_steps = []

        # Common architectural flow patterns
        flow_patterns = {
            "data_input": ["Input/Parse", "data_processing", "data_access"],
            "data_processing": ["Process/Transform", "business_logic", "data_processing"],
            "data_storage": ["Store/Persist", "data_access", "infrastructure"],
            "api_handling": ["Handle Requests", "api_layer", "communication"],
            "business_execution": ["Execute Logic", "business_logic", "application_logic"],
            "data_output": ["Output/Response", "api_layer", "ui_layer"]
        }

        # Find which patterns exist in the system
        existing_patterns = []
        for pattern_name, (description, *categories) in flow_patterns.items():
            if any(cat in system_components for cat in categories):
                existing_patterns.append((pattern_name, description))

        # Build flow based on discovered patterns
        if existing_patterns:
            # Sort by logical flow order
            flow_order = ["data_input", "data_processing", "business_execution", "data_storage", "api_handling", "data_output"]
            sorted_patterns = sorted(existing_patterns, key=lambda x: flow_order.index(x[0]) if x[0] in flow_order else 999)

            for _, description in sorted_patterns:
                flow_steps.append(description)

        return flow_steps[:5]  # Limit to 5 steps

    def _generate_chunk_based_flow(self, chunks: List[CodeChunk]) -> List[str]:
        """Generate flow based on individual chunk analysis."""
        flow_steps = []

        for chunk in chunks[:3]:
            if chunk.name:
                # Analyze function/class name to understand its role
                name_lower = chunk.name.lower()

                if any(word in name_lower for word in ["parse", "extract", "read"]):
                    flow_steps.append(f"Parse/Extract data using {chunk.name}")
                elif any(word in name_lower for word in ["process", "transform", "convert"]):
                    flow_steps.append(f"Process/Transform data with {chunk.name}")
                elif any(word in name_lower for word in ["store", "save", "persist"]):
                    flow_steps.append(f"Store/Persist data via {chunk.name}")
                elif any(word in name_lower for word in ["search", "find", "query"]):
                    flow_steps.append(f"Search/Query data using {chunk.name}")
                elif any(word in name_lower for word in ["handle", "manage", "control"]):
                    flow_steps.append(f"Handle/Manage operations with {chunk.name}")
                else:
                    flow_steps.append(f"Execute {chunk.name} functionality")
            else:
                # Analyze content for clues
                content_lower = chunk.content.lower()
                if "def " in content_lower or "function" in content_lower:
                    flow_steps.append(f"Execute function at {chunk.file_path}:{chunk.start_line}")
                elif "class " in content_lower:
                    flow_steps.append(f"Instantiate class at {chunk.file_path}:{chunk.start_line}")
                else:
                    flow_steps.append(f"Process code block at {chunk.file_path}:{chunk.start_line}")

        return flow_steps

    def _discover_key_components(self, chunks: List[CodeChunk], system_components: Dict[str, List[CodeChunk]]) -> List[Dict[str, str]]:
        """Dynamically discover key components based on actual code analysis."""
        key_components = []

        # Analyze chunks to find the most important ones
        scored_chunks = []
        for chunk in chunks[:10]:  # Analyze top 10 chunks
            score = self._calculate_component_importance(chunk, system_components)
            scored_chunks.append((chunk, score))

        # Sort by importance score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Build key components list
        for chunk, score in scored_chunks[:5]:  # Top 5 components
            component = {
                "name": self._get_component_display_name(chunk),
                "purpose": self._discover_component_purpose(chunk, system_components),
                "location": f"{chunk.file_path}:{chunk.start_line}",
                "importance_score": round(score, 2)
            }
            key_components.append(component)

        return key_components

    def _calculate_component_importance(self, chunk: CodeChunk, system_components: Dict[str, List[CodeChunk]]) -> float:
        """Calculate importance score for a component."""
        score = 0.0

        # Base score from chunk type
        type_scores = {
            "class": 1.0,
            "function": 0.8,
            "method": 0.7,
            "module": 0.9,
            "interface": 0.8
        }
        score += type_scores.get(chunk.node_type.value, 0.5)

        # Boost for main/entry point files
        file_path = chunk.file_path.lower()
        if any(name in file_path for name in ["main", "app", "index", "server", "__init__"]):
            score += 0.5

        # Boost for core functionality indicators
        name = (chunk.name or "").lower()
        content = chunk.content.lower()

        core_indicators = ["manager", "processor", "handler", "controller", "service", "client", "server"]
        if any(indicator in name for indicator in core_indicators):
            score += 0.3

        # Boost for components that appear in multiple categories
        component_categories = []
        for category, category_chunks in system_components.items():
            if chunk in category_chunks:
                component_categories.append(category)

        if len(component_categories) > 1:
            score += 0.2 * len(component_categories)

        # Boost for components with many relationships (calls, imports)
        if hasattr(chunk, 'calls') and chunk.calls:
            score += min(len(chunk.calls) * 0.1, 0.3)

        if hasattr(chunk, 'called_by') and chunk.called_by:
            score += min(len(chunk.called_by) * 0.1, 0.3)

        # Boost for larger, more complex components
        lines_of_code = chunk.end_line - chunk.start_line + 1
        if lines_of_code > 50:
            score += 0.2
        elif lines_of_code > 20:
            score += 0.1

        return score

    def _get_component_display_name(self, chunk: CodeChunk) -> str:
        """Get a meaningful display name for the component."""
        if chunk.name:
            return chunk.name

        # Extract meaningful name from file path
        file_name = chunk.file_path.split('/')[-1].split('\\')[-1]
        if file_name.endswith('.py'):
            file_name = file_name[:-3]
        elif file_name.endswith('.js') or file_name.endswith('.ts'):
            file_name = file_name[:-3]

        # If it's a class or function, try to extract from content
        content_lines = chunk.content.split('\n')
        for line in content_lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line.startswith('class '):
                class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                return class_name
            elif line.startswith('def '):
                func_name = line.split('def ')[1].split('(')[0].strip()
                return func_name
            elif line.startswith('function '):
                func_name = line.split('function ')[1].split('(')[0].strip()
                return func_name

        return f"{file_name} (line {chunk.start_line})"

    def _discover_component_purpose(self, chunk: CodeChunk, system_components: Dict[str, List[CodeChunk]]) -> str:
        """Dynamically discover the purpose of a component."""
        # Find which category this component belongs to
        component_category = None
        for category, category_chunks in system_components.items():
            if chunk in category_chunks:
                component_category = category
                break

        # Analyze the component's content and context
        name = (chunk.name or "").lower()
        content = chunk.content.lower()
        file_path = chunk.file_path.lower()

        # Generate purpose based on analysis
        if component_category:
            purpose = self._generate_category_based_purpose(component_category, name, content, file_path)
        else:
            purpose = self._generate_content_based_purpose(name, content, file_path)

        return purpose

    def _generate_category_based_purpose(self, category: str, name: str, content: str, file_path: str) -> str:
        """Generate purpose based on discovered category."""
        category_purposes = {
            "data_access": "Manages data storage and retrieval operations",
            "api_layer": "Handles API requests and responses",
            "business_logic": "Implements core business rules and logic",
            "ui_layer": "Manages user interface and interactions",
            "data_processing": "Processes and transforms data",
            "communication": "Handles inter-service communication",
            "security": "Manages authentication and authorization",
            "infrastructure": "Provides system infrastructure and utilities",
            "monitoring": "Handles logging, monitoring, and diagnostics"
        }

        base_purpose = category_purposes.get(category, f"Handles {category.replace('_', ' ')} functionality")

        # Add specific details based on name and content
        if "manager" in name:
            return f"{base_purpose} through management operations"
        elif "processor" in name:
            return f"{base_purpose} through data processing"
        elif "client" in name:
            return f"{base_purpose} through client-side operations"
        elif "server" in name:
            return f"{base_purpose} through server-side operations"

        return base_purpose

    def _generate_content_based_purpose(self, name: str, content: str, file_path: str) -> str:
        """Generate purpose based on content analysis when category is unknown."""
        # Analyze content patterns
        if any(pattern in content for pattern in ["def __init__", "class ", "constructor"]):
            return "Defines core data structures and initialization logic"
        elif any(pattern in content for pattern in ["async def", "await ", "asyncio"]):
            return "Handles asynchronous operations and concurrent processing"
        elif any(pattern in content for pattern in ["request", "response", "endpoint"]):
            return "Manages HTTP requests and API interactions"
        elif any(pattern in content for pattern in ["database", "query", "select", "insert"]):
            return "Handles database operations and data persistence"
        elif any(pattern in content for pattern in ["parse", "extract", "transform"]):
            return "Processes and transforms data structures"
        elif any(pattern in content for pattern in ["validate", "check", "verify"]):
            return "Performs data validation and verification"
        elif any(pattern in content for pattern in ["log", "error", "exception"]):
            return "Handles error management and logging"
        else:
            # Fallback based on file location
            if "test" in file_path:
                return "Provides testing and validation functionality"
            elif "config" in file_path:
                return "Manages system configuration and settings"
            elif "util" in file_path or "helper" in file_path:
                return "Provides utility functions and helper methods"
            else:
                return f"Implements {name or 'core'} functionality"

    def _determine_component_purpose(self, chunk: CodeChunk, system_components: Dict[str, List[CodeChunk]]) -> str:
        """Determine the purpose of a component in the indexing system."""
        for category, category_chunks in system_components.items():
            if chunk in category_chunks:
                purposes = {
                    "parsing": "Parses source code and extracts AST structures",
                    "chunking": "Processes code into searchable chunks with metadata",
                    "embedding": "Generates vector embeddings for semantic search",
                    "storage": "Manages database operations for vectors and graphs",
                    "search": "Handles query processing and result retrieval",
                    "api": "Provides API endpoints for system integration",
                    "frontend": "Implements user interface and visualization",
                    "analysis": "Provides intelligent code analysis and explanations"
                }
                return purposes.get(category, f"{chunk.node_type.value} in {chunk.file_path}")

        return f"{chunk.node_type.value} component in the indexing system"

    def _infer_relationships(self, chunks: List[CodeChunk]) -> List[Dict[str, str]]:
        """Infer relationships between components."""
        relationships = []

        # Simple relationship inference based on file structure and naming
        for i, chunk1 in enumerate(chunks[:3]):
            for chunk2 in chunks[i+1:4]:
                if chunk1.file_path == chunk2.file_path:
                    relationships.append({
                        "from": chunk1.name or f"line_{chunk1.start_line}",
                        "to": chunk2.name or f"line_{chunk2.start_line}",
                        "relationship": "co-located",
                        "context": "Components in the same file likely work together"
                    })
                elif "client" in chunk1.file_path and "server" in chunk2.file_path:
                    relationships.append({
                        "from": chunk1.name or "client_component",
                        "to": chunk2.name or "server_component",
                        "relationship": "communicates_with",
                        "context": "Client-server communication in the indexing system"
                    })

        return relationships[:3]  # Limit to top 3 relationships

    def _generate_contextual_recommendations(self, query: str, system_components: Dict[str, List[CodeChunk]]) -> List[str]:
        """Generate contextually relevant recommendations."""
        recommendations = []

        if "architecture" in query.lower():
            recommendations.extend([
                "Review the complete indexing pipeline: parsing → chunking → embedding → storage → search",
                "Examine the integration between Qdrant vector search and Neo4j graph relationships",
                "Consider the scalability implications of the current architecture"
            ])
        elif "performance" in query.lower():
            recommendations.extend([
                "Analyze embedding generation bottlenecks for large codebases",
                "Consider batch processing optimizations for chunk storage",
                "Review query response times and caching strategies"
            ])
        else:
            recommendations.extend([
                "Explore related components in the indexing pipeline",
                "Review the integration points with other system modules",
                "Consider the impact on search accuracy and performance"
            ])

        return recommendations[:3]  # Limit to top 3 recommendations
