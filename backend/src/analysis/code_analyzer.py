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
        graph_context: Dict[str, List[CodeChunk]]
    ) -> Dict[str, Any]:
        """
        Analyze query results and provide comprehensive explanations.
        
        Args:
            query: Original user query
            vector_results: Results from vector similarity search
            graph_context: Additional context from graph relationships
            
        Returns:
            Comprehensive analysis with explanations
        """
        try:
            # Combine and rank all relevant code chunks
            all_chunks = self._combine_and_rank_chunks(vector_results, graph_context)
            
            # Generate comprehensive analysis
            analysis = await self._generate_analysis(query, all_chunks)
            
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
        """Combine vector and graph results, ranking by relevance."""
        chunk_scores = {}
        
        # Add vector search results with their scores
        for chunk, score in vector_results:
            chunk_scores[chunk.id] = {
                "chunk": chunk,
                "vector_score": score,
                "graph_score": 0.0,
                "context_type": "primary"
            }
        
        # Add graph context with relationship-based scoring
        context_weights = {
            "parents": 0.8,
            "children": 0.7,
            "calls": 0.6,
            "called_by": 0.5
        }
        
        for context_type, chunks in graph_context.items():
            weight = context_weights.get(context_type, 0.3)
            for chunk in chunks:
                if chunk.id in chunk_scores:
                    chunk_scores[chunk.id]["graph_score"] += weight
                else:
                    chunk_scores[chunk.id] = {
                        "chunk": chunk,
                        "vector_score": 0.0,
                        "graph_score": weight,
                        "context_type": context_type
                    }
        
        # Calculate combined scores and sort
        for chunk_id, data in chunk_scores.items():
            data["combined_score"] = data["vector_score"] + data["graph_score"]
        
        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.values(), 
            key=lambda x: x["combined_score"], 
            reverse=True
        )
        
        return sorted_chunks
    
    async def _generate_analysis(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis using LLM."""
        if not self.client:
            return self._fallback_analysis_dict(query, ranked_chunks)
        
        try:
            # Prepare code context for LLM
            code_context = self._prepare_code_context(ranked_chunks[:8])  # Top 8 chunks
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(query, code_context)
            
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
    
    def _create_analysis_prompt(self, query: str, code_context: str) -> str:
        """Create analysis prompt for LLM."""
        return f"""
Analyze the following code in response to the user's query: "{query}"

{code_context}

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "summary": "Brief 2-3 sentence summary of what the code does in relation to the query",
    "detailed_explanation": "Detailed explanation of the code logic, flow, and purpose",
    "code_flow": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
    "key_components": [
        {{"name": "component_name", "purpose": "what it does", "location": "file:line"}},
        ...
    ],
    "relationships": [
        {{"from": "component1", "to": "component2", "relationship": "calls/inherits/uses"}},
        ...
    ],
    "recommendations": ["Suggestion 1", "Suggestion 2", ...]
}}

Focus on:
1. Explaining the code's purpose and logic clearly
2. Identifying key functions, classes, and their relationships
3. Describing the execution flow
4. Providing insights about the code architecture
5. Suggesting improvements or related code to explore
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for code analysis."""
        return """You are an expert code analyst and software engineer. Your job is to analyze code and provide clear, comprehensive explanations that help developers understand:

1. What the code does (purpose and functionality)
2. How it works (logic and flow)
3. How components relate to each other
4. Potential improvements or considerations

Always provide responses in valid JSON format. Use clear, professional language that's accessible to developers of all levels. Focus on practical insights and actionable information."""
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured analysis."""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: create structured response from text
                return {
                    "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                    "detailed_explanation": response_text,
                    "code_flow": [],
                    "key_components": [],
                    "relationships": [],
                    "recommendations": []
                }
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return {
                "summary": "Analysis completed but response format was invalid.",
                "detailed_explanation": response_text,
                "code_flow": [],
                "key_components": [],
                "relationships": [],
                "recommendations": []
            }
    
    def _fallback_analysis(self, query: str, vector_results: List[Tuple[CodeChunk, float]]) -> Dict[str, Any]:
        """Provide fallback analysis when LLM is not available."""
        chunks = [chunk for chunk, _ in vector_results[:5]]
        return self._fallback_analysis_dict(query, [{"chunk": chunk, "combined_score": score} for chunk, score in vector_results])
    
    def _fallback_analysis_dict(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate basic analysis without LLM."""
        chunks = [chunk_data["chunk"] for chunk_data in ranked_chunks[:5]]
        
        # Basic analysis
        file_paths = list(set(chunk.file_path for chunk in chunks))
        function_names = [chunk.name for chunk in chunks if chunk.name]
        node_types = list(set(chunk.node_type.value for chunk in chunks))
        
        return {
            "summary": f"Found {len(chunks)} relevant code chunks across {len(file_paths)} files related to '{query}'.",
            "detailed_explanation": f"The search identified code components including {', '.join(node_types)} that match your query. The most relevant results are from {', '.join(file_paths[:3])}.",
            "code_flow": [f"Step {i+1}: {chunk.name or 'Code block'} in {chunk.file_path}" for i, chunk in enumerate(chunks[:3])],
            "key_components": [
                {
                    "name": chunk.name or f"Code block at line {chunk.start_line}",
                    "purpose": f"{chunk.node_type.value} in {chunk.file_path}",
                    "location": f"{chunk.file_path}:{chunk.start_line}"
                } for chunk in chunks[:5]
            ],
            "relationships": [],
            "recommendations": [
                "Review the identified code chunks for detailed implementation",
                "Check related functions and classes for complete understanding",
                "Consider the file structure and dependencies"
            ]
        }
