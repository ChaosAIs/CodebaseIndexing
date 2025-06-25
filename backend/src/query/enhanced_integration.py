"""
Integration module for Enhanced Query Processing System.

This module provides integration between the enhanced query processor
and the existing MCP server, allowing seamless use of the sophisticated
Embedding + Graph RAG pipeline.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from ..models import CodeChunk, QueryResult
from ..database.qdrant_client import QdrantVectorStore
from ..database.neo4j_client import Neo4jGraphStore
from ..embeddings.embedding_generator import EmbeddingGenerator
from .enhanced_query_processor import EnhancedQueryProcessor, CombinedRAGResult
from .advanced_ner import ExtractedEntity


class EnhancedQueryIntegration:
    """
    Integration layer for enhanced query processing.
    
    This class provides a bridge between the existing MCP server
    and the new enhanced query processing system.
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        graph_store: Neo4jGraphStore,
        embedding_generator: EmbeddingGenerator,
        openai_client=None
    ):
        """Initialize the enhanced query integration."""
        self.enhanced_processor = EnhancedQueryProcessor(
            vector_store=vector_store,
            graph_store=graph_store,
            embedding_generator=embedding_generator,
            openai_client=openai_client
        )
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "enhanced_queries": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0
        }
    
    async def process_query_enhanced(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        limit: int = 10,
        embedding_model: str = "openai",
        use_enhanced: bool = True
    ) -> Tuple[List[Tuple[CodeChunk, float]], Dict[str, Any]]:
        """
        Process query using enhanced pipeline or fallback to basic search.
        
        Args:
            query: User's natural language query
            project_ids: Optional project filtering
            limit: Maximum number of results
            embedding_model: Model to use for embeddings
            use_enhanced: Whether to use enhanced processing
            
        Returns:
            Tuple of (search_results, metadata)
        """
        start_time = time.time()
        self.performance_stats["total_queries"] += 1
        
        try:
            if use_enhanced:
                # Use enhanced query processing
                logger.info(f"Processing query with enhanced pipeline: {query}")
                
                combined_result = await self.enhanced_processor.process_query_enhanced(
                    query=query,
                    project_ids=project_ids,
                    limit=limit,
                    embedding_model=embedding_model
                )
                
                # Convert enhanced results to standard format
                search_results = self._convert_enhanced_results(combined_result)
                
                # Create enhanced metadata
                metadata = self._create_enhanced_metadata(combined_result, start_time)
                
                self.performance_stats["enhanced_queries"] += 1
                self._update_performance_stats(combined_result.confidence_score, time.time() - start_time)
                
                return search_results, metadata
            
            else:
                # Fallback to basic embedding search
                logger.info(f"Processing query with basic pipeline: {query}")
                return await self._basic_search_fallback(query, project_ids, limit, embedding_model)
        
        except Exception as e:
            logger.error(f"Error in enhanced query processing: {e}")
            # Fallback to basic search on error
            return await self._basic_search_fallback(query, project_ids, limit, embedding_model)
    
    def _convert_enhanced_results(self, combined_result: CombinedRAGResult) -> List[Tuple[CodeChunk, float]]:
        """Convert enhanced results to standard search result format."""
        search_results = []
        
        for result_dict in combined_result.ranked_chunks:
            chunk = result_dict["chunk"]
            score = result_dict["combined_score"]
            search_results.append((chunk, score))
        
        return search_results
    
    def _create_enhanced_metadata(self, combined_result: CombinedRAGResult, start_time: float) -> Dict[str, Any]:
        """Create enhanced metadata from combined results."""
        return {
            "processing_type": "enhanced",
            "confidence_score": combined_result.confidence_score,
            "total_results": len(combined_result.ranked_chunks),
            "embedding_insights": combined_result.embedding_insights,
            "graph_insights": combined_result.graph_insights,
            "combined_context": combined_result.combined_context,
            "processing_time": time.time() - start_time,
            "performance_stats": self.performance_stats.copy()
        }
    
    async def _basic_search_fallback(
        self,
        query: str,
        project_ids: Optional[List[str]],
        limit: int,
        embedding_model: str
    ) -> Tuple[List[Tuple[CodeChunk, float]], Dict[str, Any]]:
        """Fallback to basic embedding search."""
        try:
            # Generate query embedding
            query_embeddings = await self.enhanced_processor.embedding_generator.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search similar chunks
            search_results = await self.enhanced_processor.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                project_ids=project_ids
            )
            
            metadata = {
                "processing_type": "basic_fallback",
                "confidence_score": 0.5,  # Default confidence
                "total_results": len(search_results),
                "embedding_insights": [f"Found {len(search_results)} similar chunks using basic embedding search"],
                "graph_insights": [],
                "combined_context": f"Basic search results for: {query}",
                "processing_time": 0.0
            }
            
            return search_results, metadata
        
        except Exception as e:
            logger.error(f"Error in basic search fallback: {e}")
            return [], {"error": str(e), "processing_type": "error"}
    
    def _update_performance_stats(self, confidence: float, response_time: float):
        """Update performance statistics."""
        # Update average response time
        total_queries = self.performance_stats["total_queries"]
        current_avg = self.performance_stats["avg_response_time"]
        self.performance_stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        # Update average confidence
        enhanced_queries = self.performance_stats["enhanced_queries"]
        current_conf_avg = self.performance_stats["avg_confidence"]
        self.performance_stats["avg_confidence"] = (
            (current_conf_avg * (enhanced_queries - 1) + confidence) / enhanced_queries
        )
    
    async def extract_entities_from_query(self, query: str) -> List[ExtractedEntity]:
        """Extract entities from query using advanced NER."""
        return await self.enhanced_processor.extract_entities_advanced(query)
    
    async def get_graph_context_for_entities(
        self,
        entities: List[ExtractedEntity],
        project_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get graph context for extracted entities."""
        try:
            graph_result = await self.enhanced_processor.graph_query(
                entities=entities,
                project_ids=project_ids,
                limit=20
            )
            
            return {
                "nodes": graph_result.nodes,
                "relationships": graph_result.relationships,
                "traversal_paths": graph_result.traversal_paths,
                "centrality_scores": graph_result.centrality_scores,
                "processing_time": graph_result.total_time
            }
        
        except Exception as e:
            logger.error(f"Error getting graph context: {e}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_queries": 0,
            "enhanced_queries": 0,
            "avg_response_time": 0.0,
            "avg_confidence": 0.0
        }
    
    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and recommend processing approach."""
        try:
            # Extract entities to assess complexity
            entities = await self.extract_entities_from_query(query)
            
            # Analyze query characteristics
            query_length = len(query.split())
            entity_count = len(entities)
            has_technical_terms = any(
                entity.entity_type.value in ["technology", "framework", "database", "function", "class"]
                for entity in entities
            )
            
            # Determine complexity
            complexity_score = 0
            if query_length > 10:
                complexity_score += 1
            if entity_count > 3:
                complexity_score += 1
            if has_technical_terms:
                complexity_score += 1
            
            complexity_level = "simple"
            if complexity_score >= 2:
                complexity_level = "moderate"
            if complexity_score >= 3:
                complexity_level = "complex"
            
            # Recommend processing approach
            use_enhanced = complexity_score >= 1 or has_technical_terms
            
            return {
                "complexity_level": complexity_level,
                "complexity_score": complexity_score,
                "query_length": query_length,
                "entity_count": entity_count,
                "has_technical_terms": has_technical_terms,
                "entities": [
                    {
                        "text": entity.text,
                        "type": entity.entity_type.value,
                        "confidence": entity.confidence
                    }
                    for entity in entities
                ],
                "recommended_enhanced": use_enhanced,
                "reasoning": self._get_complexity_reasoning(complexity_level, entity_count, has_technical_terms)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {e}")
            return {"error": str(e)}
    
    def _get_complexity_reasoning(self, level: str, entity_count: int, has_technical: bool) -> str:
        """Get reasoning for complexity assessment."""
        reasons = []
        
        if level == "simple":
            reasons.append("Query is straightforward with basic search terms")
        elif level == "moderate":
            reasons.append("Query has moderate complexity requiring enhanced processing")
        else:
            reasons.append("Query is complex and benefits from sophisticated analysis")
        
        if entity_count > 3:
            reasons.append(f"Multiple entities detected ({entity_count})")
        
        if has_technical:
            reasons.append("Technical terms detected requiring specialized knowledge")
        
        return ". ".join(reasons)
