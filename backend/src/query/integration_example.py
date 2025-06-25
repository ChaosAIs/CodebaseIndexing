"""
Example integration of Enhanced Query Processing with MCP Server.

This file shows how to integrate the enhanced query processing system
with the existing MCP server to provide richer, more intelligent responses.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from ..mcp_server.server import MCPServer
from ..models import SearchRequest, FlowRequest, SearchResponse, FlowResponse
from .enhanced_integration import EnhancedQueryIntegration


class EnhancedMCPServer(MCPServer):
    """
    Enhanced MCP Server with sophisticated query processing capabilities.
    
    This extends the existing MCP server to use the enhanced query processing
    pipeline for better results and richer insights.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced MCP server."""
        super().__init__(*args, **kwargs)
        
        # Initialize enhanced query integration
        self.enhanced_integration = EnhancedQueryIntegration(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            embedding_generator=self.embedding_generator,
            openai_client=getattr(self, 'openai_client', None)
        )
        
        logger.info("Enhanced MCP Server initialized with sophisticated query processing")
    
    async def search_enhanced(self, request: SearchRequest) -> SearchResponse:
        """
        Enhanced search with sophisticated query processing.
        
        This method uses the enhanced query processing pipeline to provide
        richer results with better entity understanding and graph context.
        """
        try:
            logger.info(f"Processing enhanced search: {request.query}")
            
            # Analyze query complexity to determine processing approach
            complexity_analysis = await self.enhanced_integration.analyze_query_complexity(request.query)
            use_enhanced = complexity_analysis.get("recommended_enhanced", True)
            
            logger.info(f"Query complexity: {complexity_analysis.get('complexity_level', 'unknown')}, "
                       f"using enhanced: {use_enhanced}")
            
            # Process query with enhanced pipeline
            search_results, metadata = await self.enhanced_integration.process_query_enhanced(
                query=request.query,
                project_ids=request.project_ids,
                limit=request.limit,
                embedding_model=request.model,
                use_enhanced=use_enhanced
            )
            
            # Create enhanced response
            response = SearchResponse(
                results=[
                    {
                        "chunk": chunk,
                        "score": score,
                        "metadata": {
                            "file_path": chunk.file_path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "node_type": chunk.node_type,
                            "name": chunk.name
                        }
                    }
                    for chunk, score in search_results
                ],
                metadata={
                    **metadata,
                    "complexity_analysis": complexity_analysis,
                    "enhanced_processing": use_enhanced
                }
            )
            
            logger.info(f"Enhanced search completed: {len(search_results)} results, "
                       f"confidence: {metadata.get('confidence_score', 0):.3f}")
            
            return response
        
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            # Fallback to original search method
            return await super().search(request)
    
    async def flow_enhanced(self, request: FlowRequest) -> FlowResponse:
        """
        Enhanced flow processing with sophisticated analysis.
        
        This method combines the enhanced query processing with the multi-agent
        system to provide comprehensive, flowing responses.
        """
        try:
            logger.info(f"Processing enhanced flow: {request.query}")
            
            # First, get enhanced search results
            search_request = SearchRequest(
                query=request.query,
                project_ids=request.project_ids,
                limit=request.limit,
                model=request.model
            )
            
            search_response = await self.search_enhanced(search_request)
            
            # Extract chunks for agent analysis
            chunks = [result["chunk"] for result in search_response.results]
            
            # Get enhanced metadata for agent context
            enhanced_metadata = search_response.metadata
            
            # Prepare enhanced context for agents
            agent_context = {
                "enhanced_processing": True,
                "complexity_analysis": enhanced_metadata.get("complexity_analysis", {}),
                "embedding_insights": enhanced_metadata.get("embedding_insights", []),
                "graph_insights": enhanced_metadata.get("graph_insights", []),
                "combined_context": enhanced_metadata.get("combined_context", ""),
                "confidence_score": enhanced_metadata.get("confidence_score", 0.0),
                "processing_type": enhanced_metadata.get("processing_type", "enhanced")
            }
            
            # Run multi-agent analysis with enhanced context
            flow_response = await self.agent_orchestrator.analyze_with_agents(
                query=request.query,
                chunks=chunks,
                context=agent_context
            )
            
            # Enhance the flow response with additional insights
            enhanced_flow_response = await self._enhance_flow_response(
                flow_response, enhanced_metadata, request.query
            )
            
            logger.info(f"Enhanced flow completed with {len(enhanced_flow_response.sections)} sections")
            
            return enhanced_flow_response
        
        except Exception as e:
            logger.error(f"Error in enhanced flow: {e}")
            # Fallback to original flow method
            return await super().flow(request)
    
    async def _enhance_flow_response(
        self,
        flow_response: FlowResponse,
        enhanced_metadata: Dict[str, Any],
        query: str
    ) -> FlowResponse:
        """Enhance flow response with additional insights from enhanced processing."""
        try:
            # Add enhanced insights as additional sections
            enhanced_sections = flow_response.sections.copy()
            
            # Add complexity analysis section
            complexity_analysis = enhanced_metadata.get("complexity_analysis", {})
            if complexity_analysis:
                enhanced_sections.append({
                    "title": "Query Analysis",
                    "content": self._format_complexity_analysis(complexity_analysis),
                    "type": "analysis",
                    "confidence": 0.9
                })
            
            # Add enhanced insights section
            embedding_insights = enhanced_metadata.get("embedding_insights", [])
            graph_insights = enhanced_metadata.get("graph_insights", [])
            
            if embedding_insights or graph_insights:
                insights_content = []
                
                if embedding_insights:
                    insights_content.append("**Semantic Analysis:**")
                    for insight in embedding_insights:
                        insights_content.append(f"• {insight}")
                    insights_content.append("")
                
                if graph_insights:
                    insights_content.append("**Relationship Analysis:**")
                    for insight in graph_insights:
                        insights_content.append(f"• {insight}")
                
                enhanced_sections.append({
                    "title": "Enhanced Insights",
                    "content": "\n".join(insights_content),
                    "type": "insights",
                    "confidence": enhanced_metadata.get("confidence_score", 0.8)
                })
            
            # Update metadata
            enhanced_flow_metadata = flow_response.metadata.copy()
            enhanced_flow_metadata.update({
                "enhanced_processing": True,
                "processing_confidence": enhanced_metadata.get("confidence_score", 0.0),
                "processing_type": enhanced_metadata.get("processing_type", "enhanced"),
                "performance_stats": enhanced_metadata.get("performance_stats", {})
            })
            
            return FlowResponse(
                sections=enhanced_sections,
                metadata=enhanced_flow_metadata
            )
        
        except Exception as e:
            logger.error(f"Error enhancing flow response: {e}")
            return flow_response
    
    def _format_complexity_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format complexity analysis for display."""
        lines = []
        
        complexity_level = analysis.get("complexity_level", "unknown")
        entity_count = analysis.get("entity_count", 0)
        has_technical = analysis.get("has_technical_terms", False)
        
        lines.append(f"**Query Complexity:** {complexity_level.title()}")
        lines.append(f"**Entities Detected:** {entity_count}")
        lines.append(f"**Technical Terms:** {'Yes' if has_technical else 'No'}")
        
        reasoning = analysis.get("reasoning", "")
        if reasoning:
            lines.append(f"**Analysis:** {reasoning}")
        
        entities = analysis.get("entities", [])
        if entities:
            lines.append("")
            lines.append("**Detected Entities:**")
            for entity in entities[:5]:  # Show top 5 entities
                lines.append(f"• {entity['text']} ({entity['type']}, confidence: {entity['confidence']:.2f})")
        
        return "\n".join(lines)
    
    async def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics."""
        return {
            "enhanced_integration_stats": self.enhanced_integration.get_performance_stats(),
            "agent_orchestrator_stats": getattr(self.agent_orchestrator, 'performance_stats', {}),
            "system_info": {
                "enhanced_processing_enabled": True,
                "ner_system": "Advanced NER with spaCy + patterns",
                "graph_processing": "Neo4j with centrality analysis",
                "embedding_system": "Multi-model support (OpenAI/HuggingFace/Ollama)",
                "scoring_system": "Weighted combination with learned ranking"
            }
        }


# Example usage and integration
async def example_enhanced_usage():
    """Example of how to use the enhanced MCP server."""
    
    # This would typically be initialized with proper database connections
    # enhanced_server = EnhancedMCPServer(...)
    
    # Example search request
    search_request = SearchRequest(
        query="How does the authentication system work with JWT tokens?",
        project_ids=["project-1"],
        limit=10,
        model="openai"
    )
    
    # Process with enhanced pipeline
    # search_response = await enhanced_server.search_enhanced(search_request)
    
    # Example flow request
    flow_request = FlowRequest(
        query="Analyze the scalability of the microservices architecture",
        project_ids=["project-1"],
        limit=15,
        model="openai"
    )
    
    # Process with enhanced flow
    # flow_response = await enhanced_server.flow_enhanced(flow_request)
    
    logger.info("Enhanced processing examples completed")


if __name__ == "__main__":
    asyncio.run(example_enhanced_usage())
