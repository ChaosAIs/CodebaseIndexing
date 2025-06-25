"""
Enhanced Agent Integration with Sophisticated Query Processing.

This module integrates the enhanced query processing pipeline with the
multi-agent system, allowing each agent to leverage sophisticated
embedding + graph RAG capabilities.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

from .agent_orchestrator import AgentOrchestrator, AgentRole, AgentPerspective, FlowResponse
from ..models import CodeChunk
from ..query.enhanced_integration import EnhancedQueryIntegration
from ..query.advanced_ner import ExtractedEntity


class EnhancedAgentOrchestrator(AgentOrchestrator):
    """
    Enhanced Agent Orchestrator with sophisticated query processing.
    
    This extends the existing agent orchestrator to use the enhanced
    query processing pipeline for each agent's analysis.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced agent orchestrator."""
        super().__init__(*args, **kwargs)
        
        # Enhanced query integration will be set by the MCP server
        self.enhanced_integration: Optional[EnhancedQueryIntegration] = None
        
        # Enhanced performance stats
        self.enhanced_stats = {
            "enhanced_analyses": 0,
            "entity_extractions": 0,
            "graph_queries": 0,
            "avg_entity_count": 0.0,
            "avg_graph_nodes": 0.0
        }
    
    def set_enhanced_integration(self, integration: EnhancedQueryIntegration):
        """Set the enhanced query integration."""
        self.enhanced_integration = integration
        logger.info("Enhanced query integration enabled for agent orchestrator")
    
    async def analyze_with_agents_enhanced(
        self,
        query: str,
        chunks: List[CodeChunk],
        context: Dict[str, Any] = None
    ) -> FlowResponse:
        """
        Enhanced agent analysis with sophisticated query processing.
        
        This method enhances each agent's analysis by providing:
        - Extracted entities from the query
        - Graph context for relevant entities
        - Enhanced scoring and ranking
        - Richer contextual information
        """
        if not self.enhanced_integration:
            logger.warning("Enhanced integration not available, falling back to standard analysis")
            return await super().analyze_with_agents(query, chunks, context)
        
        try:
            logger.info(f"Starting enhanced agent analysis for: {query}")
            
            # Step 1: Extract entities from query
            entities = await self.enhanced_integration.extract_entities_from_query(query)
            logger.info(f"Extracted {len(entities)} entities from query")
            
            # Step 2: Get graph context for entities
            graph_context = await self.enhanced_integration.get_graph_context_for_entities(
                entities=entities,
                project_ids=context.get("project_ids") if context else None
            )
            
            # Step 3: Enhance context with sophisticated analysis
            enhanced_context = await self._create_enhanced_context(
                query, chunks, entities, graph_context, context
            )
            
            # Step 4: Run agents with enhanced context
            flow_response = await self._run_enhanced_agents(
                query, chunks, enhanced_context
            )
            
            # Step 5: Post-process with enhanced insights
            final_response = await self._post_process_enhanced_response(
                flow_response, entities, graph_context, query
            )
            
            # Update stats
            self._update_enhanced_stats(entities, graph_context)
            
            logger.info(f"Enhanced agent analysis completed with {len(final_response.sections)} sections")
            return final_response
        
        except Exception as e:
            logger.error(f"Error in enhanced agent analysis: {e}")
            # Fallback to standard analysis
            return await super().analyze_with_agents(query, chunks, context)
    
    async def _create_enhanced_context(
        self,
        query: str,
        chunks: List[CodeChunk],
        entities: List[ExtractedEntity],
        graph_context: Dict[str, Any],
        original_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create enhanced context for agent analysis."""
        enhanced_context = original_context.copy() if original_context else {}
        
        # Add entity information
        enhanced_context["extracted_entities"] = [
            {
                "text": entity.text,
                "type": entity.entity_type.value,
                "confidence": entity.confidence,
                "aliases": entity.aliases,
                "context_keywords": self.enhanced_integration.ner_extractor.get_entity_context_keywords(entity)
            }
            for entity in entities
        ]
        
        # Add graph context
        enhanced_context["graph_context"] = graph_context
        
        # Add entity-specific insights
        enhanced_context["entity_insights"] = await self._generate_entity_insights(entities, chunks)
        
        # Add architectural analysis
        enhanced_context["architectural_analysis"] = self._analyze_architectural_context(chunks, entities)
        
        # Add query complexity analysis
        if self.enhanced_integration:
            complexity_analysis = await self.enhanced_integration.analyze_query_complexity(query)
            enhanced_context["complexity_analysis"] = complexity_analysis
        
        # Enhanced processing flag
        enhanced_context["enhanced_processing"] = True
        
        return enhanced_context
    
    async def _generate_entity_insights(
        self,
        entities: List[ExtractedEntity],
        chunks: List[CodeChunk]
    ) -> List[Dict[str, Any]]:
        """Generate insights about entities in relation to the code chunks."""
        insights = []
        
        for entity in entities:
            entity_insight = {
                "entity": entity.text,
                "type": entity.entity_type.value,
                "confidence": entity.confidence,
                "related_chunks": [],
                "relationships": []
            }
            
            # Find chunks related to this entity
            entity_text_lower = entity.text.lower()
            for chunk in chunks:
                if entity_text_lower in chunk.content.lower() or entity_text_lower in chunk.name.lower():
                    entity_insight["related_chunks"].append({
                        "file_path": chunk.file_path,
                        "name": chunk.name,
                        "node_type": chunk.node_type,
                        "relevance": "high" if entity_text_lower in chunk.name.lower() else "medium"
                    })
            
            insights.append(entity_insight)
        
        return insights
    
    def _analyze_architectural_context(
        self,
        chunks: List[CodeChunk],
        entities: List[ExtractedEntity]
    ) -> Dict[str, Any]:
        """Analyze architectural context of the code chunks and entities."""
        # Analyze file paths to determine architectural layers
        layers = {
            "api": [],
            "business": [],
            "data": [],
            "utility": [],
            "other": []
        }
        
        for chunk in chunks:
            file_path_lower = chunk.file_path.lower()
            
            if any(keyword in file_path_lower for keyword in ["controller", "api", "route", "endpoint"]):
                layers["api"].append(chunk)
            elif any(keyword in file_path_lower for keyword in ["service", "business", "logic", "manager"]):
                layers["business"].append(chunk)
            elif any(keyword in file_path_lower for keyword in ["model", "entity", "repository", "dao", "database"]):
                layers["data"].append(chunk)
            elif any(keyword in file_path_lower for keyword in ["util", "helper", "common", "shared"]):
                layers["utility"].append(chunk)
            else:
                layers["other"].append(chunk)
        
        # Analyze entity distribution across layers
        entity_distribution = {}
        for entity in entities:
            entity_distribution[entity.text] = {
                "type": entity.entity_type.value,
                "layers": []
            }
            
            entity_text_lower = entity.text.lower()
            for layer_name, layer_chunks in layers.items():
                for chunk in layer_chunks:
                    if entity_text_lower in chunk.content.lower():
                        if layer_name not in entity_distribution[entity.text]["layers"]:
                            entity_distribution[entity.text]["layers"].append(layer_name)
        
        return {
            "layer_distribution": {
                layer: len(chunks) for layer, chunks in layers.items()
            },
            "entity_distribution": entity_distribution,
            "architectural_patterns": self._detect_architectural_patterns(chunks, entities)
        }
    
    def _detect_architectural_patterns(
        self,
        chunks: List[CodeChunk],
        entities: List[ExtractedEntity]
    ) -> List[str]:
        """Detect architectural patterns in the code."""
        patterns = []
        
        # Check for common patterns based on entities and code structure
        entity_types = [entity.entity_type.value for entity in entities]
        
        if "class" in entity_types and "interface" in entity_types:
            patterns.append("Object-Oriented Design")
        
        if any("service" in chunk.name.lower() for chunk in chunks):
            patterns.append("Service Layer Pattern")
        
        if any("repository" in chunk.name.lower() for chunk in chunks):
            patterns.append("Repository Pattern")
        
        if any("controller" in chunk.name.lower() for chunk in chunks):
            patterns.append("MVC Pattern")
        
        if any("factory" in chunk.name.lower() for chunk in chunks):
            patterns.append("Factory Pattern")
        
        return patterns
    
    async def _run_enhanced_agents(
        self,
        query: str,
        chunks: List[CodeChunk],
        enhanced_context: Dict[str, Any]
    ) -> FlowResponse:
        """Run agents with enhanced context."""
        # Use the parent class method but with enhanced context
        return await super().analyze_with_agents(query, chunks, enhanced_context)
    
    async def _post_process_enhanced_response(
        self,
        flow_response: FlowResponse,
        entities: List[ExtractedEntity],
        graph_context: Dict[str, Any],
        query: str
    ) -> FlowResponse:
        """Post-process the flow response with enhanced insights."""
        enhanced_sections = flow_response.sections.copy()
        
        # Add entity analysis section
        if entities:
            entity_section = {
                "title": "Entity Analysis",
                "content": self._format_entity_analysis(entities),
                "type": "entity_analysis",
                "confidence": 0.9
            }
            enhanced_sections.insert(1, entity_section)  # Insert after overview
        
        # Add graph insights section
        if graph_context.get("nodes") or graph_context.get("relationships"):
            graph_section = {
                "title": "Code Relationships",
                "content": self._format_graph_analysis(graph_context),
                "type": "graph_analysis",
                "confidence": 0.85
            }
            enhanced_sections.append(graph_section)
        
        # Enhance metadata
        enhanced_metadata = flow_response.metadata.copy()
        enhanced_metadata.update({
            "enhanced_processing": True,
            "entity_count": len(entities),
            "graph_nodes": len(graph_context.get("nodes", [])),
            "graph_relationships": len(graph_context.get("relationships", [])),
            "enhanced_stats": self.enhanced_stats.copy()
        })
        
        return FlowResponse(
            sections=enhanced_sections,
            metadata=enhanced_metadata
        )
    
    def _format_entity_analysis(self, entities: List[ExtractedEntity]) -> str:
        """Format entity analysis for display."""
        lines = []
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity)
        
        for entity_type, type_entities in entity_groups.items():
            lines.append(f"**{entity_type.title()} Entities:**")
            for entity in sorted(type_entities, key=lambda e: e.confidence, reverse=True)[:5]:
                confidence_str = f"{entity.confidence:.2f}"
                aliases_str = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
                lines.append(f"• {entity.text} (confidence: {confidence_str}){aliases_str}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_graph_analysis(self, graph_context: Dict[str, Any]) -> str:
        """Format graph analysis for display."""
        lines = []
        
        nodes = graph_context.get("nodes", [])
        relationships = graph_context.get("relationships", [])
        centrality_scores = graph_context.get("centrality_scores", {})
        
        if nodes:
            lines.append(f"**Code Components:** {len(nodes)} nodes analyzed")
            
            # Show top nodes by centrality
            if centrality_scores:
                sorted_nodes = sorted(
                    centrality_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                
                lines.append("**Most Important Components:**")
                for node_id, score in sorted_nodes:
                    # Find node details
                    node = next((n for n in nodes if n.get("id") == node_id), None)
                    if node:
                        name = node.get("name", "unnamed")
                        node_type = node.get("node_type", "unknown")
                        lines.append(f"• {name} ({node_type}, importance: {score:.2f})")
                lines.append("")
        
        if relationships:
            # Analyze relationship types
            rel_types = {}
            for rel in relationships:
                rel_type = rel.get("type", "unknown")
                rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
            
            lines.append(f"**Relationships:** {len(relationships)} connections found")
            for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"• {rel_type}: {count}")
        
        return "\n".join(lines)
    
    def _update_enhanced_stats(self, entities: List[ExtractedEntity], graph_context: Dict[str, Any]):
        """Update enhanced processing statistics."""
        self.enhanced_stats["enhanced_analyses"] += 1
        self.enhanced_stats["entity_extractions"] += 1
        
        # Update average entity count
        current_avg = self.enhanced_stats["avg_entity_count"]
        total_analyses = self.enhanced_stats["enhanced_analyses"]
        self.enhanced_stats["avg_entity_count"] = (
            (current_avg * (total_analyses - 1) + len(entities)) / total_analyses
        )
        
        # Update average graph nodes
        nodes_count = len(graph_context.get("nodes", []))
        if nodes_count > 0:
            self.enhanced_stats["graph_queries"] += 1
            current_avg_nodes = self.enhanced_stats["avg_graph_nodes"]
            total_graph_queries = self.enhanced_stats["graph_queries"]
            self.enhanced_stats["avg_graph_nodes"] = (
                (current_avg_nodes * (total_graph_queries - 1) + nodes_count) / total_graph_queries
            )
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics."""
        return {
            **super().get_performance_stats(),
            "enhanced_stats": self.enhanced_stats.copy()
        }
