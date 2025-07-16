"""
Enhanced Query Processing System with sophisticated Embedding RAG + Graph RAG pipeline.

This module implements the advanced query processing flow:
1. Query Processing with NER entity extraction
2. Embedding Query: Convert query to embeddings and retrieve top-K documents
3. Graph Query: Use NER entities to query graph nodes and relationships
4. Combine Results: Score and rank using weighted combination
5. Context Fusion: Merge embedding and graph insights
6. Generate Response: Use combined context for rich responses
"""

import re
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import openai

from ..models import CodeChunk, QueryResult
from ..database.qdrant_client import QdrantVectorStore
from ..database.neo4j_client import Neo4jGraphStore
from ..embeddings.embedding_generator import EmbeddingGenerator
from .advanced_ner import AdvancedNERExtractor, ExtractedEntity, EntityType as NEREntityType


@dataclass
class EmbeddingRAGResult:
    """Result from embedding-based retrieval."""
    chunks: List[Tuple[CodeChunk, float]]
    query_embedding: List[float]
    total_time: float


@dataclass
class GraphRAGResult:
    """Result from graph-based retrieval."""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    traversal_paths: List[List[str]]
    centrality_scores: Dict[str, float]
    total_time: float


@dataclass
class CombinedRAGResult:
    """Combined result from both embedding and graph RAG."""
    ranked_chunks: List[Dict[str, Any]]
    embedding_insights: List[str]
    graph_insights: List[str]
    combined_context: str
    confidence_score: float


class EnhancedQueryProcessor:
    """
    Enhanced query processor implementing sophisticated Embedding + Graph RAG pipeline.
    """
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        graph_store: Neo4jGraphStore,
        embedding_generator: EmbeddingGenerator,
        openai_client: Optional[openai.OpenAI] = None
    ):
        """Initialize the enhanced query processor."""
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_generator = embedding_generator
        self.openai_client = openai_client

        # Initialize advanced NER extractor
        self.ner_extractor = AdvancedNERExtractor()

        # Scoring weights for combining results
        self.scoring_weights = {
            "embedding_score": 0.4,
            "graph_score": 0.3,
            "centrality_score": 0.15,
            "path_score": 0.15
        }
    
    async def process_query_enhanced(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        limit: int = 10,
        embedding_model: str = "openai"
    ) -> CombinedRAGResult:
        """
        Process query using enhanced Embedding + Graph RAG pipeline.
        
        Args:
            query: User's natural language query
            project_ids: Optional project filtering
            limit: Maximum number of results
            embedding_model: Model to use for embeddings
            
        Returns:
            CombinedRAGResult with ranked results and insights
        """
        logger.info(f"Processing enhanced query: {query}")
        
        # Step 1: Extract entities from query using NER
        entities = await self.extract_entities_advanced(query)
        logger.info(f"Extracted {len(entities)} entities: {[e.text for e in entities]}")
        
        # Step 2: Embedding Query - Convert query to embedding and search
        embedding_result = await self.embedding_query(
            query, project_ids, limit, embedding_model
        )
        
        # Step 3: Graph Query - Use entities to query graph
        graph_result = await self.graph_query(
            entities, project_ids, limit
        )
        
        # Step 4: Combine and rank results
        combined_result = await self.combine_and_rank_results(
            query, embedding_result, graph_result, entities
        )
        
        # Step 5: Generate context fusion and insights
        final_result = await self.generate_context_fusion(
            query, combined_result, embedding_result, graph_result
        )
        
        logger.info(f"Enhanced query processing complete. Confidence: {final_result.confidence_score:.2f}")
        return final_result
    
    async def extract_entities_advanced(self, query: str) -> List[ExtractedEntity]:
        """
        Advanced entity extraction using the new NER system.

        Args:
            query: Input query text

        Returns:
            List of extracted entities with confidence scores
        """
        # Use the advanced NER extractor
        entities = await self.ner_extractor.extract_entities(query)

        # Map entities to graph nodes
        for entity in entities:
            entity.properties["graph_node_ids"] = await self._map_entity_to_graph_nodes(entity)

        return entities

    async def _map_entity_to_graph_nodes(self, entity: ExtractedEntity) -> List[str]:
        """Map an entity to relevant graph node IDs."""
        try:
            # Query graph database to find nodes matching the entity
            node_ids = []

            # Different strategies based on entity type
            if entity.entity_type == NEREntityType.FUNCTION:
                node_ids = await self._find_function_nodes(entity.text)
            elif entity.entity_type == NEREntityType.CLASS:
                node_ids = await self._find_class_nodes(entity.text)
            elif entity.entity_type == NEREntityType.FILE:
                node_ids = await self._find_file_nodes(entity.text)
            elif entity.entity_type in [NEREntityType.PERSON, NEREntityType.ORGANIZATION]:
                # For people/orgs, search in comments, documentation
                node_ids = await self._find_nodes_by_content(entity.text)

            return node_ids
        except Exception as e:
            logger.error(f"Error mapping entity to graph nodes: {e}")
            return []

    async def _find_function_nodes(self, function_name: str) -> List[str]:
        """Find graph nodes for a specific function."""
        try:
            with self.graph_store.driver.session() as session:
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.node_type = 'function' AND c.name CONTAINS $name
                    RETURN c.id as node_id
                    LIMIT 10
                """, name=function_name)

                return [record["node_id"] for record in result]
        except Exception as e:
            logger.error(f"Error finding function nodes: {e}")
            return []

    async def _find_class_nodes(self, class_name: str) -> List[str]:
        """Find graph nodes for a specific class."""
        try:
            with self.graph_store.driver.session() as session:
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.node_type = 'class' AND c.name CONTAINS $name
                    RETURN c.id as node_id
                    LIMIT 10
                """, name=class_name)

                return [record["node_id"] for record in result]
        except Exception as e:
            logger.error(f"Error finding class nodes: {e}")
            return []

    async def _find_file_nodes(self, file_name: str) -> List[str]:
        """Find graph nodes for a specific file."""
        try:
            with self.graph_store.driver.session() as session:
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.file_path CONTAINS $name
                    RETURN c.id as node_id
                    LIMIT 10
                """, name=file_name)

                return [record["node_id"] for record in result]
        except Exception as e:
            logger.error(f"Error finding file nodes: {e}")
            return []

    async def _find_nodes_by_content(self, content: str) -> List[str]:
        """Find graph nodes containing specific content."""
        try:
            with self.graph_store.driver.session() as session:
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.content CONTAINS $content
                    RETURN c.id as node_id, c.content as content
                    LIMIT 10
                """, content=content)

                return [record["node_id"] for record in result]
        except Exception as e:
            logger.error(f"Error finding nodes by content: {e}")
            return []

    async def embedding_query(
        self,
        query: str,
        project_ids: Optional[List[str]] = None,
        limit: int = 10,
        embedding_model: str = "openai"
    ) -> EmbeddingRAGResult:
        """
        Perform embedding-based query using vector similarity.

        Args:
            query: Query text
            project_ids: Optional project filtering
            limit: Maximum results
            embedding_model: Embedding model to use

        Returns:
            EmbeddingRAGResult with similar chunks and metadata
        """
        import time
        start_time = time.time()

        try:
            # Generate query embedding
            logger.debug(f"Enhanced query processor: Generating embedding for query: {query[:100]}...")
            query_embedding = await self.embedding_generator.generate_embeddings([query])
            query_embedding = query_embedding[0]
            logger.debug(f"Enhanced query processor: Generated query embedding with dimension {len(query_embedding)}")

            # Search for similar chunks using cosine similarity
            similar_chunks = await self.vector_store.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                project_ids=project_ids
            )

            total_time = time.time() - start_time

            return EmbeddingRAGResult(
                chunks=similar_chunks,
                query_embedding=query_embedding,
                total_time=total_time
            )

        except Exception as e:
            logger.error(f"Error in embedding query: {e}")
            return EmbeddingRAGResult(
                chunks=[],
                query_embedding=[],
                total_time=time.time() - start_time
            )

    async def graph_query(
        self,
        entities: List[ExtractedEntity],
        project_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> GraphRAGResult:
        """
        Perform graph-based query using entity relationships.

        Args:
            entities: Extracted entities from query
            project_ids: Optional project filtering
            limit: Maximum results

        Returns:
            GraphRAGResult with graph nodes, relationships, and metrics
        """
        import time
        start_time = time.time()

        try:
            nodes = []
            relationships = []
            traversal_paths = []
            centrality_scores = {}

            # For each entity, find related nodes and relationships
            for entity in entities:
                graph_node_ids = entity.properties.get("graph_node_ids", [])
                if graph_node_ids:
                    # Get nodes and their relationships
                    entity_nodes, entity_rels, entity_paths = await self._explore_entity_graph(
                        entity, project_ids, limit
                    )

                    nodes.extend(entity_nodes)
                    relationships.extend(entity_rels)
                    traversal_paths.extend(entity_paths)

            # Calculate centrality scores for important nodes
            centrality_scores = await self._calculate_centrality_scores(nodes)

            # Remove duplicates
            nodes = self._deduplicate_nodes(nodes)
            relationships = self._deduplicate_relationships(relationships)

            total_time = time.time() - start_time

            return GraphRAGResult(
                nodes=nodes,
                relationships=relationships,
                traversal_paths=traversal_paths,
                centrality_scores=centrality_scores,
                total_time=total_time
            )

        except Exception as e:
            logger.error(f"Error in graph query: {e}")
            return GraphRAGResult(
                nodes=[],
                relationships=[],
                traversal_paths=[],
                centrality_scores={},
                total_time=time.time() - start_time
            )

    async def _explore_entity_graph(
        self,
        entity: ExtractedEntity,
        project_ids: Optional[List[str]] = None,
        limit: int = 10
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[List[str]]]:
        """
        Explore graph relationships for a specific entity.

        Returns:
            Tuple of (nodes, relationships, traversal_paths)
        """
        nodes = []
        relationships = []
        traversal_paths = []

        try:
            with self.graph_store.driver.session() as session:
                # Build project filter
                project_filter = ""
                graph_node_ids = entity.properties.get("graph_node_ids", [])
                params = {"entity_ids": graph_node_ids, "limit": limit}

                if project_ids:
                    project_filter = "AND c.project_id IN $project_ids"
                    params["project_ids"] = project_ids

                # Query for entity nodes and their neighbors
                query = f"""
                    MATCH (c:Chunk)
                    WHERE c.id IN $entity_ids {project_filter}

                    // Get the entity nodes
                    WITH c
                    OPTIONAL MATCH (c)-[r1]->(neighbor1:Chunk)
                    OPTIONAL MATCH (c)<-[r2]-(neighbor2:Chunk)

                    // Get second-degree neighbors for richer context
                    OPTIONAL MATCH (neighbor1)-[r3]->(neighbor3:Chunk)
                    WHERE neighbor3.id <> c.id

                    RETURN
                        c as entity_node,
                        collect(DISTINCT neighbor1) as direct_neighbors_out,
                        collect(DISTINCT neighbor2) as direct_neighbors_in,
                        collect(DISTINCT neighbor3) as second_degree_neighbors,
                        collect(DISTINCT r1) as outgoing_rels,
                        collect(DISTINCT r2) as incoming_rels,
                        collect(DISTINCT r3) as second_degree_rels
                    LIMIT $limit
                """

                result = session.run(query, params)

                for record in result:
                    # Process entity node
                    entity_node = record["entity_node"]
                    if entity_node:
                        nodes.append(self._neo4j_node_to_dict(entity_node))

                    # Process neighbors
                    for neighbor in record["direct_neighbors_out"]:
                        if neighbor:
                            nodes.append(self._neo4j_node_to_dict(neighbor))

                    for neighbor in record["direct_neighbors_in"]:
                        if neighbor:
                            nodes.append(self._neo4j_node_to_dict(neighbor))

                    for neighbor in record["second_degree_neighbors"]:
                        if neighbor:
                            nodes.append(self._neo4j_node_to_dict(neighbor))

                    # Process relationships
                    for rel in record["outgoing_rels"]:
                        if rel:
                            relationships.append(self._neo4j_rel_to_dict(rel))

                    for rel in record["incoming_rels"]:
                        if rel:
                            relationships.append(self._neo4j_rel_to_dict(rel))

                    for rel in record["second_degree_rels"]:
                        if rel:
                            relationships.append(self._neo4j_rel_to_dict(rel))

                # Generate traversal paths
                traversal_paths = await self._generate_traversal_paths(graph_node_ids, session)

        except Exception as e:
            logger.error(f"Error exploring entity graph: {e}")

        return nodes, relationships, traversal_paths

    def _neo4j_node_to_dict(self, node) -> Dict[str, Any]:
        """Convert Neo4j node to dictionary."""
        return {
            "id": node.get("id"),
            "name": node.get("name", ""),
            "node_type": node.get("node_type", ""),
            "file_path": node.get("file_path", ""),
            "content": node.get("content", ""),
            "start_line": node.get("start_line", 0),
            "end_line": node.get("end_line", 0),
            "project_id": node.get("project_id", "")
        }

    def _neo4j_rel_to_dict(self, rel) -> Dict[str, Any]:
        """Convert Neo4j relationship to dictionary."""
        return {
            "type": rel.type,
            "start_node": rel.start_node.get("id"),
            "end_node": rel.end_node.get("id"),
            "weight": rel.get("weight", 1.0),
            "properties": dict(rel)
        }

    async def _generate_traversal_paths(
        self,
        start_node_ids: List[str],
        session
    ) -> List[List[str]]:
        """Generate traversal paths from entity nodes."""
        paths = []

        try:
            for node_id in start_node_ids:
                # Find paths of length 2-3 from this node
                result = session.run("""
                    MATCH path = (start:Chunk {id: $node_id})-[*1..3]->(end:Chunk)
                    WHERE start.id <> end.id
                    RETURN [node in nodes(path) | node.id] as path_ids
                    LIMIT 5
                """, node_id=node_id)

                for record in result:
                    path_ids = record["path_ids"]
                    if len(path_ids) > 1:
                        paths.append(path_ids)

        except Exception as e:
            logger.error(f"Error generating traversal paths: {e}")

        return paths

    async def _calculate_centrality_scores(self, nodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate centrality scores for nodes."""
        centrality_scores = {}

        try:
            node_ids = [node["id"] for node in nodes if node.get("id")]

            if not node_ids:
                return centrality_scores

            with self.graph_store.driver.session() as session:
                # Calculate degree centrality (simple version)
                result = session.run("""
                    MATCH (c:Chunk)
                    WHERE c.id IN $node_ids
                    OPTIONAL MATCH (c)-[r]-(neighbor:Chunk)
                    RETURN c.id as node_id, count(r) as degree
                """, node_ids=node_ids)

                max_degree = 1
                degrees = {}

                for record in result:
                    node_id = record["node_id"]
                    degree = record["degree"] or 0
                    degrees[node_id] = degree
                    max_degree = max(max_degree, degree)

                # Normalize centrality scores
                for node_id, degree in degrees.items():
                    centrality_scores[node_id] = degree / max_degree

        except Exception as e:
            logger.error(f"Error calculating centrality scores: {e}")

        return centrality_scores

    def _deduplicate_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate nodes based on ID."""
        seen_ids = set()
        unique_nodes = []

        for node in nodes:
            node_id = node.get("id")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)

        return unique_nodes

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate relationships."""
        seen_rels = set()
        unique_rels = []

        for rel in relationships:
            rel_key = (rel.get("start_node"), rel.get("end_node"), rel.get("type"))
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                unique_rels.append(rel)

        return unique_rels

    async def combine_and_rank_results(
        self,
        query: str,
        embedding_result: EmbeddingRAGResult,
        graph_result: GraphRAGResult,
        entities: List[ExtractedEntity]
    ) -> List[Dict[str, Any]]:
        """
        Combine and rank results from embedding and graph RAG.

        Uses weighted scoring with:
        - Embedding similarity scores (cosine distance)
        - Graph traversal metrics (path length, centrality)
        - Entity relevance scores
        - Learned ranking adjustments

        Args:
            query: Original query
            embedding_result: Results from embedding RAG
            graph_result: Results from graph RAG
            entities: Extracted entities

        Returns:
            List of ranked results with combined scores
        """
        combined_results = {}

        # Process embedding results
        for chunk, embedding_score in embedding_result.chunks:
            chunk_id = chunk.id
            combined_results[chunk_id] = {
                "chunk": chunk,
                "embedding_score": embedding_score,
                "graph_score": 0.0,
                "centrality_score": 0.0,
                "path_score": 0.0,
                "entity_relevance": 0.0,
                "source": "embedding"
            }

        # Process graph results and merge with embedding results
        for node in graph_result.nodes:
            node_id = node.get("id")
            if not node_id:
                continue

            # Convert graph node to CodeChunk if not already in results
            if node_id not in combined_results:
                chunk = self._graph_node_to_code_chunk(node)
                combined_results[node_id] = {
                    "chunk": chunk,
                    "embedding_score": 0.0,
                    "graph_score": 0.0,
                    "centrality_score": 0.0,
                    "path_score": 0.0,
                    "entity_relevance": 0.0,
                    "source": "graph"
                }

            # Add graph-based scores
            result = combined_results[node_id]

            # Graph score based on relationship strength
            result["graph_score"] = self._calculate_graph_score(node, graph_result.relationships)

            # Centrality score
            result["centrality_score"] = graph_result.centrality_scores.get(node_id, 0.0)

            # Path score based on traversal paths
            result["path_score"] = self._calculate_path_score(node_id, graph_result.traversal_paths)

            # Entity relevance score
            result["entity_relevance"] = self._calculate_entity_relevance(node, entities)

        # Calculate final combined scores
        for result in combined_results.values():
            result["combined_score"] = self._calculate_combined_score(result)

        # Sort by combined score
        ranked_results = sorted(
            combined_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        return ranked_results

    def _graph_node_to_code_chunk(self, node: Dict[str, Any]) -> CodeChunk:
        """Convert graph node dictionary to CodeChunk object."""
        return CodeChunk(
            id=node.get("id", ""),
            content=node.get("content", ""),
            file_path=node.get("file_path", ""),
            start_line=node.get("start_line", 0),
            end_line=node.get("end_line", 0),
            node_type=node.get("node_type", ""),
            name=node.get("name", ""),
            project_id=node.get("project_id", "")
        )

    def _calculate_graph_score(self, node: Dict[str, Any], relationships: List[Dict[str, Any]]) -> float:
        """Calculate graph-based score for a node."""
        node_id = node.get("id")
        if not node_id:
            return 0.0

        # Count relationships involving this node
        relationship_count = 0
        total_weight = 0.0

        for rel in relationships:
            if rel.get("start_node") == node_id or rel.get("end_node") == node_id:
                relationship_count += 1
                total_weight += rel.get("weight", 1.0)

        # Normalize score (relationship count + average weight)
        if relationship_count > 0:
            avg_weight = total_weight / relationship_count
            return min(relationship_count * 0.1 + avg_weight * 0.2, 1.0)

        return 0.0

    def _calculate_path_score(self, node_id: str, traversal_paths: List[List[str]]) -> float:
        """Calculate path-based score for a node."""
        path_score = 0.0

        for path in traversal_paths:
            if node_id in path:
                # Shorter paths get higher scores
                path_length = len(path)
                if path_length > 1:
                    path_score += 1.0 / path_length

        return min(path_score, 1.0)

    def _calculate_entity_relevance(self, node: Dict[str, Any], entities: List[ExtractedEntity]) -> float:
        """Calculate entity relevance score for a node."""
        relevance_score = 0.0
        node_content = node.get("content", "").lower()
        node_name = node.get("name", "").lower()

        for entity in entities:
            entity_text = entity.text.lower()

            # Check if entity appears in node content or name
            if entity_text in node_content:
                relevance_score += entity.confidence * 0.5

            if entity_text in node_name:
                relevance_score += entity.confidence * 0.8

            # Check if node ID is in entity's mapped graph nodes
            graph_node_ids = entity.properties.get("graph_node_ids", [])
            if graph_node_ids and node.get("id") in graph_node_ids:
                relevance_score += entity.confidence * 1.0

        return min(relevance_score, 1.0)

    def _calculate_combined_score(self, result: Dict[str, Any]) -> float:
        """Calculate final combined score using weighted sum."""
        weights = self.scoring_weights

        combined_score = (
            result["embedding_score"] * weights["embedding_score"] +
            result["graph_score"] * weights["graph_score"] +
            result["centrality_score"] * weights["centrality_score"] +
            result["path_score"] * weights["path_score"]
        )

        # Add entity relevance bonus
        entity_bonus = result["entity_relevance"] * 0.2

        return combined_score + entity_bonus

    async def generate_context_fusion(
        self,
        query: str,
        ranked_results: List[Dict[str, Any]],
        embedding_result: EmbeddingRAGResult,
        graph_result: GraphRAGResult
    ) -> CombinedRAGResult:
        """
        Generate context fusion combining embedding and graph insights.

        Args:
            query: Original query
            ranked_results: Combined and ranked results
            embedding_result: Original embedding results
            graph_result: Original graph results

        Returns:
            CombinedRAGResult with fused context and insights
        """
        # Extract top chunks for context
        top_chunks = ranked_results[:10]  # Top 10 results

        # Generate embedding insights
        embedding_insights = await self._generate_embedding_insights(
            query, embedding_result, top_chunks
        )

        # Generate graph insights
        graph_insights = await self._generate_graph_insights(
            query, graph_result, top_chunks
        )

        # Create combined context
        combined_context = await self._create_combined_context(
            query, top_chunks, embedding_insights, graph_insights
        )

        # Calculate overall confidence score
        confidence_score = self._calculate_overall_confidence(
            embedding_result, graph_result, top_chunks
        )

        return CombinedRAGResult(
            ranked_chunks=top_chunks,
            embedding_insights=embedding_insights,
            graph_insights=graph_insights,
            combined_context=combined_context,
            confidence_score=confidence_score
        )

    async def _generate_embedding_insights(
        self,
        query: str,
        embedding_result: EmbeddingRAGResult,
        top_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from embedding-based retrieval."""
        insights = []

        try:
            # Analyze semantic similarity patterns
            if embedding_result.chunks:
                avg_score = np.mean([score for _, score in embedding_result.chunks])
                insights.append(f"Found {len(embedding_result.chunks)} semantically similar code chunks with average similarity of {avg_score:.3f}")

            # Analyze file distribution
            file_distribution = {}
            for result in top_chunks:
                chunk = result["chunk"]
                file_path = chunk.file_path
                file_distribution[file_path] = file_distribution.get(file_path, 0) + 1

            if file_distribution:
                most_relevant_file = max(file_distribution, key=file_distribution.get)
                insights.append(f"Most relevant code found in: {most_relevant_file} ({file_distribution[most_relevant_file]} chunks)")

            # Analyze code types
            type_distribution = {}
            for result in top_chunks:
                chunk = result["chunk"]
                node_type = chunk.node_type
                type_distribution[node_type] = type_distribution.get(node_type, 0) + 1

            if type_distribution:
                insights.append(f"Code types found: {', '.join([f'{k}({v})' for k, v in type_distribution.items()])}")

        except Exception as e:
            logger.error(f"Error generating embedding insights: {e}")
            insights.append("Unable to generate embedding insights")

        return insights

    async def _generate_graph_insights(
        self,
        query: str,
        graph_result: GraphRAGResult,
        top_chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from graph-based retrieval."""
        insights = []

        try:
            # Analyze relationship patterns
            if graph_result.relationships:
                rel_types = {}
                for rel in graph_result.relationships:
                    rel_type = rel.get("type", "unknown")
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

                insights.append(f"Found {len(graph_result.relationships)} relationships: {', '.join([f'{k}({v})' for k, v in rel_types.items()])}")

            # Analyze centrality
            if graph_result.centrality_scores:
                high_centrality_nodes = [
                    node_id for node_id, score in graph_result.centrality_scores.items()
                    if score > 0.7
                ]
                if high_centrality_nodes:
                    insights.append(f"Found {len(high_centrality_nodes)} highly connected nodes indicating architectural importance")

            # Analyze traversal paths
            if graph_result.traversal_paths:
                avg_path_length = np.mean([len(path) for path in graph_result.traversal_paths])
                insights.append(f"Code relationships span {len(graph_result.traversal_paths)} paths with average depth of {avg_path_length:.1f}")

            # Analyze architectural layers
            layers = set()
            for result in top_chunks:
                chunk = result["chunk"]
                file_path = chunk.file_path.lower()
                if "controller" in file_path or "api" in file_path:
                    layers.add("API Layer")
                elif "service" in file_path or "business" in file_path:
                    layers.add("Business Layer")
                elif "model" in file_path or "entity" in file_path:
                    layers.add("Data Layer")
                elif "util" in file_path or "helper" in file_path:
                    layers.add("Utility Layer")

            if layers:
                insights.append(f"Architectural layers involved: {', '.join(layers)}")

        except Exception as e:
            logger.error(f"Error generating graph insights: {e}")
            insights.append("Unable to generate graph insights")

        return insights

    async def _create_combined_context(
        self,
        query: str,
        top_chunks: List[Dict[str, Any]],
        embedding_insights: List[str],
        graph_insights: List[str]
    ) -> str:
        """Create combined context for response generation."""
        context_parts = []

        # Add query context
        context_parts.append(f"Query: {query}")
        context_parts.append("")

        # Add embedding insights
        if embedding_insights:
            context_parts.append("Semantic Analysis:")
            for insight in embedding_insights:
                context_parts.append(f"- {insight}")
            context_parts.append("")

        # Add graph insights
        if graph_insights:
            context_parts.append("Relationship Analysis:")
            for insight in graph_insights:
                context_parts.append(f"- {insight}")
            context_parts.append("")

        # Add top code chunks with context
        context_parts.append("Relevant Code:")
        for i, result in enumerate(top_chunks[:5], 1):  # Top 5 for context
            chunk = result["chunk"]
            score = result["combined_score"]
            context_parts.append(f"{i}. {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.3f})")
            context_parts.append(f"   Type: {chunk.node_type}, Name: {chunk.name}")

            # Add code preview
            content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            context_parts.append(f"   Code: {content_preview}")
            context_parts.append("")

        return "\n".join(context_parts)

    def _calculate_overall_confidence(
        self,
        embedding_result: EmbeddingRAGResult,
        graph_result: GraphRAGResult,
        top_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score for the combined results."""
        confidence_factors = []

        # Embedding confidence
        if embedding_result.chunks:
            avg_embedding_score = np.mean([score for _, score in embedding_result.chunks])
            confidence_factors.append(avg_embedding_score)

        # Graph confidence
        if graph_result.nodes:
            graph_confidence = min(len(graph_result.nodes) / 10.0, 1.0)  # More nodes = higher confidence
            confidence_factors.append(graph_confidence)

        # Result diversity confidence
        if top_chunks:
            files = set(result["chunk"].file_path for result in top_chunks)
            diversity_confidence = min(len(files) / 5.0, 1.0)  # More files = higher confidence
            confidence_factors.append(diversity_confidence)

        # Combined score confidence
        if top_chunks:
            top_score = top_chunks[0]["combined_score"]
            score_confidence = min(top_score, 1.0)
            confidence_factors.append(score_confidence)

        # Calculate weighted average
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.0
