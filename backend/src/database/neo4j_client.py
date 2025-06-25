"""Neo4j graph database client for storing and querying code relationships."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from neo4j import GraphDatabase, Driver, Session
from loguru import logger

from ..models import CodeChunk, RelationshipType, GraphNode, GraphEdge, GraphData
from ..config import config


class Neo4jGraphStore:
    """Neo4j graph database client for code relationships."""
    
    def __init__(self):
        """Initialize Neo4j client."""
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        try:
            with self.driver.session() as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                    "CREATE CONSTRAINT file_path_index IF NOT EXISTS FOR (c:Chunk) REQUIRE c.file_path IS NOT NULL"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"Constraint creation warning: {e}")
                
                # Create indexes
                indexes = [
                    "CREATE INDEX chunk_name_index IF NOT EXISTS FOR (c:Chunk) ON (c.name)",
                    "CREATE INDEX chunk_type_index IF NOT EXISTS FOR (c:Chunk) ON (c.node_type)",
                    "CREATE INDEX chunk_file_index IF NOT EXISTS FOR (c:Chunk) ON (c.file_path)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                    except Exception as e:
                        logger.warning(f"Index creation warning: {e}")
                
                logger.info("Neo4j schema initialized")
                
        except Exception as e:
            logger.error(f"Error initializing Neo4j schema: {e}")
            raise
    
    async def store_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Store chunks as nodes in Neo4j."""
        try:
            with self.driver.session() as session:
                # Clear existing data for files being updated
                file_paths = list(set(chunk.file_path for chunk in chunks))
                for file_path in file_paths:
                    session.run(
                        "MATCH (c:Chunk {file_path: $file_path}) DETACH DELETE c",
                        file_path=file_path
                    )

                # Also clear any chunks with the same IDs to avoid constraint violations
                chunk_ids = [chunk.id for chunk in chunks]
                for chunk_id in chunk_ids:
                    session.run(
                        "MATCH (c:Chunk {id: $chunk_id}) DETACH DELETE c",
                        chunk_id=chunk_id
                    )
                
                # Create or update nodes using MERGE
                for chunk in chunks:
                    session.run("""
                        MERGE (c:Chunk {id: $id})
                        SET c.content = $content,
                            c.file_path = $file_path,
                            c.start_line = $start_line,
                            c.end_line = $end_line,
                            c.node_type = $node_type,
                            c.name = $name,
                            c.parent_id = $parent_id,
                            c.project_id = $project_id,
                            c.calls = $calls,
                            c.called_by = $called_by,
                            c.imports = $imports,
                            c.metadata = $metadata
                    """, {
                        "id": chunk.id,
                        "content": chunk.content,
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "node_type": chunk.node_type.value,
                        "name": chunk.name,
                        "parent_id": chunk.parent_id,
                        "project_id": chunk.project_id,
                        "calls": chunk.calls,
                        "called_by": chunk.called_by,
                        "imports": chunk.imports,
                        "metadata": json.dumps(chunk.metadata) if chunk.metadata else "{}"
                    })
                
                logger.info(f"Stored {len(chunks)} chunks as nodes in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error storing chunks in Neo4j: {e}")
            return False
    
    async def create_relationships(self, chunks: List[CodeChunk]) -> bool:
        """Create relationships between chunks."""
        try:
            with self.driver.session() as session:
                # Create parent-child relationships
                for chunk in chunks:
                    if chunk.parent_id:
                        session.run("""
                            MATCH (parent:Chunk {id: $parent_id})
                            MATCH (child:Chunk {id: $child_id})
                            CREATE (parent)-[:PARENT_OF {weight: 1.0}]->(child)
                            CREATE (child)-[:CHILD_OF {weight: 1.0}]->(parent)
                        """, {
                            "parent_id": chunk.parent_id,
                            "child_id": chunk.id
                        })
                
                # Create call relationships
                for chunk in chunks:
                    for called_id in chunk.calls:
                        session.run("""
                            MATCH (caller:Chunk {id: $caller_id})
                            MATCH (callee:Chunk {id: $callee_id})
                            CREATE (caller)-[:CALLS {weight: 0.5}]->(callee)
                            CREATE (callee)-[:CALLED_BY {weight: 0.5}]->(caller)
                        """, {
                            "caller_id": chunk.id,
                            "callee_id": called_id
                        })

                # Create import relationships
                for chunk in chunks:
                    for imported_id in chunk.imports:
                        session.run("""
                            MATCH (importer:Chunk {id: $importer_id})
                            MATCH (imported:Chunk {id: $imported_id})
                            CREATE (importer)-[:IMPORTS {weight: 0.3}]->(imported)
                            CREATE (imported)-[:IMPORTED_BY {weight: 0.3}]->(importer)
                        """, {
                            "importer_id": chunk.id,
                            "imported_id": imported_id
                        })

                logger.info("Created relationships in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error creating relationships in Neo4j: {e}")
            return False
    
    async def get_chunk_context(self, chunk_id: str, max_depth: int = 2) -> Dict[str, List[CodeChunk]]:
        """Get contextual chunks using graph traversal."""
        try:
            with self.driver.session() as session:
                context = {
                    'parents': [],
                    'children': [],
                    'calls': [],
                    'called_by': [],
                    'imports': [],
                    'imported_by': [],
                    'siblings': [],
                    'architectural_context': []
                }

                # Get hierarchical parents (up to root)
                result = session.run("""
                    MATCH path = (c:Chunk {id: $chunk_id})-[:CHILD_OF*1..5]->(parent:Chunk)
                    RETURN parent, length(path) as depth
                    ORDER BY depth ASC
                """, chunk_id=chunk_id)

                for record in result:
                    parent_data = record["parent"]
                    context['parents'].append(self._record_to_chunk(parent_data))

                # Get all children (deeper exploration)
                result = session.run("""
                    MATCH path = (c:Chunk {id: $chunk_id})-[:PARENT_OF*1..3]->(child:Chunk)
                    RETURN child, length(path) as depth
                    ORDER BY depth ASC
                """, chunk_id=chunk_id)

                for record in result:
                    child_data = record["child"]
                    context['children'].append(self._record_to_chunk(child_data))

                # Get call relationships (both directions)
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CALLS]->(called:Chunk)
                    RETURN called
                """, chunk_id=chunk_id)

                for record in result:
                    called_data = record["called"]
                    context['calls'].append(self._record_to_chunk(called_data))

                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CALLED_BY]->(caller:Chunk)
                    RETURN caller
                """, chunk_id=chunk_id)

                for record in result:
                    caller_data = record["caller"]
                    context['called_by'].append(self._record_to_chunk(caller_data))

                # Get import relationships
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:IMPORTS]->(imported:Chunk)
                    RETURN imported
                """, chunk_id=chunk_id)

                for record in result:
                    imported_data = record["imported"]
                    context['imports'].append(self._record_to_chunk(imported_data))

                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:IMPORTED_BY]->(importer:Chunk)
                    RETURN importer
                """, chunk_id=chunk_id)

                for record in result:
                    importer_data = record["importer"]
                    context['imported_by'].append(self._record_to_chunk(importer_data))

                # Get siblings (same parent)
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CHILD_OF]->(parent:Chunk)<-[:CHILD_OF]-(sibling:Chunk)
                    WHERE sibling.id <> $chunk_id
                    RETURN sibling
                    LIMIT 5
                """, chunk_id=chunk_id)

                for record in result:
                    sibling_data = record["sibling"]
                    context['siblings'].append(self._record_to_chunk(sibling_data))

                # Get architectural context (same file, same module)
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (arch:Chunk)
                    WHERE arch.file_path = c.file_path
                    AND arch.id <> c.id
                    AND arch.node_type IN ['class', 'module', 'interface']
                    RETURN arch
                    LIMIT 3
                """, chunk_id=chunk_id)

                for record in result:
                    arch_data = record["arch"]
                    context['architectural_context'].append(self._record_to_chunk(arch_data))

                return context

        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {}

    async def get_comprehensive_context(self, chunk_ids: List[str], query: str = "") -> Dict[str, Any]:
        """Get comprehensive architectural context for multiple chunks."""
        try:
            with self.driver.session() as session:
                # Get system-wide architectural patterns
                architectural_context = await self._get_architectural_patterns(session, chunk_ids, query)

                # Get data flow patterns
                data_flow_context = await self._get_data_flow_patterns(session, chunk_ids)

                # Get module dependencies
                dependency_context = await self._get_dependency_patterns(session, chunk_ids)

                # Get usage patterns
                usage_context = await self._get_usage_patterns(session, chunk_ids)

                return {
                    "architectural_patterns": architectural_context,
                    "data_flow_patterns": data_flow_context,
                    "dependency_patterns": dependency_context,
                    "usage_patterns": usage_context,
                    "system_overview": await self._get_system_overview(session, chunk_ids)
                }

        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}")
            return {}

    async def _get_architectural_patterns(self, session, chunk_ids: List[str], query: str) -> List[Dict[str, Any]]:
        """Identify architectural patterns and system layers."""
        patterns = []

        # Find system layers (controllers, services, models, etc.)
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids OR
                  c.file_path CONTAINS 'controller' OR
                  c.file_path CONTAINS 'service' OR
                  c.file_path CONTAINS 'model' OR
                  c.file_path CONTAINS 'database' OR
                  c.file_path CONTAINS 'api' OR
                  c.file_path CONTAINS 'client'
            RETURN c.file_path as file_path,
                   c.node_type as node_type,
                   count(c) as component_count,
                   collect(c.name)[0..3] as sample_components
            ORDER BY component_count DESC
            LIMIT 10
        """, chunk_ids=chunk_ids)

        for record in result:
            patterns.append({
                "layer": self._identify_architectural_layer(record["file_path"]),
                "file_path": record["file_path"],
                "component_count": record["component_count"],
                "sample_components": record["sample_components"],
                "node_type": record["node_type"]
            })

        return patterns

    async def _get_data_flow_patterns(self, session, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify data flow patterns between components."""
        result = session.run("""
            MATCH (source:Chunk)-[r:CALLS|IMPORTS]->(target:Chunk)
            WHERE source.id IN $chunk_ids OR target.id IN $chunk_ids
            RETURN source.name as source_name,
                   source.file_path as source_file,
                   target.name as target_name,
                   target.file_path as target_file,
                   type(r) as relationship_type,
                   source.node_type as source_type,
                   target.node_type as target_type
            LIMIT 20
        """, chunk_ids=chunk_ids)

        flows = []
        for record in result:
            flows.append({
                "source": {
                    "name": record["source_name"],
                    "file": record["source_file"],
                    "type": record["source_type"]
                },
                "target": {
                    "name": record["target_name"],
                    "file": record["target_file"],
                    "type": record["target_type"]
                },
                "relationship": record["relationship_type"],
                "flow_direction": "outbound" if record["relationship_type"] == "CALLS" else "import"
            })

        return flows

    async def _get_dependency_patterns(self, session, chunk_ids: List[str]) -> Dict[str, Any]:
        """Analyze dependency patterns and module relationships."""
        # Get file-level dependencies
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            WITH DISTINCT c.file_path as file_path
            MATCH (source:Chunk)-[:IMPORTS|CALLS]->(target:Chunk)
            WHERE source.file_path = file_path OR target.file_path = file_path
            RETURN source.file_path as source_file,
                   target.file_path as target_file,
                   count(*) as dependency_strength
            ORDER BY dependency_strength DESC
            LIMIT 15
        """, chunk_ids=chunk_ids)

        dependencies = []
        for record in result:
            dependencies.append({
                "source_file": record["source_file"],
                "target_file": record["target_file"],
                "strength": record["dependency_strength"]
            })

        return {
            "file_dependencies": dependencies,
            "dependency_graph": self._build_dependency_graph(dependencies)
        }

    async def _get_usage_patterns(self, session, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify usage patterns and hot spots."""
        result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            OPTIONAL MATCH (c)<-[:CALLS]-(caller:Chunk)
            OPTIONAL MATCH (c)<-[:IMPORTS]-(importer:Chunk)
            RETURN c.name as component_name,
                   c.file_path as file_path,
                   c.node_type as node_type,
                   count(DISTINCT caller) as call_count,
                   count(DISTINCT importer) as import_count,
                   (count(DISTINCT caller) + count(DISTINCT importer)) as total_usage
            ORDER BY total_usage DESC
            LIMIT 10
        """, chunk_ids=chunk_ids)

        patterns = []
        for record in result:
            patterns.append({
                "component": record["component_name"],
                "file_path": record["file_path"],
                "node_type": record["node_type"],
                "call_count": record["call_count"],
                "import_count": record["import_count"],
                "total_usage": record["total_usage"],
                "usage_category": self._categorize_usage(record["total_usage"])
            })

        return patterns

    async def _get_system_overview(self, session, chunk_ids: List[str]) -> Dict[str, Any]:
        """Get high-level system overview and statistics."""
        # Get system statistics
        stats_result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            WITH collect(DISTINCT c.file_path) as file_paths
            MATCH (all_chunks:Chunk)
            WHERE all_chunks.file_path IN file_paths
            RETURN count(DISTINCT all_chunks.file_path) as total_files,
                   count(all_chunks) as total_chunks,
                   count(DISTINCT all_chunks.node_type) as node_types,
                   collect(DISTINCT all_chunks.node_type) as types_list
        """, chunk_ids=chunk_ids)

        stats = stats_result.single()

        # Get component distribution
        dist_result = session.run("""
            MATCH (c:Chunk)
            WHERE c.id IN $chunk_ids
            WITH collect(DISTINCT c.file_path) as file_paths
            MATCH (all_chunks:Chunk)
            WHERE all_chunks.file_path IN file_paths
            RETURN all_chunks.node_type as node_type, count(*) as count
            ORDER BY count DESC
        """, chunk_ids=chunk_ids)

        distribution = []
        for record in dist_result:
            distribution.append({
                "type": record["node_type"],
                "count": record["count"]
            })

        return {
            "total_files": stats["total_files"],
            "total_chunks": stats["total_chunks"],
            "node_types": stats["node_types"],
            "types_list": stats["types_list"],
            "component_distribution": distribution,
            "system_complexity": self._calculate_complexity_score(stats, distribution)
        }
    
    async def get_graph_data(self, file_path: Optional[str] = None, project_ids: Optional[List[str]] = None, limit: int = 1000) -> GraphData:
        """Get graph data for visualization."""
        try:
            with self.driver.session() as session:
                # Build query based on filters
                where_conditions = []
                params = {"limit": limit}

                if file_path:
                    where_conditions.append("c.file_path = $file_path")
                    params["file_path"] = file_path

                if project_ids:
                    where_conditions.append("c.project_id IN $project_ids")
                    params["project_ids"] = project_ids

                where_clause = " AND ".join(where_conditions)
                if where_clause:
                    where_clause = "WHERE " + where_clause

                node_query = f"""
                    MATCH (c:Chunk)
                    {where_clause}
                    RETURN c
                    LIMIT $limit
                """

                edge_query = f"""
                    MATCH (c1:Chunk)-[r]->(c2:Chunk)
                    {where_clause.replace('c.', 'c1.') if where_clause else ''}
                    RETURN c1.id as source, c2.id as target, type(r) as rel_type, r.weight as weight
                    LIMIT $limit
                """
                
                # Get nodes
                nodes = []
                result = session.run(node_query, params)
                for record in result:
                    chunk_data = record["c"]
                    node = GraphNode(
                        id=chunk_data["id"],
                        label=chunk_data.get("name", "unnamed"),
                        type=chunk_data["node_type"],
                        file_path=chunk_data["file_path"],
                        properties={
                            "start_line": chunk_data["start_line"],
                            "end_line": chunk_data["end_line"],
                            "content_preview": chunk_data["content"][:100] + "..." if len(chunk_data["content"]) > 100 else chunk_data["content"]
                        }
                    )
                    nodes.append(node)
                
                # Get edges
                edges = []
                result = session.run(edge_query, params)
                for record in result:
                    edge = GraphEdge(
                        source=record["source"],
                        target=record["target"],
                        type=self._map_relationship_type(record["rel_type"]),
                        weight=record.get("weight", 1.0)
                    )
                    edges.append(edge)
                
                return GraphData(
                    nodes=nodes,
                    edges=edges,
                    metadata={
                        "total_nodes": len(nodes),
                        "total_edges": len(edges),
                        "file_path": file_path
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting graph data: {e}")
            return GraphData(nodes=[], edges=[])

    def find_related_entities(self, entity_names: List[str], max_depth: int = 2, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity names through graph relationships.

        Args:
            entity_names: List of entity names to find relationships for
            max_depth: Maximum traversal depth
            limit: Maximum number of related entities to return

        Returns:
            List of related entities with their metadata
        """
        if not entity_names:
            return []

        try:
            with self.driver.session() as session:
                # Query to find related entities through various relationship types
                query = """
                    MATCH (source:Chunk)
                    WHERE source.name IN $entity_names
                       OR source.file_path IN $entity_names
                       OR any(name IN $entity_names WHERE source.content CONTAINS name)

                    // Find related entities through different relationship types
                    OPTIONAL MATCH (source)-[r1:CALLS|CALLED_BY|IMPORTS|IMPORTED_BY|CHILD_OF|PARENT_OF*1..2]-(related1:Chunk)
                    OPTIONAL MATCH (source)-[r2:SIMILAR_TO|DEPENDS_ON]-(related2:Chunk)

                    // Also find entities in the same file or module
                    OPTIONAL MATCH (same_file:Chunk)
                    WHERE same_file.file_path = source.file_path AND same_file.id <> source.id

                    WITH source,
                         collect(DISTINCT related1) + collect(DISTINCT related2) + collect(DISTINCT same_file) as all_related

                    UNWIND all_related as related

                    WITH source, related
                    WHERE related IS NOT NULL

                    RETURN DISTINCT
                        related.name as name,
                        related.node_type as type,
                        related.file_path as file_path,
                        related.content as content,
                        related.start_line as start_line,
                        related.end_line as end_line,
                        related.id as id,
                        // Calculate relevance score based on relationship strength
                        CASE
                            WHEN related.file_path = source.file_path THEN 0.9
                            WHEN (related)-[:CALLS|CALLED_BY]-(source) THEN 0.8
                            WHEN (related)-[:IMPORTS|IMPORTED_BY]-(source) THEN 0.7
                            WHEN (related)-[:CHILD_OF|PARENT_OF]-(source) THEN 0.85
                            ELSE 0.6
                        END as relevance_score

                    ORDER BY relevance_score DESC
                    LIMIT $limit
                """

                result = session.run(query, {
                    "entity_names": entity_names,
                    "limit": limit
                })

                related_entities = []
                for record in result:
                    entity = {
                        "name": record.get("name", ""),
                        "type": record.get("type", ""),
                        "file_path": record.get("file_path", ""),
                        "content": record.get("content", "")[:200],  # Limit content preview
                        "start_line": record.get("start_line", 0),
                        "end_line": record.get("end_line", 0),
                        "id": record.get("id", ""),
                        "relevance_score": record.get("relevance_score", 0.0)
                    }
                    related_entities.append(entity)

                logger.info(f"Found {len(related_entities)} related entities for {len(entity_names)} input entities")
                return related_entities

        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []

    def _identify_architectural_layer(self, file_path: str) -> str:
        """Identify the architectural layer based on file path."""
        file_path_lower = file_path.lower()

        if any(keyword in file_path_lower for keyword in ['controller', 'api', 'endpoint', 'route']):
            return "API Layer"
        elif any(keyword in file_path_lower for keyword in ['service', 'business', 'logic']):
            return "Business Logic Layer"
        elif any(keyword in file_path_lower for keyword in ['model', 'entity', 'schema']):
            return "Data Model Layer"
        elif any(keyword in file_path_lower for keyword in ['database', 'repository', 'dao', 'client']):
            return "Data Access Layer"
        elif any(keyword in file_path_lower for keyword in ['frontend', 'ui', 'component', 'view']):
            return "Presentation Layer"
        elif any(keyword in file_path_lower for keyword in ['util', 'helper', 'common']):
            return "Utility Layer"
        elif any(keyword in file_path_lower for keyword in ['config', 'setting']):
            return "Configuration Layer"
        else:
            return "Core Layer"

    def _build_dependency_graph(self, dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a dependency graph structure."""
        nodes = set()
        edges = []

        for dep in dependencies:
            source = dep["source_file"]
            target = dep["target_file"]
            strength = dep["strength"]

            nodes.add(source)
            nodes.add(target)
            edges.append({
                "source": source,
                "target": target,
                "weight": strength
            })

        return {
            "nodes": list(nodes),
            "edges": edges,
            "total_dependencies": len(edges),
            "unique_files": len(nodes)
        }

    def _categorize_usage(self, usage_count: int) -> str:
        """Categorize usage patterns."""
        if usage_count >= 10:
            return "High Usage (Core Component)"
        elif usage_count >= 5:
            return "Medium Usage (Important Component)"
        elif usage_count >= 2:
            return "Low Usage (Supporting Component)"
        else:
            return "Minimal Usage (Utility Component)"

    def _calculate_complexity_score(self, stats: Dict[str, Any], distribution: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate system complexity metrics."""
        total_chunks = stats.get("total_chunks", 0)
        total_files = stats.get("total_files", 1)
        node_types = stats.get("node_types", 1)

        # Calculate various complexity metrics
        chunks_per_file = total_chunks / total_files if total_files > 0 else 0
        type_diversity = node_types / 10  # Normalize to 0-1 scale

        # Calculate distribution entropy (measure of code organization)
        total = sum(item["count"] for item in distribution)
        entropy = 0
        if total > 0:
            for item in distribution:
                p = item["count"] / total
                if p > 0:
                    entropy -= p * (p ** 0.5)  # Simplified entropy calculation

        complexity_score = (chunks_per_file * 0.4) + (type_diversity * 0.3) + (entropy * 0.3)

        return {
            "complexity_score": round(complexity_score, 2),
            "chunks_per_file": round(chunks_per_file, 1),
            "type_diversity": round(type_diversity, 2),
            "organization_entropy": round(entropy, 2),
            "complexity_level": self._get_complexity_level(complexity_score)
        }

    def _get_complexity_level(self, score: float) -> str:
        """Get human-readable complexity level."""
        if score >= 3.0:
            return "Very High"
        elif score >= 2.0:
            return "High"
        elif score >= 1.0:
            return "Medium"
        elif score >= 0.5:
            return "Low"
        else:
            return "Very Low"
    
    def _record_to_chunk(self, record_data: Dict[str, Any]) -> CodeChunk:
        """Convert Neo4j record to CodeChunk."""
        return CodeChunk(
            id=record_data["id"],
            content=record_data["content"],
            file_path=record_data["file_path"],
            start_line=record_data["start_line"],
            end_line=record_data["end_line"],
            node_type=record_data["node_type"],
            name=record_data.get("name"),
            parent_id=record_data.get("parent_id"),
            project_id=record_data.get("project_id"),
            calls=record_data.get("calls", []),
            called_by=record_data.get("called_by", []),
            imports=record_data.get("imports", []),
            metadata=json.loads(record_data.get("metadata", "{}")) if record_data.get("metadata") else {}
        )
    
    def _map_relationship_type(self, neo4j_type: str) -> RelationshipType:
        """Map Neo4j relationship type to our enum."""
        mapping = {
            "PARENT_OF": RelationshipType.PARENT_CHILD,
            "CHILD_OF": RelationshipType.PARENT_CHILD,
            "CALLS": RelationshipType.CALLS,
            "CALLED_BY": RelationshipType.CALLED_BY
        }
        return mapping.get(neo4j_type, RelationshipType.CALLS)
    
    async def health_check(self) -> bool:
        """Check if Neo4j is healthy."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.driver.session() as session:
                # Get node count
                result = session.run("MATCH (c:Chunk) RETURN count(c) as node_count")
                node_count = result.single()["node_count"]
                
                # Get relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                
                # Get file count
                result = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.file_path) as file_count")
                file_count = result.single()["file_count"]
                
                return {
                    "total_chunks": node_count,
                    "total_relationships": rel_count,
                    "total_files": file_count
                }
        except Exception as e:
            logger.error(f"Error getting Neo4j statistics: {e}")
            return {}
