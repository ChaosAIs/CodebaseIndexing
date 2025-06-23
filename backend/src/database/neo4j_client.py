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
    
    async def initialize_schema(self):
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
                    'called_by': []
                }
                
                # Get parents
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CHILD_OF*1..2]->(parent:Chunk)
                    RETURN parent
                """, chunk_id=chunk_id)
                
                for record in result:
                    parent_data = record["parent"]
                    context['parents'].append(self._record_to_chunk(parent_data))
                
                # Get children
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:PARENT_OF*1..2]->(child:Chunk)
                    RETURN child
                """, chunk_id=chunk_id)
                
                for record in result:
                    child_data = record["child"]
                    context['children'].append(self._record_to_chunk(child_data))
                
                # Get called functions
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CALLS]->(called:Chunk)
                    RETURN called
                """, chunk_id=chunk_id)
                
                for record in result:
                    called_data = record["called"]
                    context['calls'].append(self._record_to_chunk(called_data))
                
                # Get calling functions
                result = session.run("""
                    MATCH (c:Chunk {id: $chunk_id})-[:CALLED_BY]->(caller:Chunk)
                    RETURN caller
                """, chunk_id=chunk_id)
                
                for record in result:
                    caller_data = record["caller"]
                    context['called_by'].append(self._record_to_chunk(caller_data))
                
                return context
                
        except Exception as e:
            logger.error(f"Error getting chunk context: {e}")
            return {}
    
    async def get_graph_data(self, file_path: Optional[str] = None, limit: int = 1000) -> GraphData:
        """Get graph data for visualization."""
        try:
            with self.driver.session() as session:
                # Build query based on filters
                if file_path:
                    node_query = """
                        MATCH (c:Chunk {file_path: $file_path})
                        RETURN c
                        LIMIT $limit
                    """
                    edge_query = """
                        MATCH (c1:Chunk {file_path: $file_path})-[r]->(c2:Chunk)
                        RETURN c1.id as source, c2.id as target, type(r) as rel_type, r.weight as weight
                        LIMIT $limit
                    """
                    params = {"file_path": file_path, "limit": limit}
                else:
                    node_query = """
                        MATCH (c:Chunk)
                        RETURN c
                        LIMIT $limit
                    """
                    edge_query = """
                        MATCH (c1:Chunk)-[r]->(c2:Chunk)
                        RETURN c1.id as source, c2.id as target, type(r) as rel_type, r.weight as weight
                        LIMIT $limit
                    """
                    params = {"limit": limit}
                
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
