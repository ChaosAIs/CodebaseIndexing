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
        logger.debug(f"🔌 Attempting to connect to Neo4j at: {config.database.neo4j_uri}")
        logger.debug(f"🔌 Using Neo4j user: {config.database.neo4j_user}")

        try:
            self.driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password)
            )
            logger.debug("🔌 Neo4j driver created successfully")

            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                test_value = result.single()[0]
                logger.debug(f"🔌 Connection test successful, returned: {test_value}")

            logger.info("✅ Connected to Neo4j database successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            logger.debug(f"❌ Connection error details: {type(e).__name__}: {str(e)}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        logger.debug("🔌 Closing Neo4j connection")
        if self.driver:
            try:
                self.driver.close()
                logger.debug("✅ Neo4j connection closed successfully")
            except Exception as e:
                logger.error(f"❌ Error closing Neo4j connection: {e}")
                logger.debug(f"❌ Close error details: {type(e).__name__}: {str(e)}")
        else:
            logger.debug("🔌 No Neo4j driver to close")
    
    async def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        logger.debug("🏗️ Starting Neo4j schema initialization")

        try:
            with self.driver.session() as session:
                logger.debug("🏗️ Neo4j session created for schema initialization")

                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
                    # Removed file_path NOT NULL constraint - requires Neo4j Enterprise Edition
                ]

                logger.debug(f"🏗️ Creating {len(constraints)} constraints")
                for i, constraint in enumerate(constraints, 1):
                    try:
                        logger.debug(f"🏗️ Creating constraint {i}/{len(constraints)}: {constraint}")
                        result = session.run(constraint)
                        logger.debug(f"✅ Constraint {i} created successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Constraint {i} creation warning: {e}")
                        logger.debug(f"⚠️ Constraint details - Type: {type(e).__name__}, Query: {constraint}")

                # Create indexes
                indexes = [
                    "CREATE INDEX chunk_name_index IF NOT EXISTS FOR (c:Chunk) ON (c.name)",
                    "CREATE INDEX chunk_type_index IF NOT EXISTS FOR (c:Chunk) ON (c.node_type)",
                    "CREATE INDEX chunk_file_index IF NOT EXISTS FOR (c:Chunk) ON (c.file_path)"
                ]

                logger.debug(f"🏗️ Creating {len(indexes)} indexes")
                for i, index in enumerate(indexes, 1):
                    try:
                        logger.debug(f"🏗️ Creating index {i}/{len(indexes)}: {index}")
                        result = session.run(index)
                        logger.debug(f"✅ Index {i} created successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Index {i} creation warning: {e}")
                        logger.debug(f"⚠️ Index details - Type: {type(e).__name__}, Query: {index}")

                logger.info("✅ Neo4j schema initialized successfully")
                logger.debug("🏗️ Schema initialization completed")

        except Exception as e:
            logger.error(f"❌ Error initializing Neo4j schema: {e}")
            logger.debug(f"❌ Schema initialization error details: {type(e).__name__}: {str(e)}")
            raise
    
    async def store_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Store chunks as nodes in Neo4j."""
        logger.debug(f"💾 Starting to store {len(chunks)} chunks in Neo4j")

        if not chunks:
            logger.debug("💾 No chunks to store, returning True")
            return True

        try:
            with self.driver.session() as session:
                logger.debug("💾 Neo4j session created for storing chunks")

                # Clear existing data for files being updated
                file_paths = list(set(chunk.file_path for chunk in chunks))
                logger.debug(f"💾 Clearing existing data for {len(file_paths)} files: {file_paths}")

                deleted_files_count = 0
                for file_path in file_paths:
                    logger.debug(f"💾 Deleting existing chunks for file: {file_path}")
                    result = session.run(
                        "MATCH (c:Chunk {file_path: $file_path}) DETACH DELETE c",
                        file_path=file_path
                    )
                    deleted_files_count += 1
                    logger.debug(f"💾 Deleted chunks for file {deleted_files_count}/{len(file_paths)}: {file_path}")

                # Also clear any chunks with the same IDs to avoid constraint violations
                chunk_ids = [chunk.id for chunk in chunks]
                logger.debug(f"💾 Clearing existing chunks by ID to avoid constraint violations: {len(chunk_ids)} IDs")

                deleted_ids_count = 0
                for chunk_id in chunk_ids:
                    logger.debug(f"💾 Deleting existing chunk with ID: {chunk_id}")
                    result = session.run(
                        "MATCH (c:Chunk {id: $chunk_id}) DETACH DELETE c",
                        chunk_id=chunk_id
                    )
                    deleted_ids_count += 1
                    if deleted_ids_count % 100 == 0:  # Log every 100 deletions
                        logger.debug(f"💾 Deleted {deleted_ids_count}/{len(chunk_ids)} chunks by ID")

                # Create or update nodes using MERGE
                logger.debug(f"💾 Creating/updating {len(chunks)} chunk nodes")
                created_count = 0

                for i, chunk in enumerate(chunks, 1):
                    try:
                        logger.debug(f"💾 Creating chunk {i}/{len(chunks)}: {chunk.id} ({chunk.name}) in {chunk.file_path}")

                        result = session.run("""
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

                        created_count += 1
                        if created_count % 50 == 0:  # Log every 50 creations
                            logger.debug(f"💾 Created {created_count}/{len(chunks)} chunks")

                    except Exception as chunk_error:
                        logger.error(f"❌ Error creating chunk {chunk.id}: {chunk_error}")
                        logger.debug(f"❌ Chunk details - File: {chunk.file_path}, Name: {chunk.name}, Type: {chunk.node_type}")
                        raise

                logger.info(f"✅ Successfully stored {len(chunks)} chunks as nodes in Neo4j")
                logger.debug(f"💾 Chunk storage completed: {created_count} chunks created/updated")
                return True

        except Exception as e:
            logger.error(f"❌ Error storing chunks in Neo4j: {e}")
            logger.debug(f"❌ Store chunks error details: {type(e).__name__}: {str(e)}")
            return False
    
    async def create_relationships(self, chunks: List[CodeChunk]) -> bool:
        """Create relationships between chunks."""
        logger.debug(f"🔗 Starting to create relationships for {len(chunks)} chunks")

        if not chunks:
            logger.debug("🔗 No chunks provided, returning True")
            return True

        try:
            with self.driver.session() as session:
                logger.debug("🔗 Neo4j session created for relationship creation")

                # Count relationships to create
                parent_child_count = sum(1 for chunk in chunks if chunk.parent_id)
                call_count = sum(len(chunk.calls) for chunk in chunks)
                import_count = sum(len(chunk.imports) for chunk in chunks)

                logger.debug(f"🔗 Relationships to create: {parent_child_count} parent-child, {call_count} calls, {import_count} imports")

                # Create parent-child relationships
                logger.debug("🔗 Creating parent-child relationships")
                created_parent_child = 0

                for chunk in chunks:
                    if chunk.parent_id:
                        try:
                            logger.debug(f"🔗 Creating parent-child: {chunk.parent_id} -> {chunk.id}")
                            result = session.run("""
                                MATCH (parent:Chunk {id: $parent_id})
                                MATCH (child:Chunk {id: $child_id})
                                CREATE (parent)-[:PARENT_OF {weight: 1.0}]->(child)
                                CREATE (child)-[:CHILD_OF {weight: 1.0}]->(parent)
                            """, {
                                "parent_id": chunk.parent_id,
                                "child_id": chunk.id
                            })
                            created_parent_child += 1
                        except Exception as rel_error:
                            logger.warning(f"⚠️ Failed to create parent-child relationship {chunk.parent_id} -> {chunk.id}: {rel_error}")

                logger.debug(f"🔗 Created {created_parent_child}/{parent_child_count} parent-child relationships")

                # Create call relationships
                logger.debug("🔗 Creating call relationships")
                created_calls = 0

                for chunk in chunks:
                    for called_id in chunk.calls:
                        try:
                            logger.debug(f"🔗 Creating call relationship: {chunk.id} -> {called_id}")
                            result = session.run("""
                                MATCH (caller:Chunk {id: $caller_id})
                                MATCH (callee:Chunk {id: $callee_id})
                                CREATE (caller)-[:CALLS {weight: 0.5}]->(callee)
                                CREATE (callee)-[:CALLED_BY {weight: 0.5}]->(caller)
                            """, {
                                "caller_id": chunk.id,
                                "callee_id": called_id
                            })
                            created_calls += 1
                        except Exception as call_error:
                            logger.warning(f"⚠️ Failed to create call relationship {chunk.id} -> {called_id}: {call_error}")

                logger.debug(f"🔗 Created {created_calls}/{call_count} call relationships")

                # Create import relationships
                logger.debug("🔗 Creating import relationships")
                created_imports = 0

                for chunk in chunks:
                    for imported_id in chunk.imports:
                        try:
                            logger.debug(f"🔗 Creating import relationship: {chunk.id} -> {imported_id}")
                            result = session.run("""
                                MATCH (importer:Chunk {id: $importer_id})
                                MATCH (imported:Chunk {id: $imported_id})
                                CREATE (importer)-[:IMPORTS {weight: 0.3}]->(imported)
                                CREATE (imported)-[:IMPORTED_BY {weight: 0.3}]->(importer)
                            """, {
                                "importer_id": chunk.id,
                                "imported_id": imported_id
                            })
                            created_imports += 1
                        except Exception as import_error:
                            logger.warning(f"⚠️ Failed to create import relationship {chunk.id} -> {imported_id}: {import_error}")

                logger.debug(f"🔗 Created {created_imports}/{import_count} import relationships")

                total_created = created_parent_child + created_calls + created_imports
                total_expected = parent_child_count + call_count + import_count

                logger.info(f"✅ Created {total_created}/{total_expected} relationships in Neo4j")
                logger.debug(f"🔗 Relationship creation completed: {created_parent_child} parent-child, {created_calls} calls, {created_imports} imports")
                return True

        except Exception as e:
            logger.error(f"❌ Error creating relationships in Neo4j: {e}")
            logger.debug(f"❌ Relationship creation error details: {type(e).__name__}: {str(e)}")
            return False
    
    async def get_chunk_context(self, chunk_id: str, max_depth: int = 2) -> Dict[str, List[CodeChunk]]:
        """Get contextual chunks using graph traversal."""
        logger.debug(f"🔍 Getting context for chunk: {chunk_id} (max_depth: {max_depth})")

        try:
            with self.driver.session() as session:
                logger.debug("🔍 Neo4j session created for context retrieval")

                context: Dict[str, List[CodeChunk]] = {
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
                logger.debug("🔍 Querying for parent chunks")
                result = session.run("""
                    MATCH path = (c:Chunk {id: $chunk_id})-[:CHILD_OF*1..5]->(parent:Chunk)
                    RETURN parent, length(path) as depth
                    ORDER BY depth ASC
                """, chunk_id=chunk_id)

                parent_count = 0
                for record in result:
                    parent_data = record["parent"]
                    parent_chunk = self._record_to_chunk(parent_data)
                    context['parents'].append(parent_chunk)
                    parent_count += 1
                    logger.debug(f"🔍 Found parent: {parent_chunk.id} ({parent_chunk.name})")

                logger.debug(f"🔍 Found {parent_count} parent chunks")

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
        logger.debug(f"📊 Getting graph data - file_path: {file_path}, project_ids: {project_ids}, limit: {limit}")

        try:
            with self.driver.session() as session:
                logger.debug("📊 Neo4j session created for graph data retrieval")

                # Build query based on filters
                where_conditions = []
                params: Dict[str, Any] = {"limit": limit}

                if file_path:
                    where_conditions.append("c.file_path = $file_path")
                    params["file_path"] = file_path
                    logger.debug(f"📊 Added file_path filter: {file_path}")

                if project_ids:
                    where_conditions.append("c.project_id IN $project_ids")
                    params["project_ids"] = project_ids
                    logger.debug(f"📊 Added project_ids filter: {project_ids}")

                where_clause = " AND ".join(where_conditions)
                if where_clause:
                    where_clause = "WHERE " + where_clause

                logger.debug(f"📊 Built where clause: {where_clause}")
                logger.debug(f"📊 Query parameters: {params}")

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
                logger.debug("📊 Executing node query")
                logger.debug(f"📊 Node query: {node_query}")

                nodes = []
                result = session.run(node_query, params)
                node_count = 0

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
                    node_count += 1

                logger.debug(f"📊 Retrieved {node_count} nodes")

                # Get edges
                logger.debug("📊 Executing edge query")
                logger.debug(f"📊 Edge query: {edge_query}")

                edges = []
                result = session.run(edge_query, params)
                edge_count = 0

                for record in result:
                    edge = GraphEdge(
                        source=record["source"],
                        target=record["target"],
                        type=self._map_relationship_type(record["rel_type"]),
                        weight=record.get("weight", 1.0)
                    )
                    edges.append(edge)
                    edge_count += 1

                logger.debug(f"📊 Retrieved {edge_count} edges")

                graph_data = GraphData(
                    nodes=nodes,
                    edges=edges,
                    metadata={
                        "total_nodes": len(nodes),
                        "total_edges": len(edges),
                        "file_path": file_path
                    }
                )

                logger.info(f"✅ Graph data retrieved: {len(nodes)} nodes, {len(edges)} edges")
                logger.debug(f"📊 Graph data metadata: {graph_data.metadata}")
                return graph_data

        except Exception as e:
            logger.error(f"❌ Error getting graph data: {e}")
            logger.debug(f"❌ Graph data error details: {type(e).__name__}: {str(e)}")
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
        logger.debug("🏥 Performing Neo4j health check")

        try:
            with self.driver.session() as session:
                logger.debug("🏥 Executing health check query")
                result = session.run("RETURN 1 as test")
                test_result = result.single()

                if test_result and test_result["test"] == 1:
                    logger.debug("✅ Neo4j health check passed")
                    return True
                else:
                    logger.warning("⚠️ Neo4j health check returned unexpected result")
                    return False

        except Exception as e:
            logger.error(f"❌ Neo4j health check failed: {e}")
            logger.debug(f"❌ Health check error details: {type(e).__name__}: {str(e)}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        logger.debug("📊 Getting Neo4j database statistics")

        try:
            with self.driver.session() as session:
                logger.debug("📊 Neo4j session created for statistics")

                # Get node count
                logger.debug("📊 Querying for node count")
                result = session.run("MATCH (c:Chunk) RETURN count(c) as node_count")
                node_count = result.single()["node_count"]
                logger.debug(f"📊 Found {node_count} nodes")

                # Get relationship count
                logger.debug("📊 Querying for relationship count")
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                logger.debug(f"📊 Found {rel_count} relationships")

                # Get file count
                logger.debug("📊 Querying for file count")
                result = session.run("MATCH (c:Chunk) RETURN count(DISTINCT c.file_path) as file_count")
                file_count = result.single()["file_count"]
                logger.debug(f"📊 Found {file_count} files")

                stats = {
                    "total_chunks": node_count,
                    "total_relationships": rel_count,
                    "total_files": file_count
                }

                logger.info(f"✅ Neo4j statistics: {stats}")
                return stats

        except Exception as e:
            logger.error(f"❌ Error getting Neo4j statistics: {e}")
            logger.debug(f"❌ Statistics error details: {type(e).__name__}: {str(e)}")
            return {}
