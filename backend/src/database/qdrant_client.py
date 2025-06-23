"""Qdrant vector database client for storing and searching embeddings."""

from typing import List, Dict, Optional, Any, Tuple
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from loguru import logger

from ..models import CodeChunk, QueryResult
from ..config import config


class QdrantVectorStore:
    """Qdrant vector database client."""
    
    def __init__(self, collection_name: str = "codebase_chunks"):
        """Initialize Qdrant client."""
        self.collection_name = collection_name
        self.client = QdrantClient(
            host=config.database.qdrant_host,
            port=config.database.qdrant_port,
            api_key=config.database.qdrant_api_key,
            https=False  # Use HTTP instead of HTTPS for local development
        )
        self.dimension = None
    
    async def initialize_collection(self, dimension: int, force_recreate: bool = False):
        """Initialize or recreate the collection."""
        self.dimension = dimension
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if collection_exists and force_recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload indexes for better filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="file_path",
                    field_schema=models.KeywordIndexParams()
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="node_type",
                    field_schema=models.KeywordIndexParams()
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="project_id",
                    field_schema=models.KeywordIndexParams()
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="name",
                    field_schema=models.KeywordIndexParams()
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    async def store_chunks(self, chunks: List[CodeChunk], embeddings: Dict[str, List[float]]) -> bool:
        """Store chunks with their embeddings in Qdrant."""
        try:
            points = []
            
            for chunk in chunks:
                if chunk.id not in embeddings:
                    logger.warning(f"No embedding found for chunk {chunk.id}")
                    continue
                
                # Prepare payload
                payload = {
                    "chunk_id": chunk.id,
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
                    "metadata": chunk.metadata
                }
                
                # Create point
                point = PointStruct(
                    id=chunk.id,
                    vector=embeddings[chunk.id],
                    payload=payload
                )
                points.append(point)
            
            # Batch upsert points
            batch_size = config.indexing.batch_size
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks in Qdrant: {e}")
            return False
    
    async def search_similar(self, query_embedding: List[float], limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None,
                           project_ids: Optional[List[str]] = None) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks using vector similarity."""
        try:
            # Build filter conditions
            filter_conditions = None
            conditions = []

            # Add project filtering
            if project_ids:
                from qdrant_client.http.models import MatchAny
                conditions.append(
                    FieldCondition(
                        key="project_id",
                        match=MatchAny(any=project_ids)
                    )
                )

            # Add other filters
            if filters:
                if "file_path" in filters:
                    conditions.append(
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=filters["file_path"])
                        )
                    )

                if "node_type" in filters:
                    conditions.append(
                        FieldCondition(
                            key="node_type",
                            match=MatchValue(value=filters["node_type"])
                        )
                    )

                if "name" in filters:
                    conditions.append(
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=filters["name"])
                        )
                    )

            if conditions:
                filter_conditions = Filter(must=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                with_payload=True
            )
            
            # Convert results to CodeChunk objects
            results = []
            for scored_point in search_result:
                payload = scored_point.payload
                
                chunk = CodeChunk(
                    id=payload["chunk_id"],
                    content=payload["content"],
                    file_path=payload["file_path"],
                    start_line=payload["start_line"],
                    end_line=payload["end_line"],
                    node_type=payload["node_type"],
                    name=payload.get("name"),
                    parent_id=payload.get("parent_id"),
                    project_id=payload.get("project_id"),
                    calls=payload.get("calls", []),
                    called_by=payload.get("called_by", []),
                    imports=payload.get("imports", []),
                    metadata=payload.get("metadata", {})
                )
                
                results.append((chunk, scored_point.score))
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return []
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get a specific chunk by ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True
            )
            
            if not result:
                return None
            
            payload = result[0].payload
            return CodeChunk(
                id=payload["chunk_id"],
                content=payload["content"],
                file_path=payload["file_path"],
                start_line=payload["start_line"],
                end_line=payload["end_line"],
                node_type=payload["node_type"],
                name=payload.get("name"),
                parent_id=payload.get("parent_id"),
                project_id=payload.get("project_id"),
                calls=payload.get("calls", []),
                called_by=payload.get("called_by", []),
                imports=payload.get("imports", []),
                metadata=payload.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    async def delete_chunks_by_file(self, file_path: str) -> bool:
        """Delete all chunks for a specific file."""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="file_path",
                        match=MatchValue(value=file_path)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_condition)
            )
            
            logger.info(f"Deleted chunks for file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunks for file {file_path}: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
