"""Tests for the MCP server."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.mcp_server.server import app
from src.models import QueryResponse, SystemStatus, GraphData


class TestMCPServer:
    """Test cases for MCP server endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "MCP Server"
    
    @patch('src.mcp_server.server.mcp_server.vector_store')
    @patch('src.mcp_server.server.mcp_server.graph_store')
    @patch('src.mcp_server.server.mcp_server.embedding_generator')
    def test_system_status_endpoint(self, mock_embedding, mock_graph, mock_vector):
        """Test system status endpoint."""
        # Mock the health checks
        mock_vector.health_check.return_value = True
        mock_graph.health_check.return_value = True
        mock_embedding.get_available_providers.return_value = {
            'ollama': True,
            'openai': False
        }
        mock_graph.get_statistics.return_value = {
            'total_files': 10,
            'total_chunks': 100
        }
        
        response = self.client.get("/mcp/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["qdrant_status"] == "healthy"
        assert data["neo4j_status"] == "healthy"
        assert "ollama" in data["available_models"]
        assert data["indexed_files"] == 10
        assert data["total_chunks"] == 100
    
    @patch('src.mcp_server.server.mcp_server.vector_store')
    @patch('src.mcp_server.server.mcp_server.graph_store')
    @patch('src.mcp_server.server.mcp_server.embedding_generator')
    def test_query_endpoint(self, mock_embedding, mock_graph, mock_vector):
        """Test query endpoint."""
        # Mock embedding generation
        mock_embedding.generate_chunk_embeddings.return_value = {
            'query': [0.1, 0.2, 0.3]
        }
        
        # Mock vector search results
        from src.models import CodeChunk, NodeType
        mock_chunk = CodeChunk(
            id="test-chunk-1",
            content="def test_function():\n    pass",
            file_path="test.py",
            start_line=1,
            end_line=2,
            node_type=NodeType.FUNCTION,
            name="test_function"
        )
        
        mock_vector.search_similar.return_value = [(mock_chunk, 0.95)]
        
        # Mock graph context
        mock_graph.get_chunk_context.return_value = {
            'parents': [],
            'children': [],
            'calls': [],
            'called_by': []
        }
        
        query_data = {
            "query": "find test functions",
            "limit": 5,
            "include_context": True
        }
        
        response = self.client.post("/mcp/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["query"] == "find test functions"
        assert data["total_results"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk"]["name"] == "test_function"
        assert data["results"][0]["score"] == 0.95
    
    @patch('src.mcp_server.server.mcp_server.graph_store')
    def test_graph_endpoint(self, mock_graph):
        """Test graph data endpoint."""
        from src.models import GraphNode, GraphEdge, NodeType, RelationshipType
        
        # Mock graph data
        mock_nodes = [
            GraphNode(
                id="node1",
                label="test_function",
                type=NodeType.FUNCTION,
                file_path="test.py"
            )
        ]
        
        mock_edges = [
            GraphEdge(
                source="node1",
                target="node2",
                type=RelationshipType.CALLS,
                weight=1.0
            )
        ]
        
        mock_graph_data = GraphData(
            nodes=mock_nodes,
            edges=mock_edges
        )
        
        mock_graph.get_graph_data.return_value = mock_graph_data
        
        response = self.client.get("/mcp/graph")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1
        assert data["nodes"][0]["label"] == "test_function"
        assert data["edges"][0]["source"] == "node1"
    
    def test_query_endpoint_validation(self):
        """Test query endpoint input validation."""
        # Test missing query
        response = self.client.post("/mcp/query", json={})
        assert response.status_code == 422
        
        # Test invalid limit
        response = self.client.post("/mcp/query", json={
            "query": "test",
            "limit": -1
        })
        assert response.status_code == 422
    
    @patch('src.mcp_server.server.mcp_server.chunk_processor')
    @patch('src.mcp_server.server.mcp_server.embedding_generator')
    @patch('src.mcp_server.server.mcp_server.vector_store')
    @patch('src.mcp_server.server.mcp_server.graph_store')
    def test_index_endpoint(self, mock_graph, mock_vector, mock_embedding, mock_processor):
        """Test indexing endpoint."""
        from src.models import CodeChunk, NodeType
        
        # Mock chunk processing
        mock_chunks = [
            CodeChunk(
                id="chunk1",
                content="def test():\n    pass",
                file_path="test.py",
                start_line=1,
                end_line=2,
                node_type=NodeType.FUNCTION,
                name="test"
            )
        ]
        
        mock_processor.process_codebase.return_value = {
            "test.py": mock_chunks
        }
        
        # Mock embedding generation
        mock_embedding.generate_chunk_embeddings.return_value = {
            "chunk1": [0.1, 0.2, 0.3]
        }
        mock_embedding.get_embedding_dimension.return_value = 768
        
        # Mock database operations
        mock_vector.initialize_collection.return_value = True
        mock_vector.store_chunks.return_value = True
        mock_graph.initialize_schema.return_value = True
        mock_graph.store_chunks.return_value = True
        mock_graph.create_relationships.return_value = True
        
        index_data = {
            "path": "/test/path",
            "languages": ["python"],
            "force_reindex": False
        }
        
        response = self.client.post("/mcp/index", json=index_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["total_files"] == 1
        assert data["total_chunks"] == 1


@pytest.mark.asyncio
async def test_server_startup_shutdown():
    """Test server startup and shutdown events."""
    from src.mcp_server.server import mcp_server
    
    # Test startup
    with patch.object(mcp_server.graph_store, 'initialize_schema') as mock_init:
        await mcp_server.startup()
        mock_init.assert_called_once()
    
    # Test shutdown
    with patch.object(mcp_server.graph_store, 'close') as mock_close:
        await mcp_server.shutdown()
        mock_close.assert_called_once()
