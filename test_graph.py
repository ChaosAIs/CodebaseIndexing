#!/usr/bin/env python3
"""Test graph data retrieval."""

import requests

def test_graph_data():
    """Test the graph data endpoint."""
    try:
        response = requests.get('http://localhost:8000/mcp/graph', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f'✅ Graph data retrieved!')
            print(f'Nodes: {len(data.get("nodes", []))}')
            print(f'Edges: {len(data.get("edges", []))}')
            print(f'Metadata: {data.get("metadata", {})}')
            
            # Show a few sample nodes
            nodes = data.get('nodes', [])[:3]
            for i, node in enumerate(nodes):
                print(f'\n📊 Node {i+1}:')
                print(f'   ID: {node.get("id")}')
                print(f'   Label: {node.get("label")}')
                print(f'   Type: {node.get("type")}')
                print(f'   File: {node.get("file_path")}')
                
            # Show a few sample edges
            edges = data.get('edges', [])[:3]
            for i, edge in enumerate(edges):
                print(f'\n🔗 Edge {i+1}:')
                print(f'   Source: {edge.get("source")}')
                print(f'   Target: {edge.get("target")}')
                print(f'   Type: {edge.get("type")}')
                print(f'   Weight: {edge.get("weight")}')
        else:
            print(f'❌ Graph data request failed: {response.status_code}')
            print(f'Error: {response.text}')
    except Exception as e:
        print(f'❌ Error getting graph data: {e}')

if __name__ == "__main__":
    test_graph_data()
