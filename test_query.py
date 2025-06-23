#!/usr/bin/env python3
"""Test querying the indexed codebase."""

import requests
import json

def test_query():
    """Test the indexed codebase with a query."""
    try:
        print('🔍 Testing query on indexed codebase...')
        
        query_data = {
            'query': 'authentication function',
            'model': 'openai',
            'limit': 5,
            'include_context': True
        }
        
        response = requests.post('http://localhost:8000/mcp/query', 
                               json=query_data, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Query successful!')
            print(f'Model used: {result.get("model_used")}')
            print(f'Processing time: {result.get("processing_time", 0):.3f}s')
            print(f'Results found: {result.get("total_results", 0)}')
            
            for i, chunk in enumerate(result.get("results", [])[:3]):
                print(f'\n📄 Result {i+1}:')
                print(f'   File: {chunk.get("file_path")}')
                print(f'   Name: {chunk.get("name")}')
                print(f'   Type: {chunk.get("node_type")}')
                print(f'   Score: {chunk.get("score", 0):.3f}')
                print(f'   Lines: {chunk.get("start_line")}-{chunk.get("end_line")}')
                
                # Show a snippet of the content
                content = chunk.get("content", "")
                if content:
                    lines = content.split('\n')
                    preview = '\n'.join(lines[:3])
                    if len(lines) > 3:
                        preview += '\n   ...'
                    print(f'   Preview:\n   {preview.replace(chr(10), chr(10) + "   ")}')
        else:
            print(f'❌ Query failed with status {response.status_code}')
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'❌ Error testing query: {e}')

if __name__ == "__main__":
    test_query()
