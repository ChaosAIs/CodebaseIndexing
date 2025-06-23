#!/usr/bin/env python3
"""Test OpenAI integration with the codebase indexing system."""

import requests
import json

def test_openai_integration():
    """Test OpenAI integration by making a simple query."""
    try:
        print('🧪 Testing OpenAI integration...')
        
        # Test system status first
        print('📊 Checking system status...')
        status_response = requests.get('http://localhost:8000/mcp/status', timeout=10)
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            print(f'Available models: {status_data.get("available_models", [])}')
            
            if 'openai' in status_data.get("available_models", []):
                print('✅ OpenAI is available!')
            else:
                print('❌ OpenAI is not available')
                return False
        else:
            print(f'❌ Failed to get system status: {status_response.status_code}')
            return False
        
        # Test query (this will fail if no data is indexed, but will test OpenAI connectivity)
        print('🔍 Testing query with OpenAI...')
        query_data = {
            'query': 'test query for OpenAI',
            'model': 'openai',
            'limit': 1,
            'include_context': False
        }
        
        response = requests.post('http://localhost:8000/mcp/query', 
                               json=query_data, 
                               timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ OpenAI query successful!')
            print(f'Model used: {result.get("model_used")}')
            print(f'Processing time: {result.get("processing_time", 0):.3f}s')
            print(f'Results found: {result.get("total_results", 0)}')
            return True
        else:
            print(f'❌ Query failed with status {response.status_code}')
            print(f'Error: {response.text}')
            return False
            
    except Exception as e:
        print(f'❌ Error testing OpenAI: {e}')
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    if success:
        print('\n🎉 OpenAI integration is working correctly!')
    else:
        print('\n❌ OpenAI integration test failed.')
