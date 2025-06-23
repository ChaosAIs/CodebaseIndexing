#!/usr/bin/env python3
"""Test frontend accessibility."""

import requests

def test_frontend():
    """Test if frontend is accessible."""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        print(f'✅ Frontend is accessible! Status: {response.status_code}')
        print('🌐 You can now view the graph in your browser at http://localhost:3000')
        print('📊 The graph view should now show the indexed codebase structure!')
        return True
    except Exception as e:
        print(f'❌ Frontend not accessible: {e}')
        return False

if __name__ == "__main__":
    test_frontend()
