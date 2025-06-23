#!/usr/bin/env python3
"""Test frontend accessibility."""

import requests

def test_frontend():
    """Test if frontend is accessible."""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        print(f'✅ Frontend is accessible! Status: {response.status_code}')
        print('🔄 Frontend should now display the enhanced analysis!')
        print('📝 Try asking: "tell me the login workflow" to see the new format')
        return True
    except Exception as e:
        print(f'❌ Frontend not accessible: {e}')
        return False

if __name__ == "__main__":
    test_frontend()
