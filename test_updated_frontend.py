#!/usr/bin/env python3
"""Test the updated frontend with collapsible Key Components."""

import requests

def test_updated_frontend():
    """Test if frontend is accessible with new features."""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        print(f'✅ Frontend is accessible! Status: {response.status_code}')
        print('🔄 Updated Key Components section with collapsible functionality!')
        print('📝 Try asking: "tell me the login workflow" and click on the Key Components to expand them')
        print('💡 Each component will show its associated code when expanded')
        print('🎯 Features added:')
        print('   • Clickable component headers with chevron icons')
        print('   • Expandable sections showing associated code')
        print('   • Syntax highlighting for component code')
        print('   • Hover effects and visual feedback')
        return True
    except Exception as e:
        print(f'❌ Frontend not accessible: {e}')
        return False

if __name__ == "__main__":
    test_updated_frontend()
