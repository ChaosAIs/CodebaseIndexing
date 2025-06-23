#!/usr/bin/env python3
"""Test the enhanced query system with intelligent analysis."""

import requests
import json

def test_enhanced_query():
    """Test the enhanced query system."""
    try:
        print('🧠 Testing enhanced query system with intelligent analysis...')
        
        query_data = {
            'query': 'tell me the login workflow',
            'model': 'openai',
            'limit': 5,
            'include_context': True
        }
        
        response = requests.post('http://localhost:8000/mcp/query', 
                               json=query_data, 
                               timeout=45)
        
        if response.status_code == 200:
            result = response.json()
            print(f'✅ Enhanced query successful!')
            print(f'Model used: {result.get("model_used")}')
            print(f'Processing time: {result.get("processing_time", 0):.3f}s')
            print(f'Results found: {result.get("total_results", 0)}')
            
            # Check if analysis is included
            analysis = result.get("analysis")
            if analysis:
                print(f'\n🧠 INTELLIGENT ANALYSIS:')
                print(f'📝 Summary: {analysis.get("summary", "N/A")}')
                print(f'\n📖 Detailed Explanation:')
                print(f'{analysis.get("detailed_explanation", "N/A")}')
                
                code_flow = analysis.get("code_flow", [])
                if code_flow:
                    print(f'\n🔄 Code Flow:')
                    for i, step in enumerate(code_flow):
                        print(f'   {i+1}. {step}')
                
                key_components = analysis.get("key_components", [])
                if key_components:
                    print(f'\n🔧 Key Components:')
                    for comp in key_components[:3]:
                        print(f'   • {comp.get("name", "N/A")}: {comp.get("purpose", "N/A")}')
                        print(f'     Location: {comp.get("location", "N/A")}')
                
                recommendations = analysis.get("recommendations", [])
                if recommendations:
                    print(f'\n💡 Recommendations:')
                    for rec in recommendations:
                        print(f'   • {rec}')
            else:
                print(f'\n❌ No analysis found in response')
            
            # Show traditional results for comparison
            print(f'\n📄 Traditional Results:')
            for i, result_item in enumerate(result.get("results", [])[:2]):
                chunk = result_item.get("chunk", {})
                print(f'   {i+1}. {chunk.get("name", "unnamed")} in {chunk.get("file_path", "unknown")}')
                print(f'      Score: {result_item.get("score", 0):.3f}')
                
        else:
            print(f'❌ Enhanced query failed with status {response.status_code}')
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'❌ Error testing enhanced query: {e}')

if __name__ == "__main__":
    test_enhanced_query()
