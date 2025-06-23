#!/usr/bin/env python3
"""Debug Tree-sitter parser to see what it's finding."""

import sys
sys.path.append('backend')

from src.parser.tree_sitter_parser import TreeSitterParser
import tree_sitter_python
from tree_sitter import Language, Parser

def debug_tree_sitter():
    """Debug Tree-sitter parsing."""
    
    # Test with simple Python code
    code = '''
def hello_world():
    """A simple function."""
    print("Hello, world!")

class MyClass:
    """A simple class."""
    
    def method(self):
        """A simple method."""
        return "Hello from method"
'''
    
    print("🔍 Debugging Tree-sitter parsing...")
    print(f"Code to parse:\n{code}")
    print("-" * 50)
    
    # Initialize parser
    parser = Parser()
    language = Language(tree_sitter_python.language())
    parser.language = language
    
    # Parse code
    tree = parser.parse(bytes(code, 'utf8'))
    
    print(f"✅ Parsed successfully!")
    print(f"Root node: {tree.root_node}")
    print(f"Root node type: {tree.root_node.type}")
    print(f"Root node children: {len(tree.root_node.children)}")
    
    # Test queries
    queries = {
        'functions': '(function_definition name: (identifier) @name) @function',
        'classes': '(class_definition name: (identifier) @name) @class',
    }
    
    for query_name, query_string in queries.items():
        print(f"\n🔍 Testing query: {query_name}")
        print(f"Query: {query_string}")
        
        try:
            query = language.query(query_string)
            matches = query.matches(tree.root_node)

            print(f"Found {len(matches)} matches:")
            for i, match in enumerate(matches):
                pattern_index, captures = match
                print(f"  {i+1}. Pattern {pattern_index}, Captures: {captures}")
                for capture_name, nodes in captures.items():
                    for node in nodes:
                        content = code[node.start_byte:node.end_byte]
                        print(f"       {capture_name}: {node.type} -> '{content.strip()}'")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Test with our parser
    print(f"\n🧪 Testing with our TreeSitterParser...")
    ts_parser = TreeSitterParser()
    
    # Write test file
    with open('test_file.py', 'w') as f:
        f.write(code)
    
    result = ts_parser.parse_file('test_file.py')
    if result:
        print(f"✅ Parsed with TreeSitterParser!")
        print(f"Chunks found: {len(result['chunks'])}")
        for chunk in result['chunks']:
            print(f"  - {chunk.node_type}: {chunk.name} (lines {chunk.start_line}-{chunk.end_line})")
    else:
        print(f"❌ Failed to parse with TreeSitterParser")

if __name__ == "__main__":
    debug_tree_sitter()
