"""Tests for the Tree-sitter parser."""

import pytest
import tempfile
import os
from pathlib import Path

from src.parser.tree_sitter_parser import TreeSitterParser
from src.parser.simple_parser import SimpleParser
from src.models import NodeType


class TestTreeSitterParser:
    """Test cases for TreeSitterParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()
    
    def test_detect_language(self):
        """Test language detection from file extensions."""
        assert self.parser.detect_language("test.py") == "python"
        assert self.parser.detect_language("test.js") == "javascript"
        assert self.parser.detect_language("test.ts") == "typescript"
        assert self.parser.detect_language("test.unknown") is None
    
    def test_parse_python_file(self):
        """Test parsing a Python file."""
        python_code = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello"

class TestClass:
    """A test class."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        """Get the value."""
        return self.value
    
    def call_hello(self):
        """Call the hello function."""
        return hello_world()
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_file = f.name
        
        try:
            result = self.parser.parse_file(temp_file)
            
            assert result is not None
            assert result['language'] == 'python'
            assert len(result['chunks']) > 0
            
            # Check for function and class chunks
            chunk_names = [chunk.name for chunk in result['chunks'] if chunk.name]
            assert 'hello_world' in chunk_names
            assert 'TestClass' in chunk_names
            assert 'get_value' in chunk_names
            
            # Check node types
            node_types = [chunk.node_type for chunk in result['chunks']]
            assert NodeType.FUNCTION in node_types
            assert NodeType.CLASS in node_types
            
        finally:
            os.unlink(temp_file)
    
    def test_get_supported_files(self):
        """Test getting supported files from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "test.py").write_text("print('hello')")
            (Path(temp_dir) / "test.js").write_text("console.log('hello');")
            (Path(temp_dir) / "test.txt").write_text("not supported")
            
            supported_files = self.parser.get_supported_files(temp_dir)
            
            assert len(supported_files) == 2
            assert any(f.endswith('test.py') for f in supported_files)
            assert any(f.endswith('test.js') for f in supported_files)
            assert not any(f.endswith('test.txt') for f in supported_files)
    
    def test_chunk_generation(self):
        """Test chunk ID generation and metadata."""
        python_code = '''
def test_function():
    """Test function."""
    x = 1
    y = 2
    return x + y
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_file = f.name
        
        try:
            result = self.parser.parse_file(temp_file)
            
            assert result is not None
            assert len(result['chunks']) > 0
            
            chunk = result['chunks'][0]
            assert chunk.id is not None
            assert chunk.file_path == temp_file
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert chunk.content is not None
            assert chunk.node_type in [NodeType.FUNCTION, NodeType.CLASS, NodeType.METHOD]
            
        finally:
            os.unlink(temp_file)


@pytest.mark.asyncio
async def test_parser_integration():
    """Integration test for parser with multiple files."""
    parser = TreeSitterParser()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple test files
        files = {
            "main.py": '''
from utils import helper_function

def main():
    """Main function."""
    result = helper_function()
    print(result)

if __name__ == "__main__":
    main()
''',
            "utils.py": '''
def helper_function():
    """Helper function."""
    return "Hello from helper!"

class UtilityClass:
    """Utility class."""
    
    def process(self, data):
        """Process data."""
        return data.upper()
''',
            "config.js": '''
const config = {
    apiUrl: "http://localhost:8000",
    timeout: 5000
};

function getConfig() {
    return config;
}

module.exports = { config, getConfig };
'''
        }
        
        for filename, content in files.items():
            (Path(temp_dir) / filename).write_text(content)
        
        # Get supported files
        supported_files = parser.get_supported_files(temp_dir)
        assert len(supported_files) == 3
        
        # Parse all files
        all_results = []
        for file_path in supported_files:
            result = parser.parse_file(file_path)
            if result:
                all_results.append(result)
        
        assert len(all_results) == 3
        
        # Check that we have chunks from all files
        total_chunks = sum(len(result['chunks']) for result in all_results)
        assert total_chunks > 0
        
        # Check languages are detected correctly
        languages = [result['language'] for result in all_results]
        assert 'python' in languages
        assert 'javascript' in languages


@pytest.mark.asyncio
async def test_node_modules_exclusion():
    """Test that node_modules and other excluded directories are ignored."""
    parser = TreeSitterParser()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main source files
        (Path(temp_dir) / "main.js").write_text('''
function main() {
    console.log("Main application");
}
''')
        (Path(temp_dir) / "utils.py").write_text('''
def utility_function():
    return "utility"
''')

        # Create node_modules directory with files that should be excluded
        node_modules_dir = Path(temp_dir) / "node_modules"
        node_modules_dir.mkdir()
        (node_modules_dir / "react" / "index.js").parent.mkdir(parents=True)
        (node_modules_dir / "react" / "index.js").write_text('''
// This should be excluded from indexing
module.exports = require('./lib/React');
''')
        (node_modules_dir / "lodash" / "lodash.js").parent.mkdir(parents=True)
        (node_modules_dir / "lodash" / "lodash.js").write_text('''
// This should also be excluded
function _() { return {}; }
''')

        # Create other excluded directories
        excluded_dirs = ['__pycache__', '.git', 'venv', 'build', 'dist']
        for excluded_dir in excluded_dirs:
            excluded_path = Path(temp_dir) / excluded_dir
            excluded_path.mkdir()
            (excluded_path / "excluded_file.py").write_text("# This should be excluded")

        # Get supported files
        supported_files = parser.get_supported_files(temp_dir)

        # Should only find the main source files, not the excluded ones
        assert len(supported_files) == 2

        # Verify the correct files are found
        file_names = [Path(f).name for f in supported_files]
        assert "main.js" in file_names
        assert "utils.py" in file_names

        # Verify no files from excluded directories are found
        for file_path in supported_files:
            assert "node_modules" not in file_path
            assert "__pycache__" not in file_path
            assert ".git" not in file_path
            assert "venv" not in file_path
            assert "build" not in file_path
            assert "dist" not in file_path


@pytest.mark.asyncio
async def test_simple_parser_node_modules_exclusion():
    """Test that SimpleParser also excludes node_modules and other directories."""
    parser = SimpleParser()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create main source files
        (Path(temp_dir) / "main.ts").write_text('''
class MainApp {
    run() {
        console.log("Running app");
    }
}
''')
        (Path(temp_dir) / "helper.py").write_text('''
def helper():
    return "help"
''')

        # Create node_modules with TypeScript/JavaScript files
        node_modules_dir = Path(temp_dir) / "node_modules"
        node_modules_dir.mkdir()
        (node_modules_dir / "@types" / "node" / "index.d.ts").parent.mkdir(parents=True)
        (node_modules_dir / "@types" / "node" / "index.d.ts").write_text('''
// TypeScript definitions - should be excluded
declare module "fs" {
    export function readFile(path: string): string;
}
''')

        # Get supported files
        supported_files = parser.get_supported_files(temp_dir)

        # Should only find the main source files
        assert len(supported_files) == 2

        # Verify correct files are found
        file_names = [Path(f).name for f in supported_files]
        assert "main.ts" in file_names
        assert "helper.py" in file_names

        # Verify no node_modules files are included
        for file_path in supported_files:
            assert "node_modules" not in file_path
