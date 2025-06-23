"""Tree-sitter based code parser for extracting ASTs and code structures."""

import os
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any
from pathlib import Path
import tree_sitter
from tree_sitter import Parser, Node, Language
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_java
import tree_sitter_cpp
import tree_sitter_c
import tree_sitter_go
import tree_sitter_rust

from ..models import CodeChunk, NodeType
from ..config import config
from loguru import logger


class LanguageParser:
    """Language-specific parser configuration."""
    
    LANGUAGE_MAP = {
        'python': {
            'language': tree_sitter_python.language(),
            'extensions': ['.py'],
            'queries': {
                'functions': '(function_definition name: (identifier) @name) @function',
                'classes': '(class_definition name: (identifier) @name) @class',
                'methods': '(function_definition name: (identifier) @name) @method',
                'calls': '(call function: (identifier) @name) @call',
                'imports': '(import_statement) @import'
            }
        },
        'javascript': {
            'language': tree_sitter_javascript.language(),
            'extensions': ['.js', '.jsx'],
            'queries': {
                'functions': '(function_declaration name: (identifier) @name) @function',
                'classes': '(class_declaration name: (identifier) @name) @class',
                'methods': '(method_definition name: (property_identifier) @name) @method',
                'calls': '(call_expression function: (identifier) @name) @call',
                'imports': '(import_statement) @import'
            }
        },
        'typescript': {
            'language': tree_sitter_typescript.language_typescript(),
            'extensions': ['.ts', '.tsx'],
            'queries': {
                'functions': '(function_declaration name: (identifier) @name) @function',
                'classes': '(class_declaration name: (type_identifier) @name) @class',
                'methods': '(method_definition name: (property_identifier) @name) @method',
                'calls': '(call_expression function: (identifier) @name) @call',
                'imports': '(import_statement) @import'
            }
        }
    }
    
    def __init__(self, language: str):
        """Initialize language parser."""
        if language not in self.LANGUAGE_MAP:
            raise ValueError(f"Unsupported language: {language}")
        
        self.language = language
        self.config = self.LANGUAGE_MAP[language]
        self.parser = Parser(self.config['language'])
        
        # Compile queries
        self.queries = {}
        lang_obj = Language(self.config['language'])
        for query_name, query_string in self.config['queries'].items():
            try:
                self.queries[query_name] = lang_obj.query(query_string)
            except Exception as e:
                logger.warning(f"Failed to compile query {query_name} for {language}: {e}")
    
    def get_extensions(self) -> List[str]:
        """Get file extensions for this language."""
        return self.config['extensions']
    
    def parse(self, code: str) -> tree_sitter.Tree:
        """Parse code into AST."""
        return self.parser.parse(bytes(code, 'utf8'))


class TreeSitterParser:
    """Main Tree-sitter parser for extracting code structures."""
    
    def __init__(self):
        """Initialize the parser."""
        self.language_parsers = {}
        self.supported_languages = list(LanguageParser.LANGUAGE_MAP.keys())
        
        # Initialize language parsers
        for lang in self.supported_languages:
            try:
                self.language_parsers[lang] = LanguageParser(lang)
                logger.info(f"Initialized parser for {lang}")
            except Exception as e:
                logger.error(f"Failed to initialize parser for {lang}: {e}")
    
    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        for lang, parser in self.language_parsers.items():
            if ext in parser.get_extensions():
                return lang
        return None
    
    def get_supported_files(self, directory: str) -> List[str]:
        """Get all supported source files in directory, excluding common directories to ignore."""
        supported_files = []
        supported_extensions = set()

        # Get directories to exclude from configuration
        excluded_dirs = config.indexing.excluded_dirs

        for parser in self.language_parsers.values():
            supported_extensions.update(parser.get_extensions())

        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from dirs list to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    supported_files.append(os.path.join(root, file))

        return supported_files
    
    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parse a single file and extract code structures."""
        try:
            # Detect language
            language = self.detect_language(file_path)
            if not language:
                logger.warning(f"Unsupported file type: {file_path}")
                return None
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse with Tree-sitter
            parser = self.language_parsers[language]
            tree = parser.parse(content)
            
            # Extract structures
            result = {
                'file_path': file_path,
                'language': language,
                'content': content,
                'tree': tree,
                'chunks': [],
                'call_graph': {}
            }
            
            # Extract code chunks
            chunks = self._extract_chunks(tree, content, file_path, language)
            result['chunks'] = chunks
            
            # Build call graph
            call_graph = self._build_call_graph(tree, content, language)
            result['call_graph'] = call_graph
            
            logger.info(f"Parsed {file_path}: {len(chunks)} chunks, {len(call_graph)} calls")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def _extract_chunks(self, tree: tree_sitter.Tree, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Extract code chunks from AST using direct node traversal."""
        chunks = []

        # Use direct node traversal instead of queries
        self._traverse_node(tree.root_node, content, file_path, language, chunks)

        return chunks

    def _traverse_node(self, node: Node, content: str, file_path: str, language: str, chunks: List[CodeChunk]):
        """Traverse AST nodes to find functions, classes, and methods."""
        # Check if this node is a function, class, or method
        if language == 'python':
            if node.type == 'function_definition':
                chunk = self._create_chunk(node, content, file_path, language, 'function')
                if chunk:
                    chunks.append(chunk)
            elif node.type == 'class_definition':
                chunk = self._create_chunk(node, content, file_path, language, 'class')
                if chunk:
                    chunks.append(chunk)

        # Recursively traverse children
        for child in node.children:
            self._traverse_node(child, content, file_path, language, chunks)
    
    def _create_chunk(self, node: Node, content: str, file_path: str, language: str, node_type: str) -> Optional[CodeChunk]:
        """Create a code chunk from a Tree-sitter node."""
        try:
            # Get node boundaries
            start_line = node.start_point[0] + 1  # 1-indexed
            end_line = node.end_point[0] + 1
            
            # Extract content
            lines = content.split('\n')
            chunk_content = '\n'.join(lines[start_line-1:end_line])
            
            # Generate unique ID
            chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
            
            # Extract name
            name = self._extract_node_name(node, content)
            
            # Map node type
            mapped_type = self._map_node_type(node_type)
            
            return CodeChunk(
                id=chunk_id,
                content=chunk_content,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                node_type=mapped_type,
                name=name,
                parent_id=None,  # Will be set later if needed
                metadata={
                    'language': language,
                    'raw_node_type': node.type,
                    'byte_range': (node.start_byte, node.end_byte)
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating chunk: {e}")
            return None
    
    def _extract_node_name(self, node: Node, content: str) -> Optional[str]:
        """Extract the name of a node (function, class, etc.)."""
        try:
            # Look for identifier nodes in children
            for child in node.children:
                if child.type == 'identifier' or 'name' in child.type:
                    return content[child.start_byte:child.end_byte]
                
                # Recursively search in children
                for grandchild in child.children:
                    if grandchild.type == 'identifier':
                        return content[grandchild.start_byte:grandchild.end_byte]
            
            return None
        except Exception:
            return None
    
    def _map_node_type(self, node_type: str) -> NodeType:
        """Map Tree-sitter node type to our NodeType enum."""
        mapping = {
            'function': NodeType.FUNCTION,
            'class': NodeType.CLASS,
            'method': NodeType.METHOD
        }
        return mapping.get(node_type, NodeType.FUNCTION)
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generate unique ID for a chunk."""
        data = f"{file_path}:{start_line}:{end_line}:{content}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _build_call_graph(self, tree: tree_sitter.Tree, content: str, language: str) -> Dict[str, List[str]]:
        """Build call graph from function calls using direct traversal."""
        call_graph = {}

        # For now, return empty call graph to avoid query issues
        # This can be enhanced later with proper call detection
        return call_graph
    
    def _find_containing_function(self, node: Node, content: str) -> Optional[str]:
        """Find the function that contains the given node."""
        current = node.parent
        while current:
            if current.type in ['function_definition', 'method_definition']:
                # Find the function name
                for child in current.children:
                    if child.type == 'identifier':
                        return content[child.start_byte:child.end_byte]
            current = current.parent
        return None
