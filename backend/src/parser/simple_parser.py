"""Simple fallback parser that doesn't rely on Tree-sitter."""

import os
import re
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger

from ..models import CodeChunk, NodeType
from ..config import config


class SimpleParser:
    """Simple regex-based parser for basic code structure extraction."""
    
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx'}
    
    def __init__(self):
        """Initialize the simple parser."""
        pass
    
    def get_supported_files(self, directory: str) -> List[str]:
        """Get all supported source files in directory, excluding common directories to ignore."""
        supported_files = []

        # Get directories to exclude from configuration
        excluded_dirs = config.indexing.excluded_dirs

        for root, dirs, files in os.walk(directory):
            # Remove excluded directories from dirs list to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                    supported_files.append(os.path.join(root, file))

        return supported_files
    
    def parse_file(self, file_path: str) -> Optional[Dict]:
        """Parse a single file and extract basic code structures."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect language
            language = self._detect_language(file_path)
            if not language:
                return None
            
            # Extract chunks based on language
            chunks = []
            if language == 'python':
                chunks = self._parse_python(content, file_path)
            elif language in ['javascript', 'typescript']:
                chunks = self._parse_javascript(content, file_path)
            
            result = {
                'file_path': file_path,
                'language': language,
                'content': content,
                'chunks': chunks,
                'call_graph': {}  # Simple parser doesn't build call graphs
            }
            
            logger.info(f"Parsed {file_path}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.py':
            return 'python'
        elif ext in ['.js', '.jsx']:
            return 'javascript'
        elif ext in ['.ts', '.tsx']:
            return 'typescript'
        return None
    
    def _parse_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python code using regex patterns."""
        chunks = []
        lines = content.split('\n')
        
        # Find class definitions
        class_pattern = r'^class\s+(\w+).*?:'
        for i, line in enumerate(lines):
            match = re.match(class_pattern, line.strip())
            if match:
                class_name = match.group(1)
                start_line = i + 1
                end_line = self._find_block_end(lines, i)
                
                chunk_content = '\n'.join(lines[i:end_line])
                chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
                
                chunk = CodeChunk(
                    id=chunk_id,
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    node_type=NodeType.CLASS,
                    name=class_name,
                    metadata={'language': 'python'}
                )
                chunks.append(chunk)
        
        # Find function definitions
        func_pattern = r'^def\s+(\w+).*?:'
        for i, line in enumerate(lines):
            match = re.match(func_pattern, line.strip())
            if match:
                func_name = match.group(1)
                start_line = i + 1
                end_line = self._find_block_end(lines, i)
                
                chunk_content = '\n'.join(lines[i:end_line])
                chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
                
                # Determine if it's a method (inside a class) or function
                node_type = NodeType.METHOD if self._is_inside_class(lines, i) else NodeType.FUNCTION
                
                chunk = CodeChunk(
                    id=chunk_id,
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    node_type=node_type,
                    name=func_name,
                    metadata={'language': 'python'}
                )
                chunks.append(chunk)
        
        return chunks
    
    def _parse_javascript(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript code using regex patterns."""
        chunks = []
        lines = content.split('\n')
        
        # Find class definitions
        class_pattern = r'^class\s+(\w+)'
        for i, line in enumerate(lines):
            match = re.match(class_pattern, line.strip())
            if match:
                class_name = match.group(1)
                start_line = i + 1
                end_line = self._find_js_block_end(lines, i)
                
                chunk_content = '\n'.join(lines[i:end_line])
                chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
                
                chunk = CodeChunk(
                    id=chunk_id,
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    node_type=NodeType.CLASS,
                    name=class_name,
                    metadata={'language': 'javascript'}
                )
                chunks.append(chunk)
        
        # Find function definitions
        func_patterns = [
            r'^function\s+(\w+)',
            r'^const\s+(\w+)\s*=\s*\(',
            r'^(\w+)\s*\(',  # Method definitions
        ]
        
        for i, line in enumerate(lines):
            for pattern in func_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    func_name = match.group(1)
                    start_line = i + 1
                    end_line = self._find_js_block_end(lines, i)
                    
                    chunk_content = '\n'.join(lines[i:end_line])
                    chunk_id = self._generate_chunk_id(file_path, start_line, end_line, chunk_content)
                    
                    chunk = CodeChunk(
                        id=chunk_id,
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        node_type=NodeType.FUNCTION,
                        name=func_name,
                        metadata={'language': 'javascript'}
                    )
                    chunks.append(chunk)
                    break
        
        return chunks
    
    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a Python block based on indentation."""
        if start_idx >= len(lines):
            return len(lines)
        
        start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':  # Skip empty lines
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= start_indent and line.strip():
                return i
        
        return len(lines)
    
    def _find_js_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a JavaScript block based on braces."""
        if start_idx >= len(lines):
            return len(lines)
        
        brace_count = 0
        found_opening = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    found_opening = True
                elif char == '}':
                    brace_count -= 1
                    if found_opening and brace_count == 0:
                        return i + 1
        
        return len(lines)
    
    def _is_inside_class(self, lines: List[str], func_idx: int) -> bool:
        """Check if a function is inside a class definition."""
        func_indent = len(lines[func_idx]) - len(lines[func_idx].lstrip())
        
        for i in range(func_idx - 1, -1, -1):
            line = lines[i]
            if line.strip() == '':
                continue
            
            line_indent = len(line) - len(line.lstrip())
            if line_indent < func_indent and line.strip().startswith('class '):
                return True
            elif line_indent <= func_indent and not line.strip().startswith('class '):
                return False
        
        return False
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int, content: str) -> str:
        """Generate unique ID for a chunk."""
        data = f"{file_path}:{start_line}:{end_line}:{content}"
        return hashlib.md5(data.encode()).hexdigest()
