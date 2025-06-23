"""Chunk processor for creating structured chunks with bidirectional relationships."""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import re
from loguru import logger

from ..models import CodeChunk, NodeType, RelationshipType
from ..parser.tree_sitter_parser import TreeSitterParser
from ..parser.simple_parser import SimpleParser


class ChunkProcessor:
    """Process parsed code into structured chunks with relationships."""
    
    def __init__(self):
        """Initialize the chunk processor."""
        self.parser = TreeSitterParser()
        self.simple_parser = SimpleParser()  # Fallback parser
        self.chunk_registry: Dict[str, CodeChunk] = {}
        self.relationships: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    def process_codebase(self, directory: str) -> Dict[str, List[CodeChunk]]:
        """Process entire codebase and return structured chunks."""
        logger.info(f"Processing codebase: {directory}")
        
        # Get all supported files - try tree-sitter first, fallback to simple parser
        files = self.parser.get_supported_files(directory)
        if not files:
            logger.info("Tree-sitter parser found no files, trying simple parser")
            files = self.simple_parser.get_supported_files(directory)
        logger.info(f"Found {len(files)} supported files")
        
        all_chunks = {}
        file_call_graphs = {}
        
        # Parse each file
        for file_path in files:
            result = self.parser.parse_file(file_path)
            if not result:
                # Try simple parser as fallback
                result = self.simple_parser.parse_file(file_path)

            if result:
                chunks = result['chunks']
                call_graph = result['call_graph']

                # Process chunks for this file
                processed_chunks = self._process_file_chunks(chunks, call_graph, file_path)
                all_chunks[file_path] = processed_chunks
                file_call_graphs[file_path] = call_graph
        
        # Build cross-file relationships
        self._build_cross_file_relationships(all_chunks, file_call_graphs)
        
        # Update chunks with relationship information
        self._update_chunks_with_relationships(all_chunks)
        
        logger.info(f"Processed {len(all_chunks)} files with {sum(len(chunks) for chunks in all_chunks.values())} total chunks")
        return all_chunks
    
    def _process_file_chunks(self, chunks: List[CodeChunk], call_graph: Dict[str, List[str]], file_path: str) -> List[CodeChunk]:
        """Process chunks for a single file."""
        # Register chunks
        for chunk in chunks:
            self.chunk_registry[chunk.id] = chunk
        
        # Build parent-child relationships
        self._build_parent_child_relationships(chunks)
        
        # Build call relationships within file
        self._build_call_relationships(chunks, call_graph)
        
        return chunks
    
    def _build_parent_child_relationships(self, chunks: List[CodeChunk]) -> None:
        """Build parent-child relationships between chunks."""
        # Sort chunks by start line to process in order
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        
        for i, chunk in enumerate(sorted_chunks):
            # Find parent chunk (containing chunk)
            parent = self._find_parent_chunk(chunk, sorted_chunks[:i])
            if parent:
                chunk.parent_id = parent.id
                self.relationships[parent.id][RelationshipType.PARENT_CHILD].append(chunk.id)
    
    def _find_parent_chunk(self, chunk: CodeChunk, potential_parents: List[CodeChunk]) -> Optional[CodeChunk]:
        """Find the parent chunk that contains the given chunk."""
        # Look for the most specific parent (smallest containing chunk)
        best_parent = None
        best_size = float('inf')
        
        for parent in potential_parents:
            # Check if parent contains this chunk
            if (parent.start_line <= chunk.start_line and 
                parent.end_line >= chunk.end_line and
                parent.file_path == chunk.file_path):
                
                # Prefer more specific parents (smaller size)
                size = parent.end_line - parent.start_line
                if size < best_size:
                    best_parent = parent
                    best_size = size
        
        return best_parent
    
    def _build_call_relationships(self, chunks: List[CodeChunk], call_graph: Dict[str, List[str]]) -> None:
        """Build call relationships between chunks."""
        # Create name to chunk mapping
        name_to_chunk = {}
        for chunk in chunks:
            if chunk.name:
                name_to_chunk[chunk.name] = chunk
        
        # Process call graph
        for caller_name, callees in call_graph.items():
            caller_chunk = name_to_chunk.get(caller_name)
            if not caller_chunk:
                continue
            
            for callee_name in callees:
                callee_chunk = name_to_chunk.get(callee_name)
                if callee_chunk:
                    # Add bidirectional call relationship
                    caller_chunk.calls.append(callee_chunk.id)
                    callee_chunk.called_by.append(caller_chunk.id)
                    
                    self.relationships[caller_chunk.id][RelationshipType.CALLS].append(callee_chunk.id)
                    self.relationships[callee_chunk.id][RelationshipType.CALLED_BY].append(caller_chunk.id)
    
    def _build_cross_file_relationships(self, all_chunks: Dict[str, List[CodeChunk]], file_call_graphs: Dict[str, Dict[str, List[str]]]) -> None:
        """Build relationships across files (imports, cross-file calls)."""
        # Build global name registry
        global_name_registry = {}
        for file_path, chunks in all_chunks.items():
            for chunk in chunks:
                if chunk.name:
                    if chunk.name not in global_name_registry:
                        global_name_registry[chunk.name] = []
                    global_name_registry[chunk.name].append(chunk)
        
        # Process cross-file calls
        for file_path, call_graph in file_call_graphs.items():
            file_chunks = all_chunks[file_path]
            file_chunk_names = {chunk.name: chunk for chunk in file_chunks if chunk.name}
            
            for caller_name, callees in call_graph.items():
                caller_chunk = file_chunk_names.get(caller_name)
                if not caller_chunk:
                    continue
                
                for callee_name in callees:
                    # Check if callee is in another file
                    if callee_name not in file_chunk_names and callee_name in global_name_registry:
                        for callee_chunk in global_name_registry[callee_name]:
                            if callee_chunk.file_path != file_path:
                                # Cross-file call
                                caller_chunk.calls.append(callee_chunk.id)
                                callee_chunk.called_by.append(caller_chunk.id)
                                
                                self.relationships[caller_chunk.id][RelationshipType.CALLS].append(callee_chunk.id)
                                self.relationships[callee_chunk.id][RelationshipType.CALLED_BY].append(caller_chunk.id)
    
    def _update_chunks_with_relationships(self, all_chunks: Dict[str, List[CodeChunk]]) -> None:
        """Update chunks with final relationship information."""
        for file_path, chunks in all_chunks.items():
            for chunk in chunks:
                # Remove duplicates
                chunk.calls = list(set(chunk.calls))
                chunk.called_by = list(set(chunk.called_by))
                
                # Update chunk registry
                self.chunk_registry[chunk.id] = chunk
    
    def get_chunk_context(self, chunk_id: str, max_depth: int = 2) -> Dict[str, List[CodeChunk]]:
        """Get contextual chunks for a given chunk."""
        if chunk_id not in self.chunk_registry:
            return {}
        
        context = {
            'parents': [],
            'children': [],
            'calls': [],
            'called_by': []
        }
        
        chunk = self.chunk_registry[chunk_id]
        
        # Get parent
        if chunk.parent_id and chunk.parent_id in self.chunk_registry:
            context['parents'].append(self.chunk_registry[chunk.parent_id])
        
        # Get children
        for rel_type, related_ids in self.relationships[chunk_id].items():
            if rel_type == RelationshipType.PARENT_CHILD:
                for related_id in related_ids:
                    if related_id in self.chunk_registry:
                        context['children'].append(self.chunk_registry[related_id])
        
        # Get calls
        for call_id in chunk.calls:
            if call_id in self.chunk_registry:
                context['calls'].append(self.chunk_registry[call_id])
        
        # Get called_by
        for caller_id in chunk.called_by:
            if caller_id in self.chunk_registry:
                context['called_by'].append(self.chunk_registry[caller_id])
        
        return context
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        """Get chunk by ID."""
        return self.chunk_registry.get(chunk_id)
    
    def get_all_chunks(self) -> List[CodeChunk]:
        """Get all processed chunks."""
        return list(self.chunk_registry.values())
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks for a specific file."""
        return [chunk for chunk in self.chunk_registry.values() if chunk.file_path == file_path]
    
    def get_chunks_by_type(self, node_type: NodeType) -> List[CodeChunk]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunk_registry.values() if chunk.node_type == node_type]
    
    def search_chunks_by_name(self, name_pattern: str) -> List[CodeChunk]:
        """Search chunks by name pattern."""
        pattern = re.compile(name_pattern, re.IGNORECASE)
        return [chunk for chunk in self.chunk_registry.values() 
                if chunk.name and pattern.search(chunk.name)]
