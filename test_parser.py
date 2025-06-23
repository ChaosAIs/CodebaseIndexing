#!/usr/bin/env python3

import sys
import os
sys.path.append('backend/src')

from chunking.chunk_processor import ChunkProcessor

def test_chunk_processor():
    processor = ChunkProcessor()
    
    # Test with our test project
    test_path = "test_project"
    print(f"Testing chunk processor with path: {test_path}")
    
    try:
        chunks = processor.process_codebase(test_path)
        print(f"Found {len(chunks)} files")
        
        total_chunks = 0
        for file_path, file_chunks in chunks.items():
            print(f"File: {file_path} - {len(file_chunks)} chunks")
            total_chunks += len(file_chunks)
            
            for chunk in file_chunks:
                print(f"  - {chunk.node_type.value}: {chunk.name} (lines {chunk.start_line}-{chunk.end_line})")
        
        print(f"Total chunks: {total_chunks}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chunk_processor()
