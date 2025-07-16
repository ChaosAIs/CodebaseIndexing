"""Command-line indexer for processing codebases."""

import asyncio
import click
from collections import Counter
from pathlib import Path
from loguru import logger

from src.chunking.chunk_processor import ChunkProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.database.qdrant_client import QdrantVectorStore
from src.database.neo4j_client import Neo4jGraphStore
from src.config import config


@click.command()
@click.option('--path', '-p', required=True, help='Path to the codebase to index')
@click.option('--model', '-m', default='local', help='Embedding model to use (local/cloud)')
@click.option('--force', '-f', is_flag=True, help='Force reindexing of existing data')
@click.option('--languages', '-l', multiple=True, default=['python'], help='Programming languages to index')
def main(path: str, model: str, force: bool, languages: tuple):
    """Index a codebase for semantic search and graph analysis."""
    asyncio.run(index_codebase(path, model, force, list(languages)))


async def index_codebase(path: str, model: str, force: bool, languages: list):
    """Index a codebase asynchronously."""
    try:
        logger.info(f"Starting indexing of codebase: {path}")
        logger.info(f"Languages: {languages}")
        logger.info(f"Embedding model: {model}")
        logger.info(f"Force reindex: {force}")
        
        # Validate path
        codebase_path = Path(path)
        if not codebase_path.exists():
            logger.error(f"Path does not exist: {path}")
            return
        
        if not codebase_path.is_dir():
            logger.error(f"Path is not a directory: {path}")
            return
        
        # Initialize components
        chunk_processor = ChunkProcessor()
        embedding_generator = EmbeddingGenerator()
        vector_store = QdrantVectorStore()
        graph_store = Neo4jGraphStore()
        
        # Process codebase
        logger.info("Processing codebase with Tree-sitter...")
        all_chunks = chunk_processor.process_codebase(str(codebase_path))
        
        if not all_chunks:
            logger.error("No supported files found in the codebase")
            return
        
        # Flatten chunks
        chunks = []
        for file_chunks in all_chunks.values():
            chunks.extend(file_chunks)
        
        logger.info(f"Extracted {len(chunks)} chunks from {len(all_chunks)} files")
        
        # Generate embeddings
        logger.info(f"Generating embeddings using {model} model...")
        logger.debug(f"Chunk types distribution: {dict(Counter(chunk.node_type.value for chunk in chunks))}")

        embeddings = await embedding_generator.generate_chunk_embeddings(chunks, model)

        logger.info(f"Embedding generation completed for standalone indexer")
        
        if not embeddings:
            logger.error("Failed to generate embeddings")
            return
        
        # Get embedding dimension
        dimension = embedding_generator.get_embedding_dimension(model)
        logger.info(f"Embedding dimension: {dimension}")
        
        # Initialize databases
        logger.info("Initializing databases...")
        await vector_store.initialize_collection(dimension, force)
        await graph_store.initialize_schema()
        
        # Store in Qdrant
        logger.info("Storing embeddings in Qdrant...")
        success = await vector_store.store_chunks(chunks, embeddings)
        if not success:
            logger.error("Failed to store chunks in Qdrant")
            return
        
        # Store in Neo4j
        logger.info("Storing chunks in Neo4j...")
        success = await graph_store.store_chunks(chunks)
        if not success:
            logger.error("Failed to store chunks in Neo4j")
            return
        
        # Create relationships
        logger.info("Creating relationships in Neo4j...")
        success = await graph_store.create_relationships(chunks)
        if not success:
            logger.error("Failed to create relationships in Neo4j")
            return
        
        # Get final statistics
        qdrant_info = await vector_store.get_collection_info()
        neo4j_stats = await graph_store.get_statistics()
        
        logger.info("Indexing completed successfully!")
        logger.info(f"Total files processed: {len(all_chunks)}")
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Qdrant vectors: {qdrant_info.get('points_count', 'unknown')}")
        logger.info(f"Neo4j nodes: {neo4j_stats.get('total_chunks', 'unknown')}")
        logger.info(f"Neo4j relationships: {neo4j_stats.get('total_relationships', 'unknown')}")
        
        # Close connections
        graph_store.close()
        
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise


if __name__ == "__main__":
    main()
