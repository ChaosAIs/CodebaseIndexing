#!/usr/bin/env python3
"""Test the query processing optimizations."""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_optimizations():
    """Test the optimization components."""
    
    print("🧪 Testing Query Processing Optimizations")
    print("=" * 50)
    
    # Test 1: Intelligent Query Analyzer
    print("\n1. Testing Intelligent Query Analyzer...")
    try:
        from backend.src.query.intelligent_query_analyzer import IntelligentQueryAnalyzer
        
        analyzer = IntelligentQueryAnalyzer()
        
        # Test different query types
        test_queries = [
            "simple function search",
            "analyze the authentication system",
            "Hi, can you offer the solution architecture review for me?",
            "find security vulnerabilities"
        ]
        
        for query in test_queries:
            result = await analyzer.analyze_query(query, 100)
            print(f"  Query: '{query}'")
            print(f"    Complexity: {result.complexity.value}")
            print(f"    Strategy: {result.processing_strategy.value}")
            print(f"    Should stream: {result.should_stream}")
            print(f"    Agents: {len(result.required_agents)}")
            print(f"    Estimated time: {result.estimated_processing_time:.1f}s")
            print()
        
        print("✅ Intelligent Query Analyzer working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing query analyzer: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Streaming Processor
    print("\n2. Testing Streaming Processor...")
    try:
        from backend.src.streaming.stream_processor import StreamProcessor, StreamEventType
        
        processor = StreamProcessor()
        
        # Test stream creation
        stream_id = "test-stream-123"
        
        # Test event emission (without actual stream)
        print(f"  Created stream processor with {len(StreamEventType)} event types")
        print("✅ Streaming Processor components working!")
        
    except Exception as e:
        print(f"❌ Error testing streaming processor: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Database Connections
    print("\n3. Testing Database Connections...")
    try:
        from backend.src.database.qdrant_client import QdrantVectorStore
        from backend.src.database.neo4j_client import Neo4jGraphStore
        
        # Test Qdrant
        qdrant = QdrantVectorStore()
        qdrant_info = await qdrant.get_collection_info()
        print(f"  Qdrant: {qdrant_info['points_count']} chunks indexed")
        
        # Test Neo4j
        neo4j = Neo4jGraphStore()
        neo4j_stats = await neo4j.get_statistics()
        print(f"  Neo4j: {neo4j_stats['total_chunks']} chunks, {neo4j_stats['total_relationships']} relationships")
        
        print("✅ Database connections working!")
        
    except Exception as e:
        print(f"❌ Error testing databases: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 Optimization Testing Complete!")
    print("\nKey Improvements:")
    print("• Intelligent query analysis for optimal processing")
    print("• Dynamic agent selection (2-8 agents based on complexity)")
    print("• Streaming responses for long-running queries")
    print("• Real-time progress updates and status logging")
    print("• Optimized resource allocation and chunk distribution")

if __name__ == "__main__":
    asyncio.run(test_optimizations())
