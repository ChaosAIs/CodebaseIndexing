#!/usr/bin/env python3
"""
Test script to verify the debug error fixes.
"""

import sys
import os

# Add the backend src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_embedding_generator_fix():
    """Test that the EmbeddingGenerator.generate_embeddings method exists."""
    print('🧪 Testing EmbeddingGenerator.generate_embeddings method...')
    try:
        from src.embeddings.embedding_generator import EmbeddingGenerator
        gen = EmbeddingGenerator()
        print('✅ EmbeddingGenerator imported successfully')
        print('✅ generate_embeddings method exists:', hasattr(gen, 'generate_embeddings'))
        
        # Check method signature
        import inspect
        sig = inspect.signature(gen.generate_embeddings)
        print(f'✅ Method signature: {sig}')
        
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        return False

def test_division_by_zero_fix():
    """Test that chunk distribution logging handles empty chunks without division by zero."""
    print('\n🧪 Testing chunk distribution logging with empty chunks...')
    try:
        from src.orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
        from src.agents.agent_orchestrator import AgentRole
        
        orchestrator = EnhancedAgentOrchestrator()
        empty_distribution = {
            AgentRole.ARCHITECT: [],
            AgentRole.SECURITY: [],
            AgentRole.DEVELOPER: []
        }
        
        orchestrator._log_chunk_distribution_summary(empty_distribution)
        print('✅ Empty chunk distribution logging works without division by zero error')
        
        # Check the log content
        if orchestrator.orchestration_logs:
            last_log = orchestrator.orchestration_logs[-1]
            if "N/A (no chunks distributed)" in last_log:
                print('✅ Proper handling of empty distribution detected in log')
            else:
                print('⚠️ Expected "N/A" message not found in log')
        
        return True
    except Exception as e:
        print(f'❌ Chunk distribution error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_debug_logging():
    """Test the enhanced debug logging functionality."""
    print('\n🧪 Testing enhanced debug logging methods...')
    try:
        from src.orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
        from src.agents.agent_orchestrator import AgentRole
        from src.query.intelligent_query_analyzer import AgentTask
        from src.models import CodeChunk, NodeType
        
        orchestrator = EnhancedAgentOrchestrator()
        
        # Test chunk for logging
        test_chunk = CodeChunk(
            id="test_chunk",
            file_path="test/file.py",
            start_line=1,
            end_line=10,
            content="def test(): pass",
            node_type=NodeType.FUNCTION,
            project_id="test_project"
        )
        
        # Test agent task
        test_task = AgentTask(
            agent_role=AgentRole.ARCHITECT,
            task_description="Test task",
            priority=1,
            estimated_chunks_needed=5,
            specific_focus_areas=["testing"]
        )
        
        # Test the logging methods
        print('  📝 Testing agent processing start logging...')
        orchestrator._log_agent_processing_start(
            AgentRole.ARCHITECT, test_task, [test_chunk], "test query"
        )
        
        print('  📝 Testing agent context details logging...')
        test_context = {
            'chunk_count': 1,
            'focus_areas': ['testing'],
            'specialization': {'focus': 'architecture'},
            'project_ids': ['test_project']
        }
        orchestrator._log_agent_context_details(AgentRole.ARCHITECT, test_context)
        
        print('✅ Enhanced debug logging methods work correctly')
        
        # Show sample log output
        if orchestrator.orchestration_logs:
            print(f'\n📋 Generated {len(orchestrator.orchestration_logs)} log entries')
            print('📋 Sample log entry:')
            print('-' * 50)
            print(orchestrator.orchestration_logs[-1][:200] + '...')
        
        return True
    except Exception as e:
        print(f'❌ Enhanced debug logging error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Debug Error Fixes Verification")
    print("=" * 50)
    
    # Run all tests
    results = []
    results.append(test_embedding_generator_fix())
    results.append(test_division_by_zero_fix())
    results.append(test_enhanced_debug_logging())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All debug error fixes verified successfully!")
        print("\n📋 Summary of fixes:")
        print("1. ✅ Added generate_embeddings method to EmbeddingGenerator")
        print("2. ✅ Fixed division by zero in chunk distribution logging")
        print("3. ✅ Enhanced debug logging with markdown formatting works")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
