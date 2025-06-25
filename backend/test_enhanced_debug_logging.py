#!/usr/bin/env python3
"""
Test script for enhanced debug logging in multi-agent orchestration.

This script tests the new debug logging functionality to ensure that
agent roles and available chunks are properly logged with markdown formatting.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add the backend src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
from src.agents.agent_orchestrator import AgentRole
from src.query.intelligent_query_analyzer import AgentTask, QueryAnalysisResult, QueryComplexity, ProcessingStrategy
from src.models import CodeChunk, NodeType


def create_mock_chunks() -> List[CodeChunk]:
    """Create mock code chunks for testing."""
    return [
        CodeChunk(
            id="chunk_1",
            file_path="backend/src/models/user.py",
            start_line=1,
            end_line=50,
            content="class User:\n    def __init__(self, name):\n        self.name = name",
            node_type=NodeType.CLASS,
            project_id="test_project"
        ),
        CodeChunk(
            id="chunk_2",
            file_path="backend/src/services/auth_service.py",
            start_line=10,
            end_line=80,
            content="def authenticate_user(username, password):\n    # Authentication logic",
            node_type=NodeType.FUNCTION,
            project_id="test_project"
        ),
        CodeChunk(
            id="chunk_3",
            file_path="frontend/src/components/LoginForm.js",
            start_line=1,
            end_line=100,
            content="const LoginForm = () => {\n    // React component",
            node_type=NodeType.FUNCTION,  # Using FUNCTION for JS component
            project_id="test_project"
        ),
        CodeChunk(
            id="chunk_4",
            file_path="backend/src/config/database.py",
            start_line=1,
            end_line=30,
            content="DATABASE_CONFIG = {\n    'host': 'localhost'",
            node_type=NodeType.VARIABLE,
            project_id="test_project"
        ),
        CodeChunk(
            id="chunk_5",
            file_path="backend/tests/test_auth.py",
            start_line=1,
            end_line=60,
            content="def test_user_authentication():\n    # Test code",
            node_type=NodeType.FUNCTION,
            project_id="test_project"
        )
    ]


def create_mock_agent_tasks() -> List[AgentTask]:
    """Create mock agent tasks for testing."""
    return [
        AgentTask(
            agent_role=AgentRole.ARCHITECT,
            task_description="Analyze system architecture and design patterns",
            priority=1,
            estimated_chunks_needed=15,
            specific_focus_areas=["system_design", "patterns", "structure"]
        ),
        AgentTask(
            agent_role=AgentRole.SECURITY,
            task_description="Review authentication and security mechanisms",
            priority=2,
            estimated_chunks_needed=12,
            specific_focus_areas=["authentication", "authorization", "security"]
        ),
        AgentTask(
            agent_role=AgentRole.DEVELOPER,
            task_description="Examine code quality and implementation details",
            priority=1,
            estimated_chunks_needed=18,
            specific_focus_areas=["code_quality", "implementation", "best_practices"]
        )
    ]


async def test_enhanced_debug_logging():
    """Test the enhanced debug logging functionality."""
    print("🧪 Testing Enhanced Debug Logging for Multi-Agent Orchestration")
    print("=" * 70)
    
    # Create orchestrator
    orchestrator = EnhancedAgentOrchestrator()
    
    # Create mock data
    chunks = create_mock_chunks()
    agent_tasks = create_mock_agent_tasks()
    
    query = "How does the authentication system work and what are the security considerations?"
    context = {
        "project_ids": ["test_project"],
        "user_id": "test_user"
    }
    
    # Create mock analysis result
    analysis_result = QueryAnalysisResult(
        complexity=QueryComplexity.MODERATE,
        processing_strategy=ProcessingStrategy.MULTI_PERSPECTIVE,
        required_agents=agent_tasks,
        estimated_processing_time=30.0,
        should_stream=False,
        search_optimization_hints={"focus": "authentication"},
        explanation="Multi-agent analysis for authentication system review"
    )
    
    print(f"📝 Query: {query}")
    print(f"🎯 Agent Tasks: {len(agent_tasks)}")
    print(f"📦 Available Chunks: {len(chunks)}")
    print("\n" + "=" * 70)
    
    try:
        # Test the orchestration with enhanced logging
        print("🚀 Starting orchestration with enhanced debug logging...")
        
        result = await orchestrator.orchestrate_agents(
            query=query,
            analysis_result=analysis_result,
            context=context,
            stream_id=None  # No streaming for this test
        )
        
        print("\n" + "=" * 70)
        print("✅ Orchestration completed successfully!")
        
        if result:
            print(f"📊 Result Summary:")
            print(f"   - Successful agents: {len([r for r in result.agent_results if r.success])}")
            print(f"   - Failed agents: {len([r for r in result.agent_results if not r.success])}")
            print(f"   - Total processing time: {result.total_processing_time:.2f}s")
            print(f"   - Has flow response: {result.flow_response is not None}")
        
        # Display the orchestration logs
        print("\n" + "=" * 70)
        print("📋 Orchestration Debug Logs:")
        print("=" * 70)
        
        for log_entry in orchestrator.orchestration_logs:
            print(log_entry)
            
    except Exception as e:
        print(f"❌ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()


def test_chunk_distribution_logging():
    """Test the chunk distribution logging functionality."""
    print("\n" + "=" * 70)
    print("🧪 Testing Chunk Distribution Logging")
    print("=" * 70)
    
    orchestrator = EnhancedAgentOrchestrator()
    chunks = create_mock_chunks()
    
    # Create mock chunk distribution
    chunk_distribution = {
        AgentRole.ARCHITECT: chunks[:2],
        AgentRole.SECURITY: chunks[1:3],
        AgentRole.DEVELOPER: chunks[2:5]
    }
    
    print("📊 Testing chunk distribution summary logging...")
    orchestrator._log_chunk_distribution_summary(chunk_distribution)
    
    print("\n📋 Chunk Distribution Logs:")
    print("-" * 50)
    for log_entry in orchestrator.orchestration_logs[-10:]:  # Show last 10 entries
        print(log_entry)


if __name__ == "__main__":
    print("🔍 Enhanced Debug Logging Test Suite")
    print("=" * 70)
    
    # Run the tests
    asyncio.run(test_enhanced_debug_logging())
    test_chunk_distribution_logging()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("Check the output above for detailed debug logging examples.")
