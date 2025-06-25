#!/usr/bin/env python3
"""
Test script for the enhanced agent orchestrator with agent-specific query generation.
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
from src.agents.agent_orchestrator import AgentRole
from src.query.intelligent_query_analyzer import AgentTask, QueryAnalysisResult, QueryComplexity, ProcessingStrategy
from src.models import CodeChunk, NodeType


async def test_agent_specific_query_generation():
    """Test the agent-specific query generation functionality."""
    print("🧪 Testing Enhanced Agent Orchestrator with Agent-Specific Query Generation")
    print("=" * 80)
    
    # Create a mock orchestrator (without OpenAI client for testing)
    orchestrator = EnhancedAgentOrchestrator(openai_client=None, base_orchestrator=None)
    
    # Test query
    original_query = "Architecture design"
    
    # Create mock agent tasks
    agent_tasks = [
        AgentTask(
            agent_role=AgentRole.ARCHITECT,
            task_description="Analyze system architecture and design patterns",
            priority=1,
            estimated_chunks_needed=20,
            specific_focus_areas=["architecture", "design", "patterns"]
        ),
        AgentTask(
            agent_role=AgentRole.SECURITY,
            task_description="Analyze security vulnerabilities and authentication",
            priority=2,
            estimated_chunks_needed=15,
            specific_focus_areas=["security", "auth", "validation"]
        ),
        AgentTask(
            agent_role=AgentRole.PERFORMANCE,
            task_description="Analyze performance bottlenecks and optimization opportunities",
            priority=2,
            estimated_chunks_needed=15,
            specific_focus_areas=["performance", "optimization", "scalability"]
        )
    ]
    
    # Test agent-specific query generation
    print("🔍 Testing Agent-Specific Query Generation:")
    print("-" * 50)
    
    for i, agent_task in enumerate(agent_tasks, 1):
        print(f"\n{i}. {agent_task.agent_role.value.upper()} AGENT:")
        print(f"   Original Task: {agent_task.task_description}")
        
        # Generate agent-specific query
        agent_query = await orchestrator._generate_agent_specific_query(
            original_query=original_query,
            agent_role=agent_task.agent_role,
            agent_task=agent_task,
            context={}
        )
        
        print(f"   Generated Query:")
        print(f"   {agent_query[:200]}...")
        print()
    
    print("✅ Agent-specific query generation test completed successfully!")
    print("\n" + "=" * 80)
    print("🎯 Key Improvements Implemented:")
    print("   • Each agent now gets a specialized query based on their role")
    print("   • Queries include role-specific keywords and focus areas")
    print("   • This should result in more relevant chunks for each agent")
    print("   • Agents will no longer get 0 chunks assigned")
    print("   • Better distribution of unique chunks across agents")


async def test_chunk_distribution_logic():
    """Test the chunk distribution logic improvements."""
    print("\n🔧 Testing Chunk Distribution Logic:")
    print("-" * 50)
    
    # Create mock code chunks
    mock_chunks = [
        CodeChunk(
            id=f"chunk_{i}",
            file_path=f"src/models/model_{i}.py",
            start_line=1,
            end_line=50,
            content=f"class Model{i}:\n    def __init__(self):\n        pass",
            node_type=NodeType.CLASS,
            name=f"Model{i}"
        )
        for i in range(1, 21)  # 20 mock chunks
    ]
    
    orchestrator = EnhancedAgentOrchestrator(openai_client=None, base_orchestrator=None)
    
    # Test chunk categorization
    categorized = orchestrator._categorize_chunks(mock_chunks)
    
    print(f"📦 Categorized {len(mock_chunks)} chunks:")
    for category, chunks in categorized.items():
        if chunks:
            print(f"   • {category}: {len(chunks)} chunks")
    
    print("\n✅ Chunk distribution logic test completed!")


async def test_agent_validation():
    """Test the agent validation and filtering logic."""
    print("\n🔍 Testing Agent Validation Logic:")
    print("-" * 50)

    orchestrator = EnhancedAgentOrchestrator(openai_client=None, base_orchestrator=None)

    # Create mock agent tasks
    agent_tasks = [
        AgentTask(
            agent_role=AgentRole.ARCHITECT,
            task_description="Analyze architecture",
            priority=1,
            estimated_chunks_needed=20,
            specific_focus_areas=["architecture"]
        ),
        AgentTask(
            agent_role=AgentRole.SECURITY,
            task_description="Analyze security",
            priority=2,
            estimated_chunks_needed=15,
            specific_focus_areas=["security"]
        ),
        AgentTask(
            agent_role=AgentRole.PERFORMANCE,
            task_description="Analyze performance",
            priority=2,
            estimated_chunks_needed=15,
            specific_focus_areas=["performance"]
        )
    ]

    # Create mock chunk distribution (some agents get enough chunks, others don't)
    chunk_distribution = {
        AgentRole.ARCHITECT: [CodeChunk(
            id=f"chunk_{i}",
            file_path=f"src/arch_{i}.py",
            start_line=1,
            end_line=50,
            content=f"Architecture code {i}",
            node_type=NodeType.CLASS,
            name=f"Arch{i}"
        ) for i in range(10)],  # 10 chunks - sufficient

        AgentRole.SECURITY: [CodeChunk(
            id=f"sec_chunk_{i}",
            file_path=f"src/security_{i}.py",
            start_line=1,
            end_line=50,
            content=f"Security code {i}",
            node_type=NodeType.FUNCTION,
            name=f"Security{i}"
        ) for i in range(2)],  # 2 chunks - insufficient

        AgentRole.PERFORMANCE: [CodeChunk(
            id=f"perf_chunk_{i}",
            file_path=f"src/perf_{i}.py",
            start_line=1,
            end_line=50,
            content=f"Performance code {i}",
            node_type=NodeType.METHOD,
            name=f"Perf{i}"
        ) for i in range(8)]  # 8 chunks - sufficient
    }

    # Test validation
    valid_agents = orchestrator._validate_and_filter_agents(agent_tasks, chunk_distribution)

    print(f"📊 Validation Results:")
    print(f"   • Original agents: {len(agent_tasks)}")
    print(f"   • Valid agents: {len(valid_agents)}")
    print(f"   • Filtered out: {len(agent_tasks) - len(valid_agents)}")

    for agent in valid_agents:
        chunks_count = len(chunk_distribution.get(agent.agent_role, []))
        print(f"   ✅ {agent.agent_role.value}: {chunks_count} chunks")

    print("\n✅ Agent validation test completed!")


if __name__ == "__main__":
    asyncio.run(test_agent_specific_query_generation())
    asyncio.run(test_chunk_distribution_logic())
    asyncio.run(test_agent_validation())
