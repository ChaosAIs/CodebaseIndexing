#!/usr/bin/env python3
"""
Simple test for enhanced debug logging functionality.

This test focuses on the debug logging methods without requiring
full orchestration setup.
"""

import sys
import os
from typing import List, Dict, Any

# Add the backend src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestration.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
from src.agents.agent_orchestrator import AgentRole, AgentPerspective
from src.query.intelligent_query_analyzer import AgentTask
from src.models import CodeChunk, NodeType


def create_test_chunks() -> List[CodeChunk]:
    """Create test code chunks."""
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
            file_path="frontend/src/components/LoginForm.tsx",
            start_line=1,
            end_line=100,
            content="const LoginForm = () => {\n    // React component",
            node_type=NodeType.FUNCTION,
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
        ),
        CodeChunk(
            id="chunk_6",
            file_path="backend/src/middleware/auth_middleware.py",
            start_line=5,
            end_line=45,
            content="class AuthMiddleware:\n    def process_request(self, request):",
            node_type=NodeType.CLASS,
            project_id="test_project"
        )
    ]


def create_test_agent_task(role: AgentRole) -> AgentTask:
    """Create a test agent task."""
    task_descriptions = {
        AgentRole.ARCHITECT: "Analyze system architecture and design patterns",
        AgentRole.SECURITY: "Review authentication and security mechanisms",
        AgentRole.DEVELOPER: "Examine code quality and implementation details",
        AgentRole.PERFORMANCE: "Analyze performance bottlenecks and optimization opportunities",
        AgentRole.MAINTAINER: "Review code maintainability and technical debt"
    }
    
    focus_areas = {
        AgentRole.ARCHITECT: ["system_design", "patterns", "structure"],
        AgentRole.SECURITY: ["authentication", "authorization", "security"],
        AgentRole.DEVELOPER: ["code_quality", "implementation", "best_practices"],
        AgentRole.PERFORMANCE: ["optimization", "bottlenecks", "scalability"],
        AgentRole.MAINTAINER: ["maintainability", "technical_debt", "refactoring"]
    }
    
    return AgentTask(
        agent_role=role,
        task_description=task_descriptions.get(role, f"Analyze from {role.value} perspective"),
        priority=1,
        estimated_chunks_needed=10,
        specific_focus_areas=focus_areas.get(role, ["general_analysis"])
    )


def create_test_perspective(role: AgentRole) -> AgentPerspective:
    """Create a test agent perspective."""
    return AgentPerspective(
        role=role,
        analysis=f"Detailed analysis from {role.value} perspective covering multiple aspects of the codebase.",
        key_insights=[
            f"Key insight 1 from {role.value} analysis",
            f"Key insight 2 from {role.value} analysis",
            f"Key insight 3 from {role.value} analysis"
        ],
        recommendations=[
            f"Recommendation 1 from {role.value}",
            f"Recommendation 2 from {role.value}"
        ],
        confidence=0.85,
        focus_areas=["authentication", "security", "architecture"]
    )


def test_agent_processing_start_logging():
    """Test the agent processing start logging."""
    print("🧪 Testing Agent Processing Start Logging")
    print("=" * 60)
    
    orchestrator = EnhancedAgentOrchestrator()
    chunks = create_test_chunks()
    
    # Test different agent roles
    test_roles = [AgentRole.ARCHITECT, AgentRole.SECURITY, AgentRole.DEVELOPER]
    
    for role in test_roles:
        agent_task = create_test_agent_task(role)
        query = "How does the authentication system work and what are the security considerations?"
        
        print(f"\n🤖 Testing {role.value.upper()} Agent Processing Start:")
        print("-" * 50)
        
        orchestrator._log_agent_processing_start(role, agent_task, chunks[:4], query)
        
        # Show the last log entry
        if orchestrator.orchestration_logs:
            print("📋 Generated Log:")
            print(orchestrator.orchestration_logs[-1])


def test_chunk_distribution_logging():
    """Test the chunk distribution logging."""
    print("\n🧪 Testing Chunk Distribution Logging")
    print("=" * 60)
    
    orchestrator = EnhancedAgentOrchestrator()
    chunks = create_test_chunks()
    
    # Create different distribution scenarios
    scenarios = [
        {
            "name": "Balanced Distribution",
            "distribution": {
                AgentRole.ARCHITECT: chunks[:2],
                AgentRole.SECURITY: chunks[2:4],
                AgentRole.DEVELOPER: chunks[4:6]
            }
        },
        {
            "name": "Uneven Distribution",
            "distribution": {
                AgentRole.ARCHITECT: chunks[:1],
                AgentRole.SECURITY: chunks[1:4],
                AgentRole.DEVELOPER: chunks[4:6]
            }
        },
        {
            "name": "Overlapping Distribution",
            "distribution": {
                AgentRole.ARCHITECT: chunks[:3],
                AgentRole.SECURITY: chunks[1:4],
                AgentRole.DEVELOPER: chunks[2:5]
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Testing {scenario['name']}:")
        print("-" * 40)
        
        orchestrator._log_chunk_distribution_summary(scenario['distribution'])
        
        # Show the last log entry
        if orchestrator.orchestration_logs:
            print("📋 Generated Log:")
            print(orchestrator.orchestration_logs[-1])


def test_agent_analysis_results_logging():
    """Test the agent analysis results logging."""
    print("\n🧪 Testing Agent Analysis Results Logging")
    print("=" * 60)
    
    orchestrator = EnhancedAgentOrchestrator()
    
    # Test different confidence levels and result types
    test_cases = [
        {"role": AgentRole.ARCHITECT, "confidence": 0.95},
        {"role": AgentRole.SECURITY, "confidence": 0.75},
        {"role": AgentRole.DEVELOPER, "confidence": 0.55},
        {"role": AgentRole.PERFORMANCE, "confidence": 0.35}
    ]
    
    for case in test_cases:
        role = case["role"]
        confidence = case["confidence"]
        
        print(f"\n🔍 Testing {role.value.upper()} Analysis Results (Confidence: {confidence}):")
        print("-" * 50)
        
        perspective = create_test_perspective(role)
        perspective.confidence = confidence
        
        orchestrator._log_agent_analysis_results(role, perspective)
        
        # Show the last log entry
        if orchestrator.orchestration_logs:
            print("📋 Generated Log:")
            print(orchestrator.orchestration_logs[-1])


def test_empty_chunk_distribution():
    """Test logging with empty chunk distribution."""
    print("\n🧪 Testing Empty Chunk Distribution")
    print("=" * 60)
    
    orchestrator = EnhancedAgentOrchestrator()
    
    # Test empty distribution
    empty_distribution = {
        AgentRole.ARCHITECT: [],
        AgentRole.SECURITY: [],
        AgentRole.DEVELOPER: []
    }
    
    print("📊 Testing Empty Distribution:")
    print("-" * 30)
    
    orchestrator._log_chunk_distribution_summary(empty_distribution)
    
    # Show the last log entry
    if orchestrator.orchestration_logs:
        print("📋 Generated Log:")
        print(orchestrator.orchestration_logs[-1])


if __name__ == "__main__":
    print("🔍 Enhanced Debug Logging Test Suite - Simple Version")
    print("=" * 70)
    
    # Run all tests
    test_agent_processing_start_logging()
    test_chunk_distribution_logging()
    test_agent_analysis_results_logging()
    test_empty_chunk_distribution()
    
    print("\n" + "=" * 70)
    print("✅ All debug logging tests completed!")
    print("The enhanced debug logging provides detailed markdown-formatted")
    print("information about agent roles, available chunks, and processing details.")
