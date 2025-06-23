#!/usr/bin/env python3
"""
Demo script to show project filtering functionality in the Chat Interface.
This script demonstrates how the backend handles project-specific queries.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.database.sqlite_client import ProjectManager
from src.models import QueryRequest, EmbeddingModel


async def demo_project_filtering():
    """Demonstrate project filtering functionality."""
    print("🚀 CHAT INTERFACE PROJECT FILTERING DEMO")
    print("=" * 60)
    
    # Initialize project manager
    project_manager = ProjectManager()
    
    # List all projects
    print("\n📋 AVAILABLE PROJECTS:")
    print("-" * 30)
    projects = await project_manager.list_projects()
    
    if not projects:
        print("❌ No projects found. Please create and index some projects first.")
        return
    
    indexed_projects = [p for p in projects if p.status == 'indexed']
    
    if not indexed_projects:
        print("❌ No indexed projects found. Please index some projects first.")
        return
    
    for i, project in enumerate(indexed_projects, 1):
        print(f"  {i}. {project.name}")
        print(f"     ID: {project.id}")
        print(f"     Status: {project.status}")
        print(f"     Files: {project.total_files}, Chunks: {project.total_chunks}")
        print(f"     Path: {project.source_path}")
        print()
    
    print("\n🔍 PROJECT FILTERING EXAMPLES:")
    print("-" * 40)
    
    # Example 1: Query all projects
    print("\n1️⃣ QUERY ALL PROJECTS (no filter):")
    query_all = QueryRequest(
        query="find authentication functions",
        limit=5,
        include_context=True,
        project_ids=None  # No filter - search all projects
    )
    print(f"   Query: {query_all.query}")
    print(f"   Project Filter: None (searches all indexed projects)")
    print(f"   Expected: Results from all {len(indexed_projects)} indexed projects")
    
    # Example 2: Query specific project
    if len(indexed_projects) >= 1:
        selected_project = indexed_projects[0]
        print(f"\n2️⃣ QUERY SPECIFIC PROJECT ({selected_project.name}):")
        query_specific = QueryRequest(
            query="find authentication functions",
            limit=5,
            include_context=True,
            project_ids=[selected_project.id]  # Filter to specific project
        )
        print(f"   Query: {query_specific.query}")
        print(f"   Project Filter: [{selected_project.name}]")
        print(f"   Expected: Results only from '{selected_project.name}' project")
    
    # Example 3: Query multiple projects
    if len(indexed_projects) >= 2:
        selected_projects = indexed_projects[:2]
        print(f"\n3️⃣ QUERY MULTIPLE PROJECTS:")
        query_multiple = QueryRequest(
            query="find error handling code",
            limit=5,
            include_context=True,
            project_ids=[p.id for p in selected_projects]
        )
        print(f"   Query: {query_multiple.query}")
        print(f"   Project Filter: {[p.name for p in selected_projects]}")
        print(f"   Expected: Results only from selected {len(selected_projects)} projects")
    
    print("\n💾 PERSISTENCE FEATURES:")
    print("-" * 30)
    print("✅ Selected projects are saved to localStorage")
    print("✅ Selection persists across page refreshes")
    print("✅ Selection persists across browser sessions")
    print("✅ Clear selection button available in settings")
    
    print("\n🎯 CHAT INTERFACE FEATURES:")
    print("-" * 35)
    print("✅ Visual project selection dropdown")
    print("✅ Multi-project selection with checkboxes")
    print("✅ 'Select All' and individual project toggles")
    print("✅ Project status indicators (indexed/indexing/error)")
    print("✅ Real-time selection status display")
    print("✅ Warning when no projects selected")
    print("✅ Project count indicator in input area")
    
    print("\n🔧 BACKEND FILTERING:")
    print("-" * 25)
    print("✅ Vector store filtering by project_id")
    print("✅ Graph store filtering by project_id")
    print("✅ Embedding search scoped to selected projects")
    print("✅ Context retrieval limited to selected projects")
    
    print("\n📊 API USAGE EXAMPLES:")
    print("-" * 25)
    
    # Show API request examples
    api_examples = [
        {
            "description": "Search all projects",
            "request": {
                "query": "find authentication functions",
                "limit": 10,
                "include_context": True,
                "project_ids": None
            }
        },
        {
            "description": "Search specific project",
            "request": {
                "query": "show me error handling",
                "limit": 5,
                "include_context": True,
                "project_ids": [indexed_projects[0].id] if indexed_projects else []
            }
        },
        {
            "description": "Search multiple projects",
            "request": {
                "query": "find database connections",
                "limit": 8,
                "include_context": True,
                "project_ids": [p.id for p in indexed_projects[:2]] if len(indexed_projects) >= 2 else []
            }
        }
    ]
    
    for i, example in enumerate(api_examples, 1):
        print(f"\n{i}. {example['description']}:")
        print("   POST /mcp/query")
        print("   Content-Type: application/json")
        print("   Body:")
        print(json.dumps(example['request'], indent=6))
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT FILTERING DEMO COMPLETE!")
    print("\nThe Chat Interface now supports:")
    print("• Project-specific knowledge filtering")
    print("• Persistent project selection")
    print("• Multi-project search capabilities")
    print("• Visual project selection UI")
    print("• Real-time filtering status")


if __name__ == "__main__":
    asyncio.run(demo_project_filtering())
