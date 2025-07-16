#!/usr/bin/env python3
"""Script to clean up all databases (Neo4j, Qdrant, SQLite)."""

import sqlite3
import sys
import os
from pathlib import Path

# Direct database connections
from neo4j import GraphDatabase
from qdrant_client import QdrantClient as QdrantClientLib

def cleanup_neo4j():
    """Clean up Neo4j database."""
    print("🧹 Cleaning up Neo4j database...")
    try:
        # Neo4j connection settings from .env
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "enhanced_password_123"

        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Delete all nodes and relationships
        with driver.session() as session:
            # Delete all relationships first
            result = session.run("MATCH ()-[r]->() DELETE r RETURN count(r) as deleted_relationships")
            deleted_rels = result.single()["deleted_relationships"]
            print(f"   ✅ Deleted {deleted_rels} relationships")

            # Delete all nodes
            result = session.run("MATCH (n) DELETE n RETURN count(n) as deleted_nodes")
            deleted_nodes = result.single()["deleted_nodes"]
            print(f"   ✅ Deleted {deleted_nodes} nodes")

        driver.close()
        print("   ✅ Neo4j cleanup completed")
        return True

    except Exception as e:
        print(f"   ❌ Error cleaning up Neo4j: {e}")
        return False

def cleanup_qdrant():
    """Clean up Qdrant database."""
    print("🧹 Cleaning up Qdrant database...")
    try:
        # Default Qdrant connection settings
        qdrant_client = QdrantClientLib(host="localhost", port=6333)

        # Get all collections
        collections = qdrant_client.get_collections()

        if collections and hasattr(collections, 'collections'):
            for collection in collections.collections:
                collection_name = collection.name
                try:
                    # Delete the collection
                    qdrant_client.delete_collection(collection_name)
                    print(f"   ✅ Deleted collection: {collection_name}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not delete collection {collection_name}: {e}")
        else:
            print("   ℹ️  No collections found in Qdrant")

        print("   ✅ Qdrant cleanup completed")
        return True

    except Exception as e:
        print(f"   ❌ Error cleaning up Qdrant: {e}")
        return False

def cleanup_sqlite():
    """Clean up SQLite database."""
    print("🧹 Cleaning up SQLite database...")
    try:
        db_path = Path("data/projects.db")
        
        if not db_path.exists():
            print("   ℹ️  SQLite database file not found, nothing to clean")
            return True
        
        with sqlite3.connect(str(db_path)) as conn:
            # Get table names
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            deleted_records = 0
            for table in tables:
                table_name = table[0]
                if table_name != 'sqlite_sequence':  # Skip system table
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    cursor = conn.execute(f"DELETE FROM {table_name}")
                    deleted_records += count
                    print(f"   ✅ Cleared table '{table_name}': {count} records")
            
            conn.commit()
            print(f"   ✅ SQLite cleanup completed - {deleted_records} total records deleted")
            return True
            
    except Exception as e:
        print(f"   ❌ Error cleaning up SQLite: {e}")
        return False

def main():
    """Main cleanup function."""
    print("🚀 Starting database cleanup...")
    print("=" * 50)
    
    success_count = 0
    total_count = 3
    
    # Cleanup Neo4j
    if cleanup_neo4j():
        success_count += 1
    
    print()
    
    # Cleanup Qdrant
    if cleanup_qdrant():
        success_count += 1
    
    print()
    
    # Cleanup SQLite
    if cleanup_sqlite():
        success_count += 1
    
    print()
    print("=" * 50)
    
    if success_count == total_count:
        print("🎉 All databases cleaned successfully!")
        print("💡 You can now restart the backend server for a fresh start.")
    else:
        print(f"⚠️  Cleanup completed with {total_count - success_count} errors.")
        print("   Please check the error messages above.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
