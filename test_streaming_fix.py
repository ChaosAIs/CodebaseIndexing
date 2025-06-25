#!/usr/bin/env python3
"""
Test script to verify streaming functionality is working.
This script tests both the simple test endpoint and the main streaming endpoint.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime


async def test_simple_streaming():
    """Test the simple test streaming endpoint."""
    print("🧪 Testing Simple Streaming Endpoint")
    print("=" * 50)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post('http://localhost:8000/mcp/test/stream') as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    return False
                
                print("📡 Reading streaming events...")
                event_count = 0
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        try:
                            event_data = json.loads(line_str[6:])
                            event_count += 1
                            print(f"  📨 Event {event_count}: {event_data.get('event_type')} - {event_data.get('message')}")
                            
                            if event_data.get('event_type') == 'test_complete':
                                print("✅ Test streaming completed successfully!")
                                return True
                                
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON decode error: {e}")
                            print(f"Raw line: {line_str}")
                
                print(f"📊 Total events received: {event_count}")
                return event_count > 0
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_main_streaming():
    """Test the main streaming query endpoint."""
    print("\n🧪 Testing Main Streaming Query Endpoint")
    print("=" * 50)
    
    try:
        query_data = {
            "query": "What is the main function?",
            "limit": 5,
            "include_context": True,
            "project_ids": None,
            "model": None
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/mcp/query/stream',
                json=query_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}")
                    text = await response.text()
                    print(f"Error details: {text}")
                    return False
                
                print("📡 Reading streaming events...")
                event_count = 0
                timeout_count = 0
                max_timeout = 30  # 30 seconds timeout
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        try:
                            event_data = json.loads(line_str[6:])
                            event_count += 1
                            event_type = event_data.get('event_type', 'unknown')
                            message = event_data.get('message', 'No message')
                            progress = event_data.get('progress_percentage', 0)
                            
                            print(f"  📨 Event {event_count}: {event_type} ({progress}%) - {message}")
                            
                            if event_type in ['processing_complete', 'complete', 'PROCESSING_COMPLETE']:
                                print("✅ Main streaming completed successfully!")
                                return True
                                
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON decode error: {e}")
                            print(f"Raw line: {line_str}")
                    
                    # Simple timeout mechanism
                    timeout_count += 1
                    if timeout_count > max_timeout * 10:  # Rough timeout
                        print("⏰ Timeout reached")
                        break
                
                print(f"📊 Total events received: {event_count}")
                return event_count > 0
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_frontend_compatibility():
    """Test streaming in a way that mimics the frontend implementation."""
    print("\n🧪 Testing Frontend-Compatible Streaming")
    print("=" * 50)
    
    try:
        query_data = {
            "query": "Simple test query",
            "limit": 3,
            "include_context": False,
            "project_ids": [],
            "model": None
        }
        
        # Use the same approach as the frontend
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8000/mcp/query/stream',
                json=query_data
            ) as response:
                
                if response.status != 200:
                    print(f"❌ HTTP Error: {response.status}")
                    return False
                
                print("📡 Streaming like frontend...")
                
                async for chunk in response.content.iter_chunked(1024):
                    chunk_str = chunk.decode('utf-8')
                    lines = chunk_str.split('\n')
                    
                    for line in lines:
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])
                                print(f"  📨 {event_data.get('event_type')}: {event_data.get('message')}")
                                
                                if event_data.get('event_type') in ['processing_complete', 'complete']:
                                    print("✅ Frontend-compatible streaming works!")
                                    return True
                                    
                            except json.JSONDecodeError:
                                continue
                
                return False
                
    except Exception as e:
        print(f"❌ Frontend compatibility test failed: {e}")
        return False


async def main():
    """Run all streaming tests."""
    print("🔍 Streaming Functionality Test Suite")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now()}")
    print()
    
    # Test 1: Simple streaming endpoint
    test1_result = await test_simple_streaming()
    
    # Test 2: Main streaming endpoint
    test2_result = await test_main_streaming()
    
    # Test 3: Frontend compatibility
    test3_result = await test_frontend_compatibility()
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    print(f"Simple Streaming: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Main Streaming: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"Frontend Compatible: {'✅ PASS' if test3_result else '❌ FAIL'}")
    
    overall_success = test1_result or test2_result or test3_result
    print(f"\nOverall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    if not overall_success:
        print("\n🔧 Troubleshooting Tips:")
        print("1. Make sure the backend server is running on port 8000")
        print("2. Check if there are any CORS issues")
        print("3. Verify the streaming endpoints are properly configured")
        print("4. Check the backend logs for any errors")
    
    return overall_success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
