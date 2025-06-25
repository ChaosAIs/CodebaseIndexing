"""
Streaming response processor for real-time query processing updates.

This module handles streaming responses to provide real-time feedback
to users during long-running query processing operations.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from ..models import CodeChunk, QueryResult
from ..agents.agent_orchestrator import AgentRole


class StreamEventType(Enum):
    """Types of streaming events."""
    PROCESSING_START = "processing_start"
    QUERY_ANALYSIS_START = "query_analysis_start"
    QUERY_ANALYSIS_COMPLETE = "query_analysis_complete"
    ORCHESTRATION_START = "orchestration_start"
    CHUNK_DISTRIBUTION_START = "chunk_distribution_start"
    CHUNK_DISTRIBUTION_COMPLETE = "chunk_distribution_complete"
    SEARCH_START = "search_start"
    SEARCH_PROGRESS = "search_progress"
    SEARCH_COMPLETE = "search_complete"
    AGENT_SETUP_START = "agent_setup_start"
    AGENT_SETUP_COMPLETE = "agent_setup_complete"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    SYNTHESIS_START = "synthesis_start"
    SYNTHESIS_PROGRESS = "synthesis_progress"
    SYNTHESIS_COMPLETE = "synthesis_complete"
    PROCESSING_COMPLETE = "processing_complete"
    ERROR = "error"
    LOG = "log"
    USER_MESSAGE = "user_message"


@dataclass
class StreamEvent:
    """A single streaming event."""
    event_type: StreamEventType
    timestamp: float
    data: Dict[str, Any]
    message: str
    progress_percentage: Optional[float] = None


@dataclass
class ProcessingStatus:
    """Current processing status."""
    stage: str
    progress: float
    message: str
    agents_completed: int
    agents_total: int
    estimated_time_remaining: Optional[float] = None


class StreamProcessor:
    """
    Handles streaming responses for query processing.
    """
    
    def __init__(self):
        """Initialize the stream processor."""
        self.active_streams = {}
        self.processing_stats = {}
    
    async def create_stream(self, stream_id: str) -> AsyncGenerator[str, None]:
        """
        Create a new streaming response generator.
        
        Args:
            stream_id: Unique identifier for this stream
            
        Yields:
            JSON-encoded streaming events
        """
        logger.info(f"Creating stream: {stream_id}")
        
        # Initialize stream state
        self.active_streams[stream_id] = {
            "start_time": time.time(),
            "events": [],
            "status": ProcessingStatus(
                stage="initializing",
                progress=0.0,
                message="Initializing query processing...",
                agents_completed=0,
                agents_total=0
            )
        }
        
        try:
            # Create event queue for this stream
            event_queue = asyncio.Queue()
            self.active_streams[stream_id]["queue"] = event_queue

            # Send initial event directly to queue to avoid race condition
            initial_event = StreamEvent(
                event_type=StreamEventType.PROCESSING_START,
                timestamp=time.time(),
                data={"stream_id": stream_id, "timestamp": time.time()},
                message="Starting query processing...",
                progress_percentage=0.0
            )
            await event_queue.put(initial_event)
            logger.info(f"Added initial event to queue for stream {stream_id}")

            # Send a test event immediately to verify streaming works
            test_event = StreamEvent(
                event_type=StreamEventType.LOG,
                timestamp=time.time(),
                data={"test": True, "stream_id": stream_id},
                message="Stream connection established",
                progress_percentage=5.0
            )
            await event_queue.put(test_event)
            logger.info(f"Added test event to queue for stream {stream_id}")

            # Stream events as they come
            while stream_id in self.active_streams:
                try:
                    # Wait for next event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    
                    if event is None:  # End of stream signal
                        break
                    
                    # Convert event to JSON and yield
                    try:
                        event_dict = {
                            "event_type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                            "timestamp": event.timestamp,
                            "data": event.data,
                            "message": event.message,
                            "progress_percentage": event.progress_percentage
                        }
                        event_json = json.dumps(event_dict, default=str)
                        yield f"data: {event_json}\n\n"
                        logger.debug(f"Yielded event to stream {stream_id}: {event.event_type.value}")
                    except Exception as json_error:
                        logger.error(f"JSON serialization error for stream {stream_id}: {json_error}")
                        # Send a simple error event instead
                        error_dict = {
                            "event_type": "error",
                            "timestamp": time.time(),
                            "data": {"error": "Serialization error"},
                            "message": "Error processing event",
                            "progress_percentage": None
                        }
                        yield f"data: {json.dumps(error_dict)}\n\n"
                    
                    # Check if this is a completion event
                    if event.event_type == StreamEventType.PROCESSING_COMPLETE:
                        break
                        
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    try:
                        heartbeat_dict = {
                            "event_type": "log",
                            "timestamp": time.time(),
                            "data": {"type": "heartbeat"},
                            "message": "Processing continues...",
                            "progress_percentage": self.active_streams[stream_id]["status"].progress
                        }
                        event_json = json.dumps(heartbeat_dict, default=str)
                        yield f"data: {event_json}\n\n"
                        logger.debug(f"Sent heartbeat to stream {stream_id}")
                    except Exception as heartbeat_error:
                        logger.error(f"Heartbeat error for stream {stream_id}: {heartbeat_error}")
                        # Continue without heartbeat
                    
        except Exception as e:
            logger.error(f"Error in stream {stream_id}: {e}")
            try:
                error_dict = {
                    "event_type": "error",
                    "timestamp": time.time(),
                    "data": {"error": str(e)},
                    "message": f"Stream error: {e}",
                    "progress_percentage": None
                }
                event_json = json.dumps(error_dict, default=str)
                yield f"data: {event_json}\n\n"
            except Exception as error_serialization:
                logger.error(f"Failed to serialize error event: {error_serialization}")
                # Send minimal error response
                yield f"data: {{'event_type': 'error', 'message': 'Stream processing error'}}\n\n"
        finally:
            # Clean up stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            logger.info(f"Stream {stream_id} closed")
    
    async def emit_event(self, stream_id: str, event_type: StreamEventType, 
                        data: Dict[str, Any], message: str, 
                        progress_percentage: Optional[float] = None):
        """
        Emit an event to a specific stream.
        
        Args:
            stream_id: Stream identifier
            event_type: Type of event
            data: Event data
            message: Human-readable message
            progress_percentage: Optional progress percentage
        """
        if stream_id not in self.active_streams:
            logger.warning(f"Attempted to emit event to non-existent stream: {stream_id}")
            return

        logger.debug(f"Emitting {event_type.value} event to stream {stream_id}: {message}")
        
        event = StreamEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            message=message,
            progress_percentage=progress_percentage
        )
        
        # Add to event history
        self.active_streams[stream_id]["events"].append(event)
        
        # Update status if progress provided
        if progress_percentage is not None:
            self.active_streams[stream_id]["status"].progress = progress_percentage
            self.active_streams[stream_id]["status"].message = message
        
        # Send to queue
        queue = self.active_streams[stream_id]["queue"]
        await queue.put(event)
        logger.debug(f"Successfully queued {event_type.value} event for stream {stream_id}")

        # Log queue size for debugging
        queue_size = queue.qsize()
        logger.debug(f"Queue size for stream {stream_id}: {queue_size}")
    
    async def emit_query_analysis(self, stream_id: str, analysis_result: Any):
        """Emit query analysis results."""
        await self.emit_event(
            stream_id,
            StreamEventType.QUERY_ANALYSIS_COMPLETE,
            {
                "complexity": analysis_result.complexity.value,
                "strategy": analysis_result.processing_strategy.value,
                "agents_count": len(analysis_result.required_agents),
                "estimated_time": analysis_result.estimated_processing_time,
                "should_stream": analysis_result.should_stream,
                "explanation": analysis_result.explanation
            },
            f"Query analyzed: {analysis_result.explanation}",
            5.0
        )
    
    async def emit_search_start(self, stream_id: str, search_terms: List[str]):
        """Emit search start event."""
        await self.emit_event(
            stream_id,
            StreamEventType.SEARCH_START,
            {
                "search_terms": search_terms,
                "terms_count": len(search_terms)
            },
            f"Starting search with {len(search_terms)} terms...",
            10.0
        )
    
    async def emit_search_progress(self, stream_id: str, completed_terms: int, total_terms: int, found_chunks: int):
        """Emit search progress event."""
        progress = 10.0 + (completed_terms / total_terms) * 20.0  # 10-30% for search
        await self.emit_event(
            stream_id,
            StreamEventType.SEARCH_PROGRESS,
            {
                "completed_terms": completed_terms,
                "total_terms": total_terms,
                "found_chunks": found_chunks
            },
            f"Search progress: {completed_terms}/{total_terms} terms, {found_chunks} chunks found",
            progress
        )
    
    async def emit_search_complete(self, stream_id: str, total_chunks: int):
        """Emit search completion event."""
        await self.emit_event(
            stream_id,
            StreamEventType.SEARCH_COMPLETE,
            {
                "total_chunks": total_chunks
            },
            f"Search complete: {total_chunks} relevant chunks found",
            30.0
        )
    
    async def emit_agent_start(self, stream_id: str, agent_role: AgentRole, task_description: str, agent_index: int, total_agents: int):
        """Emit agent start event."""
        # Update total agents in status
        self.active_streams[stream_id]["status"].agents_total = total_agents
        
        progress = 30.0 + (agent_index / total_agents) * 60.0  # 30-90% for agents
        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_START,
            {
                "agent_role": agent_role.value,
                "task_description": task_description,
                "agent_index": agent_index,
                "total_agents": total_agents
            },
            f"Starting {agent_role.value} analysis: {task_description}",
            progress
        )
    
    async def emit_agent_progress(self, stream_id: str, agent_role: AgentRole, progress_message: str, agent_index: int, total_agents: int):
        """Emit agent progress event."""
        base_progress = 30.0 + (agent_index / total_agents) * 60.0
        agent_progress = base_progress + (60.0 / total_agents) * 0.5  # Mid-point of agent processing
        
        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_PROGRESS,
            {
                "agent_role": agent_role.value,
                "progress_message": progress_message,
                "agent_index": agent_index,
                "total_agents": total_agents
            },
            f"{agent_role.value}: {progress_message}",
            agent_progress
        )
    
    async def emit_agent_complete(self, stream_id: str, agent_role: AgentRole, confidence: float, agent_index: int, total_agents: int):
        """Emit agent completion event."""
        # Update completed agents count
        self.active_streams[stream_id]["status"].agents_completed = agent_index + 1
        
        progress = 30.0 + ((agent_index + 1) / total_agents) * 60.0
        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_COMPLETE,
            {
                "agent_role": agent_role.value,
                "confidence": confidence,
                "agent_index": agent_index,
                "total_agents": total_agents
            },
            f"{agent_role.value} analysis complete (confidence: {confidence:.2f})",
            progress
        )
    
    async def emit_synthesis_start(self, stream_id: str, perspectives_count: int):
        """Emit synthesis start event."""
        await self.emit_event(
            stream_id,
            StreamEventType.SYNTHESIS_START,
            {
                "perspectives_count": perspectives_count
            },
            f"Synthesizing {perspectives_count} agent perspectives...",
            90.0
        )
    
    async def emit_synthesis_complete(self, stream_id: str, final_results: List[QueryResult]):
        """Emit synthesis completion event."""
        await self.emit_event(
            stream_id,
            StreamEventType.SYNTHESIS_COMPLETE,
            {
                "results_count": len(final_results)
            },
            f"Synthesis complete: {len(final_results)} results generated",
            95.0
        )
    
    async def emit_processing_complete(self, stream_id: str, response_data: Any = None, total_time: float = 0.0, results_count: int = 0):
        """Emit processing completion event with final response data."""

        # Debug log final streaming result
        self._log_streaming_final_result_debug(stream_id, response_data, total_time, results_count)

        await self.emit_event(
            stream_id,
            StreamEventType.PROCESSING_COMPLETE,
            {
                "total_time": total_time,
                "results_count": results_count,
                "stream_id": stream_id,
                "final_response": response_data,  # Include the actual response data
                "response_type": "flow_response" if hasattr(response_data, 'agent_perspectives') else "query_results"
            },
            f"Processing complete in {total_time:.2f}s with {results_count} results",
            100.0
        )

        # Signal end of stream
        queue = self.active_streams[stream_id]["queue"]
        await queue.put(None)
    
    async def emit_log(self, stream_id: str, log_level: str, message: str):
        """Emit a log message."""
        await self.emit_event(
            stream_id,
            StreamEventType.LOG,
            {
                "level": log_level,
                "internal_message": message
            },
            f"[{log_level.upper()}] {message}"
        )

    async def emit_orchestration_log(self, stream_id: str, orchestration_logs: List[str]):
        """Emit orchestration logs for detailed monitoring."""
        await self.emit_event(
            stream_id,
            StreamEventType.LOG,
            {
                "level": "orchestration",
                "orchestration_logs": orchestration_logs,
                "log_count": len(orchestration_logs)
            },
            f"Orchestration logs: {len(orchestration_logs)} entries"
        )

    # User-friendly streaming methods for each workflow step

    async def emit_processing_start(self, stream_id: str, query: str):
        """Emit processing start event with user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.PROCESSING_START,
            {"query": query},
            f"🚀 Starting analysis for: '{query[:50]}{'...' if len(query) > 50 else ''}'",
            0.0
        )

    async def emit_user_message(self, stream_id: str, message: str, progress: Optional[float] = None):
        """Emit a user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.USER_MESSAGE,
            {"user_friendly": True},
            message,
            progress
        )

    async def emit_query_analysis_start(self, stream_id: str, query: str):
        """Emit query analysis start with user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.QUERY_ANALYSIS_START,
            {"query": query},
            f"🔍 Analyzing your query: '{query[:50]}{'...' if len(query) > 50 else ''}'",
            5.0
        )

    async def emit_query_analysis_complete(self, stream_id: str, analysis_result: Any):
        """Emit query analysis completion with user-friendly explanation."""
        complexity = analysis_result.complexity.value
        agent_count = len(analysis_result.required_agents)

        # Create user-friendly complexity explanation
        complexity_msg = {
            "simple": "This looks like a straightforward question",
            "moderate": "This requires some analysis across multiple areas",
            "complex": "This needs comprehensive analysis from multiple perspectives",
            "architectural": "This requires deep architectural analysis"
        }.get(complexity, "This needs analysis")

        message = f"✅ {complexity_msg}. I'll use {agent_count} specialized experts to help answer this."

        await self.emit_event(
            stream_id,
            StreamEventType.QUERY_ANALYSIS_COMPLETE,
            {
                "complexity": complexity,
                "agent_count": agent_count,
                "estimated_time": analysis_result.estimated_processing_time,
                "explanation": analysis_result.explanation
            },
            message,
            15.0
        )

    async def emit_orchestration_start(self, stream_id: str, agent_count: int):
        """Emit orchestration start with user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.ORCHESTRATION_START,
            {"agent_count": agent_count},
            f"🎭 Setting up {agent_count} expert analysts to work on your question...",
            20.0
        )

    async def emit_chunk_distribution_start(self, stream_id: str, total_chunks: int, agent_count: int):
        """Emit chunk distribution start."""
        await self.emit_event(
            stream_id,
            StreamEventType.CHUNK_DISTRIBUTION_START,
            {"total_chunks": total_chunks, "agent_count": agent_count},
            f"📦 Distributing {total_chunks} code sections among {agent_count} experts for focused analysis...",
            25.0
        )

    async def emit_chunk_distribution_complete(self, stream_id: str, distribution_summary: Dict[str, int]):
        """Emit chunk distribution completion."""
        total_assigned = sum(distribution_summary.values())
        await self.emit_event(
            stream_id,
            StreamEventType.CHUNK_DISTRIBUTION_COMPLETE,
            {"distribution": distribution_summary, "total_assigned": total_assigned},
            f"✅ Successfully assigned {total_assigned} code sections to experts based on their specializations",
            30.0
        )

    async def emit_agent_setup_start(self, stream_id: str, agent_count: int):
        """Emit agent setup start."""
        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_SETUP_START,
            {"agent_count": agent_count},
            f"⚙️ Preparing {agent_count} expert analysts with their specialized tasks...",
            35.0
        )

    async def emit_agent_setup_complete(self, stream_id: str, agents_ready: List[str]):
        """Emit agent setup completion."""
        agent_list = ", ".join(agents_ready)
        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_SETUP_COMPLETE,
            {"agents_ready": agents_ready},
            f"✅ Expert team ready: {agent_list}",
            40.0
        )

    async def emit_agent_start_friendly(self, stream_id: str, agent_role: AgentRole, task_description: str, agent_index: int, total_agents: int):
        """Emit agent start with user-friendly message."""
        # Create user-friendly role descriptions
        role_descriptions = {
            "architect": "🏗️ Architecture Expert",
            "developer": "👨‍💻 Development Expert",
            "security": "🔒 Security Expert",
            "performance": "⚡ Performance Expert",
            "maintainer": "🔧 Maintenance Expert",
            "business": "💼 Business Logic Expert",
            "integration": "🔗 Integration Expert",
            "data": "📊 Data Expert"
        }

        friendly_role = role_descriptions.get(agent_role.value, f"🤖 {agent_role.value.title()} Expert")
        progress = 40.0 + (agent_index / total_agents) * 50.0

        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_START,
            {
                "agent_role": agent_role.value,
                "task_description": task_description,
                "agent_index": agent_index,
                "total_agents": total_agents,
                "friendly_role": friendly_role
            },
            f"{friendly_role} is analyzing the code from their specialized perspective...",
            progress
        )

    async def emit_agent_progress_friendly(self, stream_id: str, agent_role: AgentRole, progress_message: str, agent_index: int, total_agents: int):
        """Emit agent progress with user-friendly message."""
        role_descriptions = {
            "architect": "🏗️ Architecture Expert",
            "developer": "👨‍💻 Development Expert",
            "security": "🔒 Security Expert",
            "performance": "⚡ Performance Expert",
            "maintainer": "🔧 Maintenance Expert",
            "business": "💼 Business Logic Expert",
            "integration": "🔗 Integration Expert",
            "data": "📊 Data Expert"
        }

        friendly_role = role_descriptions.get(agent_role.value, f"🤖 {agent_role.value.title()} Expert")
        base_progress = 40.0 + (agent_index / total_agents) * 50.0
        agent_progress = base_progress + (50.0 / total_agents) * 0.5

        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_PROGRESS,
            {
                "agent_role": agent_role.value,
                "progress_message": progress_message,
                "agent_index": agent_index,
                "total_agents": total_agents,
                "friendly_role": friendly_role
            },
            f"{friendly_role}: {progress_message}",
            agent_progress
        )

    async def emit_agent_complete_friendly(self, stream_id: str, agent_role: AgentRole, confidence: float, insights_count: int, agent_index: int, total_agents: int):
        """Emit agent completion with user-friendly message."""
        role_descriptions = {
            "architect": "🏗️ Architecture Expert",
            "developer": "👨‍💻 Development Expert",
            "security": "🔒 Security Expert",
            "performance": "⚡ Performance Expert",
            "maintainer": "🔧 Maintenance Expert",
            "business": "💼 Business Logic Expert",
            "integration": "🔗 Integration Expert",
            "data": "📊 Data Expert"
        }

        friendly_role = role_descriptions.get(agent_role.value, f"🤖 {agent_role.value.title()} Expert")
        progress = 40.0 + ((agent_index + 1) / total_agents) * 50.0

        confidence_desc = "high confidence" if confidence > 0.8 else "good confidence" if confidence > 0.6 else "moderate confidence"

        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_COMPLETE,
            {
                "agent_role": agent_role.value,
                "confidence": confidence,
                "insights_count": insights_count,
                "agent_index": agent_index,
                "total_agents": total_agents,
                "friendly_role": friendly_role
            },
            f"✅ {friendly_role} completed analysis with {confidence_desc} and {insights_count} key insights",
            progress
        )

    async def emit_agent_chunk_retrieval(self, stream_id: str, agent_role: AgentRole, chunks_retrieved: int, agent_index: int, total_agents: int):
        """Emit agent chunk retrieval event with user-friendly message."""
        role_descriptions = {
            "architect": "🏗️ Architecture Expert",
            "developer": "👨‍💻 Development Expert",
            "security": "🔒 Security Expert",
            "performance": "⚡ Performance Expert",
            "maintainer": "🔧 Maintenance Expert",
            "business": "💼 Business Expert",
            "integration": "🔗 Integration Expert",
            "data": "📊 Data Expert",
            "ui_ux": "🎨 UI/UX Expert",
            "devops": "🚀 DevOps Expert",
            "testing": "🧪 Testing Expert",
            "compliance": "📋 Compliance Expert"
        }

        friendly_role = role_descriptions.get(agent_role.value, f"{agent_role.value.title()} Expert")
        progress = 20.0 + (agent_index / total_agents) * 15.0  # 20-35% for chunk retrieval

        await self.emit_event(
            stream_id,
            StreamEventType.AGENT_PROGRESS,
            {
                "agent_role": agent_role.value,
                "chunks_retrieved": chunks_retrieved,
                "agent_index": agent_index,
                "total_agents": total_agents,
                "friendly_role": friendly_role
            },
            f"📦 {friendly_role} retrieved {chunks_retrieved} specialized code sections for analysis",
            progress
        )

    async def emit_agent_debug_info(self, stream_id: str, agent_role: AgentRole, debug_data: Dict[str, Any]):
        """Emit detailed debug information for agent processing."""
        role_descriptions = {
            "architect": "🏗️ Architecture Expert",
            "developer": "👨‍💻 Development Expert",
            "security": "🔒 Security Expert",
            "performance": "⚡ Performance Expert",
            "maintainer": "🔧 Maintenance Expert",
            "business": "💼 Business Expert",
            "integration": "🔗 Integration Expert",
            "data": "📊 Data Expert",
            "ui_ux": "🎨 UI/UX Expert",
            "devops": "🚀 DevOps Expert",
            "testing": "🧪 Testing Expert",
            "compliance": "📋 Compliance Expert"
        }

        friendly_role = role_descriptions.get(agent_role.value, f"{agent_role.value.title()} Expert")

        await self.emit_event(
            stream_id,
            StreamEventType.LOG,
            {
                "agent_role": agent_role.value,
                "friendly_role": friendly_role,
                "debug_type": "agent_processing",
                **debug_data
            },
            f"🔍 Debug: {friendly_role} processing details",
            None
        )

    async def emit_synthesis_start_friendly(self, stream_id: str, perspectives_count: int):
        """Emit synthesis start with user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.SYNTHESIS_START,
            {"perspectives_count": perspectives_count},
            f"🔄 Combining insights from {perspectives_count} expert analyses into a comprehensive response...",
            90.0
        )

    async def emit_synthesis_progress(self, stream_id: str, step: str, progress_pct: float):
        """Emit synthesis progress updates."""
        await self.emit_event(
            stream_id,
            StreamEventType.SYNTHESIS_PROGRESS,
            {"synthesis_step": step},
            f"🔄 {step}",
            90.0 + (progress_pct * 0.08)  # 90-98% range
        )

    async def emit_synthesis_complete_friendly(self, stream_id: str, final_insights: int, recommendations: int):
        """Emit synthesis completion with user-friendly message."""
        await self.emit_event(
            stream_id,
            StreamEventType.SYNTHESIS_COMPLETE,
            {
                "final_insights": final_insights,
                "recommendations": recommendations
            },
            f"✅ Analysis complete! Generated {final_insights} key insights and {recommendations} recommendations",
            98.0
        )

    def _log_streaming_final_result_debug(self, stream_id: str, response_data: Any, total_time: float, results_count: int):
        """Log comprehensive debug information about the final streaming result."""
        logger.info("📡 === STREAMING FINAL RESULT DEBUG ===")
        logger.info(f"🆔 Stream ID: {stream_id}")
        logger.info(f"⏱️ Total Time: {total_time:.3f}s")
        logger.info(f"📊 Results Count: {results_count}")

        # Determine response type and log accordingly
        if isinstance(response_data, dict):
            if 'agent_perspectives' in response_data:
                # Flow response dictionary
                logger.info("📋 Response Type: Flow Response Dictionary")
                logger.info(f"  📄 Executive Summary: {len(response_data.get('executive_summary', ''))} chars")
                logger.info(f"  📖 Detailed Analysis: {len(response_data.get('detailed_analysis', ''))} chars")
                logger.info(f"  🔄 Synthesis: {len(response_data.get('synthesis', ''))} chars")

                perspectives = response_data.get('agent_perspectives', [])
                logger.info(f"  👁️ Agent Perspectives: {len(perspectives)} perspectives")
                for i, perspective in enumerate(perspectives):
                    role = perspective.get('role', 'unknown')
                    confidence = perspective.get('confidence', 0)
                    insights = len(perspective.get('key_insights', []))
                    recommendations = len(perspective.get('recommendations', []))
                    logger.info(f"    Perspective {i+1}: {role} (conf: {confidence:.2f}, insights: {insights}, recs: {recommendations})")

                logger.info(f"  ✅ Action Items: {len(response_data.get('action_items', []))}")
                logger.info(f"  ❓ Follow-up Questions: {len(response_data.get('follow_up_questions', []))}")

                # Calculate total content size
                total_content = (
                    len(response_data.get('executive_summary', '')) +
                    len(response_data.get('detailed_analysis', '')) +
                    len(response_data.get('synthesis', '')) +
                    sum(len(p.get('analysis', '')) for p in perspectives)
                )
                logger.info(f"📏 Total Content Size: {total_content} characters")

            else:
                # Other dictionary response
                logger.info("📋 Response Type: Dictionary Response")
                logger.info(f"  🔑 Keys: {list(response_data.keys())}")

        elif isinstance(response_data, list):
            # Query results list
            logger.info("📋 Response Type: Query Results List")
            logger.info(f"  📊 Results: {len(response_data)} items")

        elif hasattr(response_data, 'agent_perspectives'):
            # Flow response object
            logger.info("📋 Response Type: Flow Response Object")
            logger.info(f"  👁️ Agent Perspectives: {len(response_data.agent_perspectives)} perspectives")

        else:
            # Unknown response type
            logger.info(f"📋 Response Type: {type(response_data).__name__}")
            logger.info(f"  📊 Data: {str(response_data)[:200]}{'...' if len(str(response_data)) > 200 else ''}")

        # Log streaming performance metrics
        if total_time > 0:
            throughput = results_count / total_time
            logger.info(f"⚡ Performance: {throughput:.1f} results/second")

        # Log stream status
        if stream_id in self.active_streams:
            stream_info = self.active_streams[stream_id]
            events_count = len(stream_info.get("events", []))
            logger.info(f"📡 Stream Events: {events_count} events emitted")

        logger.info("📡 === END STREAMING FINAL RESULT DEBUG ===")

    def get_stream_status(self, stream_id: str) -> Optional[ProcessingStatus]:
        """Get current status of a stream."""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]["status"]
        return None
    
    def get_stream_events(self, stream_id: str) -> List[StreamEvent]:
        """Get all events for a stream."""
        if stream_id in self.active_streams:
            return self.active_streams[stream_id]["events"]
        return []


# Global stream processor instance
stream_processor = StreamProcessor()
