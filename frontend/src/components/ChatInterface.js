import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Code, FileText, Search, Settings, ChevronDown, ChevronUp, FolderOpen, X, Users, Brain, Target, CheckCircle, AlertTriangle } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { apiService } from '../services/apiService';
import ProjectSelector from './ProjectSelector';
import { usePersistedProjectSelection } from '../hooks/usePersistedState';
import MarkdownRenderer from './MarkdownRenderer';

// Streaming message component - removed memo to ensure re-renders
const StreamingMessage = ({ message, forceUpdate, renderCount }) => {
  // Enhanced logging to debug rendering
  console.log('🎨 StreamingMessage render:', message.id, 'Status:', message.content.status, 'Events:', message.content.events?.length || 0, 'LastUpdated:', message.content.lastUpdated, 'ForceUpdate:', forceUpdate, 'RenderCount:', renderCount);

  return (
    <div className="max-w-5xl bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 px-4 py-3 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="animate-spin">
              <Loader2 className="w-4 h-4 text-blue-600" />
            </div>
            <span className="text-sm font-medium text-gray-700">
              {message.content.status} {message.content.lastUpdated && `(${new Date(message.content.lastUpdated).toLocaleTimeString()})`}
            </span>

            <span className="text-xs text-gray-500 ml-2">
              {message.content.events.length} events • {message.content.lastUpdated ? new Date(message.content.lastUpdated).toLocaleTimeString() : 'No updates'}
            </span>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      {message.content.progress > 0 && (
        <div className="px-4 py-2 bg-gray-50">
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${message.content.progress}%` }}
            ></div>
          </div>
          <div className="text-xs text-gray-600 mt-1">{message.content.progress}% complete</div>
        </div>
      )}

      {/* Events */}
      {message.content.events && message.content.events.length > 0 && (
        <div className="px-4 py-3 max-h-60 overflow-y-auto">
          <div className="text-xs text-gray-600 mb-2">Processing Log:</div>
          {message.content.events.slice(-10).map((event, index) => (
            <div key={index} className="text-xs text-gray-600 flex items-start space-x-2 mb-1">
              <span className="text-gray-400 flex-shrink-0">
                {event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : ''}
              </span>
              <span className="flex-1 font-mono">
                {/* Add emoji for different event types */}
                {event.event_type === 'processing_start' && '🚀 '}
                {event.event_type === 'query_analysis_start' && '🔍 '}
                {event.event_type === 'query_analysis_complete' && '✅ '}
                {event.event_type === 'search_start' && '🔎 '}
                {event.event_type === 'search_complete' && '✅ '}
                {event.event_type === 'orchestration_start' && '🎭 '}
                {event.event_type === 'agent_start' && '🤖 '}
                {event.event_type === 'agent_progress' && '⚙️ '}
                {event.event_type === 'agent_complete' && '✅ '}
                {event.event_type === 'synthesis_start' && '🔄 '}
                {event.event_type === 'synthesis_progress' && '📈 '}
                {event.event_type === 'synthesis_complete' && '✅ '}
                {event.event_type === 'processing_complete' && '🎉 '}
                {event.message}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Final Response (if available) */}
      {message.content.final_response && (
        <div className="border-t border-gray-200 px-4 py-3">
          <div className="text-sm text-green-600 font-medium mb-2">
            ✅ Processing Complete
          </div>
          <div className="text-sm text-gray-700">
            Final response available - converting to final message...
          </div>
        </div>
      )}
    </div>
  );
};

const ChatInterface = ({ systemStatus }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isMultiAgentAnalysis, setIsMultiAgentAnalysis] = useState(false);
  // selectedModel removed - using .env configuration only
  const [showSettings, setShowSettings] = useState(false);
  const [includeContext, setIncludeContext] = useState(true);
  const [resultLimit, setResultLimit] = useState(10);
  const [selectedProjectIds, setSelectedProjectIds] = usePersistedProjectSelection();
  const [expandedComponents, setExpandedComponents] = useState({});
  const [expandedResults, setExpandedResults] = useState({});
  const [expandedAgentPerspectives, setExpandedAgentPerspectives] = useState({});
  const [projects, setProjects] = useState([]);
  const [forceUpdate, setForceUpdate] = useState(0);
  const [renderCount, setRenderCount] = useState(0);
  const [lastRequestTime, setLastRequestTime] = useState(0);
  const [requestQueue, setRequestQueue] = useState([]);
  const [isProcessingRequest, setIsProcessingRequest] = useState(false);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Track renders (disabled to reduce log noise)
  // useEffect(() => {
  //   setRenderCount(prev => prev + 1);
  // });

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load projects for name mapping
  useEffect(() => {
    const loadProjects = async () => {
      try {
        const projectList = await apiService.listProjects();
        setProjects(projectList);
      } catch (error) {
        console.error('Failed to load projects:', error);
      }
    };
    loadProjects();
  }, []);

  // Model selection removed - using .env configuration only

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Request throttling to prevent too many concurrent requests
  const throttleRequest = async (requestFn) => {
    const now = Date.now();
    const minInterval = 1000; // Minimum 1 second between requests

    if (isProcessingRequest) {
      console.log('Request blocked: Another request is already processing');
      return null;
    }

    if (now - lastRequestTime < minInterval) {
      const waitTime = minInterval - (now - lastRequestTime);
      console.log(`Request throttled: Waiting ${waitTime}ms`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    setIsProcessingRequest(true);
    setLastRequestTime(Date.now());

    try {
      const result = await requestFn();
      return result;
    } finally {
      setIsProcessingRequest(false);
    }
  };

  // Get selected project names for display
  const getSelectedProjectNames = () => {
    const selectedProjects = projects.filter(p => selectedProjectIds.includes(p.id));
    return selectedProjects.map(p => p.name);
  };

  const handleExpandResults = (messageId) => {
    setExpandedResults(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

  const handleExpandAgentPerspective = (messageId, agentIndex) => {
    const key = `${messageId}-${agentIndex}`;
    setExpandedAgentPerspectives(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const handleSubmit = async (e, useFlowAnalysis = false) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading || isProcessingRequest) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      flowAnalysis: useFlowAnalysis
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setIsMultiAgentAnalysis(useFlowAnalysis);

    const response = await throttleRequest(async () => {
      if (useFlowAnalysis) {
        return await apiService.queryCodebaseFlow(userMessage.content, {
          model: null, // Use .env configuration
          limit: resultLimit,
          includeContext: includeContext,
          projectIds: selectedProjectIds.length > 0 ? selectedProjectIds : null
        });
      } else {
        return await apiService.queryCodebase(userMessage.content, {
          model: null, // Use .env configuration
          limit: resultLimit,
          includeContext: includeContext,
          projectIds: selectedProjectIds.length > 0 ? selectedProjectIds : null
        });
      }
    });

    if (!response) {
      setIsLoading(false);
      return;
    }

    try {

      // Check if response suggests streaming
      if (response.use_streaming) {
        // Handle streaming response
        await handleStreamingQuery(userMessage.content, useFlowAnalysis);
        return;
      }

      const assistantMessage = {
        id: Date.now() + 1,
        type: useFlowAnalysis ? 'flow_assistant' : 'assistant',
        content: response,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: error.message,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsMultiAgentAnalysis(false);
      inputRef.current?.focus();
    }
  };

  const handleStreamingQuery = async (query, useFlowAnalysis = false) => {
    // Create a streaming message placeholder
    const streamingMessageId = Date.now() + 1;
    const streamingMessage = {
      id: streamingMessageId,
      type: 'streaming',
      content: {
        status: 'Starting analysis...',
        progress: 0,
        events: [],
        final_response: null
      },
      timestamp: new Date()
    };

    setMessages(prev => {
      const newMessages = [...prev, streamingMessage];
      // Reduced logging to prevent spam
      // console.log('🔥 Added streaming message with ID:', streamingMessageId);
      // console.log('🔥 Total messages after adding:', newMessages.length);
      // console.log('🔥 Streaming message content:', streamingMessage.content);
      return newMessages;
    });

    // Turn off the hardcoded loading state since we now have a streaming message
    setIsLoading(false);

    try {
      // Reduced logging to prevent spam
      // console.log('Making streaming request to /mcp/query/stream');
      // console.log('Request payload:', { query, model: null, limit: resultLimit });

      // Start streaming query
      console.log('🔥 Making streaming request to /mcp/query/stream');
      const response = await fetch('/mcp/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          model: null, // Use .env configuration
          limit: resultLimit,
          include_context: includeContext,
          project_ids: selectedProjectIds.length > 0 ? selectedProjectIds : null
        }),
        // Add signal for timeout control
        signal: AbortSignal.timeout(300000) // 5 minute timeout
      });

      // Reduced logging to prevent spam
      console.log('Response received:', response.status, response.statusText);

      if (!response.ok) {
        console.error('Response not OK:', response.status, response.statusText);
        if (response.status === 429) {
          throw new Error('Too many requests. Please wait a moment before trying again.');
        }
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      console.log('🔥 Starting to read response stream...');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('🔥 Stream ended');
          break;
        }

        // Decode the chunk and add to buffer
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines
        const lines = buffer.split('\n');
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || '';

        console.log('🔥 Processing', lines.length, 'complete lines');

        for (const line of lines) {
          if (line.trim() === '') continue; // Skip empty lines

          console.log('🔥 Processing line:', line);
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue; // Skip empty data lines

              console.log('🔥 Parsing JSON:', jsonStr);
              const eventData = JSON.parse(jsonStr);

              // Enhanced debug logging
              console.log('🔥 Received streaming event:', eventData.event_type, eventData.message);
              console.log('🔥 Event data:', JSON.stringify(eventData, null, 2));

              // Update the streaming message with new event
              const currentTime = Date.now();

              setMessages(prev => {
                const newMessages = prev.map(msg => {
                  if (msg.id === streamingMessageId) {
                    const updatedMsg = {
                      ...msg,
                      content: {
                        ...msg.content,
                        status: eventData.message,
                        progress: eventData.progress_percentage || msg.content.progress,
                        events: [...msg.content.events, eventData],
                        lastUpdated: currentTime
                      }
                    };
                    return updatedMsg;
                  }
                  return msg;
                });

                return newMessages;
              });

              // Force a re-render by updating a separate state
              setForceUpdate(prev => prev + 1);

              // Check if processing is complete
              if (eventData.event_type === 'processing_complete' ||
                  eventData.event_type === 'StreamEventType.COMPLETE' ||
                  eventData.event_type === 'StreamEventType.PROCESSING_COMPLETE' ||
                  eventData.event_type === 'complete' ||
                  eventData.event_type === 'COMPLETE') {

                const finalResponse = eventData.data?.final_response || eventData.data;

                // Convert to final message
                setMessages(prev => prev.map(msg => {
                  if (msg.id === streamingMessageId) {
                    const updatedMsg = {
                      ...msg,
                      type: useFlowAnalysis ? 'flow_assistant' : 'assistant',
                      content: {
                        ...msg.content,
                        final_response: finalResponse,
                        status: 'Complete',
                        // Ensure arrays are preserved
                        events: msg.content.events || [],
                        results: msg.content.results || []
                      }
                    };
                    return updatedMsg;
                  }
                  return msg;
                }));
              }
            } catch (e) {
              console.error('Error parsing streaming event:', e, 'Line:', line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      // Update message to show error
      setMessages(prev => prev.map(msg => {
        if (msg.id === streamingMessageId) {
          return {
            ...msg,
            type: 'error',
            content: `Streaming error: ${error.message}`
          };
        }
        return msg;
      }));
    }
  };

  const testStreaming = async () => {
    console.log('🧪 Testing streaming functionality...');

    try {
      const response = await fetch('/mcp/test/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        console.error('❌ Test streaming failed:', response.status, response.statusText);
        return;
      }

      console.log('✅ Test streaming response received');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let eventCount = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          console.log('🧪 Test streaming completed, received', eventCount, 'events');
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(line.slice(6));
              eventCount++;
              console.log('🧪 Test event:', eventData.event_type, '-', eventData.message);
            } catch (e) {
              console.error('🧪 Test JSON parse error:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('🧪 Test streaming error:', error);
    }
  };

  const formatFilePath = (filePath) => {
    if (!filePath) return '';
    const parts = filePath.split('/');
    if (parts.length > 2) {
      return `.../${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
    }
    return filePath;
  };

  const getLanguageFromPath = (filePath) => {
    if (!filePath) return 'text';
    const extension = filePath.split('.').pop()?.toLowerCase();
    const languageMap = {
      'py': 'python',
      'js': 'javascript',
      'jsx': 'javascript',
      'ts': 'typescript',
      'tsx': 'typescript',
      'java': 'java',
      'cpp': 'cpp',
      'c': 'c',
      'go': 'go',
      'rs': 'rust'
    };
    return languageMap[extension] || 'text';
  };

  const formatAgentRole = (role) => {
    return role.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getAgentRoleIcon = (role) => {
    const iconMap = {
      'architect': Settings,
      'developer': Code,
      'security': AlertTriangle,
      'performance': Target,
      'maintainer': CheckCircle,
      'business': Users,
      'integration': FileText,
      'data': FileText,
      'ui_ux': Settings,
      'devops': Settings,
      'testing': CheckCircle,
      'compliance': AlertTriangle
    };
    return iconMap[role] || Brain;
  };

  const getAgentRoleColorClasses = (role) => {
    const colorMap = {
      'architect': {
        bg: 'bg-blue-50',
        border: 'border-blue-200',
        hover: 'hover:bg-blue-100',
        text: 'text-blue-900',
        textSecondary: 'text-blue-800',
        textTertiary: 'text-blue-600',
        bgSecondary: 'bg-blue-100',
        dot: 'bg-blue-600'
      },
      'developer': {
        bg: 'bg-green-50',
        border: 'border-green-200',
        hover: 'hover:bg-green-100',
        text: 'text-green-900',
        textSecondary: 'text-green-800',
        textTertiary: 'text-green-600',
        bgSecondary: 'bg-green-100',
        dot: 'bg-green-600'
      },
      'security': {
        bg: 'bg-red-50',
        border: 'border-red-200',
        hover: 'hover:bg-red-100',
        text: 'text-red-900',
        textSecondary: 'text-red-800',
        textTertiary: 'text-red-600',
        bgSecondary: 'bg-red-100',
        dot: 'bg-red-600'
      },
      'performance': {
        bg: 'bg-purple-50',
        border: 'border-purple-200',
        hover: 'hover:bg-purple-100',
        text: 'text-purple-900',
        textSecondary: 'text-purple-800',
        textTertiary: 'text-purple-600',
        bgSecondary: 'bg-purple-100',
        dot: 'bg-purple-600'
      },
      'maintainer': {
        bg: 'bg-yellow-50',
        border: 'border-yellow-200',
        hover: 'hover:bg-yellow-100',
        text: 'text-yellow-900',
        textSecondary: 'text-yellow-800',
        textTertiary: 'text-yellow-600',
        bgSecondary: 'bg-yellow-100',
        dot: 'bg-yellow-600'
      },
      'business': {
        bg: 'bg-indigo-50',
        border: 'border-indigo-200',
        hover: 'hover:bg-indigo-100',
        text: 'text-indigo-900',
        textSecondary: 'text-indigo-800',
        textTertiary: 'text-indigo-600',
        bgSecondary: 'bg-indigo-100',
        dot: 'bg-indigo-600'
      },
      'integration': {
        bg: 'bg-pink-50',
        border: 'border-pink-200',
        hover: 'hover:bg-pink-100',
        text: 'text-pink-900',
        textSecondary: 'text-pink-800',
        textTertiary: 'text-pink-600',
        bgSecondary: 'bg-pink-100',
        dot: 'bg-pink-600'
      },
      'data': {
        bg: 'bg-cyan-50',
        border: 'border-cyan-200',
        hover: 'hover:bg-cyan-100',
        text: 'text-cyan-900',
        textSecondary: 'text-cyan-800',
        textTertiary: 'text-cyan-600',
        bgSecondary: 'bg-cyan-100',
        dot: 'bg-cyan-600'
      },
      'ui_ux': {
        bg: 'bg-orange-50',
        border: 'border-orange-200',
        hover: 'hover:bg-orange-100',
        text: 'text-orange-900',
        textSecondary: 'text-orange-800',
        textTertiary: 'text-orange-600',
        bgSecondary: 'bg-orange-100',
        dot: 'bg-orange-600'
      },
      'devops': {
        bg: 'bg-gray-50',
        border: 'border-gray-200',
        hover: 'hover:bg-gray-100',
        text: 'text-gray-900',
        textSecondary: 'text-gray-800',
        textTertiary: 'text-gray-600',
        bgSecondary: 'bg-gray-100',
        dot: 'bg-gray-600'
      },
      'testing': {
        bg: 'bg-emerald-50',
        border: 'border-emerald-200',
        hover: 'hover:bg-emerald-100',
        text: 'text-emerald-900',
        textSecondary: 'text-emerald-800',
        textTertiary: 'text-emerald-600',
        bgSecondary: 'bg-emerald-100',
        dot: 'bg-emerald-600'
      },
      'compliance': {
        bg: 'bg-rose-50',
        border: 'border-rose-200',
        hover: 'hover:bg-rose-100',
        text: 'text-rose-900',
        textSecondary: 'text-rose-800',
        textTertiary: 'text-rose-600',
        bgSecondary: 'bg-rose-100',
        dot: 'bg-rose-600'
      }
    };
    return colorMap[role] || colorMap['devops']; // Default to gray
  };

  const toggleComponentExpansion = (messageId, componentIndex) => {
    const key = `${messageId}-${componentIndex}`;
    setExpandedComponents(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const findComponentCode = (component, results) => {
    // For multi-agent analysis, try to find relevant code chunks based on component content
    if (!results || results.length === 0) return null;

    // Try to find chunks that match the component's purpose or name
    const matchingChunks = results.filter(result => {
      const chunk = result.chunk;
      const componentName = component.name.toLowerCase();
      const componentPurpose = component.purpose.toLowerCase();
      const chunkContent = chunk.content.toLowerCase();
      const chunkName = (chunk.name || '').toLowerCase();

      // Check if the chunk is relevant to this component
      return (
        // Direct name match
        chunkName.includes(componentName.replace(/\s+/g, '')) ||
        // Content relevance
        chunkContent.includes(componentName.replace(/\s+/g, '')) ||
        // Purpose-based matching for common patterns
        (componentPurpose.includes('dto') && (chunkName.includes('dto') || chunkContent.includes('dto'))) ||
        (componentPurpose.includes('model') && (chunkName.includes('model') || chunkContent.includes('class'))) ||
        (componentPurpose.includes('request') && (chunkName.includes('request') || chunkContent.includes('request'))) ||
        (componentPurpose.includes('response') && (chunkName.includes('response') || chunkContent.includes('response'))) ||
        (componentPurpose.includes('validation') && (chunkName.includes('valid') || chunkContent.includes('valid'))) ||
        (componentPurpose.includes('pydantic') && chunkContent.includes('pydantic')) ||
        (componentPurpose.includes('architecture') && (chunkName.includes('server') || chunkName.includes('main'))) ||
        (componentPurpose.includes('separation of concerns') && (chunkName.includes('server') || chunkName.includes('client')))
      );
    });

    // Return the highest scoring match
    return matchingChunks.length > 0 ?
      matchingChunks.sort((a, b) => b.score - a.score)[0] :
      null;
  };

  const renderMessage = (message) => {
    switch (message.type) {
      case 'user':
        return (
          <div className="flex justify-end mb-4">
            <div className="max-w-3xl bg-primary-600 text-white rounded-lg px-4 py-2">
              <p className="whitespace-pre-wrap">{message.content}</p>
              <div className="flex items-center justify-between mt-2">
                <div className="text-xs text-primary-200">
                  {message.timestamp.toLocaleTimeString()}
                </div>
                {message.flowAnalysis && (
                  <div className="flex items-center space-x-1 text-xs text-primary-200">
                    <Brain className="w-3 h-3" />
                    <span>Multi-Agent Analysis</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        );

      case 'assistant':
        return (
          <div key={message.id} className="flex justify-start mb-6">
            <div className="max-w-5xl bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              {/* Header */}
              <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Search className="w-4 h-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">
                      Found {message.content.total_results} results
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {(message.content.processing_time * 1000).toFixed(0)}ms • {message.content.model_used}
                  </div>
                </div>
              </div>

              {/* Intelligent Analysis */}
              {message.content.analysis && (
                <div className="p-4 space-y-4">
                  {/* Summary */}
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Search className="w-5 h-5 text-blue-600" />
                      <h3 className="text-lg font-medium text-blue-900">Analysis Summary</h3>
                    </div>
                    <div className="text-blue-800">
                      <MarkdownRenderer content={message.content.analysis.summary} />
                    </div>
                  </div>

                  {/* Detailed Explanation */}
                  {message.content.analysis.detailed_explanation && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <FileText className="w-5 h-5 text-green-600" />
                        <h3 className="text-lg font-medium text-green-900">How It Works</h3>
                      </div>
                      <div className="text-green-800">
                        <MarkdownRenderer content={message.content.analysis.detailed_explanation} />
                      </div>
                    </div>
                  )}

                  {/* Code Flow */}
                  {message.content.analysis.code_flow && message.content.analysis.code_flow.length > 0 && (
                    <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-3">
                        <Code className="w-5 h-5 text-purple-600" />
                        <h3 className="text-lg font-medium text-purple-900">Execution Flow</h3>
                      </div>
                      <ol className="space-y-2">
                        {message.content.analysis.code_flow.map((step, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <span className="flex-shrink-0 w-6 h-6 bg-purple-600 text-white text-xs rounded-full flex items-center justify-center font-medium">
                              {index + 1}
                            </span>
                            <span className="text-purple-800">{step}</span>
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Key Components */}
                  {message.content.analysis.key_components && message.content.analysis.key_components.length > 0 && (
                    <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-3">
                        <Settings className="w-5 h-5 text-orange-600" />
                        <h3 className="text-lg font-medium text-orange-900">Key Components</h3>
                      </div>
                      <div className="space-y-3">
                        {message.content.analysis.key_components.map((component, index) => {
                          const componentKey = `${message.id}-${index}`;
                          const isExpanded = expandedComponents[componentKey];
                          const associatedCode = findComponentCode(component, message.content.results);

                          return (
                            <div key={index} className="bg-white rounded-lg border border-orange-200 overflow-hidden">
                              {/* Component Header - Clickable */}
                              <div
                                className="p-3 cursor-pointer hover:bg-orange-50 transition-colors"
                                onClick={() => toggleComponentExpansion(message.id, index)}
                              >
                                <div className="flex items-center justify-between">
                                  <div className="flex items-center space-x-2">
                                    <h4 className="font-medium text-orange-900">{component.name}</h4>
                                    <span className="text-xs text-orange-600 bg-orange-100 px-2 py-1 rounded">
                                      {component.location}
                                    </span>
                                  </div>
                                  <div className="flex items-center space-x-2">
                                    {associatedCode && (
                                      <span className="text-xs text-orange-500">
                                        {isExpanded ? 'Click to hide code' : 'Click to view code'}
                                      </span>
                                    )}
                                    {isExpanded ? (
                                      <ChevronUp className="w-4 h-4 text-orange-600" />
                                    ) : (
                                      <ChevronDown className="w-4 h-4 text-orange-600" />
                                    )}
                                  </div>
                                </div>
                                <p className="text-orange-800 text-sm mt-1">{component.purpose}</p>
                              </div>

                              {/* Expanded Content */}
                              {isExpanded && (
                                <div className="border-t border-orange-200">
                                  {associatedCode ? (
                                    <div className="p-0">
                                      {/* Code Header */}
                                      <div className="bg-orange-100 px-3 py-2 border-b border-orange-200">
                                        <div className="flex items-center justify-between">
                                          <div className="flex items-center space-x-2">
                                            <Code className="w-4 h-4 text-orange-600" />
                                            <span className="text-sm font-medium text-orange-800">
                                              {associatedCode.chunk.name || 'Code Implementation'}
                                            </span>
                                            <span className="text-xs text-orange-600">
                                              ({associatedCode.chunk.node_type})
                                            </span>
                                          </div>
                                          <span className="text-xs text-orange-600">
                                            Score: {(associatedCode.score * 100).toFixed(1)}%
                                          </span>
                                        </div>
                                      </div>

                                      {/* Code Content */}
                                      <div className="relative">
                                        <SyntaxHighlighter
                                          language={getLanguageFromPath(associatedCode.chunk.file_path)}
                                          style={vscDarkPlus}
                                          customStyle={{
                                            margin: 0,
                                            fontSize: '12px',
                                            maxHeight: '300px',
                                            overflow: 'auto'
                                          }}
                                          showLineNumbers={true}
                                          startingLineNumber={associatedCode.chunk.start_line}
                                        >
                                          {associatedCode.chunk.content}
                                        </SyntaxHighlighter>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="p-3 text-center text-orange-600">
                                      <Code className="w-8 h-8 mx-auto mb-2 opacity-50" />
                                      <p className="text-sm">No associated code found for this component</p>
                                      <p className="text-xs text-orange-500 mt-1">
                                        The code might be in a different file or not indexed
                                      </p>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Recommendations */}
                  {message.content.analysis.recommendations && message.content.analysis.recommendations.length > 0 && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-3">
                        <Loader2 className="w-5 h-5 text-yellow-600" />
                        <h3 className="text-lg font-medium text-yellow-900">Recommendations</h3>
                      </div>
                      <ul className="space-y-2">
                        {message.content.analysis.recommendations.map((recommendation, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <span className="flex-shrink-0 w-2 h-2 bg-yellow-600 rounded-full mt-2"></span>
                            <span className="text-yellow-800">{recommendation}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Supporting Code Evidence */}
              <div className="border-t border-gray-200">
                <div className="bg-gray-50 px-4 py-2">
                  <h3 className="text-sm font-medium text-gray-700 flex items-center">
                    <Code className="w-4 h-4 mr-2" />
                    Supporting Code Evidence
                  </h3>
                </div>
                <div className="p-4 space-y-4">
                  {message.content.results && message.content.results.slice(0, expandedResults[message.id] ? message.content.results.length : 3).map((result, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg overflow-hidden">
                      {/* Result Header */}
                      <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <Code className="w-4 h-4 text-gray-500" />
                            <span className="text-sm font-medium text-gray-700">
                              {result.chunk.name || 'Unnamed'}
                            </span>
                            <span className="text-xs text-gray-500">
                              ({result.chunk.node_type})
                            </span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-500">
                              Score: {(result.score * 100).toFixed(1)}%
                            </span>
                            <span className="text-xs text-gray-500">
                              {formatFilePath(result.chunk.file_path)}:{result.chunk.start_line}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Code Content */}
                      <div className="relative">
                        <SyntaxHighlighter
                          language={getLanguageFromPath(result.chunk.file_path)}
                          style={vscDarkPlus}
                          customStyle={{
                            margin: 0,
                            fontSize: '13px',
                            maxHeight: '200px',
                            overflow: 'auto'
                          }}
                          showLineNumbers={true}
                          startingLineNumber={result.chunk.start_line}
                        >
                          {result.chunk.content}
                        </SyntaxHighlighter>
                      </div>

                      {/* Context */}
                      {result.context_chunks && result.context_chunks.length > 0 && (
                        <div className="bg-blue-50 px-3 py-2 border-t border-gray-200">
                          <div className="text-xs text-blue-700 font-medium mb-1">
                            Related Context ({result.context_chunks.length} items):
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {result.context_chunks && result.context_chunks.slice(0, 5).map((contextChunk, contextIndex) => (
                              <span
                                key={contextIndex}
                                className="inline-flex items-center px-2 py-1 rounded text-xs bg-blue-100 text-blue-700"
                              >
                                <FileText className="w-3 h-3 mr-1" />
                                {contextChunk.name || 'Unnamed'}
                              </span>
                            ))}
                            {result.context_chunks.length > 5 && (
                              <span className="text-xs text-blue-600">
                                +{result.context_chunks.length - 5} more
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}

                  {message.content.results.length > 3 && !expandedResults[message.id] && (
                    <div className="text-center py-2">
                      <button
                        onClick={() => handleExpandResults(message.id)}
                        className="inline-flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 hover:underline focus:outline-none focus:underline transition-colors"
                      >
                        <ChevronDown className="w-4 h-4" />
                        <span>+{message.content.results.length - 3} more results available</span>
                      </button>
                    </div>
                  )}

                  {message.content.results.length > 3 && expandedResults[message.id] && (
                    <div className="text-center py-2">
                      <button
                        onClick={() => handleExpandResults(message.id)}
                        className="inline-flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 hover:underline focus:outline-none focus:underline transition-colors"
                      >
                        <ChevronUp className="w-4 h-4" />
                        <span>Show fewer results</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );

      case 'flow_assistant':
        return (
          <div key={message.id} className="flex justify-start mb-6">
            <div className="max-w-6xl bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 px-4 py-3 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-blue-600" />
                    <span className="text-lg font-semibold text-gray-800">
                      Multi-Agent Analysis
                    </span>
                    <span className="text-sm text-gray-600">
                      ({message.content.agent_perspectives?.length || 0} expert perspectives)
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {(message.content.processing_time * 1000).toFixed(0)}ms
                  </div>
                </div>
              </div>

              {/* Executive Summary */}
              {message.content.executive_summary && (
                <div className="p-4 bg-blue-50 border-b border-blue-200">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-5 h-5 text-blue-600" />
                    <h3 className="text-lg font-medium text-blue-900">Executive Summary</h3>
                  </div>
                  <div className="text-blue-800 leading-relaxed">
                    <MarkdownRenderer content={message.content.executive_summary} />
                  </div>
                </div>
              )}

              {/* Agent Perspectives */}
              {message.content.agent_perspectives && message.content.agent_perspectives.length > 0 && (
                <div className="p-4 space-y-4">
                  <div className="flex items-center space-x-2 mb-4">
                    <Users className="w-5 h-5 text-gray-600" />
                    <h3 className="text-lg font-medium text-gray-900">Expert Perspectives</h3>
                  </div>

                  <div className="grid gap-4">
                    {message.content.agent_perspectives.map((perspective, index) => {
                      const perspectiveKey = `${message.id}-${index}`;
                      const isExpanded = expandedAgentPerspectives[perspectiveKey];
                      const IconComponent = getAgentRoleIcon(perspective.role);
                      const colors = getAgentRoleColorClasses(perspective.role);

                      return (
                        <div key={index} className={`${colors.bg} ${colors.border} rounded-lg overflow-hidden`}>
                          {/* Agent Header - Clickable */}
                          <div
                            className={`p-4 cursor-pointer ${colors.hover} transition-colors`}
                            onClick={() => handleExpandAgentPerspective(message.id, index)}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <div className={`p-2 ${colors.bgSecondary} rounded-lg`}>
                                  <IconComponent className={`w-5 h-5 ${colors.textTertiary}`} />
                                </div>
                                <div>
                                  <h4 className={`font-semibold ${colors.text}`}>
                                    {formatAgentRole(perspective.role)}
                                  </h4>
                                  <div className="flex items-center space-x-2 mt-1">
                                    <span className={`text-xs ${colors.textTertiary} ${colors.bgSecondary} px-2 py-1 rounded`}>
                                      Confidence: {(perspective.confidence * 100).toFixed(0)}%
                                    </span>
                                    {perspective.focus_areas && perspective.focus_areas.length > 0 && (
                                      <span className={`text-xs ${colors.textTertiary}`}>
                                        Focus: {perspective.focus_areas.join(', ')}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className={`text-xs ${colors.textTertiary}`}>
                                  {isExpanded ? 'Click to collapse' : 'Click to expand'}
                                </span>
                                {isExpanded ? (
                                  <ChevronUp className={`w-4 h-4 ${colors.textTertiary}`} />
                                ) : (
                                  <ChevronDown className={`w-4 h-4 ${colors.textTertiary}`} />
                                )}
                              </div>
                            </div>

                            {/* Brief Analysis Preview */}
                            <div className={`${colors.textSecondary} text-sm mt-2 line-clamp-2`}>
                              <MarkdownRenderer content={perspective.analysis} />
                            </div>
                          </div>

                          {/* Expanded Content */}
                          {isExpanded && (
                            <div className={`${colors.border} border-t p-4 space-y-4`}>
                              {/* Full Analysis */}
                              <div>
                                <h5 className={`font-medium ${colors.text} mb-2`}>Detailed Analysis</h5>
                                <div className={`${colors.textSecondary} leading-relaxed`}>
                                  <MarkdownRenderer content={perspective.analysis} />
                                </div>
                              </div>

                              {/* Key Insights */}
                              {perspective.key_insights && perspective.key_insights.length > 0 && (
                                <div>
                                  <h5 className={`font-medium ${colors.text} mb-2`}>Key Insights</h5>
                                  <ul className="space-y-1">
                                    {perspective.key_insights.map((insight, insightIndex) => (
                                      <li key={insightIndex} className="flex items-start space-x-2">
                                        <span className={`flex-shrink-0 w-2 h-2 ${colors.dot} rounded-full mt-2`}></span>
                                        <span className={`${colors.textSecondary} text-sm`}>{insight}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}

                              {/* Recommendations */}
                              {perspective.recommendations && perspective.recommendations.length > 0 && (
                                <div>
                                  <h5 className={`font-medium ${colors.text} mb-2`}>Recommendations</h5>
                                  <ul className="space-y-1">
                                    {perspective.recommendations.map((recommendation, recIndex) => (
                                      <li key={recIndex} className="flex items-start space-x-2">
                                        <CheckCircle className={`flex-shrink-0 w-4 h-4 ${colors.textTertiary} mt-0.5`} />
                                        <span className={`${colors.textSecondary} text-sm`}>{recommendation}</span>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Detailed Analysis */}
              {message.content.detailed_analysis && (
                <div className="p-4 bg-green-50 border-t border-green-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <FileText className="w-5 h-5 text-green-600" />
                    <h3 className="text-lg font-medium text-green-900">Comprehensive Analysis</h3>
                  </div>
                  <div className="text-green-800 leading-relaxed">
                    <MarkdownRenderer content={message.content.detailed_analysis} />
                  </div>
                </div>
              )}

              {/* Synthesis */}
              {message.content.synthesis && (
                <div className="p-4 bg-purple-50 border-t border-purple-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <Brain className="w-5 h-5 text-purple-600" />
                    <h3 className="text-lg font-medium text-purple-900">Synthesis</h3>
                  </div>
                  <div className="text-purple-800 leading-relaxed">
                    <MarkdownRenderer content={message.content.synthesis} />
                  </div>
                </div>
              )}

              {/* Action Items */}
              {message.content.action_items && message.content.action_items.length > 0 && (
                <div className="p-4 bg-yellow-50 border-t border-yellow-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <Target className="w-5 h-5 text-yellow-600" />
                    <h3 className="text-lg font-medium text-yellow-900">Action Items</h3>
                  </div>
                  <ol className="space-y-2">
                    {message.content.action_items.map((item, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="flex-shrink-0 w-6 h-6 bg-yellow-600 text-white text-xs rounded-full flex items-center justify-center font-medium">
                          {index + 1}
                        </span>
                        <span className="text-yellow-800">{item}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {/* Follow-up Questions */}
              {message.content.follow_up_questions && message.content.follow_up_questions.length > 0 && (
                <div className="p-4 bg-indigo-50 border-t border-indigo-200">
                  <div className="flex items-center space-x-2 mb-3">
                    <Search className="w-5 h-5 text-indigo-600" />
                    <h3 className="text-lg font-medium text-indigo-900">Follow-up Questions</h3>
                  </div>
                  <ul className="space-y-2">
                    {message.content.follow_up_questions.map((question, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="flex-shrink-0 w-2 h-2 bg-indigo-600 rounded-full mt-2"></span>
                        <span className="text-indigo-800">{question}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        );

      case 'error':
        return (
          <div className="flex justify-start mb-4">
            <div className="max-w-3xl bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-2">
              <p className="font-medium">Error:</p>
              <p>{message.content}</p>
              <div className="text-xs text-red-500 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        );

      case 'streaming':
        return (
          <div className="flex justify-start mb-6">
            <StreamingMessage
              message={message}
              forceUpdate={forceUpdate}
              renderCount={renderCount}
            />
          </div>
        );

      default:
        return null;
    }
  };

  const renderFinalResponse = (finalResponse) => {
    console.log('Rendering final response:', finalResponse);

    // Handle null or undefined response
    if (!finalResponse) {
      return (
        <div className="text-sm text-gray-500">
          No response data available.
        </div>
      );
    }

    // Check if it's a flow response (from agents) or search results
    if (finalResponse.executive_summary || finalResponse.agent_perspectives) {
      // Flow response from agents
      return (
        <div className="space-y-4">
          {finalResponse.executive_summary && (
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Executive Summary</h4>
              <div className="text-sm text-gray-700">
                <MarkdownRenderer content={finalResponse.executive_summary} />
              </div>
            </div>
          )}

          {finalResponse.agent_perspectives && finalResponse.agent_perspectives.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Expert Analysis ({finalResponse.agent_perspectives.length} perspectives)</h4>
              <div className="space-y-3">
                {finalResponse.agent_perspectives.map((perspective, index) => (
                  <div key={index} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center mb-2">
                      <span className="text-sm font-medium text-blue-600">
                        {perspective.role.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())} Expert
                      </span>
                      <span className="ml-2 text-xs text-gray-500">
                        Confidence: {Math.round(perspective.confidence * 100)}%
                      </span>
                    </div>
                    <div className="text-sm text-gray-700 mb-2">
                      <MarkdownRenderer content={perspective.analysis} />
                    </div>

                    {perspective.key_insights && perspective.key_insights.length > 0 && (
                      <div className="mb-2">
                        <span className="text-xs font-medium text-gray-600">Key Insights:</span>
                        <ul className="text-xs text-gray-600 ml-2 mt-1">
                          {perspective.key_insights.map((insight, i) => (
                            <li key={i} className="mb-1">• {insight}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {perspective.recommendations && perspective.recommendations.length > 0 && (
                      <div>
                        <span className="text-xs font-medium text-gray-600">Recommendations:</span>
                        <ul className="text-xs text-gray-600 ml-2 mt-1">
                          {perspective.recommendations.map((rec, i) => (
                            <li key={i} className="mb-1">• {rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {finalResponse.synthesis && (
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Synthesis</h4>
              <div className="text-sm text-gray-700">
                <MarkdownRenderer content={finalResponse.synthesis} />
              </div>
            </div>
          )}

          {finalResponse.action_items && finalResponse.action_items.length > 0 && (
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Action Items</h4>
              <ul className="text-sm text-gray-700 ml-4">
                {finalResponse.action_items.map((item, index) => (
                  <li key={index} className="mb-1">• {item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    } else if (Array.isArray(finalResponse)) {
      // Search results
      return (
        <div>
          <h4 className="font-medium text-gray-900 mb-2">Search Results ({finalResponse.length} items)</h4>
          <div className="space-y-2">
            {finalResponse && finalResponse.slice(0, 5).map((result, index) => (
              <div key={index} className="border border-gray-200 rounded p-2">
                <div className="text-sm font-medium text-gray-900">{result.chunk.name || 'Code Section'}</div>
                <div className="text-xs text-gray-500">{result.chunk.file_path}</div>
                <div className="text-xs text-gray-600 mt-1">Score: {(result.score * 100).toFixed(1)}%</div>
              </div>
            ))}
            {finalResponse.length > 5 && (
              <div className="text-xs text-gray-500">... and {finalResponse.length - 5} more results</div>
            )}
          </div>
        </div>
      );
    } else {
      // Fallback for other response types
      return (
        <div className="text-sm text-gray-700">
          Response ready - {finalResponse.results_count || 'Unknown'} items found
        </div>
      );
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Query Settings</h3>
            <button
              onClick={() => setShowSettings(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          <div className="space-y-4">
            {/* Project Selection */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <FolderOpen className="w-4 h-4 text-blue-600" />
                <label className="block text-sm font-medium text-blue-900">
                  Project Knowledge Selection
                </label>
              </div>
              <ProjectSelector
                selectedProjectIds={selectedProjectIds}
                onSelectionChange={setSelectedProjectIds}
                className="w-full"
              />
              <div className="mt-2 text-xs">
                {selectedProjectIds.length === 0 ? (
                  <p className="text-amber-600 bg-amber-50 px-2 py-1 rounded border border-amber-200">
                    ⚠️ No projects selected - searching across all indexed projects
                  </p>
                ) : (
                  <p className="text-blue-600">
                    ✅ Chat will use knowledge from {selectedProjectIds.length} selected project{selectedProjectIds.length > 1 ? 's' : ''}
                  </p>
                )}
              </div>
              <div className="flex items-center justify-between mt-2">
                <p className="text-xs text-gray-600">
                  💡 Select specific projects to focus the chat on relevant codebase knowledge.
                  Unselected projects will be excluded from search results.
                </p>
                {selectedProjectIds.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setSelectedProjectIds([])}
                    className="text-xs text-blue-600 hover:text-blue-800 underline ml-2"
                  >
                    Clear selection
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Embedding Model selection removed - now uses .env configuration only */}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Result Limit
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={resultLimit}
                onChange={(e) => setResultLimit(parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Options
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={includeContext}
                  onChange={(e) => setIncludeContext(e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Include context</span>
              </label>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Selected Projects Display */}
        {selectedProjectIds.length > 0 && (
          <div className="mb-4">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FolderOpen className="w-4 h-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-900">
                    Active Projects:
                  </span>
                </div>
                <button
                  onClick={() => setSelectedProjectIds([])}
                  className="text-blue-600 hover:text-blue-800 p-1 rounded transition-colors"
                  title="Clear project selection"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {getSelectedProjectNames().map((projectName, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                  >
                    {projectName}
                  </span>
                ))}
              </div>
              <p className="text-xs text-blue-600 mt-2">
                Chat responses will be limited to knowledge from these projects only
              </p>
            </div>
          </div>
        )}

        {messages.length === 0 && (
          <div className="text-center py-12">
            <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Search Your Codebase
            </h3>
            <p className="text-gray-600 max-w-md mx-auto mb-4">
              Ask questions about your code using natural language.
              Use quick search for fast results, or comprehensive analysis for multi-expert perspectives.
            </p>

            {/* Project Selection Status */}
            <div className="max-w-md mx-auto">
              {selectedProjectIds.length === 0 ? (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                  <div className="flex items-center justify-center space-x-2 text-amber-700">
                    <FolderOpen className="w-4 h-4" />
                    <span className="text-sm font-medium">Searching all projects</span>
                  </div>
                  <p className="text-xs text-amber-600 mt-1">
                    Click the settings button to select specific projects
                  </p>
                </div>
              ) : (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <div className="flex items-center justify-center space-x-2 text-blue-700">
                    <FolderOpen className="w-4 h-4" />
                    <span className="text-sm font-medium">
                      Searching {selectedProjectIds.length} selected project{selectedProjectIds.length > 1 ? 's' : ''}
                    </span>
                  </div>
                  <p className="text-xs text-blue-600 mt-1">
                    Knowledge limited to selected projects only
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={`${message.id}-${message.content?.lastUpdated || 0}-${forceUpdate}`}>
            {renderMessage(message)}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 px-4 py-3">
              <div className="flex items-center space-x-2">
                {isMultiAgentAnalysis ? (
                  <>
                    <Brain className="w-4 h-4 animate-pulse text-purple-600" />
                    <span className="text-gray-600">
                      Running multi-agent analysis... This may take up to 2 minutes.
                    </span>
                  </>
                ) : (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-primary-600" />
                    <span className="text-gray-600">Searching codebase...</span>
                  </>
                )}
              </div>
              {isMultiAgentAnalysis && (
                <div className="mt-2 text-xs text-gray-500">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    <span className="ml-2">Consulting 6-10 expert agents for comprehensive insights</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <form onSubmit={handleSubmit} className="flex items-end space-x-2">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask about your codebase... (e.g., 'find authentication functions', 'show me error handling')"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 resize-none"
              rows="2"
              disabled={isLoading}
            />
          </div>

          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
            title="Query Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          <button
            type="button"
            onClick={testStreaming}
            className="p-2 text-orange-400 hover:text-orange-600 hover:bg-orange-100 rounded-lg transition-colors"
            title="Test Streaming (Debug)"
          >
            <span className="text-xs">🧪</span>
          </button>

          <button
            type="button"
            onClick={(e) => handleSubmit(e, true)}
            disabled={!inputValue.trim() || isLoading}
            className="px-3 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-1"
            title="Multi-Agent Analysis - Comprehensive analysis from multiple expert perspectives"
          >
            <Brain className="w-4 h-4" />
            <span className="text-sm font-medium">Analyze</span>
          </button>

          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading || isProcessingRequest}
            className="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title={isProcessingRequest ? "Request throttled - please wait" : "Quick Search"}
          >
            <Send className="w-5 h-5" />
          </button>
        </form>

        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <span>Press Enter for quick search • Use "Analyze" for comprehensive multi-agent analysis</span>
            {selectedProjectIds.length > 0 && (
              <div className="flex items-center space-x-1 text-blue-600">
                <FolderOpen className="w-3 h-3" />
                <span>{selectedProjectIds.length} project{selectedProjectIds.length > 1 ? 's' : ''} selected</span>
              </div>
            )}
          </div>
          {/* Model selection removed - using .env configuration */}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
