import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Code, FileText, Search, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { apiService } from '../services/apiService';

const ChatInterface = ({ systemStatus }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [includeContext, setIncludeContext] = useState(true);
  const [resultLimit, setResultLimit] = useState(10);
  const [expandedComponents, setExpandedComponents] = useState({});
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Set default model when system status loads
    if (systemStatus?.available_models?.length > 0 && !selectedModel) {
      setSelectedModel(systemStatus.available_models[0]);
    }
  }, [systemStatus, selectedModel]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiService.queryCodebase(userMessage.content, {
        model: selectedModel || null,
        limit: resultLimit,
        includeContext: includeContext
      });

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
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
      inputRef.current?.focus();
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

  const toggleComponentExpansion = (messageId, componentIndex) => {
    const key = `${messageId}-${componentIndex}`;
    setExpandedComponents(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const findComponentCode = (component, results) => {
    // Find the code chunk that matches this component
    const locationParts = component.location.split(':');
    const filePath = locationParts[0];
    const lineNumber = parseInt(locationParts[1]);

    // Look for matching chunks in results
    const matchingChunks = results.filter(result => {
      const chunk = result.chunk;
      return chunk.file_path.includes(filePath) &&
             chunk.start_line <= lineNumber &&
             chunk.end_line >= lineNumber &&
             (chunk.name === component.name || chunk.content.includes(component.name));
    });

    return matchingChunks.length > 0 ? matchingChunks[0] : null;
  };

  const renderMessage = (message) => {
    switch (message.type) {
      case 'user':
        return (
          <div key={message.id} className="flex justify-end mb-4">
            <div className="max-w-3xl bg-primary-600 text-white rounded-lg px-4 py-2">
              <p className="whitespace-pre-wrap">{message.content}</p>
              <div className="text-xs text-primary-200 mt-1">
                {message.timestamp.toLocaleTimeString()}
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
                    <p className="text-blue-800">{message.content.analysis.summary}</p>
                  </div>

                  {/* Detailed Explanation */}
                  {message.content.analysis.detailed_explanation && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <div className="flex items-center space-x-2 mb-2">
                        <FileText className="w-5 h-5 text-green-600" />
                        <h3 className="text-lg font-medium text-green-900">How It Works</h3>
                      </div>
                      <p className="text-green-800 whitespace-pre-wrap">{message.content.analysis.detailed_explanation}</p>
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
                  {message.content.results.slice(0, 3).map((result, index) => (
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
                            {result.context_chunks.slice(0, 5).map((contextChunk, contextIndex) => (
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

                  {message.content.results.length > 3 && (
                    <div className="text-center py-2">
                      <span className="text-sm text-gray-500">
                        +{message.content.results.length - 3} more results available
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );

      case 'error':
        return (
          <div key={message.id} className="flex justify-start mb-4">
            <div className="max-w-3xl bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-2">
              <p className="font-medium">Error:</p>
              <p>{message.content}</p>
              <div className="text-xs text-red-500 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        );

      default:
        return null;
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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Embedding Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">Default</option>
                {systemStatus?.available_models?.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>

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
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="text-center py-12">
            <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Search Your Codebase
            </h3>
            <p className="text-gray-600 max-w-md mx-auto">
              Ask questions about your code using natural language.
              Try queries like "find authentication functions" or "show me error handling code".
            </p>
          </div>
        )}

        {messages.map(renderMessage)}

        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 px-4 py-3">
              <div className="flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin text-primary-600" />
                <span className="text-gray-600">Searching codebase...</span>
              </div>
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
          >
            <Settings className="w-5 h-5" />
          </button>

          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="p-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>

        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <span>Press Enter to send, Shift+Enter for new line</span>
          {selectedModel && (
            <span>Using: {selectedModel}</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
