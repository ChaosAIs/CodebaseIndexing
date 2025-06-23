import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { MessageSquare, Network, Activity, Database, Cpu } from 'lucide-react';

const Sidebar = ({ systemStatus }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      id: 'chat',
      label: 'Chat Interface',
      icon: MessageSquare,
      path: '/chat',
      description: 'Query your codebase with natural language'
    },
    {
      id: 'graph',
      label: 'Graph View',
      icon: Network,
      path: '/graph',
      description: 'Visualize code relationships and dependencies'
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
      case 'available':
        return 'text-green-500';
      case 'unhealthy':
      case 'unavailable':
        return 'text-red-500';
      default:
        return 'text-yellow-500';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
      case 'available':
        return '●';
      case 'unhealthy':
      case 'unavailable':
        return '●';
      default:
        return '●';
    }
  };

  return (
    <div className="w-64 bg-white shadow-lg border-r border-gray-200 flex flex-col">
      {/* Logo/Title */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
            <Database className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">CodeIndex</h2>
            <p className="text-xs text-gray-500">v1.0.0</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <button
                key={item.id}
                onClick={() => navigate(item.path)}
                className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 border border-primary-200'
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                }`}
              >
                <Icon className="w-5 h-5" />
                <div className="flex-1">
                  <div className="font-medium">{item.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">
                    {item.description}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </nav>

      {/* System Status */}
      {systemStatus && (
        <div className="p-4 border-t border-gray-200">
          <h3 className="text-sm font-medium text-gray-900 mb-3 flex items-center">
            <Activity className="w-4 h-4 mr-2" />
            System Status
          </h3>
          
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Qdrant</span>
              <span className={`flex items-center ${getStatusColor(systemStatus.qdrant_status)}`}>
                {getStatusIcon(systemStatus.qdrant_status)}
                <span className="ml-1">{systemStatus.qdrant_status}</span>
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Neo4j</span>
              <span className={`flex items-center ${getStatusColor(systemStatus.neo4j_status)}`}>
                {getStatusIcon(systemStatus.neo4j_status)}
                <span className="ml-1">{systemStatus.neo4j_status}</span>
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Ollama</span>
              <span className={`flex items-center ${getStatusColor(systemStatus.ollama_status)}`}>
                {getStatusIcon(systemStatus.ollama_status)}
                <span className="ml-1">{systemStatus.ollama_status}</span>
              </span>
            </div>
          </div>

          {/* Statistics */}
          <div className="mt-4 pt-3 border-t border-gray-100">
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>Files Indexed:</span>
                <span className="font-medium">{systemStatus.indexed_files || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Total Chunks:</span>
                <span className="font-medium">{systemStatus.total_chunks || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Models:</span>
                <span className="font-medium">{systemStatus.available_models?.length || 0}</span>
              </div>
            </div>
          </div>

          {/* Available Models */}
          {systemStatus.available_models && systemStatus.available_models.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-100">
              <div className="text-xs text-gray-600 mb-2">Available Models:</div>
              <div className="flex flex-wrap gap-1">
                {systemStatus.available_models.map((model) => (
                  <span
                    key={model}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary-100 text-primary-700"
                  >
                    <Cpu className="w-3 h-3 mr-1" />
                    {model}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Sidebar;
