import React, { useState } from 'react';
import { 
  Database, 
  BarChart3, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  TrendingUp
} from 'lucide-react';
import ProjectList from './ProjectList';
import { useProjectsStatus } from '../hooks/useProjectStatus';

const Projects = ({ systemStatus }) => {
  const [selectedProjectIds, setSelectedProjectIds] = useState([]);

  // Get real-time project statistics
  const { projects } = useProjectsStatus();

  const getProjectStats = () => {
    const stats = {
      total: projects.length,
      indexed: projects.filter(p => p.status === 'indexed').length,
      indexing: projects.filter(p => p.status === 'indexing').length,
      error: projects.filter(p => p.status === 'error').length,
      totalFiles: projects.reduce((sum, p) => sum + (p.total_files || 0), 0),
      totalChunks: projects.reduce((sum, p) => sum + (p.total_chunks || 0), 0)
    };
    return stats;
  };

  const stats = getProjectStats();

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
              <Database className="w-6 h-6 text-primary-600" />
              <span>Project Management</span>
            </h1>
            <p className="text-sm text-gray-600 mt-1">
              Manage and index your codebase projects
            </p>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Database className="w-5 h-5 text-gray-400" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Total Projects</p>
                <p className="text-lg font-semibold text-gray-900">{stats.total}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <CheckCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Indexed</p>
                <p className="text-lg font-semibold text-gray-900">{stats.indexed}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Clock className="w-5 h-5 text-blue-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Indexing</p>
                <p className="text-lg font-semibold text-gray-900">{stats.indexing}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <AlertCircle className="w-5 h-5 text-red-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Errors</p>
                <p className="text-lg font-semibold text-gray-900">{stats.error}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <BarChart3 className="w-5 h-5 text-purple-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Total Files</p>
                <p className="text-lg font-semibold text-gray-900">{stats.totalFiles.toLocaleString()}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="w-5 h-5 text-indigo-500" />
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-500">Total Chunks</p>
                <p className="text-lg font-semibold text-gray-900">{stats.totalChunks.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full p-6">
          <ProjectList 
            onProjectSelect={setSelectedProjectIds}
            selectedProjectIds={selectedProjectIds}
          />
        </div>
      </div>

      {/* System Status Indicator */}
      {systemStatus && (
        <div className="bg-white border-t border-gray-200 px-6 py-3">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus.qdrant_status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-gray-600">Vector DB</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus.neo4j_status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <span className="text-gray-600">Graph DB</span>
              </div>
            </div>
            <div className="text-gray-500">
              {systemStatus.indexed_files} files • {systemStatus.total_chunks} chunks indexed
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Projects;
