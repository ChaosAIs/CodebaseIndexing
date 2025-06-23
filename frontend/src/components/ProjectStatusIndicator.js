import React from 'react';
import { 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  Loader2, 
  FileText, 
  Database,
  Calendar,
  AlertTriangle
} from 'lucide-react';

const ProjectStatusIndicator = ({ 
  project, 
  showDetails = false, 
  className = '',
  size = 'md' 
}) => {
  if (!project) return null;

  const getStatusConfig = (status) => {
    switch (status) {
      case 'indexed':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          label: 'Indexed',
          description: 'Project is fully indexed and searchable'
        };
      case 'indexing':
        return {
          icon: Loader2,
          color: 'text-blue-600',
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          label: 'Indexing...',
          description: 'Project is currently being indexed',
          animate: true
        };
      case 'error':
        return {
          icon: AlertCircle,
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          label: 'Error',
          description: 'Indexing failed with errors'
        };
      default:
        return {
          icon: Clock,
          color: 'text-gray-500',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          label: 'Not Indexed',
          description: 'Project has not been indexed yet'
        };
    }
  };

  const getSizeClasses = (size) => {
    switch (size) {
      case 'sm':
        return {
          container: 'p-2',
          icon: 'w-3 h-3',
          text: 'text-xs',
          title: 'text-sm'
        };
      case 'lg':
        return {
          container: 'p-4',
          icon: 'w-6 h-6',
          text: 'text-sm',
          title: 'text-lg'
        };
      default: // md
        return {
          container: 'p-3',
          icon: 'w-4 h-4',
          text: 'text-sm',
          title: 'text-base'
        };
    }
  };

  const statusConfig = getStatusConfig(project.status);
  const sizeClasses = getSizeClasses(size);
  const StatusIcon = statusConfig.icon;

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (!showDetails) {
    // Simple status badge
    return (
      <div className={`inline-flex items-center ${sizeClasses.container} ${statusConfig.bgColor} ${statusConfig.borderColor} border rounded-full ${className}`}>
        <StatusIcon 
          className={`${sizeClasses.icon} ${statusConfig.color} ${statusConfig.animate ? 'animate-spin' : ''}`} 
        />
        <span className={`ml-1 ${sizeClasses.text} ${statusConfig.color} font-medium`}>
          {statusConfig.label}
        </span>
      </div>
    );
  }

  // Detailed status card
  return (
    <div className={`${statusConfig.bgColor} ${statusConfig.borderColor} border rounded-lg ${sizeClasses.container} ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <StatusIcon 
            className={`${sizeClasses.icon} ${statusConfig.color} ${statusConfig.animate ? 'animate-spin' : ''}`} 
          />
          <span className={`${sizeClasses.title} font-semibold ${statusConfig.color}`}>
            {statusConfig.label}
          </span>
        </div>
        
        {project.status === 'indexed' && project.indexed_at && (
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <Calendar className="w-3 h-3" />
            <span>{formatDate(project.indexed_at)}</span>
          </div>
        )}
      </div>

      {/* Description */}
      <p className={`${sizeClasses.text} text-gray-600 mb-3`}>
        {statusConfig.description}
      </p>

      {/* Statistics */}
      {project.status === 'indexed' && (
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4 text-gray-400" />
            <div>
              <div className="text-sm font-medium text-gray-900">
                {project.total_files?.toLocaleString() || 0}
              </div>
              <div className="text-xs text-gray-500">Files</div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-gray-400" />
            <div>
              <div className="text-sm font-medium text-gray-900">
                {project.total_chunks?.toLocaleString() || 0}
              </div>
              <div className="text-xs text-gray-500">Chunks</div>
            </div>
          </div>
        </div>
      )}

      {/* Embedding Model */}
      {project.status === 'indexed' && project.embedding_model && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">Embedding Model</div>
          <div className="text-sm font-mono text-gray-700 bg-white px-2 py-1 rounded border">
            {project.embedding_model}
          </div>
        </div>
      )}

      {/* Error Details */}
      {project.status === 'error' && project.indexing_error && (
        <div className="mt-3 p-3 bg-red-100 border border-red-200 rounded">
          <div className="flex items-start space-x-2">
            <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
            <div>
              <div className="text-sm font-medium text-red-800 mb-1">Error Details</div>
              <div className="text-sm text-red-700 font-mono">
                {project.indexing_error}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Progress for indexing status */}
      {project.status === 'indexing' && (
        <div className="mt-3">
          <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
            <span>Processing files...</span>
            <span className="animate-pulse">●</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
          </div>
        </div>
      )}

      {/* Last Updated */}
      <div className="mt-3 pt-3 border-t border-gray-200">
        <div className="text-xs text-gray-500">
          Last updated: {formatDate(project.updated_at)}
        </div>
      </div>
    </div>
  );
};

export default ProjectStatusIndicator;
