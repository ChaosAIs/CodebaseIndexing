import React, { useState } from 'react';
import { RefreshCw, AlertCircle, CheckCircle, Clock } from 'lucide-react';

const StatusPanel = ({ systemStatus, onRefresh }) => {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await onRefresh();
    } finally {
      setRefreshing(false);
    }
  };

  const getOverallStatus = () => {
    if (!systemStatus) return 'unknown';
    
    const { qdrant_status, neo4j_status } = systemStatus;
    
    if (qdrant_status === 'healthy' && neo4j_status === 'healthy') {
      return 'healthy';
    } else if (qdrant_status === 'unhealthy' || neo4j_status === 'unhealthy') {
      return 'unhealthy';
    } else {
      return 'partial';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'unhealthy':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'partial':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'healthy':
        return 'All Systems Operational';
      case 'unhealthy':
        return 'System Issues Detected';
      case 'partial':
        return 'Partial System Availability';
      default:
        return 'Status Unknown';
    }
  };

  const overallStatus = getOverallStatus();

  return (
    <div className="flex items-center space-x-4">
      {/* Status Indicator */}
      <div className="flex items-center space-x-2">
        {getStatusIcon(overallStatus)}
        <span className="text-sm font-medium text-gray-700">
          {getStatusText(overallStatus)}
        </span>
      </div>

      {/* Refresh Button */}
      <button
        onClick={handleRefresh}
        disabled={refreshing}
        className="flex items-center space-x-1 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50"
      >
        <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
        <span>Refresh</span>
      </button>

      {/* Quick Stats */}
      {systemStatus && (
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <div className="flex items-center space-x-1">
            <span className="font-medium">{systemStatus.indexed_files || 0}</span>
            <span>files</span>
          </div>
          <div className="flex items-center space-x-1">
            <span className="font-medium">{systemStatus.total_chunks || 0}</span>
            <span>chunks</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default StatusPanel;
