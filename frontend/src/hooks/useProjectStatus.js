import { useState, useEffect, useCallback } from 'react';
import { apiService } from '../services/apiService';

/**
 * Custom hook for tracking project indexing status with real-time updates
 */
export const useProjectStatus = (projectId, pollingInterval = 2000) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isPolling, setIsPolling] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!projectId) return;

    try {
      setError(null);
      const project = await apiService.getProject(projectId);
      setStatus(project);
      
      // If project is indexing, continue polling
      if (project.status === 'indexing') {
        setIsPolling(true);
      } else {
        setIsPolling(false);
      }
    } catch (err) {
      setError(err.message);
      setIsPolling(false);
    }
  }, [projectId]);

  const startIndexing = useCallback(async (options = {}) => {
    if (!projectId) return;

    setLoading(true);
    setError(null);
    
    try {
      await apiService.indexProject(projectId, options);
      setIsPolling(true);
      await fetchStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [projectId, fetchStatus]);

  // Polling effect
  useEffect(() => {
    let intervalId;

    if (isPolling && projectId) {
      intervalId = setInterval(fetchStatus, pollingInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isPolling, projectId, pollingInterval, fetchStatus]);

  // Initial fetch
  useEffect(() => {
    if (projectId) {
      fetchStatus();
    }
  }, [projectId, fetchStatus]);

  return {
    status,
    loading,
    error,
    isPolling,
    startIndexing,
    refreshStatus: fetchStatus
  };
};

/**
 * Custom hook for tracking multiple projects status
 */
export const useProjectsStatus = (pollingInterval = 5000) => {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasIndexingProjects, setHasIndexingProjects] = useState(false);

  const fetchProjects = useCallback(async () => {
    try {
      setError(null);
      const projectList = await apiService.listProjects();
      setProjects(projectList);
      
      // Check if any projects are currently indexing
      const indexingProjects = projectList.filter(p => p.status === 'indexing');
      setHasIndexingProjects(indexingProjects.length > 0);
    } catch (err) {
      setError(err.message);
      setHasIndexingProjects(false);
    }
  }, []);

  const startProjectIndexing = useCallback(async (projectId, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      await apiService.indexProject(projectId, options);
      await fetchProjects();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [fetchProjects]);

  // Polling effect for indexing projects
  useEffect(() => {
    let intervalId;

    if (hasIndexingProjects) {
      intervalId = setInterval(fetchProjects, pollingInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [hasIndexingProjects, pollingInterval, fetchProjects]);

  // Initial fetch
  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  return {
    projects,
    loading,
    error,
    hasIndexingProjects,
    startProjectIndexing,
    refreshProjects: fetchProjects
  };
};

export default useProjectStatus;
