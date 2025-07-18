import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 120000, // Increased to 2 minutes for multi-agent analysis
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Query the codebase
  async queryCodebase(query, options = {}) {
    try {
      const response = await this.client.post('/mcp/query', {
        query,
        model: options.model || null,
        limit: options.limit || 10,
        include_context: options.includeContext !== false,
        project_ids: options.projectIds || null,
      }, {
        timeout: 120000  // 2 minutes for multi-agent analysis
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to query codebase');
    }
  }

  // Query the codebase with enhanced flow analysis
  async queryCodebaseFlow(query, options = {}) {
    try {
      const response = await this.client.post('/mcp/query/flow', {
        query,
        model: options.model || null,
        limit: options.limit || 10,
        include_context: options.includeContext !== false,
        project_ids: options.projectIds || null,
      }, {
        timeout: 120000  // 2 minutes for multi-agent flow analysis
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to query codebase with flow analysis');
    }
  }

  // Get graph data
  async getGraphData(filePath = null, projectIds = null, limit = 1000) {
    try {
      const params = { limit };
      if (filePath) {
        params.file_path = filePath;
      }
      if (projectIds && projectIds.length > 0) {
        params.project_ids = projectIds.join(',');
      }

      const response = await this.client.get('/mcp/graph', { params });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get graph data');
    }
  }

  // Get system status
  async getSystemStatus() {
    try {
      const response = await this.client.get('/mcp/status');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get system status');
    }
  }

  // Index a codebase
  async indexCodebase(path, options = {}) {
    try {
      const response = await this.client.post('/mcp/index', {
        path,
        languages: options.languages || ['python'],
        embedding_model: options.embeddingModel || null,
        force_reindex: options.forceReindex || false,
      }, {
        timeout: 300000  // 5 minutes for indexing operations
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to index codebase');
    }
  }

  // Project Management Methods

  // List all projects
  async listProjects(skip = 0, limit = 100) {
    try {
      const response = await this.client.get('/projects', {
        params: { skip, limit }
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to list projects');
    }
  }

  // Create a new project
  async createProject(projectData) {
    try {
      const response = await this.client.post('/projects', projectData);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to create project');
    }
  }

  // Get project by ID
  async getProject(projectId) {
    try {
      const response = await this.client.get(`/projects/${projectId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to get project');
    }
  }

  // Update project
  async updateProject(projectId, projectData) {
    try {
      const response = await this.client.put(`/projects/${projectId}`, projectData);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to update project');
    }
  }

  // Delete project
  async deleteProject(projectId) {
    try {
      const response = await this.client.delete(`/projects/${projectId}`);
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to delete project');
    }
  }

  // Index a project
  async indexProject(projectId, options = {}) {
    try {
      const response = await this.client.post(`/projects/${projectId}/index`, {
        languages: options.languages || ['python', 'javascript', 'typescript'],
        embedding_model: options.embeddingModel || null,
        force_reindex: options.forceReindex || false,
      }, {
        timeout: 300000  // 5 minutes for project indexing operations
      });
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Failed to index project');
    }
  }

  // Health check
  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw this.handleError(error, 'Health check failed');
    }
  }

  // Error handler
  handleError(error, defaultMessage) {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || defaultMessage;
      return new Error(`${message} (${error.response.status})`);
    } else if (error.request) {
      // Request was made but no response received
      return new Error('No response from server. Please check if the MCP server is running.');
    } else {
      // Something else happened
      return new Error(error.message || defaultMessage);
    }
  }

  // Utility method to check if server is reachable
  async isServerReachable() {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      return false;
    }
  }

  // Get available embedding models from status
  async getAvailableModels() {
    try {
      const status = await this.getSystemStatus();
      return status.available_models || [];
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  // Format file path for display
  formatFilePath(filePath) {
    if (!filePath) return '';

    // Get just the filename and parent directory
    const parts = filePath.split('/');
    if (parts.length > 2) {
      return `.../${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
    }
    return filePath;
  }

  // File system operations
  async browseDirectory(path = '', showHidden = false) {
    try {
      const params = new URLSearchParams();
      if (path) params.append('path', path);
      if (showHidden) params.append('show_hidden', 'true');

      const response = await fetch(`${API_BASE_URL}/filesystem/browse?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to browse directory:', error);
      throw error;
    }
  }

  // Format code content for display
  formatCodeContent(content, maxLength = 200) {
    if (!content) return '';
    
    if (content.length <= maxLength) {
      return content;
    }
    
    return content.substring(0, maxLength) + '...';
  }

  // Extract language from file path
  getLanguageFromPath(filePath) {
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
      'rs': 'rust',
      'php': 'php',
      'rb': 'ruby',
      'cs': 'csharp',
      'swift': 'swift',
      'kt': 'kotlin',
      'scala': 'scala',
      'sh': 'bash',
      'sql': 'sql',
      'json': 'json',
      'xml': 'xml',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'yaml': 'yaml',
      'yml': 'yaml',
      'md': 'markdown',
    };
    
    return languageMap[extension] || 'text';
  }
}

// Create and export a singleton instance
export const apiService = new ApiService();
export default apiService;
