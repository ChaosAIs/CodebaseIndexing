import React, { useState, useEffect } from 'react';
import { 
  Plus, 
  Edit, 
  Trash2, 
  FolderOpen, 
  Clock, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Play,
  RefreshCw
} from 'lucide-react';
import { apiService } from '../services/apiService';
import ProjectForm from './ProjectForm';
import ProjectStatusIndicator from './ProjectStatusIndicator';
import { useProjectsStatus } from '../hooks/useProjectStatus';

const ProjectList = ({ onProjectSelect, selectedProjectIds = [] }) => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingProject, setEditingProject] = useState(null);

  // Use the projects status hook for real-time updates
  const {
    projects,
    loading,
    error,
    hasIndexingProjects,
    startProjectIndexing,
    refreshProjects
  } = useProjectsStatus();



  const handleCreateProject = async (projectData) => {
    try {
      await apiService.createProject(projectData);
      setShowCreateForm(false);
      await refreshProjects();
    } catch (err) {
      throw new Error(err.message);
    }
  };

  const handleUpdateProject = async (projectId, projectData) => {
    try {
      await apiService.updateProject(projectId, projectData);
      setEditingProject(null);
      await refreshProjects();
    } catch (err) {
      throw new Error(err.message);
    }
  };

  const handleDeleteProject = async (projectId) => {
    if (!window.confirm('Are you sure you want to delete this project? This action cannot be undone.')) {
      return;
    }

    try {
      await apiService.deleteProject(projectId);
      await refreshProjects();
    } catch (err) {
      // Error will be handled by the hook
      console.error('Delete project error:', err);
    }
  };

  const handleIndexProject = async (projectId) => {
    try {
      await startProjectIndexing(projectId, {
        languages: ['python', 'javascript', 'typescript', 'java', 'cpp'],
        force_reindex: false
      });
    } catch (err) {
      // Error will be handled by the hook
      console.error('Index project error:', err);
    }
  };



  const handleProjectToggle = (projectId) => {
    if (onProjectSelect) {
      const isSelected = selectedProjectIds.includes(projectId);
      if (isSelected) {
        onProjectSelect(selectedProjectIds.filter(id => id !== projectId));
      } else {
        onProjectSelect([...selectedProjectIds, projectId]);
      }
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
        <span className="ml-2 text-gray-600">Loading projects...</span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Projects</h2>
          <p className="text-sm text-gray-600">Manage your codebase projects</p>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={refreshProjects}
            className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50 flex items-center space-x-1"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => setShowCreateForm(true)}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Project</span>
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Project List */}
      {projects.length === 0 ? (
        <div className="text-center py-12">
          <FolderOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
          <p className="text-gray-600 mb-4">Create your first project to get started with codebase indexing.</p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
          >
            Create Project
          </button>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <div
              key={project.id}
              className={`bg-white rounded-lg border-2 p-4 hover:shadow-md transition-shadow ${
                selectedProjectIds.includes(project.id) 
                  ? 'border-primary-500 bg-primary-50' 
                  : 'border-gray-200'
              }`}
            >
              {/* Project Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedProjectIds.includes(project.id)}
                      onChange={() => handleProjectToggle(project.id)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <h3 className="font-medium text-gray-900 truncate">{project.name}</h3>
                  </div>
                  {project.description && (
                    <p className="text-sm text-gray-600 mt-1 line-clamp-2">{project.description}</p>
                  )}
                </div>
                <div className="flex space-x-1 ml-2">
                  <button
                    onClick={() => setEditingProject(project)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                    title="Edit project"
                  >
                    <Edit className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDeleteProject(project.id)}
                    className="p-1 text-gray-400 hover:text-red-600"
                    title="Delete project"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Project Details */}
              <div className="space-y-2 text-sm">
                <div className="flex items-center text-gray-600">
                  <FolderOpen className="w-4 h-4 mr-2" />
                  <span className="truncate" title={project.source_path}>
                    {project.source_path}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <ProjectStatusIndicator
                    project={project}
                    size="sm"
                    className="flex-1"
                  />

                  {project.status !== 'indexing' && (
                    <button
                      onClick={() => handleIndexProject(project.id)}
                      className="flex items-center space-x-1 px-2 py-1 text-xs bg-primary-100 text-primary-700 rounded hover:bg-primary-200 ml-2"
                      title="Index project"
                    >
                      <Play className="w-3 h-3" />
                      <span>Index</span>
                    </button>
                  )}
                </div>

                {project.status === 'indexed' && (
                  <div className="flex justify-between text-xs text-gray-500 mt-2">
                    <span>{project.total_files} files</span>
                    <span>{project.total_chunks} chunks</span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create/Edit Project Modal */}
      {(showCreateForm || editingProject) && (
        <ProjectForm
          project={editingProject}
          onSubmit={editingProject ? 
            (data) => handleUpdateProject(editingProject.id, data) : 
            handleCreateProject
          }
          onCancel={() => {
            setShowCreateForm(false);
            setEditingProject(null);
          }}
        />
      )}
    </div>
  );
};



export default ProjectList;
