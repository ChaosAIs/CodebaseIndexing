import React, { useState, useEffect, useRef } from 'react';
import { 
  ChevronDown, 
  Check, 
  FolderOpen, 
  X,
  CheckCircle,
  AlertCircle,
  Clock
} from 'lucide-react';
import { apiService } from '../services/apiService';

const ProjectSelector = ({ selectedProjectIds = [], onSelectionChange, className = '' }) => {
  const [projects, setProjects] = useState([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const dropdownRef = useRef(null);

  useEffect(() => {
    loadProjects();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const loadProjects = async () => {
    setLoading(true);
    setError(null);
    try {
      const projectList = await apiService.listProjects();
      setProjects(projectList);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleProjectToggle = (projectId) => {
    const isSelected = selectedProjectIds.includes(projectId);
    let newSelection;
    
    if (isSelected) {
      newSelection = selectedProjectIds.filter(id => id !== projectId);
    } else {
      newSelection = [...selectedProjectIds, projectId];
    }
    
    onSelectionChange(newSelection);
  };

  const handleSelectAll = () => {
    const indexedProjects = projects.filter(p => p.status === 'indexed');
    const allIndexedIds = indexedProjects.map(p => p.id);
    
    if (selectedProjectIds.length === allIndexedIds.length) {
      // Deselect all
      onSelectionChange([]);
    } else {
      // Select all indexed projects
      onSelectionChange(allIndexedIds);
    }
  };

  const clearSelection = () => {
    onSelectionChange([]);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'indexed':
        return <CheckCircle className="w-3 h-3 text-green-500" />;
      case 'indexing':
        return <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
      case 'error':
        return <AlertCircle className="w-3 h-3 text-red-500" />;
      default:
        return <Clock className="w-3 h-3 text-gray-400" />;
    }
  };

  const getSelectedProjectNames = () => {
    const selectedProjects = projects.filter(p => selectedProjectIds.includes(p.id));
    return selectedProjects.map(p => p.name);
  };

  const indexedProjects = projects.filter(p => p.status === 'indexed');
  const selectedNames = getSelectedProjectNames();

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 text-left bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
      >
        <div className="flex items-center space-x-2 flex-1 min-w-0">
          <FolderOpen className="w-4 h-4 text-gray-400 flex-shrink-0" />
          <span className="truncate text-sm">
            {selectedProjectIds.length === 0 
              ? 'Select projects...' 
              : selectedProjectIds.length === 1
                ? selectedNames[0]
                : `${selectedProjectIds.length} projects selected`
            }
          </span>
        </div>
        <div className="flex items-center space-x-1">
          {selectedProjectIds.length > 0 && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                clearSelection();
              }}
              className="p-1 hover:bg-gray-100 rounded"
              title="Clear selection"
            >
              <X className="w-3 h-3 text-gray-400" />
            </button>
          )}
          <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
          {loading ? (
            <div className="px-3 py-2 text-sm text-gray-500">Loading projects...</div>
          ) : error ? (
            <div className="px-3 py-2 text-sm text-red-600">Error: {error}</div>
          ) : projects.length === 0 ? (
            <div className="px-3 py-2 text-sm text-gray-500">No projects found</div>
          ) : (
            <>
              {/* Select All Option */}
              {indexedProjects.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={handleSelectAll}
                    className="w-full px-3 py-2 text-left text-sm hover:bg-gray-50 flex items-center space-x-2"
                  >
                    <div className="w-4 h-4 flex items-center justify-center">
                      {selectedProjectIds.length === indexedProjects.length ? (
                        <Check className="w-3 h-3 text-primary-600" />
                      ) : selectedProjectIds.length > 0 ? (
                        <div className="w-2 h-2 bg-primary-600 rounded-sm" />
                      ) : null}
                    </div>
                    <span className="font-medium">
                      {selectedProjectIds.length === indexedProjects.length ? 'Deselect All' : 'Select All Indexed'}
                    </span>
                  </button>
                  <div className="border-t border-gray-100" />
                </>
              )}

              {/* Project Options */}
              {projects.map((project) => {
                const isSelected = selectedProjectIds.includes(project.id);
                const isIndexed = project.status === 'indexed';
                
                return (
                  <button
                    key={project.id}
                    type="button"
                    onClick={() => isIndexed && handleProjectToggle(project.id)}
                    disabled={!isIndexed}
                    className={`w-full px-3 py-2 text-left text-sm flex items-center space-x-2 ${
                      isIndexed 
                        ? 'hover:bg-gray-50 cursor-pointer' 
                        : 'cursor-not-allowed opacity-50'
                    } ${isSelected ? 'bg-primary-50' : ''}`}
                  >
                    <div className="w-4 h-4 flex items-center justify-center">
                      {isSelected && <Check className="w-3 h-3 text-primary-600" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="truncate">{project.name}</span>
                        {getStatusIcon(project.status)}
                      </div>
                      {project.description && (
                        <div className="text-xs text-gray-500 truncate mt-0.5">
                          {project.description}
                        </div>
                      )}
                    </div>
                    {isIndexed && (
                      <div className="text-xs text-gray-400">
                        {project.total_chunks} chunks
                      </div>
                    )}
                  </button>
                );
              })}

              {/* No Indexed Projects Message */}
              {indexedProjects.length === 0 && (
                <div className="px-3 py-2 text-sm text-gray-500 text-center">
                  No indexed projects available.
                  <br />
                  Index projects first to enable search.
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* Selected Projects Display (when multiple selected) */}
      {selectedProjectIds.length > 1 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {selectedNames.slice(0, 3).map((name, index) => (
            <span
              key={index}
              className="inline-flex items-center px-2 py-1 text-xs bg-primary-100 text-primary-800 rounded"
            >
              {name}
            </span>
          ))}
          {selectedNames.length > 3 && (
            <span className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
              +{selectedNames.length - 3} more
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default ProjectSelector;
