import React, { useState, useEffect } from 'react';
import { X, Folder, FolderOpen, HardDrive, ChevronRight, Home } from 'lucide-react';
import { apiService } from '../services/apiService';

const FolderPicker = ({ isOpen, onClose, onSelect, initialPath = '' }) => {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPath, setSelectedPath] = useState(initialPath);

  useEffect(() => {
    if (isOpen) {
      loadDirectory(initialPath);
    }
  }, [isOpen, initialPath]);

  const loadDirectory = async (path = '') => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.browseDirectory(path, false);
      setItems(response.items || []);
      setCurrentPath(response.current_path || path);
      if (!selectedPath && response.current_path) {
        setSelectedPath(response.current_path);
      }
    } catch (err) {
      setError(err.message);
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  const handleItemClick = (item) => {
    if (item.is_directory) {
      if (item.type === 'parent') {
        loadDirectory(item.path);
      } else {
        loadDirectory(item.path);
      }
    }
  };

  const handleItemSelect = (item) => {
    if (item.is_directory) {
      setSelectedPath(item.path);
    }
  };

  const handleConfirm = () => {
    if (selectedPath) {
      onSelect(selectedPath);
      onClose();
    }
  };

  const getItemIcon = (item) => {
    if (item.type === 'drive') {
      return <HardDrive className="w-4 h-4 text-blue-500" />;
    } else if (item.type === 'parent') {
      return <ChevronRight className="w-4 h-4 text-gray-500 transform rotate-180" />;
    } else if (item.is_directory) {
      return <Folder className="w-4 h-4 text-blue-500" />;
    } else {
      return <div className="w-4 h-4" />; // Placeholder for files
    }
  };

  const formatPath = (path) => {
    if (!path) return 'Computer';
    // On Windows, show drive letters nicely
    if (path.match(/^[A-Z]:\\$/)) {
      return `${path.charAt(0)}: Drive`;
    }
    return path;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Select Folder</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Current Path */}
        <div className="px-4 py-2 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Home className="w-4 h-4" />
            <span className="truncate">{formatPath(currentPath)}</span>
          </div>
        </div>

        {/* Directory Contents */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-gray-500">Loading...</div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-red-600">Error: {error}</div>
            </div>
          ) : items.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-gray-500">No folders found</div>
            </div>
          ) : (
            <div className="space-y-1">
              {items.map((item, index) => (
                <div
                  key={index}
                  className={`flex items-center space-x-3 p-2 rounded cursor-pointer hover:bg-gray-100 ${
                    selectedPath === item.path ? 'bg-blue-50 border border-blue-200' : ''
                  }`}
                  onClick={() => {
                    handleItemSelect(item);
                    if (item.is_directory && item.type !== 'parent') {
                      // Double-click behavior: navigate into directory
                    }
                  }}
                  onDoubleClick={() => handleItemClick(item)}
                >
                  {getItemIcon(item)}
                  <span className="flex-1 text-sm text-gray-900 truncate">
                    {item.name}
                  </span>
                  {item.is_directory && item.type !== 'parent' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleItemClick(item);
                      }}
                      className="p-1 hover:bg-gray-200 rounded"
                    >
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Selected Path Display */}
        {selectedPath && (
          <div className="px-4 py-2 bg-gray-50 border-t border-gray-200">
            <div className="text-sm text-gray-600">
              <span className="font-medium">Selected:</span> {selectedPath}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-end space-x-3 p-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={!selectedPath}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Select Folder
          </button>
        </div>
      </div>
    </div>
  );
};

export default FolderPicker;
