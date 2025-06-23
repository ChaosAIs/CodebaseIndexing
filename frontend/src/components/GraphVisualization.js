import React, { useState, useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import dagre from 'cytoscape-dagre';
import coseBilkent from 'cytoscape-cose-bilkent';
import { 
  Network, 
  Filter, 
  Download, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Settings,
  FileText,
  Code,
  Loader2
} from 'lucide-react';
import { apiService } from '../services/apiService';
import ProjectSelector from './ProjectSelector';

// Register cytoscape extensions
cytoscape.use(dagre);
cytoscape.use(coseBilkent);

const GraphVisualization = ({ systemStatus }) => {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState('');
  const [selectedProjectIds, setSelectedProjectIds] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [layout, setLayout] = useState('dagre');
  const [nodeLimit, setNodeLimit] = useState(500);
  
  const cyRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    loadGraphData();
  }, [selectedFile, selectedProjectIds, nodeLimit]);

  useEffect(() => {
    if (graphData && containerRef.current) {
      initializeCytoscape();
    }
    
    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graphData, layout]);

  const loadGraphData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await apiService.getGraphData(
        selectedFile || null,
        selectedProjectIds.length > 0 ? selectedProjectIds : null,
        nodeLimit
      );
      setGraphData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const initializeCytoscape = () => {
    if (!graphData || !containerRef.current) return;

    // Destroy existing instance
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // Prepare nodes
    const nodes = graphData.nodes.map(node => ({
      data: {
        id: node.id,
        label: node.label,
        type: node.type,
        filePath: node.file_path,
        ...node.properties
      }
    }));

    // Prepare edges
    const edges = graphData.edges.map(edge => ({
      data: {
        id: `${edge.source}-${edge.target}`,
        source: edge.source,
        target: edge.target,
        type: edge.type,
        weight: edge.weight
      }
    }));

    // Initialize cytoscape
    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [...nodes, ...edges],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => getNodeColor(ele.data('type')),
            'label': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '12px',
            'font-weight': 'bold',
            'color': '#333',
            'text-outline-width': 2,
            'text-outline-color': '#fff',
            'width': (ele) => Math.max(30, ele.data('label').length * 8),
            'height': '30px',
            'border-width': 2,
            'border-color': '#666',
            'shape': (ele) => getNodeShape(ele.data('type'))
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#3b82f6',
            'border-width': 3,
            'background-color': '#dbeafe'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': (ele) => Math.max(1, ele.data('weight') * 3),
            'line-color': (ele) => getEdgeColor(ele.data('type')),
            'target-arrow-color': (ele) => getEdgeColor(ele.data('type')),
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1.2,
            'opacity': 0.7
          }
        },
        {
          selector: 'edge:selected',
          style: {
            'line-color': '#3b82f6',
            'target-arrow-color': '#3b82f6',
            'opacity': 1,
            'width': 3
          }
        }
      ],
      layout: getLayoutConfig(layout),
      minZoom: 0.1,
      maxZoom: 3,
      wheelSensitivity: 0.2
    });

    // Event handlers
    cyRef.current.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNode({
        id: node.data('id'),
        label: node.data('label'),
        type: node.data('type'),
        filePath: node.data('filePath'),
        startLine: node.data('start_line'),
        endLine: node.data('end_line'),
        contentPreview: node.data('content_preview')
      });
    });

    cyRef.current.on('tap', (evt) => {
      if (evt.target === cyRef.current) {
        setSelectedNode(null);
      }
    });
  };

  const getNodeColor = (type) => {
    const colors = {
      'function_definition': '#10b981',
      'class_definition': '#3b82f6',
      'method_definition': '#8b5cf6',
      'variable_definition': '#f59e0b',
      'import_statement': '#ef4444',
      'module': '#6b7280'
    };
    return colors[type] || '#6b7280';
  };

  const getNodeShape = (type) => {
    const shapes = {
      'function_definition': 'round-rectangle',
      'class_definition': 'rectangle',
      'method_definition': 'ellipse',
      'variable_definition': 'diamond',
      'import_statement': 'triangle',
      'module': 'hexagon'
    };
    return shapes[type] || 'ellipse';
  };

  const getEdgeColor = (type) => {
    const colors = {
      'parent_child': '#3b82f6',
      'calls': '#10b981',
      'called_by': '#10b981',
      'imports': '#ef4444',
      'imported_by': '#ef4444'
    };
    return colors[type] || '#6b7280';
  };

  const getLayoutConfig = (layoutName) => {
    const layouts = {
      dagre: {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 50,
        rankSep: 100,
        animate: true,
        animationDuration: 500
      },
      'cose-bilkent': {
        name: 'cose-bilkent',
        animate: true,
        animationDuration: 1000,
        nodeRepulsion: 4500,
        idealEdgeLength: 100,
        edgeElasticity: 0.45
      },
      circle: {
        name: 'circle',
        animate: true,
        animationDuration: 500
      },
      grid: {
        name: 'grid',
        animate: true,
        animationDuration: 500
      }
    };
    return layouts[layoutName] || layouts.dagre;
  };

  const handleZoomIn = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.2);
    }
  };

  const handleZoomOut = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 0.8);
    }
  };

  const handleReset = () => {
    if (cyRef.current) {
      cyRef.current.fit();
      cyRef.current.center();
    }
  };

  const handleExport = () => {
    if (cyRef.current) {
      const png = cyRef.current.png({ scale: 2, full: true });
      const link = document.createElement('a');
      link.download = 'codebase-graph.png';
      link.href = png;
      link.click();
    }
  };

  const applyLayout = (layoutName) => {
    if (cyRef.current) {
      setLayout(layoutName);
      cyRef.current.layout(getLayoutConfig(layoutName)).run();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading graph data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="text-center">
          <Network className="w-12 h-12 text-red-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Failed to Load Graph</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={loadGraphData}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full bg-gray-50">
      {/* Main Graph Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h2 className="text-lg font-medium text-gray-900 flex items-center">
                <Network className="w-5 h-5 mr-2" />
                Code Graph
              </h2>
              
              {graphData && (
                <div className="text-sm text-gray-600">
                  {graphData.nodes.length} nodes, {graphData.edges.length} edges
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                <Settings className="w-5 h-5" />
              </button>
              
              <button
                onClick={handleZoomIn}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                <ZoomIn className="w-5 h-5" />
              </button>
              
              <button
                onClick={handleZoomOut}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                <ZoomOut className="w-5 h-5" />
              </button>
              
              <button
                onClick={handleReset}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
              
              <button
                onClick={handleExport}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                <Download className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          {/* Settings Panel */}
          {showSettings && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="space-y-4">
                {/* Project Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Filter by Projects
                  </label>
                  <ProjectSelector
                    selectedProjectIds={selectedProjectIds}
                    onSelectionChange={setSelectedProjectIds}
                    className="w-full"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    {selectedProjectIds.length === 0
                      ? 'Showing graph for all indexed projects'
                      : `Showing graph for ${selectedProjectIds.length} selected project${selectedProjectIds.length > 1 ? 's' : ''}`
                    }
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Layout
                    </label>
                    <select
                      value={layout}
                      onChange={(e) => applyLayout(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      <option value="dagre">Hierarchical</option>
                      <option value="cose-bilkent">Force-directed</option>
                      <option value="circle">Circle</option>
                      <option value="grid">Grid</option>
                    </select>
                  </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    File Filter
                  </label>
                  <input
                    type="text"
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    placeholder="Filter by file path..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Node Limit
                  </label>
                  <input
                    type="number"
                    min="50"
                    max="2000"
                    value={nodeLimit}
                    onChange={(e) => setNodeLimit(parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Actions
                  </label>
                  <button
                    onClick={loadGraphData}
                    className="w-full px-3 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
                  >
                    Refresh
                  </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Graph Container */}
        <div className="flex-1 relative">
          <div
            ref={containerRef}
            className="w-full h-full graph-container"
            style={{ background: '#fafafa' }}
          />
        </div>
      </div>
      
      {/* Node Details Panel */}
      {selectedNode && (
        <div className="w-80 bg-white border-l border-gray-200 p-4 overflow-y-auto">
          <div className="mb-4">
            <h3 className="text-lg font-medium text-gray-900 mb-2">Node Details</h3>
            
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-700">Name:</label>
                <p className="text-sm text-gray-900 font-mono">{selectedNode.label}</p>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700">Type:</label>
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800 ml-2">
                  {selectedNode.type}
                </span>
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700">File:</label>
                <p className="text-sm text-gray-600 break-all">{selectedNode.filePath}</p>
              </div>
              
              {selectedNode.startLine && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Lines:</label>
                  <p className="text-sm text-gray-600">
                    {selectedNode.startLine} - {selectedNode.endLine}
                  </p>
                </div>
              )}
              
              {selectedNode.contentPreview && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Preview:</label>
                  <div className="mt-1 p-2 bg-gray-50 rounded text-xs font-mono text-gray-700 max-h-32 overflow-y-auto">
                    {selectedNode.contentPreview}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphVisualization;
