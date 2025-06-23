import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import GraphVisualization from './components/GraphVisualization';
import StatusPanel from './components/StatusPanel';
import { apiService } from './services/apiService';

function App() {
  const [currentView, setCurrentView] = useState('chat');
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading Codebase Indexing Solution...</p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex">
        <Sidebar 
          currentView={currentView} 
          setCurrentView={setCurrentView}
          systemStatus={systemStatus}
        />
        
        <main className="flex-1 flex flex-col">
          <header className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Codebase Indexing Solution
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Semantic search and graph visualization for codebases
                </p>
              </div>
              <StatusPanel systemStatus={systemStatus} onRefresh={loadSystemStatus} />
            </div>
          </header>

          <div className="flex-1 overflow-hidden">
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route 
                path="/chat" 
                element={
                  <ChatInterface 
                    systemStatus={systemStatus}
                    onStatusUpdate={loadSystemStatus}
                  />
                } 
              />
              <Route 
                path="/graph" 
                element={
                  <GraphVisualization 
                    systemStatus={systemStatus}
                  />
                } 
              />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
