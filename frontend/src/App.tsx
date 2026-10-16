import React, { useState, useEffect } from 'react';
import type { Project } from './types';
import { api } from './services/api';
import { Header } from './components/Header';
import { OverviewWorkspace } from './features/dashboard/OverviewWorkspace';
import { NetworkBuilderWorkspace } from './features/network-builder/NetworkBuilderWorkspace';
import { DatasetStudioWorkspace } from './features/datasets/DatasetStudioWorkspace';
import { TrainingLabWorkspace } from './features/training/TrainingLabWorkspace';
import { ModelInspectorWorkspace } from './features/model-inspector/ModelInspectorWorkspace';
import { ExperimentCenterWorkspace } from './features/experiments/ExperimentCenterWorkspace';
import { X, Plus } from 'lucide-react';

export const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [activeWorkspace, setActiveWorkspace] = useState<string>('overview');
  const [backendHealthy, setBackendHealthy] = useState<boolean>(true);

  // Inspector cross-workspace selection
  const [inspectorExperimentId, setInspectorExperimentId] = useState<string>('');

  // New project modal
  const [showNewProjectModal, setShowNewProjectModal] = useState<boolean>(false);
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [newProjectDesc, setNewProjectDesc] = useState<string>('');
  const [creatingProject, setCreatingProject] = useState<boolean>(false);

  // Health check and project loading
  useEffect(() => {
    const checkHealthAndLoadProjects = async () => {
      try {
        const health = await api.getHealth();
        setBackendHealthy(health.status === 'healthy');

        const projList = await api.listProjects();
        if (projList.length === 0) {
          // Auto-create initial default project
          const initialProj = await api.createProject({
            name: 'Industrial Sensor Telemetry Research',
            description: 'Deep learning classification of multi-axis vibration, temperature, and pressure sensor anomalies.'
          });
          setProjects([initialProj]);
          setActiveProject(initialProj);
        } else {
          setProjects(projList);
          setActiveProject(projList[0]);
        }
      } catch (err) {
        console.error('Failed to communicate with NeuralForge backend', err);
        setBackendHealthy(false);
      }
    };

    checkHealthAndLoadProjects();
    const interval = setInterval(checkHealthAndLoadProjects, 15000);
    return () => clearInterval(interval);
  }, []);

  // Handle project creation
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;

    setCreatingProject(true);
    try {
      const created = await api.createProject({
        name: newProjectName.trim(),
        description: newProjectDesc.trim()
      });
      setProjects(prev => [created, ...prev]);
      setActiveProject(created);
      setShowNewProjectModal(false);
      setNewProjectName('');
      setNewProjectDesc('');
    } catch (err: any) {
      alert(`Failed to create project: ${err.message}`);
    } finally {
      setCreatingProject(false);
    }
  };

  const handleOpenInspector = (expId: string) => {
    setInspectorExperimentId(expId);
    setActiveWorkspace('inspector');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', backgroundColor: 'var(--bg-app)' }}>
      {/* Universal Top Header */}
      <Header
        projects={projects}
        activeProject={activeProject}
        onSelectProject={(p) => setActiveProject(p)}
        onNewProject={() => setShowNewProjectModal(true)}
        activeWorkspace={activeWorkspace}
        onSelectWorkspace={(ws) => setActiveWorkspace(ws)}
        backendHealthy={backendHealthy}
      />

      {/* Workspace Display Area */}
      <main style={{ flex: 1, overflow: 'hidden' }}>
        {activeWorkspace === 'overview' && (
          <OverviewWorkspace
            activeProject={activeProject}
            onNavigate={(ws) => setActiveWorkspace(ws)}
            onSelectExperimentForInspector={handleOpenInspector}
          />
        )}

        {activeWorkspace === 'network-builder' && (
          <NetworkBuilderWorkspace
            activeProject={activeProject}
          />
        )}

        {activeWorkspace === 'datasets' && (
          <DatasetStudioWorkspace
            activeProject={activeProject}
          />
        )}

        {activeWorkspace === 'training' && (
          <TrainingLabWorkspace
            activeProject={activeProject}
            onTrainingFinished={() => {
              // Refresh projects counts
              api.listProjects().then(setProjects).catch(console.error);
            }}
          />
        )}

        {activeWorkspace === 'inspector' && (
          <ModelInspectorWorkspace
            activeProject={activeProject}
            selectedExperimentId={inspectorExperimentId}
          />
        )}

        {activeWorkspace === 'experiments' && (
          <ExperimentCenterWorkspace
            activeProject={activeProject}
            onSelectExperimentForInspector={handleOpenInspector}
          />
        )}
      </main>

      {/* New Project Modal */}
      {showNewProjectModal && (
        <div className="nf-modal-overlay">
          <div className="nf-modal">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ fontSize: '15px', fontWeight: 700 }}>Create New Research Project</div>
              <button
                className="nf-btn nf-btn-secondary"
                style={{ padding: '3px' }}
                onClick={() => setShowNewProjectModal(false)}
              >
                <X size={15} />
              </button>
            </div>

            <form onSubmit={handleCreateProject}>
              <div className="nf-form-group">
                <label className="nf-label">Project Name</label>
                <input
                  type="text"
                  className="nf-input"
                  placeholder="e.g. Predictive Maintenance AI"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  required
                />
              </div>

              <div className="nf-form-group">
                <label className="nf-label">Description</label>
                <textarea
                  className="nf-input"
                  rows={3}
                  placeholder="Goals, dataset references, and modeling hypotheses..."
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
                <button
                  type="button"
                  className="nf-btn nf-btn-secondary"
                  onClick={() => setShowNewProjectModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="nf-btn nf-btn-primary"
                  disabled={creatingProject || !newProjectName.trim()}
                >
                  <Plus size={14} /> {creatingProject ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
