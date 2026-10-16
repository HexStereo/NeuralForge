import React from 'react';
import type { Project } from '../types';
import { Layers, Database, PlayCircle, Eye, Activity, BarChart2, Plus } from 'lucide-react';

interface HeaderProps {
  projects: Project[];
  activeProject: Project | null;
  onSelectProject: (p: Project) => void;
  onNewProject: () => void;
  activeWorkspace: string;
  onSelectWorkspace: (ws: string) => void;
  backendHealthy: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  projects,
  activeProject,
  onSelectProject,
  onNewProject,
  activeWorkspace,
  onSelectWorkspace,
  backendHealthy
}) => {
  return (
    <header className="nf-header">
      <div className="nf-header-left">
        <div className="nf-logo">
          <div className="nf-logo-icon">NF</div>
          <span>NeuralForge</span>
        </div>

        <div className="nf-project-select">
          <span>Project:</span>
          <select
            value={activeProject?.id || ''}
            onChange={(e) => {
              const p = projects.find(item => item.id === e.target.value);
              if (p) onSelectProject(p);
            }}
          >
            {projects.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
          <button
            className="nf-btn nf-btn-secondary"
            style={{ padding: '2px 6px', fontSize: '11px', marginLeft: '4px' }}
            onClick={onNewProject}
            title="Create New Project"
          >
            <Plus size={12} /> New
          </button>
        </div>

        <nav className="nf-nav-tabs">
          <button
            className={`nf-nav-tab ${activeWorkspace === 'overview' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('overview')}
          >
            <Activity size={14} /> Overview
          </button>
          <button
            className={`nf-nav-tab ${activeWorkspace === 'network-builder' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('network-builder')}
          >
            <Layers size={14} /> Network Builder
          </button>
          <button
            className={`nf-nav-tab ${activeWorkspace === 'datasets' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('datasets')}
          >
            <Database size={14} /> Dataset Studio
          </button>
          <button
            className={`nf-nav-tab ${activeWorkspace === 'training' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('training')}
          >
            <PlayCircle size={14} /> Training Lab
          </button>
          <button
            className={`nf-nav-tab ${activeWorkspace === 'inspector' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('inspector')}
          >
            <Eye size={14} /> Model Inspector
          </button>
          <button
            className={`nf-nav-tab ${activeWorkspace === 'experiments' ? 'active' : ''}`}
            onClick={() => onSelectWorkspace('experiments')}
          >
            <BarChart2 size={14} /> Experiment Center
          </button>
        </nav>
      </div>

      <div className="nf-header-right" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div
          title="NeuralForge AI Studio - Created by HEXSTEREO"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 700,
            background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(6, 182, 212, 0.2))',
            border: '1px solid rgba(139, 92, 246, 0.4)',
            color: '#c4b5fd',
            letterSpacing: '0.8px',
            fontFamily: 'monospace',
            textShadow: '0 0 8px rgba(139, 92, 246, 0.6)'
          }}
        >
          <span style={{ color: '#06b6d4', fontSize: '10px' }}>⚡</span>
          <span>HEXSTEREO</span>
        </div>

        {backendHealthy ? (
          <span className="nf-badge nf-badge-success" title="FastAPI and NeuralForge Core computation engine online">
            ● Core Ready
          </span>
        ) : (
          <span className="nf-badge nf-badge-danger" title="Backend connection offline">
            ● Core Offline
          </span>
        )}
      </div>
    </header>
  );
};
