import React, { useEffect, useState } from 'react';
import type { Project, Experiment } from '../../types';
import { api } from '../../services/api';
import { Layers, Database, PlayCircle, Eye, Clock, ArrowRight } from 'lucide-react';

interface OverviewWorkspaceProps {
  activeProject: Project | null;
  onNavigate: (ws: string) => void;
  onSelectExperimentForInspector?: (expId: string) => void;
}

export const OverviewWorkspace: React.FC<OverviewWorkspaceProps> = ({
  activeProject,
  onNavigate,
  onSelectExperimentForInspector
}) => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    if (activeProject) {
      api.listExperiments(activeProject.id)
        .then(setExperiments)
        .catch(console.error);
    }
  }, [activeProject]);

  const completedCount = experiments.filter(e => e.status === 'completed').length;

  return (
    <div className="nf-workspace">
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '6px' }}>
          {activeProject ? activeProject.name : 'Research Workspace'}
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
          {activeProject?.description || 'Deep learning neural network architecture design, training telemetry, and model inspection.'}
        </p>
      </div>

      {/* Metrics Row */}
      <div className="nf-grid-4" style={{ marginBottom: '24px' }}>
        <div className="nf-stat-box">
          <div className="nf-stat-label">Architectures</div>
          <div className="nf-stat-value">{activeProject?.architecture_count ?? 0}</div>
        </div>
        <div className="nf-stat-box">
          <div className="nf-stat-label">Datasets</div>
          <div className="nf-stat-value">{activeProject?.dataset_count ?? 0}</div>
        </div>
        <div className="nf-stat-box">
          <div className="nf-stat-label">Experiments</div>
          <div className="nf-stat-value">{experiments.length}</div>
        </div>
        <div className="nf-stat-box">
          <div className="nf-stat-label">Completed Runs</div>
          <div className="nf-stat-value" style={{ color: 'var(--color-success)' }}>
            {completedCount}
          </div>
        </div>
      </div>

      {/* Quick Launch Cards */}
      <div className="nf-grid-3" style={{ marginBottom: '24px' }}>
        <div
          className="nf-card"
          style={{ cursor: 'pointer', transition: 'border-color 0.15s ease' }}
          onClick={() => onNavigate('network-builder')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <div style={{ padding: '8px', background: 'var(--accent-purple-dim)', borderRadius: '4px', color: 'var(--accent-purple-hover)' }}>
              <Layers size={18} />
            </div>
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Visual Network Builder</h3>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Graph-based layer composition</span>
            </div>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '12px' }}>
            Assemble feed-forward architectures with Dense, ReLU, Sigmoid, Tanh, and Softmax.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-purple-hover)', fontSize: '12px', fontWeight: 500 }}>
            Open Builder <ArrowRight size={14} />
          </div>
        </div>

        <div
          className="nf-card"
          style={{ cursor: 'pointer', transition: 'border-color 0.15s ease' }}
          onClick={() => onNavigate('datasets')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <div style={{ padding: '8px', background: 'var(--accent-cyan-dim)', borderRadius: '4px', color: 'var(--accent-cyan)' }}>
              <Database size={18} />
            </div>
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Dataset Studio</h3>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>CSV ingestion & normalization</span>
            </div>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '12px' }}>
            Profile data distributions, select features, and configure leak-free train/val splits.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)', fontSize: '12px', fontWeight: 500 }}>
            Open Datasets <ArrowRight size={14} />
          </div>
        </div>

        <div
          className="nf-card"
          style={{ cursor: 'pointer', transition: 'border-color 0.15s ease' }}
          onClick={() => onNavigate('training')}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
            <div style={{ padding: '8px', background: 'rgba(16, 185, 129, 0.15)', borderRadius: '4px', color: 'var(--color-success)' }}>
              <PlayCircle size={18} />
            </div>
            <div>
              <h3 style={{ fontSize: '14px', fontWeight: 600 }}>Training Lab</h3>
              <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>Real-time telemetry & controls</span>
            </div>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginBottom: '12px' }}>
            Configure Adam/SGD optimizers and stream live losses and accuracies via WebSockets.
          </p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--color-success)', fontSize: '12px', fontWeight: 500 }}>
            Open Training Lab <ArrowRight size={14} />
          </div>
        </div>
      </div>

      {/* Recent Experiments Table */}
      <div className="nf-card">
        <div className="nf-card-header">
          <div className="nf-card-title">
            <Clock size={16} /> Recent Experiments
          </div>
          <button className="nf-btn nf-btn-secondary" onClick={() => onNavigate('experiments')}>
            View All Experiments
          </button>
        </div>

        {experiments.length === 0 ? (
          <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-dim)' }}>
            No training experiments recorded yet for this project. Start your first run in the Training Lab!
          </div>
        ) : (
          <div className="nf-table-container">
            <table className="nf-table">
              <thead>
                <tr>
                  <th>Experiment Name</th>
                  <th>Status</th>
                  <th>Epochs</th>
                  <th>Final Train Loss</th>
                  <th>Final Metric</th>
                  <th>Created At</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {experiments.slice(0, 5).map(exp => {
                  const finalMetrics = exp.final_metrics || {};
                  return (
                    <tr key={exp.id}>
                      <td style={{ fontWeight: 600 }}>{exp.name}</td>
                      <td>
                        <span className={`nf-badge nf-badge-${
                          exp.status === 'completed' ? 'success' :
                          exp.status === 'running' ? 'warning' :
                          exp.status === 'failed' ? 'danger' : 'muted'
                        }`}>
                          {exp.status}
                        </span>
                      </td>
                      <td className="nf-mono">{exp.history?.length ?? 0}</td>
                      <td className="nf-mono">
                        {finalMetrics.final_train_loss !== undefined
                          ? Number(finalMetrics.final_train_loss).toFixed(4)
                          : '—'}
                      </td>
                      <td className="nf-mono">
                        {finalMetrics.final_train_metric !== undefined
                          ? `${(Number(finalMetrics.final_train_metric) * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td style={{ color: 'var(--text-dim)' }}>
                        {new Date(exp.created_at).toLocaleDateString()} {new Date(exp.created_at).toLocaleTimeString()}
                      </td>
                      <td>
                        {exp.status === 'completed' && onSelectExperimentForInspector && (
                          <button
                            className="nf-btn nf-btn-secondary"
                            style={{ padding: '3px 8px', fontSize: '11px' }}
                            onClick={() => onSelectExperimentForInspector(exp.id)}
                          >
                            <Eye size={12} /> Inspect
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
