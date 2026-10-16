import React, { useState, useEffect } from 'react';
import type { Project, Experiment } from '../../types';
import { api } from '../../services/api';
import {
  BarChart2, Trash2, Download, Eye, GitCompare, X
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend
} from 'recharts';

interface ExperimentCenterWorkspaceProps {
  activeProject: Project | null;
  onSelectExperimentForInspector?: (expId: string) => void;
}

export const ExperimentCenterWorkspace: React.FC<ExperimentCenterWorkspaceProps> = ({
  activeProject,
  onSelectExperimentForInspector
}) => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [loading, setLoading] = useState(false);

  // Load experiments
  const fetchExperiments = () => {
    if (!activeProject) return;
    setLoading(true);
    api.listExperiments(activeProject.id)
      .then(setExperiments)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchExperiments();
  }, [activeProject]);

  // Toggle selection for comparison
  const toggleSelect = (id: string) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter(i => i !== id));
    } else {
      setSelectedIds([...selectedIds, id]);
    }
  };

  // Compare selected experiments
  const handleCompare = async () => {
    if (selectedIds.length === 0) return;
    try {
      const data = await api.compareExperiments(selectedIds);
      setComparisonData(data);
      setShowComparisonModal(true);
    } catch (err: any) {
      alert(`Comparison failed: ${err.message}`);
    }
  };

  // Delete experiment
  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`Are you sure you want to delete experiment "${name}"?`)) return;
    try {
      await api.deleteExperiment(id);
      setExperiments(prev => prev.filter(e => e.id !== id));
      setSelectedIds(prev => prev.filter(i => i !== id));
    } catch (err: any) {
      alert(`Delete failed: ${err.message}`);
    }
  };

  // Export experiment JSON
  const handleExport = (exp: Experiment) => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exp, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `${exp.name}_config.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="nf-workspace">
      {/* Top Bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={18} color="var(--accent-purple-hover)" /> Experiment Center
          </h1>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Experiment history, hyperparameter tracking, multi-run comparison, and checkpoint management
          </span>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            className="nf-btn nf-btn-primary"
            onClick={handleCompare}
            disabled={selectedIds.length < 2}
          >
            <GitCompare size={14} /> Compare Selected ({selectedIds.length})
          </button>
        </div>
      </div>

      {/* Experiments Table */}
      <div className="nf-card">
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-dim)' }}>
            Loading experiment records...
          </div>
        ) : experiments.length === 0 ? (
          <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-dim)' }}>
            No experiments logged yet. Train a model in the Training Lab to record your first experiment!
          </div>
        ) : (
          <div className="nf-table-container">
            <table className="nf-table">
              <thead>
                <tr>
                  <th style={{ width: '40px', textAlign: 'center' }}>
                    <input
                      type="checkbox"
                      checked={selectedIds.length === experiments.length && experiments.length > 0}
                      onChange={(e) => {
                        if (e.target.checked) setSelectedIds(experiments.map(exp => exp.id));
                        else setSelectedIds([]);
                      }}
                    />
                  </th>
                  <th>Experiment Name</th>
                  <th>Status</th>
                  <th>Optimizer</th>
                  <th>Learning Rate</th>
                  <th>Batch / Epochs</th>
                  <th>Final Train Loss</th>
                  <th>Final Accuracy</th>
                  <th>Checkpoint</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => {
                  const hp = exp.hyperparameters || {};
                  const fm = exp.final_metrics || {};
                  const isChecked = selectedIds.includes(exp.id);

                  return (
                    <tr key={exp.id}>
                      <td style={{ textAlign: 'center' }}>
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => toggleSelect(exp.id)}
                        />
                      </td>
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
                      <td className="nf-mono">{String(hp.optimizer || 'adam').toUpperCase()}</td>
                      <td className="nf-mono">{hp.learning_rate || '—'}</td>
                      <td className="nf-mono">{hp.batch_size} / {hp.epochs}</td>
                      <td className="nf-mono">
                        {fm.final_train_loss !== undefined ? Number(fm.final_train_loss).toFixed(4) : '—'}
                      </td>
                      <td className="nf-mono">
                        {fm.final_train_metric !== undefined ? `${(Number(fm.final_train_metric) * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ fontSize: '11px', color: exp.checkpoint_path ? 'var(--color-success)' : 'var(--text-dim)' }}>
                        {exp.checkpoint_path ? 'Saved (.npz)' : 'None'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px' }}>
                          {exp.status === 'completed' && onSelectExperimentForInspector && (
                            <button
                              className="nf-btn nf-btn-secondary"
                              style={{ padding: '3px 6px', fontSize: '11px' }}
                              onClick={() => onSelectExperimentForInspector(exp.id)}
                              title="Inspect model activations and weights"
                            >
                              <Eye size={12} />
                            </button>
                          )}
                          <button
                            className="nf-btn nf-btn-secondary"
                            style={{ padding: '3px 6px', fontSize: '11px' }}
                            onClick={() => handleExport(exp)}
                            title="Export config JSON"
                          >
                            <Download size={12} />
                          </button>
                          <button
                            className="nf-btn nf-btn-danger"
                            style={{ padding: '3px 6px', fontSize: '11px' }}
                            onClick={() => handleDelete(exp.id, exp.name)}
                            title="Delete experiment"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Side-by-Side Comparison Modal */}
      {showComparisonModal && (
        <div className="nf-modal-overlay">
          <div className="nf-modal" style={{ width: '900px', maxWidth: '95vw', maxHeight: '90vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '16px', fontWeight: 700 }}>
                <GitCompare size={18} color="var(--accent-purple-hover)" /> Experiment Comparison
              </div>
              <button
                className="nf-btn nf-btn-secondary"
                style={{ padding: '4px' }}
                onClick={() => setShowComparisonModal(false)}
              >
                <X size={16} />
              </button>
            </div>

            {/* Hyperparameters Comparison Table */}
            <div style={{ marginBottom: '20px' }}>
              <h4 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Hyperparameter & Metric Matrix</h4>
              <div className="nf-table-container">
                <table className="nf-table">
                  <thead>
                    <tr>
                      <th>Experiment</th>
                      <th>Architecture</th>
                      <th>Optimizer</th>
                      <th>LR</th>
                      <th>Batch</th>
                      <th>Epochs</th>
                      <th>Final Loss</th>
                      <th>Final Metric</th>
                      <th>Runtime</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonData.map(c => (
                      <tr key={c.id}>
                        <td style={{ fontWeight: 600 }}>{c.name}</td>
                        <td>{c.architecture_name}</td>
                        <td className="nf-mono">{c.hyperparameters?.optimizer?.toUpperCase()}</td>
                        <td className="nf-mono">{c.hyperparameters?.learning_rate}</td>
                        <td className="nf-mono">{c.hyperparameters?.batch_size}</td>
                        <td className="nf-mono">{c.hyperparameters?.epochs}</td>
                        <td className="nf-mono" style={{ color: 'var(--accent-purple-hover)', fontWeight: 700 }}>
                          {c.final_metrics?.final_train_loss ? Number(c.final_metrics.final_train_loss).toFixed(4) : '—'}
                        </td>
                        <td className="nf-mono" style={{ color: 'var(--color-success)', fontWeight: 700 }}>
                          {c.final_metrics?.final_train_metric ? `${(Number(c.final_metrics.final_train_metric) * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td className="nf-mono">
                          {c.final_metrics?.total_duration ? `${c.final_metrics.total_duration}s` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Loss Comparison Overlay Chart */}
            <div style={{ height: '320px', display: 'flex', flexDirection: 'column' }}>
              <h4 style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                Convergence Curve Comparison (Loss vs Epoch)
              </h4>
              <div style={{ flex: 1 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#242b36" />
                    <XAxis
                      dataKey="epoch"
                      type="number"
                      domain={[1, 'auto']}
                      stroke="#64748b"
                      fontSize={11}
                    />
                    <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#13171e', borderColor: '#333d4d', fontSize: '11px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                    {comparisonData.map((c, idx) => {
                      const colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899'];
                      const strokeColor = colors[idx % colors.length];

                      return (
                        <Line
                          key={c.id}
                          data={c.loss_curve}
                          type="monotone"
                          dataKey="train_loss"
                          name={`${c.name} (Loss)`}
                          stroke={strokeColor}
                          strokeWidth={2}
                          dot={false}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
