import React, { useState, useEffect } from 'react';
import type { Project, Experiment, ModelInspectorReport } from '../../types';
import { api } from '../../services/api';
import { Eye, Grid, Activity, Cpu, Award } from 'lucide-react';

interface ModelInspectorWorkspaceProps {
  activeProject: Project | null;
  selectedExperimentId?: string;
}

export const ModelInspectorWorkspace: React.FC<ModelInspectorWorkspaceProps> = ({
  activeProject,
  selectedExperimentId: initialExpId
}) => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string>(initialExpId || '');
  const [report, setReport] = useState<ModelInspectorReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'evaluation' | 'parameters' | 'activations'>('evaluation');

  // Load completed experiments
  useEffect(() => {
    if (activeProject) {
      api.listExperiments(activeProject.id)
        .then(list => {
          const completed = list.filter(e => e.status === 'completed');
          setExperiments(completed);
          if (completed.length > 0 && !selectedExpId) {
            setSelectedExpId(completed[0].id);
          }
        })
        .catch(console.error);
    }
  }, [activeProject]);

  // Load inspection report
  useEffect(() => {
    if (!selectedExpId) {
      setReport(null);
      return;
    }
    setLoading(true);
    api.inspectExperiment(selectedExpId)
      .then(setReport)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedExpId]);

  const evalData = report?.evaluation;
  const cm = evalData?.confusion_matrix;
  const classesReport = evalData?.classes || {};
  const targetClasses = evalData?.target_classes || [];

  return (
    <div className="nf-workspace">
      {/* Header & Experiment Picker */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Eye size={18} color="var(--accent-purple-hover)" /> Model Inspector
          </h1>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Real forward-pass activations, parameter distributions, confusion matrix, and precision/recall diagnostics
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '12px', fontWeight: 600 }}>Experiment:</span>
          <select
            className="nf-select"
            value={selectedExpId}
            onChange={(e) => setSelectedExpId(e.target.value)}
            style={{ minWidth: '220px' }}
          >
            <option value="">-- Select Completed Experiment --</option>
            {experiments.map(e => (
              <option key={e.id} value={e.id}>{e.name} (ID: {e.id.slice(0, 8)})</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="nf-card" style={{ textAlign: 'center', padding: '60px', color: 'var(--text-dim)' }}>
          Computing real activation passes and parameter statistics...
        </div>
      ) : report ? (
        <>
          {/* Summary Badges Bar */}
          <div className="nf-grid-4" style={{ marginBottom: '20px' }}>
            <div className="nf-stat-box">
              <div className="nf-stat-label">Architecture</div>
              <div style={{ fontSize: '14px', fontWeight: 700 }}>{report.architecture_name}</div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">Dataset</div>
              <div style={{ fontSize: '14px', fontWeight: 700 }}>{report.dataset_name}</div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">Model Accuracy</div>
              <div className="nf-stat-value nf-mono" style={{ color: 'var(--color-success)' }}>
                {evalData?.accuracy !== undefined ? `${(evalData.accuracy * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">Inspected Layers</div>
              <div className="nf-stat-value nf-mono" style={{ color: 'var(--accent-purple-hover)' }}>
                {report.layers.length}
              </div>
            </div>
          </div>

          {/* Tab Selector */}
          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '8px' }}>
            <button
              className={`nf-btn ${activeTab === 'evaluation' ? 'nf-btn-primary' : 'nf-btn-secondary'}`}
              onClick={() => setActiveTab('evaluation')}
            >
              <Grid size={14} /> Evaluation & Confusion Matrix
            </button>
            <button
              className={`nf-btn ${activeTab === 'parameters' ? 'nf-btn-primary' : 'nf-btn-secondary'}`}
              onClick={() => setActiveTab('parameters')}
            >
              <Cpu size={14} /> Layer Parameters & Distributions
            </button>
            <button
              className={`nf-btn ${activeTab === 'activations' ? 'nf-btn-primary' : 'nf-btn-secondary'}`}
              onClick={() => setActiveTab('activations')}
            >
              <Activity size={14} /> Real Layer Activations
            </button>
          </div>

          {/* Tab 1: Evaluation & Confusion Matrix */}
          {activeTab === 'evaluation' && (
            <div className="nf-grid-2">
              {/* Confusion Matrix Heatmap */}
              <div className="nf-card">
                <div className="nf-card-header">
                  <div className="nf-card-title">
                    <Grid size={15} /> Confusion Matrix (Actual vs Predicted)
                  </div>
                </div>

                {cm && cm.length > 0 ? (
                  <div style={{ overflowX: 'auto', padding: '10px 0' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: `auto repeat(${cm.length}, 60px)`, gap: '4px', textAlign: 'center' }}>
                      <div />
                      {cm.map((_, j) => (
                        <div key={j} style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 600 }}>
                          Pred {targetClasses[j] || j}
                        </div>
                      ))}

                      {cm.map((row, i) => (
                        <React.Fragment key={i}>
                          <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '8px' }}>
                            True {targetClasses[i] || i}
                          </div>
                          {row.map((val, j) => {
                            const isDiag = i === j;
                            const maxVal = Math.max(...cm.flat(), 1);
                            const intensity = val / maxVal;
                            const bg = isDiag
                              ? `rgba(16, 185, 129, ${0.15 + intensity * 0.7})`
                              : val > 0
                              ? `rgba(239, 68, 68, ${0.15 + intensity * 0.5})`
                              : 'var(--bg-panel)';

                            return (
                              <div
                                key={j}
                                className="nf-mono"
                                style={{
                                  backgroundColor: bg,
                                  height: '48px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontWeight: 700,
                                  fontSize: '12px',
                                  borderRadius: '3px',
                                  border: '1px solid var(--border-subtle)',
                                  color: val > 0 ? '#fff' : 'var(--text-dim)'
                                }}
                              >
                                {val}
                              </div>
                            );
                          })}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ color: 'var(--text-dim)', padding: '20px', textAlign: 'center' }}>
                    Confusion matrix only available for classification tasks
                  </div>
                )}
              </div>

              {/* Classification Report */}
              <div className="nf-card">
                <div className="nf-card-header">
                  <div className="nf-card-title">
                    <Award size={15} /> Classification Report
                  </div>
                </div>

                <div className="nf-table-container">
                  <table className="nf-table">
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(classesReport).map(([clsIdx, metrics]) => (
                        <tr key={clsIdx}>
                          <td style={{ fontWeight: 600 }}>{targetClasses[parseInt(clsIdx)] || `Class ${clsIdx}`}</td>
                          <td className="nf-mono">{(metrics.precision * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(metrics.recall * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(metrics.f1_score * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{metrics.support}</td>
                        </tr>
                      ))}
                      {evalData?.macro_avg && (
                        <tr style={{ background: 'var(--bg-panel)', fontWeight: 600 }}>
                          <td>Macro Avg</td>
                          <td className="nf-mono">{(evalData.macro_avg.precision * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(evalData.macro_avg.recall * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(evalData.macro_avg.f1_score * 100).toFixed(1)}%</td>
                          <td className="nf-mono">—</td>
                        </tr>
                      )}
                      {evalData?.weighted_avg && (
                        <tr style={{ background: 'var(--bg-panel)', fontWeight: 600 }}>
                          <td>Weighted Avg</td>
                          <td className="nf-mono">{(evalData.weighted_avg.precision * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(evalData.weighted_avg.recall * 100).toFixed(1)}%</td>
                          <td className="nf-mono">{(evalData.weighted_avg.f1_score * 100).toFixed(1)}%</td>
                          <td className="nf-mono">—</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Layer Parameters & Distributions */}
          {activeTab === 'parameters' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {report.layers.map((layer) => (
                <div key={layer.index} className="nf-card">
                  <div className="nf-card-header">
                    <div className="nf-card-title">
                      Layer {layer.index}: {layer.type}
                      <span className="nf-badge nf-badge-purple" style={{ marginLeft: '8px' }}>
                        {layer.param_count.toLocaleString()} parameters
                      </span>
                    </div>
                  </div>

                  {Object.keys(layer.parameters).length === 0 ? (
                    <div style={{ color: 'var(--text-dim)', fontSize: '12px' }}>
                      Non-parametric layer
                    </div>
                  ) : (
                    <div className="nf-grid-2">
                      {Object.entries(layer.parameters).map(([pName, pStat]) => {
                        const counts = pStat.histogram.counts;
                        const maxCount = Math.max(...counts, 1);

                        return (
                          <div key={pName} className="nf-stat-box">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                              <strong style={{ fontSize: '13px' }}>{pName}</strong>
                              <span className="nf-mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                                Shape: [{pStat.shape.join(', ')}]
                              </span>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', fontSize: '11px', marginBottom: '12px' }}>
                              <div><span className="nf-label">Mean:</span> <span className="nf-mono">{pStat.mean}</span></div>
                              <div><span className="nf-label">Std:</span> <span className="nf-mono">{pStat.std}</span></div>
                              <div><span className="nf-label">Min:</span> <span className="nf-mono">{pStat.min}</span></div>
                              <div><span className="nf-label">Max:</span> <span className="nf-mono">{pStat.max}</span></div>
                            </div>

                            {/* Histogram Visualization */}
                            <div>
                              <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginBottom: '4px' }}>Distribution Histogram:</div>
                              <div style={{ display: 'flex', alignItems: 'flex-end', height: '40px', gap: '2px', background: 'var(--bg-app)', padding: '4px', borderRadius: '4px' }}>
                                {counts.map((c, idx) => (
                                  <div
                                    key={idx}
                                    style={{
                                      flex: 1,
                                      height: `${(c / maxCount) * 100}%`,
                                      backgroundColor: 'var(--accent-purple)',
                                      borderRadius: '1px'
                                    }}
                                    title={`Bin: ${pStat.histogram.bin_edges[idx]} to ${pStat.histogram.bin_edges[idx+1]} (count: ${c})`}
                                  />
                                ))}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Tab 3: Real Layer Activations */}
          {activeTab === 'activations' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Intermediate activations captured from a test batch evaluation through NeuralForge Core.
              </p>

              {report.activations.map((act) => {
                const counts = act.histogram.counts;
                const maxCount = Math.max(...counts, 1);

                return (
                  <div key={act.layer_index} className="nf-card">
                    <div className="nf-card-header">
                      <div className="nf-card-title">
                        Layer {act.layer_index}: {act.layer_type}
                        <span className="nf-mono" style={{ fontSize: '11px', color: 'var(--text-dim)', marginLeft: '10px' }}>
                          Output Shape: [{act.output_shape.join(', ')}]
                        </span>
                      </div>
                      <span className="nf-badge nf-badge-muted">
                        Sparsity: {act.sparsity_percentage}% Zeros
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', fontSize: '11px', marginBottom: '14px' }}>
                      <div className="nf-stat-box"><span className="nf-label">Mean Activation:</span> <span className="nf-mono">{act.mean}</span></div>
                      <div className="nf-stat-box"><span className="nf-label">Std Dev:</span> <span className="nf-mono">{act.std}</span></div>
                      <div className="nf-stat-box"><span className="nf-label">Min Value:</span> <span className="nf-mono">{act.min}</span></div>
                      <div className="nf-stat-box"><span className="nf-label">Max Value:</span> <span className="nf-mono">{act.max}</span></div>
                    </div>

                    <div>
                      <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '4px' }}>Activation Value Histogram:</div>
                      <div style={{ display: 'flex', alignItems: 'flex-end', height: '48px', gap: '2px', background: 'var(--bg-app)', padding: '4px', borderRadius: '4px' }}>
                        {counts.map((c, idx) => (
                          <div
                            key={idx}
                            style={{
                              flex: 1,
                              height: `${(c / maxCount) * 100}%`,
                              backgroundColor: 'var(--accent-cyan)',
                              borderRadius: '1px'
                            }}
                            title={`Bin: ${act.histogram.bin_edges[idx]} to ${act.histogram.bin_edges[idx+1]} (count: ${c})`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      ) : (
        <div className="nf-card" style={{ textAlign: 'center', padding: '60px' }}>
          <Eye size={36} color="var(--border-strong)" style={{ marginBottom: '16px' }} />
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>No Completed Experiment Selected</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
            Run a training experiment to inspect its real activations, weight distributions, and confusion matrix.
          </p>
        </div>
      )}
    </div>
  );
};
