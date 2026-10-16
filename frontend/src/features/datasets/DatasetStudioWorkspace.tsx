import React, { useState, useEffect } from 'react';
import type { Project, Dataset } from '../../types';
import { api } from '../../services/api';
import { Database, Upload, Sliders, FileText } from 'lucide-react';

interface DatasetStudioWorkspaceProps {
  activeProject: Project | null;
  onDatasetUpdated?: (ds: Dataset) => void;
}

export const DatasetStudioWorkspace: React.FC<DatasetStudioWorkspaceProps> = ({
  activeProject,
  onDatasetUpdated
}) => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  const [uploading, setUploading] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);

  // Configuration edit state
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>('classification');
  const [trainRatio, setTrainRatio] = useState<number>(0.7);
  const [valRatio, setValRatio] = useState<number>(0.15);
  const [testRatio, setTestRatio] = useState<number>(0.15);
  const [randomSeed, setRandomSeed] = useState<number>(42);

  // Refresh datasets
  const fetchDatasets = async () => {
    if (!activeProject) return;
    try {
      const list = await api.listDatasets(activeProject.id);
      setDatasets(list);
      if (list.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(list[0].id);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, [activeProject]);

  // Sync selected dataset details
  useEffect(() => {
    if (!selectedDatasetId) {
      setSelectedDataset(null);
      return;
    }
    api.getDataset(selectedDatasetId)
      .then(ds => {
        setSelectedDataset(ds);
        setSelectedFeatures(ds.feature_columns || []);
        setSelectedTarget(ds.target_column || '');
        setTaskType(ds.task_type || 'classification');
        if (ds.split_config) {
          setTrainRatio(ds.split_config.train_ratio ?? 0.7);
          setValRatio(ds.split_config.val_ratio ?? 0.15);
          setTestRatio(ds.split_config.test_ratio ?? 0.15);
          setRandomSeed(ds.split_config.random_seed ?? 42);
        }
      })
      .catch(console.error);
  }, [selectedDatasetId]);

  // Handle CSV file upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0 || !activeProject) return;
    const file = e.target.files[0];
    const name = file.name.replace(/\.csv$/i, '');

    setUploading(true);
    try {
      const created = await api.uploadDataset(activeProject.id, name, file);
      setDatasets(prev => [created, ...prev]);
      setSelectedDatasetId(created.id);
      alert(`Dataset "${created.name}" uploaded and profiled successfully!`);
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  // Load sample dataset
  const handleLoadSample = async () => {
    if (!activeProject) return;
    setUploading(true);
    try {
      const created = await api.createSampleDataset(activeProject.id);
      setDatasets(prev => [created, ...prev]);
      setSelectedDatasetId(created.id);
      alert('Sample Sensor Telemetry dataset loaded!');
    } catch (err: any) {
      alert(`Failed to load sample dataset: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  // Toggle feature checkbox
  const toggleFeature = (colName: string) => {
    if (selectedFeatures.includes(colName)) {
      setSelectedFeatures(selectedFeatures.filter(f => f !== colName));
    } else {
      setSelectedFeatures([...selectedFeatures, colName]);
    }
  };

  // Save configuration
  const handleSaveConfig = async () => {
    if (!selectedDatasetId) return;
    setSavingConfig(true);
    try {
      const updated = await api.updateDatasetConfig(selectedDatasetId, {
        task_type: taskType,
        feature_columns: selectedFeatures,
        target_column: selectedTarget,
        split_config: {
          train_ratio: trainRatio,
          val_ratio: valRatio,
          test_ratio: testRatio,
          random_seed: randomSeed,
          shuffle: true
        }
      });
      setSelectedDataset(updated);
      if (onDatasetUpdated) onDatasetUpdated(updated);
      alert('Dataset configuration updated successfully!');
    } catch (err: any) {
      alert(`Update failed: ${err.message}`);
    } finally {
      setSavingConfig(false);
    }
  };

  const columns = selectedDataset?.summary_stats?.columns || [];

  return (
    <div className="nf-workspace">
      {/* Top action bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div>
          <h1 style={{ fontSize: '18px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Database size={18} color="var(--accent-cyan)" /> Dataset Studio
          </h1>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            CSV upload, automated statistical profiling, feature encoding, and leak-free splitting
          </span>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            className="nf-btn nf-btn-secondary"
            onClick={handleLoadSample}
            disabled={uploading || !activeProject}
          >
            <Database size={14} /> Load Sample Dataset
          </button>

          <label className="nf-btn nf-btn-primary" style={{ cursor: uploading ? 'not-allowed' : 'pointer' }}>
            <Upload size={14} /> {uploading ? 'Profiling...' : 'Upload CSV'}
            <input
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              disabled={uploading || !activeProject}
            />
          </label>
        </div>
      </div>

      {/* Dataset Picker and Stats */}
      <div className="nf-card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '12px', fontWeight: 600 }}>Active Dataset:</span>
            <select
              className="nf-select"
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              style={{ minWidth: '220px' }}
            >
              <option value="">-- Select Dataset --</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows, {d.column_count} cols)</option>
              ))}
            </select>
          </div>

          {selectedDataset && (
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
              <div>
                <span className="nf-label">Rows:</span>{' '}
                <span className="nf-mono" style={{ fontWeight: 600 }}>{selectedDataset.row_count.toLocaleString()}</span>
              </div>
              <div>
                <span className="nf-label">Columns:</span>{' '}
                <span className="nf-mono" style={{ fontWeight: 600 }}>{selectedDataset.column_count}</span>
              </div>
              <div>
                <span className="nf-label">Task Type:</span>{' '}
                <span className="nf-badge nf-badge-purple">{taskType}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {selectedDataset ? (
        <>
          {/* Column Selection & Preprocessing Panel */}
          <div className="nf-grid-2" style={{ marginBottom: '20px' }}>
            {/* Feature and Target Selection */}
            <div className="nf-card">
              <div className="nf-card-header">
                <div className="nf-card-title">
                  <Sliders size={15} /> Feature & Target Selection
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="nf-btn nf-btn-secondary"
                    style={{ padding: '2px 8px', fontSize: '11px' }}
                    onClick={() => setSelectedFeatures(columns.map(c => c.name).filter(n => n !== selectedTarget))}
                  >
                    Select All
                  </button>
                  <button
                    className="nf-btn nf-btn-secondary"
                    style={{ padding: '2px 8px', fontSize: '11px' }}
                    onClick={() => setSelectedFeatures([])}
                  >
                    Clear
                  </button>
                </div>
              </div>

              <div className="nf-form-group">
                <label className="nf-label">Supervised Task Type</label>
                <select
                  className="nf-select"
                  value={taskType}
                  onChange={(e) => setTaskType(e.target.value as any)}
                >
                  <option value="classification">Classification (Cross-Entropy Loss)</option>
                  <option value="regression">Regression (Mean Squared Error Loss)</option>
                </select>
              </div>

              <div style={{ maxHeight: '240px', overflowY: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '4px' }}>
                <table className="nf-table">
                  <thead>
                    <tr>
                      <th style={{ width: '40px' }}>Feature</th>
                      <th style={{ width: '40px' }}>Target</th>
                      <th>Column Name</th>
                      <th>Data Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {columns.map(col => {
                      const isFeature = selectedFeatures.includes(col.name);
                      const isTarget = selectedTarget === col.name;

                      return (
                        <tr key={col.name}>
                          <td style={{ textAlign: 'center' }}>
                            <input
                              type="checkbox"
                              checked={isFeature}
                              disabled={isTarget}
                              onChange={() => toggleFeature(col.name)}
                            />
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <input
                              type="radio"
                              name="target_col_radio"
                              checked={isTarget}
                              onChange={() => {
                                setSelectedTarget(col.name);
                                setSelectedFeatures(selectedFeatures.filter(f => f !== col.name));
                              }}
                            />
                          </td>
                          <td style={{ fontWeight: 600 }}>{col.name}</td>
                          <td className="nf-mono" style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
                            {col.dtype}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Train / Validation / Test Splitting */}
            <div className="nf-card">
              <div className="nf-card-header">
                <div className="nf-card-title">
                  <Sliders size={15} /> Leak-Free Data Splitting
                </div>
              </div>

              <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '14px' }}>
                Scalers are fitted strictly on the Training set only, preventing information leakage into Validation and Test sets.
              </p>

              <div className="nf-form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <label className="nf-label">Train Ratio: {(trainRatio * 100).toFixed(0)}%</label>
                  <span className="nf-mono" style={{ fontSize: '11px' }}>
                    {Math.round(selectedDataset.row_count * trainRatio)} samples
                  </span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.85"
                  step="0.05"
                  value={trainRatio}
                  onChange={(e) => {
                    const t = parseFloat(e.target.value);
                    setTrainRatio(t);
                    const rem = parseFloat((1.0 - t).toFixed(2));
                    setValRatio(parseFloat((rem / 2).toFixed(2)));
                    setTestRatio(parseFloat((rem / 2).toFixed(2)));
                  }}
                />
              </div>

              <div style={{ display: 'flex', gap: '12px' }}>
                <div className="nf-form-group" style={{ flex: 1 }}>
                  <label className="nf-label">Val Ratio: {(valRatio * 100).toFixed(0)}%</label>
                  <span className="nf-mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                    {Math.round(selectedDataset.row_count * valRatio)} samples
                  </span>
                </div>

                <div className="nf-form-group" style={{ flex: 1 }}>
                  <label className="nf-label">Test Ratio: {(testRatio * 100).toFixed(0)}%</label>
                  <span className="nf-mono" style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
                    {Math.round(selectedDataset.row_count * testRatio)} samples
                  </span>
                </div>
              </div>

              <div className="nf-form-group">
                <label className="nf-label">Random Split Seed</label>
                <input
                  type="number"
                  className="nf-input nf-mono"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(parseInt(e.target.value) || 42)}
                />
              </div>

              <button
                className="nf-btn nf-btn-primary"
                style={{ width: '100%', marginTop: '8px' }}
                onClick={handleSaveConfig}
                disabled={savingConfig || selectedFeatures.length === 0 || !selectedTarget}
              >
                {savingConfig ? 'Saving...' : 'Save Dataset Configuration'}
              </button>
            </div>
          </div>

          {/* Statistical Profiling Table */}
          <div className="nf-card" style={{ marginBottom: '20px' }}>
            <div className="nf-card-header">
              <div className="nf-card-title">
                <FileText size={15} /> Column Statistical Profiling
              </div>
            </div>

            <div className="nf-table-container">
              <table className="nf-table">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Unique</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map(col => (
                    <tr key={col.name}>
                      <td style={{ fontWeight: 600 }}>{col.name}</td>
                      <td className="nf-mono" style={{ color: 'var(--text-dim)' }}>{col.dtype}</td>
                      <td className="nf-mono">{col.missing_count} ({col.missing_percentage}%)</td>
                      <td className="nf-mono">{col.unique_count}</td>
                      <td className="nf-mono">{col.min_val !== undefined ? col.min_val : '—'}</td>
                      <td className="nf-mono">{col.max_val !== undefined ? col.max_val : '—'}</td>
                      <td className="nf-mono">{col.mean_val !== undefined ? col.mean_val : '—'}</td>
                      <td className="nf-mono">{col.std_val !== undefined ? col.std_val : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Dataset Preview */}
          <div className="nf-card">
            <div className="nf-card-header">
              <div className="nf-card-title">
                <Database size={15} /> Dataset Preview (First 10 Rows)
              </div>
            </div>

            {selectedDataset.preview_rows && selectedDataset.preview_rows.length > 0 ? (
              <div className="nf-table-container">
                <table className="nf-table">
                  <thead>
                    <tr>
                      {Object.keys(selectedDataset.preview_rows[0]).map(col => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {selectedDataset.preview_rows.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((val: any, cIdx) => (
                          <td key={cIdx} className="nf-mono">
                            {val !== null ? String(val) : '<null>'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '20px' }}>
                Preview data unavailable
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="nf-card" style={{ textAlign: 'center', padding: '60px' }}>
          <Database size={36} color="var(--border-strong)" style={{ marginBottom: '16px' }} />
          <h3 style={{ fontSize: '15px', fontWeight: 600, marginBottom: '8px' }}>No Dataset Selected</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '13px', marginBottom: '20px' }}>
            Upload a CSV file or load the built-in sensor telemetry dataset to begin feature engineering.
          </p>
          <button className="nf-btn nf-btn-primary" onClick={handleLoadSample} disabled={uploading || !activeProject}>
            <Database size={14} /> Load Built-In Sensor Dataset
          </button>
        </div>
      )}
    </div>
  );
};
