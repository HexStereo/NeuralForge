import React, { useState, useEffect, useRef } from 'react';
import type {
  Project, Dataset, Architecture, Hyperparameters,
  LiveTelemetryMessage, MetricHistoryPoint
} from '../../types';
import { api } from '../../services/api';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend
} from 'recharts';
import {
  Play, Square, Activity, TrendingDown, Award, Terminal
} from 'lucide-react';

interface TrainingLabWorkspaceProps {
  activeProject: Project | null;
  onTrainingFinished?: (experimentId: string) => void;
}

export const TrainingLabWorkspace: React.FC<TrainingLabWorkspaceProps> = ({
  activeProject,
  onTrainingFinished
}) => {
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // Selected config
  const [experimentName, setExperimentName] = useState('Experiment_1');
  const [selectedArchId, setSelectedArchId] = useState('');
  const [selectedDatasetId, setSelectedDatasetId] = useState('');

  // Hyperparameters
  const [epochs, setEpochs] = useState<number>(20);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<number>(0.02);
  const [optimizer, setOptimizer] = useState<'adam' | 'sgd'>('adam');
  const [lossFunction, setLossFunction] = useState<'cross_entropy' | 'mse'>('cross_entropy');
  const [randomSeed, setRandomSeed] = useState<number>(42);

  // Live execution state
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [currentAcc, setCurrentAcc] = useState<number | null>(null);
  const [currentValLoss, setCurrentValLoss] = useState<number | null>(null);
  const [currentValAcc, setCurrentValAcc] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [etaSeconds, setEtaSeconds] = useState<number | null>(null);

  // Chart data
  const [history, setHistory] = useState<MetricHistoryPoint[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  // Load project datasets and architectures
  useEffect(() => {
    if (activeProject) {
      api.listArchitectures(activeProject.id)
        .then(list => {
          setArchitectures(list);
          if (list.length > 0 && !selectedArchId) setSelectedArchId(list[0].id);
        })
        .catch(console.error);

      api.listDatasets(activeProject.id)
        .then(list => {
          setDatasets(list);
          if (list.length > 0 && !selectedDatasetId) {
            setSelectedDatasetId(list[0].id);
            if (list[0].task_type === 'regression') {
              setLossFunction('mse');
            } else {
              setLossFunction('cross_entropy');
            }
          }
        })
        .catch(console.error);
    }
  }, [activeProject]);

  // Scroll terminal logs to bottom
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Start Training Handler
  const handleStartTraining = async () => {
    if (!activeProject || !selectedArchId || !selectedDatasetId) return;

    setStatus('starting');
    setHistory([]);
    setLogs([`[INIT] Starting training: ${experimentName}`]);
    setCurrentEpoch(0);
    setCurrentLoss(null);
    setCurrentAcc(null);
    setCurrentValLoss(null);
    setCurrentValAcc(null);

    const hyperparams: Hyperparameters = {
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      optimizer,
      loss_function: lossFunction,
      random_seed: randomSeed
    };

    try {
      const res = await api.startTraining({
        project_id: activeProject.id,
        name: experimentName,
        architecture_id: selectedArchId,
        dataset_id: selectedDatasetId,
        hyperparameters: hyperparams
      });

      const expId = res.experiment_id;
      setActiveExperimentId(expId);
      setStatus('running');

      // Connect WebSocket telemetry stream
      if (wsRef.current) wsRef.current.close();

      const ws = api.connectTelemetryStream(
        expId,
        (msg: LiveTelemetryMessage) => {
          if (msg.type === 'connected') {
            setLogs(prev => [...prev, `[WS] Telemetry connected to ${expId.slice(0, 8)}`]);
          } else if (msg.type === 'batch_metric') {
            setCurrentStep(msg.step || 0);
            setTotalSteps(msg.total_steps || 1);
            if (msg.batch_loss !== undefined) setCurrentLoss(msg.batch_loss);
            if (msg.batch_metric !== undefined) setCurrentAcc(msg.batch_metric);
            if (msg.elapsed_seconds !== undefined) setElapsedSeconds(msg.elapsed_seconds);
          } else if (msg.type === 'metric') {
            setCurrentEpoch(msg.epoch || 0);
            if (msg.train_loss !== undefined) setCurrentLoss(msg.train_loss);
            if (msg.train_acc !== undefined) setCurrentAcc(msg.train_acc);
            if (msg.val_loss !== undefined) setCurrentValLoss(msg.val_loss);
            if (msg.val_acc !== undefined) setCurrentValAcc(msg.val_acc);
            if (msg.elapsed_seconds !== undefined) setElapsedSeconds(msg.elapsed_seconds);
            if (msg.eta_seconds !== undefined) setEtaSeconds(msg.eta_seconds);

            setHistory(prev => [
              ...prev,
              {
                epoch: msg.epoch || prev.length + 1,
                train_loss: msg.train_loss || 0,
                val_loss: msg.val_loss,
                train_metric: msg.train_acc,
                val_metric: msg.val_acc,
                elapsed_seconds: msg.elapsed_seconds
              }
            ]);

            setLogs(prev => [
              ...prev,
              `[EPOCH ${msg.epoch}/${epochs}] Train Loss: ${msg.train_loss?.toFixed(4)}${
                msg.val_loss !== undefined && msg.val_loss !== null ? ` | Val Loss: ${msg.val_loss.toFixed(4)}` : ''
              }${msg.train_acc !== undefined && msg.train_acc !== null ? ` | Train Acc: ${(msg.train_acc * 100).toFixed(1)}%` : ''}`
            ]);
          } else if (msg.type === 'status') {
            setStatus(msg.status || 'completed');
            setLogs(prev => [...prev, `[STATUS] Training run ${msg.status}.`]);
            if (msg.status === 'completed' || msg.status === 'cancelled' || msg.status === 'failed') {
              if (onTrainingFinished) onTrainingFinished(expId);
            }
          } else if (msg.type === 'error') {
            setStatus('failed');
            setLogs(prev => [...prev, `[ERROR] ${msg.error}`]);
          }
        },
        () => {
          setLogs(prev => [...prev, '[WS] WebSocket connection error']);
        }
      );

      wsRef.current = ws;
    } catch (err: any) {
      setStatus('failed');
      setLogs(prev => [...prev, `[LAUNCH ERROR] ${err.message}`]);
      alert(`Failed to start training: ${err.message}`);
    }
  };

  // Cancel Training Handler
  const handleCancelTraining = async () => {
    if (!activeExperimentId) return;
    try {
      await api.cancelTraining(activeExperimentId);
      setLogs(prev => [...prev, '[CANCEL] Cancellation requested by user...']);
    } catch (err: any) {
      alert(`Cancel failed: ${err.message}`);
    }
  };

  const isTrainingActive = status === 'running' || status === 'starting';

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 50px)', overflow: 'hidden' }}>
      {/* Left Configuration Sidebar */}
      <div style={{
        width: '340px',
        backgroundColor: 'var(--bg-panel)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
        padding: '16px'
      }}>
        <div style={{ marginBottom: '16px' }}>
          <h2 style={{ fontSize: '15px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} color="var(--accent-purple-hover)" /> Training Configuration
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            Hyperparameters & execution controls
          </span>
        </div>

        {/* Experiment Name */}
        <div className="nf-form-group">
          <label className="nf-label">Experiment Name</label>
          <input
            type="text"
            className="nf-input"
            value={experimentName}
            disabled={isTrainingActive}
            onChange={(e) => setExperimentName(e.target.value)}
          />
        </div>

        {/* Architecture Selector */}
        <div className="nf-form-group">
          <label className="nf-label">Architecture</label>
          <select
            className="nf-select"
            value={selectedArchId}
            disabled={isTrainingActive}
            onChange={(e) => setSelectedArchId(e.target.value)}
          >
            <option value="">-- Select Architecture --</option>
            {architectures.map(a => (
              <option key={a.id} value={a.id}>{a.name} ({a.parameter_count} params)</option>
            ))}
          </select>
        </div>

        {/* Dataset Selector */}
        <div className="nf-form-group">
          <label className="nf-label">Dataset</label>
          <select
            className="nf-select"
            value={selectedDatasetId}
            disabled={isTrainingActive}
            onChange={(e) => {
              setSelectedDatasetId(e.target.value);
              const ds = datasets.find(d => d.id === e.target.value);
              if (ds?.task_type === 'regression') setLossFunction('mse');
              else setLossFunction('cross_entropy');
            }}
          >
            <option value="">-- Select Dataset --</option>
            {datasets.map(d => (
              <option key={d.id} value={d.id}>{d.name} ({d.row_count} samples)</option>
            ))}
          </select>
        </div>

        {/* Hyperparameters Form */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
          <div className="nf-form-group" style={{ margin: 0 }}>
            <label className="nf-label">Epochs</label>
            <input
              type="number"
              className="nf-input nf-mono"
              value={epochs}
              min={1}
              max={1000}
              disabled={isTrainingActive}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
            />
          </div>

          <div className="nf-form-group" style={{ margin: 0 }}>
            <label className="nf-label">Batch Size</label>
            <input
              type="number"
              className="nf-input nf-mono"
              value={batchSize}
              min={1}
              max={2048}
              disabled={isTrainingActive}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '12px' }}>
          <div className="nf-form-group" style={{ margin: 0 }}>
            <label className="nf-label">Learning Rate</label>
            <input
              type="number"
              step="0.005"
              className="nf-input nf-mono"
              value={learningRate}
              disabled={isTrainingActive}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.001)}
            />
          </div>

          <div className="nf-form-group" style={{ margin: 0 }}>
            <label className="nf-label">Random Seed</label>
            <input
              type="number"
              className="nf-input nf-mono"
              value={randomSeed}
              disabled={isTrainingActive}
              onChange={(e) => setRandomSeed(parseInt(e.target.value) || 42)}
            />
          </div>
        </div>

        <div className="nf-form-group">
          <label className="nf-label">Optimizer</label>
          <select
            className="nf-select"
            value={optimizer}
            disabled={isTrainingActive}
            onChange={(e) => setOptimizer(e.target.value as any)}
          >
            <option value="adam">Adam (Adaptive Moments)</option>
            <option value="sgd">SGD (with Momentum)</option>
          </select>
        </div>

        <div className="nf-form-group">
          <label className="nf-label">Loss Function</label>
          <select
            className="nf-select"
            value={lossFunction}
            disabled={isTrainingActive}
            onChange={(e) => setLossFunction(e.target.value as any)}
          >
            <option value="cross_entropy">Cross-Entropy (Classification)</option>
            <option value="mse">Mean Squared Error (Regression)</option>
          </select>
        </div>

        {/* Start / Cancel Action Buttons */}
        <div style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)', display: 'flex', gap: '8px' }}>
          <button
            className="nf-btn nf-btn-primary"
            style={{ flex: 1, height: '36px', fontWeight: 600 }}
            onClick={handleStartTraining}
            disabled={isTrainingActive || !selectedArchId || !selectedDatasetId}
          >
            <Play size={14} /> Start Training
          </button>

          {isTrainingActive && (
            <button
              className="nf-btn nf-btn-danger"
              style={{ height: '36px' }}
              onClick={handleCancelTraining}
              title="Stop Training"
            >
              <Square size={14} /> Stop
            </button>
          )}
        </div>
      </div>

      {/* Right Telemetry Dashboard */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Top Progress & Metrics Callout */}
        <div className="nf-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className={`nf-badge nf-badge-${
                status === 'running' ? 'warning' :
                status === 'completed' ? 'success' :
                status === 'failed' ? 'danger' : 'muted'
              }`}>
                ● {status.toUpperCase()}
              </span>
              <span style={{ fontWeight: 600, fontSize: '13px' }}>
                {activeExperimentId ? `Run: ${experimentName}` : 'Ready to train'}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: 'var(--text-muted)' }}>
              <span>Elapsed: <strong className="nf-mono">{elapsedSeconds.toFixed(1)}s</strong></span>
              {etaSeconds !== null && isTrainingActive && (
                <span>ETA: <strong className="nf-mono">{etaSeconds}s</strong></span>
              )}
            </div>
          </div>

          {/* Progress Bar */}
          <div style={{ marginBottom: '14px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '4px', color: 'var(--text-dim)' }}>
              <span>Epoch {currentEpoch} of {epochs}</span>
              <span>Batch {currentStep} / {totalSteps || '—'}</span>
            </div>
            <div style={{ height: '6px', background: 'var(--bg-app)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{
                height: '100%',
                width: `${epochs > 0 ? (currentEpoch / epochs) * 100 : 0}%`,
                background: 'linear-gradient(90deg, var(--accent-purple) 0%, var(--accent-cyan) 100%)',
                transition: 'width 0.2s ease'
              }} />
            </div>
          </div>

          {/* Stats Bar */}
          <div className="nf-grid-4">
            <div className="nf-stat-box">
              <div className="nf-stat-label">Train Loss</div>
              <div className="nf-stat-value nf-mono">
                {currentLoss !== null ? currentLoss.toFixed(4) : '—'}
              </div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">Validation Loss</div>
              <div className="nf-stat-value nf-mono">
                {currentValLoss !== null ? currentValLoss.toFixed(4) : '—'}
              </div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">{lossFunction === 'mse' ? 'Train MAE' : 'Train Accuracy'}</div>
              <div className="nf-stat-value nf-mono" style={{ color: 'var(--accent-purple-hover)' }}>
                {currentAcc !== null ? (lossFunction === 'mse' ? currentAcc.toFixed(4) : `${(currentAcc * 100).toFixed(1)}%`) : '—'}
              </div>
            </div>
            <div className="nf-stat-box">
              <div className="nf-stat-label">{lossFunction === 'mse' ? 'Val MAE' : 'Val Accuracy'}</div>
              <div className="nf-stat-value nf-mono" style={{ color: 'var(--color-success)' }}>
                {currentValAcc !== null ? (lossFunction === 'mse' ? currentValAcc.toFixed(4) : `${(currentValAcc * 100).toFixed(1)}%`) : '—'}
              </div>
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div className="nf-grid-2">
          {/* Loss Curve */}
          <div className="nf-card" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
            <div className="nf-card-title" style={{ marginBottom: '10px' }}>
              <TrendingDown size={14} color="var(--accent-purple)" /> Loss Convergence
            </div>
            <div style={{ flex: 1, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#242b36" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#13171e', borderColor: '#333d4d', fontSize: '11px' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#06b6d4" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Metric Curve */}
          <div className="nf-card" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
            <div className="nf-card-title" style={{ marginBottom: '10px' }}>
              <Award size={14} color="var(--color-success)" /> Metric Progress
            </div>
            <div style={{ flex: 1, width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#242b36" />
                  <XAxis dataKey="epoch" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#13171e', borderColor: '#333d4d', fontSize: '11px' }}
                    labelStyle={{ color: '#f1f5f9' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                  <Line type="monotone" dataKey="train_metric" name="Train Metric" stroke="#10b981" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="val_metric" name="Val Metric" stroke="#f59e0b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Live Step Telemetry Log Terminal */}
        <div className="nf-card" style={{ flex: 1, minHeight: '160px', display: 'flex', flexDirection: 'column' }}>
          <div className="nf-card-header" style={{ marginBottom: '8px' }}>
            <div className="nf-card-title">
              <Terminal size={14} /> Telemetry Event Log
            </div>
            <button
              className="nf-btn nf-btn-secondary"
              style={{ padding: '2px 8px', fontSize: '11px' }}
              onClick={() => setLogs([])}
            >
              Clear Log
            </button>
          </div>

          <div style={{
            flex: 1,
            background: 'var(--bg-app)',
            borderRadius: '4px',
            border: '1px solid var(--border-subtle)',
            padding: '10px 14px',
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            overflowY: 'auto',
            maxHeight: '200px',
            color: 'var(--text-muted)'
          }}>
            {logs.length === 0 ? (
              <span style={{ color: 'var(--text-dim)' }}>Awaiting training telemetry...</span>
            ) : (
              logs.map((line, idx) => (
                <div key={idx} style={{
                  color: line.includes('[ERROR]') ? 'var(--color-danger)' :
                         line.includes('[EPOCH') ? 'var(--text-main)' :
                         line.includes('[STATUS]') ? 'var(--accent-purple-hover)' : 'var(--text-dim)',
                  marginBottom: '2px'
                }}>
                  {line}
                </div>
              ))
            )}
            <div ref={terminalEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
};
