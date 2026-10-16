import React, { useState, useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  Handle,
  Position,
  BackgroundVariant
} from '@xyflow/react';
import type { Connection, Edge, Node, NodeProps } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type { Project, Architecture, ArchitectureConfig, ArchitectureValidationResult } from '../../types';
import { api } from '../../services/api';
import { Layers, Plus, Save, Trash2 } from 'lucide-react';

// Custom Node Component with Handles and Parameter Controls
const CustomLayerNode: React.FC<NodeProps> = ({ data, isConnectable }) => {
  const type = String(data.type || 'dense').toLowerCase();
  const isInput = type === 'input';
  const isOutput = type === 'output';

  return (
    <div className="nf-flow-node" style={{
      borderColor: isInput ? 'var(--accent-cyan)' : isOutput ? 'var(--color-success)' : 'var(--border-strong)'
    }}>
      {!isInput && (
        <Handle
          type="target"
          position={Position.Top}
          isConnectable={isConnectable}
          style={{ background: 'var(--accent-purple)', width: '8px', height: '8px' }}
        />
      )}

      <div className="nf-node-header">
        <span style={{ textTransform: 'uppercase', fontSize: '11px', letterSpacing: '0.5px' }}>
          {String(data.label || type)}
        </span>
        <span className={`nf-badge nf-badge-${isInput ? 'cyan' : isOutput ? 'success' : 'purple'}`} style={{ fontSize: '9px' }}>
          {type}
        </span>
      </div>

      <div className="nf-node-body">
        {type === 'dense' && (
          <div>
            <div>In: {String(data.in_features || 0)} → Out: {String(data.out_features || 0)}</div>
            <div style={{ color: 'var(--text-dim)', fontSize: '10px' }}>
              Bias: {data.bias !== false ? 'True' : 'False'} | Params: {((Number(data.in_features) || 0) * (Number(data.out_features) || 0) + (data.bias !== false ? (Number(data.out_features) || 0) : 0))}
            </div>
          </div>
        )}
        {type === 'input' && (
          <div>Input Dim: {String(data.dim || 4)}</div>
        )}
        {type === 'output' && (
          <div>Target Classes: {String(data.dim || 3)}</div>
        )}
        {type === 'dropout' && (
          <div>Drop Rate: {String(data.p || 0.5)}</div>
        )}
        {['relu', 'sigmoid', 'tanh', 'softmax'].includes(type) && (
          <div style={{ color: 'var(--text-dim)' }}>Element-wise activation</div>
        )}
      </div>

      {!isOutput && (
        <Handle
          type="source"
          position={Position.Bottom}
          isConnectable={isConnectable}
          style={{ background: 'var(--accent-purple)', width: '8px', height: '8px' }}
        />
      )}
    </div>
  );
};

interface NetworkBuilderWorkspaceProps {
  activeProject: Project | null;
  onArchitectureSaved?: (arch: Architecture) => void;
}

export const NetworkBuilderWorkspace: React.FC<NetworkBuilderWorkspaceProps> = ({
  activeProject,
  onArchitectureSaved
}) => {
  const [archName, setArchName] = useState('Feedforward MLP Classifier');
  const [archDesc, setArchDesc] = useState('Fully-connected network with ReLU activations');
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [selectedArchId, setSelectedArchId] = useState<string>('');

  const [inputDim, setInputDim] = useState<number>(4);
  const [outputDim, setOutputDim] = useState<number>(3);

  // Default initial graph
  const initialNodes: Node[] = [
    { id: 'node_in', type: 'customLayer', position: { x: 250, y: 30 }, data: { type: 'input', label: 'Input Features', dim: 4 } },
    { id: 'node_d1', type: 'customLayer', position: { x: 250, y: 130 }, data: { type: 'dense', label: 'Dense 1', in_features: 4, out_features: 16, bias: true } },
    { id: 'node_r1', type: 'customLayer', position: { x: 250, y: 240 }, data: { type: 'relu', label: 'ReLU 1' } },
    { id: 'node_d2', type: 'customLayer', position: { x: 250, y: 330 }, data: { type: 'dense', label: 'Dense 2', in_features: 16, out_features: 8, bias: true } },
    { id: 'node_r2', type: 'customLayer', position: { x: 250, y: 440 }, data: { type: 'relu', label: 'ReLU 2' } },
    { id: 'node_d3', type: 'customLayer', position: { x: 250, y: 530 }, data: { type: 'dense', label: 'Dense Output', in_features: 8, out_features: 3, bias: true } },
    { id: 'node_out', type: 'customLayer', position: { x: 250, y: 640 }, data: { type: 'output', label: 'Network Output', dim: 3 } }
  ];

  const initialEdges: Edge[] = [
    { id: 'e1', source: 'node_in', target: 'node_d1', animated: true },
    { id: 'e2', source: 'node_d1', target: 'node_r1' },
    { id: 'e3', source: 'node_r1', target: 'node_d2' },
    { id: 'e4', source: 'node_d2', target: 'node_r2' },
    { id: 'e5', source: 'node_r2', target: 'node_d3' },
    { id: 'e6', source: 'node_d3', target: 'node_out' }
  ];

  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const [validationResult, setValidationResult] = useState<ArchitectureValidationResult | null>(null);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);

  const nodeTypes = useMemo(() => ({ customLayer: CustomLayerNode }), []);

  const onNodesChange = useCallback((changes: any) => setNodes((nds) => applyNodeChanges(changes, nds)), []);
  const onEdgesChange = useCallback((changes: any) => setEdges((eds) => applyEdgeChanges(changes, eds)), []);
  const onConnect = useCallback((connection: Connection) => setEdges((eds) => addEdge(connection, eds)), []);

  const onNodeClick = useCallback((_: any, node: Node) => {
    setSelectedNodeId(node.id);
  }, []);

  // Fetch architectures for project
  useEffect(() => {
    if (activeProject) {
      api.listArchitectures(activeProject.id)
        .then(setArchitectures)
        .catch(console.error);
    }
  }, [activeProject]);

  // Construct config object
  const buildCurrentConfig = useCallback((): ArchitectureConfig => {
    return {
      input_dim: inputDim,
      output_dim: outputDim,
      nodes: nodes.map(n => ({
        id: n.id,
        type: String(n.data.type || 'dense'),
        data: n.data,
        position: n.position
      })),
      edges: edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target
      }))
    };
  }, [nodes, edges, inputDim, outputDim]);

  // Validate architecture with backend
  const validateCurrentArch = useCallback(async () => {
    setValidating(true);
    try {
      const cfg = buildCurrentConfig();
      const res = await api.validateArchitecture(cfg);
      setValidationResult(res);
    } catch (err: any) {
      setValidationResult({
        is_valid: false,
        errors: [err.message || 'Validation request failed'],
        total_parameters: 0,
        layer_summaries: []
      });
    } finally {
      setValidating(false);
    }
  }, [buildCurrentConfig]);

  // Validate whenever nodes or edges change
  useEffect(() => {
    const timer = setTimeout(() => {
      validateCurrentArch();
    }, 400);
    return () => clearTimeout(timer);
  }, [nodes, edges, inputDim, outputDim, validateCurrentArch]);

  // Save Architecture
  const handleSave = async () => {
    if (!activeProject) return;
    setSaving(true);
    try {
      const cfg = buildCurrentConfig();
      const created = await api.createArchitecture({
        project_id: activeProject.id,
        name: archName,
        description: archDesc,
        config: cfg
      });
      setArchitectures(prev => [created, ...prev]);
      setSelectedArchId(created.id);
      if (onArchitectureSaved) onArchitectureSaved(created);
      alert(`Architecture "${created.name}" saved successfully!`);
    } catch (err: any) {
      alert(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // Load selected architecture
  const handleLoadArch = (archId: string) => {
    const arch = architectures.find(a => a.id === archId);
    if (!arch) return;
    setArchName(arch.name);
    setArchDesc(arch.description);
    setInputDim(arch.config.input_dim);
    setOutputDim(arch.config.output_dim);

    const loadedNodes: Node[] = (arch.config.nodes || []).map(n => ({
      id: n.id,
      type: 'customLayer',
      position: n.position || { x: 250, y: 100 },
      data: n.data
    }));

    const loadedEdges: Edge[] = (arch.config.edges || []).map(e => ({
      id: e.id,
      source: e.source,
      target: e.target
    }));

    setNodes(loadedNodes);
    setEdges(loadedEdges);
    setSelectedArchId(archId);
  };

  // Node editing handlers
  const selectedNode = nodes.find(n => n.id === selectedNodeId);

  const updateSelectedNodeData = (field: string, val: any) => {
    if (!selectedNodeId) return;
    setNodes(nds => nds.map(n => {
      if (n.id === selectedNodeId) {
        return {
          ...n,
          data: {
            ...n.data,
            [field]: val
          }
        };
      }
      return n;
    }));
  };

  // Add new node to canvas
  const addNode = (type: string) => {
    const newId = `node_${type}_${Date.now().toString().slice(-4)}`;
    const yPos = nodes.length > 0 ? Math.max(...nodes.map(n => n.position.y)) + 90 : 100;

    let initialData: Record<string, any> = { type, label: type.toUpperCase() };
    if (type === 'dense') {
      initialData = { type: 'dense', label: 'Dense', in_features: 8, out_features: 8, bias: true };
    } else if (type === 'dropout') {
      initialData = { type: 'dropout', label: 'Dropout', p: 0.2 };
    }

    const newNode: Node = {
      id: newId,
      type: 'customLayer',
      position: { x: 250, y: yPos },
      data: initialData
    };

    setNodes(nds => [...nds, newNode]);
    setSelectedNodeId(newId);
  };

  // Delete selected node
  const deleteSelectedNode = () => {
    if (!selectedNodeId) return;
    setNodes(nds => nds.filter(n => n.id !== selectedNodeId));
    setEdges(eds => eds.filter(e => e.source !== selectedNodeId && e.target !== selectedNodeId));
    setSelectedNodeId(null);
  };

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 50px)', overflow: 'hidden' }}>
      {/* Left Sidebar */}
      <div style={{
        width: '320px',
        backgroundColor: 'var(--bg-panel)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
        padding: '16px'
      }}>
        <div style={{ marginBottom: '16px' }}>
          <h2 style={{ fontSize: '15px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Layers size={16} color="var(--accent-purple-hover)" /> Architecture Graph
          </h2>
          <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
            Sequential feedforward topology
          </span>
        </div>

        {/* Saved Architectures Picker */}
        <div className="nf-form-group">
          <label className="nf-label">Saved Architecture</label>
          <div style={{ display: 'flex', gap: '6px' }}>
            <select
              className="nf-select"
              style={{ flex: 1 }}
              value={selectedArchId}
              onChange={(e) => handleLoadArch(e.target.value)}
            >
              <option value="">-- Select Saved Architecture --</option>
              {architectures.map(a => (
                <option key={a.id} value={a.id}>{a.name} ({a.parameter_count} params)</option>
              ))}
            </select>
          </div>
        </div>

        {/* Architecture Metadata */}
        <div className="nf-form-group">
          <label className="nf-label">Model Name</label>
          <input
            type="text"
            className="nf-input"
            value={archName}
            onChange={(e) => setArchName(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '16px' }}>
          <div className="nf-form-group" style={{ flex: 1, margin: 0 }}>
            <label className="nf-label">Input Dim</label>
            <input
              type="number"
              className="nf-input nf-mono"
              value={inputDim}
              onChange={(e) => setInputDim(parseInt(e.target.value) || 1)}
            />
          </div>
          <div className="nf-form-group" style={{ flex: 1, margin: 0 }}>
            <label className="nf-label">Output Dim</label>
            <input
              type="number"
              className="nf-input nf-mono"
              value={outputDim}
              onChange={(e) => setOutputDim(parseInt(e.target.value) || 1)}
            />
          </div>
        </div>

        {/* Add Layer Palette */}
        <div style={{ marginBottom: '20px' }}>
          <label className="nf-label">Add Layers</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', marginTop: '6px' }}>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('dense')}>
              <Plus size={12} /> Dense
            </button>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('relu')}>
              <Plus size={12} /> ReLU
            </button>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('sigmoid')}>
              <Plus size={12} /> Sigmoid
            </button>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('tanh')}>
              <Plus size={12} /> Tanh
            </button>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('softmax')}>
              <Plus size={12} /> Softmax
            </button>
            <button className="nf-btn nf-btn-secondary" onClick={() => addNode('dropout')}>
              <Plus size={12} /> Dropout
            </button>
          </div>
        </div>

        {/* Selected Node Properties Editor */}
        {selectedNode && (
          <div className="nf-card" style={{ marginBottom: '20px', padding: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <span style={{ fontWeight: 600, fontSize: '12px' }}>Edit: {String(selectedNode.data.label || selectedNode.id)}</span>
              <button
                className="nf-btn nf-btn-danger"
                style={{ padding: '2px 6px', fontSize: '11px' }}
                onClick={deleteSelectedNode}
                title="Delete node"
              >
                <Trash2 size={12} />
              </button>
            </div>

            {selectedNode.data.type === 'dense' && (
              <>
                <div className="nf-form-group">
                  <label className="nf-label">In Features</label>
                  <input
                    type="number"
                    className="nf-input nf-mono"
                    value={Number(selectedNode.data.in_features) || 0}
                    onChange={(e) => updateSelectedNodeData('in_features', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="nf-form-group">
                  <label className="nf-label">Out Features</label>
                  <input
                    type="number"
                    className="nf-input nf-mono"
                    value={Number(selectedNode.data.out_features) || 0}
                    onChange={(e) => updateSelectedNodeData('out_features', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px' }}>
                  <input
                    type="checkbox"
                    id="node_bias_check"
                    checked={selectedNode.data.bias !== false}
                    onChange={(e) => updateSelectedNodeData('bias', e.target.checked)}
                  />
                  <label htmlFor="node_bias_check">Include Bias Parameter</label>
                </div>
              </>
            )}

            {selectedNode.data.type === 'dropout' && (
              <div className="nf-form-group">
                <label className="nf-label">Drop Rate (0 to 1)</label>
                <input
                  type="number"
                  step="0.05"
                  className="nf-input nf-mono"
                  value={Number(selectedNode.data.p) || 0.5}
                  onChange={(e) => updateSelectedNodeData('p', parseFloat(e.target.value) || 0.0)}
                />
              </div>
            )}
          </div>
        )}

        {/* Validation Status & Parameter Summary */}
        <div style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Status:</span>
            {validating ? (
              <span className="nf-badge nf-badge-muted">Validating...</span>
            ) : validationResult?.is_valid ? (
              <span className="nf-badge nf-badge-success">● Valid Architecture</span>
            ) : (
              <span className="nf-badge nf-badge-danger">● Invalid Topology</span>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Trainable Parameters:</span>
            <span className="nf-mono" style={{ fontWeight: 700, color: 'var(--accent-purple-hover)' }}>
              {validationResult?.total_parameters?.toLocaleString() || 0}
            </span>
          </div>

          {validationResult?.errors && validationResult.errors.length > 0 && (
            <div style={{
              background: 'var(--color-danger-dim)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '4px',
              padding: '8px',
              fontSize: '11px',
              color: 'var(--color-danger)',
              marginBottom: '12px',
              maxHeight: '100px',
              overflowY: 'auto'
            }}>
              {validationResult.errors.map((err, i) => (
                <div key={i}>• {err}</div>
              ))}
            </div>
          )}

          <button
            className="nf-btn nf-btn-primary"
            style={{ width: '100%' }}
            onClick={handleSave}
            disabled={saving || !validationResult?.is_valid}
          >
            <Save size={14} /> {saving ? 'Saving...' : 'Save Architecture'}
          </button>
        </div>
      </div>

      {/* Main React Flow Canvas */}
      <div style={{ flex: 1, position: 'relative', height: '100%', background: '#090b0e' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          fitView
        >
          <Background color="#1d232e" gap={20} size={1} variant={BackgroundVariant.Dots} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
};
