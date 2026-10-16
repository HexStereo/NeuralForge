import type {
  Project, Dataset, Architecture, ArchitectureConfig,
  ArchitectureValidationResult, Hyperparameters, Experiment,
  ModelInspectorReport, LiveTelemetryMessage
} from '../types';

const API_BASE = '/api';

export async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Accept': 'application/json',
      ...(options?.headers || {})
    }
  });

  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (_) {}
    throw new Error(errorDetail);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const api = {
  // Projects
  getHealth: () => request<{ status: string; service: string; version: string; environment: any }>('/health'),
  listProjects: () => request<Project[]>('/projects'),
  createProject: (data: { name: string; description?: string }) =>
    request<Project>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }),
  getProject: (id: string) => request<Project>(`/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: 'DELETE' }),

  // Datasets
  listDatasets: (projectId?: string) =>
    request<Dataset[]>(projectId ? `/datasets?project_id=${projectId}` : '/datasets'),
  uploadDataset: (projectId: string, name: string, file: File) => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('name', name);
    formData.append('file', file);
    return request<Dataset>('/datasets/upload', {
      method: 'POST',
      body: formData
    });
  },
  createSampleDataset: (projectId: string) => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    return request<Dataset>('/datasets/sample', {
      method: 'POST',
      body: formData
    });
  },
  getDataset: (id: string) => request<Dataset>(`/datasets/${id}`),
  updateDatasetConfig: (id: string, config: any) =>
    request<Dataset>(`/datasets/${id}/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    }),
  deleteDataset: (id: string) => request<void>(`/datasets/${id}`, { method: 'DELETE' }),

  // Architectures
  listArchitectures: (projectId?: string) =>
    request<Architecture[]>(projectId ? `/architectures?project_id=${projectId}` : '/architectures'),
  validateArchitecture: (config: ArchitectureConfig) =>
    request<ArchitectureValidationResult>('/architectures/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    }),
  createArchitecture: (data: { project_id: string; name: string; description?: string; config: ArchitectureConfig }) =>
    request<Architecture>('/architectures', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }),
  getArchitecture: (id: string) => request<Architecture>(`/architectures/${id}`),
  deleteArchitecture: (id: string) => request<void>(`/architectures/${id}`, { method: 'DELETE' }),

  // Training
  startTraining: (data: { project_id: string; name: string; architecture_id: string; dataset_id: string; hyperparameters: Hyperparameters }) =>
    request<{ experiment_id: string; status: string; message: string }>('/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }),
  cancelTraining: (experimentId: string) =>
    request<{ experiment_id: string; status: string }>(`/training/${experimentId}/cancel`, { method: 'POST' }),
  getTrainingStatus: (experimentId: string) =>
    request<{ id: string; status: string; error_message: string; final_metrics: any; epochs_completed: number; history: any[] }>(`/training/${experimentId}/status`),

  // Experiments
  listExperiments: (projectId?: string) =>
    request<Experiment[]>(projectId ? `/experiments?project_id=${projectId}` : '/experiments'),
  getExperiment: (id: string) => request<Experiment>(`/experiments/${id}`),
  getCheckpoints: (experimentId: string) =>
    request<Array<{ id: string; epoch: number; step: number; train_loss: number; val_loss?: number; file_path: string; created_at: string }>>(`/experiments/${experimentId}/checkpoints`),
  compareExperiments: (ids: string[]) =>
    request<any[]>('/experiments/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ids)
    }),
  deleteExperiment: (id: string) => request<void>(`/experiments/${id}`, { method: 'DELETE' }),

  // Model Inspector
  inspectExperiment: (experimentId: string, sampleSize: number = 50) =>
    request<ModelInspectorReport>(`/inspector/${experimentId}?sample_size=${sampleSize}`),

  // WebSocket Connection
  connectTelemetryStream: (
    experimentId: string,
    onMessage: (msg: LiveTelemetryMessage) => void,
    onError?: (err: Event) => void,
    onClose?: () => void
  ) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/training/${experimentId}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        onMessage(parsed);
      } catch (err) {
        console.error('Failed to parse WebSocket telemetry frame', err);
      }
    };

    if (onError) ws.onerror = onError;
    if (onClose) ws.onclose = onClose;

    return ws;
  }
};
