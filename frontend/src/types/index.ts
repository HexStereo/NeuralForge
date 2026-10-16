export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  dataset_count: number;
  architecture_count: number;
  experiment_count: number;
}

export interface ColumnStat {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  sample_values: any[];
  min_val?: number;
  max_val?: number;
  mean_val?: number;
  std_val?: number;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  filename: string;
  row_count: number;
  column_count: number;
  task_type: 'classification' | 'regression';
  feature_columns: string[];
  target_column: string;
  split_config: {
    train_ratio: number;
    val_ratio: number;
    test_ratio: number;
    random_seed: number;
    shuffle?: boolean;
  };
  summary_stats: {
    columns: ColumnStat[];
    row_count: number;
    column_count: number;
  };
  preview_rows: Record<string, any>[];
  created_at: string;
}

export interface ArchitectureNodeData {
  [key: string]: any;
  label?: string;
  in_features?: number;
  out_features?: number;
  activation?: string;
  bias?: boolean;
  p?: number;
  dim?: number;
}

export interface ArchitectureNode {
  id: string;
  type: string;
  data: ArchitectureNodeData;
  position?: { x: number; y: number };
}

export interface ArchitectureEdge {
  id: string;
  source: string;
  target: string;
}

export interface ArchitectureConfig {
  input_dim: number;
  output_dim: number;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
}

export interface LayerSummary {
  index: number;
  name: string;
  type: string;
  input_shape: number[];
  output_shape: number[];
  param_count: number;
}

export interface ArchitectureValidationResult {
  is_valid: boolean;
  errors: string[];
  total_parameters: number;
  layer_summaries: LayerSummary[];
}

export interface Architecture {
  id: string;
  project_id: string;
  name: string;
  description: string;
  config: ArchitectureConfig;
  parameter_count: number;
  is_valid: boolean;
  validation_errors: string[];
  created_at: string;
}

export interface Hyperparameters {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: 'adam' | 'sgd';
  optimizer_params?: Record<string, any>;
  loss_function: 'cross_entropy' | 'mse';
  random_seed: number;
  checkpoint_frequency?: number;
}

export interface MetricHistoryPoint {
  epoch: number;
  train_loss: number;
  train_metric?: number;
  val_loss?: number;
  val_metric?: number;
  elapsed_seconds?: number;
}

export interface LiveTelemetryMessage {
  type: 'metric' | 'batch_metric' | 'status' | 'error' | 'connected';
  experiment_id: string;
  status?: string;
  epoch?: number;
  total_epochs?: number;
  step?: number;
  total_steps?: number;
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  batch_loss?: number;
  batch_metric?: number;
  elapsed_seconds?: number;
  eta_seconds?: number;
  message?: string;
  error?: string;
}

export interface Experiment {
  id: string;
  project_id: string;
  name: string;
  architecture_id?: string;
  dataset_id?: string;
  status: 'queued' | 'running' | 'completed' | 'cancelled' | 'failed';
  hyperparameters: Hyperparameters;
  final_metrics: Record<string, any>;
  history: MetricHistoryPoint[];
  checkpoint_path: string;
  error_message: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
}

export interface ParameterDistribution {
  shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  histogram: {
    counts: number[];
    bin_edges: number[];
  };
}

export interface InspectedLayer {
  index: number;
  type: string;
  trainable: boolean;
  param_count: number;
  parameters: Record<string, ParameterDistribution>;
}

export interface InspectedActivation {
  layer_index: number;
  layer_type: string;
  output_shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  sparsity_percentage: number;
  histogram: {
    counts: number[];
    bin_edges: number[];
  };
}

export interface ModelInspectorReport {
  experiment_id: string;
  architecture_name: string;
  dataset_name: string;
  task_type: string;
  layers: InspectedLayer[];
  activations: InspectedActivation[];
  evaluation: {
    accuracy?: number;
    classes?: Record<string, {
      precision: number;
      recall: number;
      f1_score: number;
      support: number;
    }>;
    macro_avg?: { precision: number; recall: number; f1_score: number };
    weighted_avg?: { precision: number; recall: number; f1_score: number };
    confusion_matrix?: number[][];
    target_classes?: string[];
    mae?: number;
    rmse?: number;
    r2_score?: number;
  };
}
