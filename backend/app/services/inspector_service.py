"""Inspection service analyzing real layer weights, activations, and evaluation metrics."""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from backend.app.models.entities import ExperimentRecord
from backend.app.services.architecture_service import ArchitectureService
from backend.app.services.dataset_service import DatasetService
from neuralforge_core import (
    Sequential, CheckpointManager, Tensor,
    accuracy, confusion_matrix, classification_report,
    mean_absolute_error, root_mean_squared_error, r2_score
)


class InspectorService:
    @staticmethod
    def inspect_experiment(experiment: ExperimentRecord, sample_size: int = 50) -> Dict[str, Any]:
        """Perform comprehensive inspection on weights, activations, and test predictions."""
        if not experiment.checkpoint_path or not Path(experiment.checkpoint_path).exists():
            raise FileNotFoundError("Model checkpoint not found for this experiment")

        # 1. Build and restore model
        arch_cfg = json.loads(experiment.architecture.config_json)
        layer_specs = arch_cfg.get("compiled_layers", [])
        model: Sequential = ArchitectureService.build_core_model(layer_specs)

        CheckpointManager.load(experiment.checkpoint_path, model=model)
        model.eval()

        # 2. Extract weight & bias statistics and distributions
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_data = {
                "index": i,
                "type": layer.__class__.__name__,
                "trainable": len(layer.parameters()) > 0,
                "param_count": sum(p.size for p in layer.parameters()),
                "parameters": {}
            }

            for p_name, param in layer.named_parameters():
                arr = param.data
                hist, bin_edges = np.histogram(arr, bins=10)
                layer_data["parameters"][p_name] = {
                    "shape": list(arr.shape),
                    "mean": round(float(np.mean(arr)), 5),
                    "std": round(float(np.std(arr)), 5),
                    "min": round(float(np.min(arr)), 5),
                    "max": round(float(np.max(arr)), 5),
                    "histogram": {
                        "counts": hist.tolist(),
                        "bin_edges": [round(float(b), 4) for b in bin_edges]
                    }
                }
            layers_info.append(layer_data)

        # 3. Load test dataset for real inference
        train_loader, val_loader, test_loader, meta = DatasetService.prepare_data_loaders(
            dataset=experiment.dataset,
            batch_size=sample_size,
            seed=42
        )
        eval_loader = test_loader or val_loader or train_loader

        # 4. Capture intermediate layer activations from real inference
        activations_info = []
        sample_batch_x, sample_batch_y = next(iter(eval_loader))

        curr_input = sample_batch_x
        for i, layer in enumerate(model.layers):
            curr_output = layer(curr_input)
            act_arr = curr_output.data

            hist, bin_edges = np.histogram(act_arr, bins=10)
            sparsity = float(np.mean(act_arr == 0.0)) * 100.0

            activations_info.append({
                "layer_index": i,
                "layer_type": layer.__class__.__name__,
                "output_shape": list(act_arr.shape),
                "mean": round(float(np.mean(act_arr)), 5),
                "std": round(float(np.std(act_arr)), 5),
                "min": round(float(np.min(act_arr)), 5),
                "max": round(float(np.max(act_arr)), 5),
                "sparsity_percentage": round(sparsity, 2),
                "histogram": {
                    "counts": hist.tolist(),
                    "bin_edges": [round(float(b), 4) for b in bin_edges]
                }
            })
            curr_input = curr_output

        # 5. Compute full evaluation metrics on eval set
        all_preds = []
        all_trues = []
        for bx, by in eval_loader:
            preds = model(bx)
            all_preds.append(preds.data)
            all_trues.append(by)

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)

        task_type = experiment.dataset.task_type
        eval_report = {}

        if task_type == "classification":
            num_classes = meta["output_dim"]
            eval_report = classification_report(y_pred, y_true, num_classes=num_classes)
            eval_report["target_classes"] = meta["target_info"].get("classes", [])
        else:
            eval_report = {
                "mae": round(mean_absolute_error(y_pred, y_true), 4),
                "rmse": round(root_mean_squared_error(y_pred, y_true), 4),
                "r2_score": round(r2_score(y_pred, y_true), 4)
            }

        return {
            "experiment_id": experiment.id,
            "architecture_name": experiment.architecture.name,
            "dataset_name": experiment.dataset.name,
            "task_type": task_type,
            "layers": layers_info,
            "activations": activations_info,
            "evaluation": eval_report
        }
