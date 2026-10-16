"""Secure Checkpoint Manager saving weights and optimizer state without pickle."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from neuralforge_core.module import Module
from neuralforge_core.optimizers import Optimizer


class CheckpointManager:
    """Manages saving and loading model checkpoints using secure np.savez format."""

    @staticmethod
    def save(
        filepath: str,
        model: Module,
        optimizer: Optional[Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save model parameters and optimizer states to an NPZ archive."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays_to_save: Dict[str, np.ndarray] = {}

        # 1. Model weights
        model_state = model.state_dict()
        for name, arr in model_state.items():
            arrays_to_save[f"model_param__{name}"] = arr

        # 2. Optimizer state
        opt_meta = {}
        if optimizer is not None:
            opt_state = optimizer.state_dict()
            opt_meta["defaults"] = opt_state.get("defaults", {})
            for param_idx, p_data in opt_state.get("state", {}).items():
                for sub_k, sub_v in p_data.items():
                    if isinstance(sub_v, np.ndarray):
                        arrays_to_save[f"opt_state__{param_idx}__{sub_k}"] = sub_v
                    else:
                        opt_meta[f"opt_scalar__{param_idx}__{sub_k}"] = sub_v

        # 3. Metadata
        header = {
            "epoch": epoch,
            "metrics": metrics or {},
            "metadata": metadata or {},
            "optimizer_meta": opt_meta,
            "param_keys": list(model_state.keys())
        }
        header_json = json.dumps(header)
        arrays_to_save["__checkpoint_header__"] = np.array(header_json)

        np.savez_compressed(str(path), **arrays_to_save)
        return str(path)

    @staticmethod
    def load(
        filepath: str,
        model: Module,
        optimizer: Optional[Optimizer] = None
    ) -> Dict[str, Any]:
        """Restore model parameters and optimizer state from an NPZ archive."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        # allow_pickle=False enforces strict security against arbitrary code execution
        with np.load(str(path), allow_pickle=False) as data:
            if "__checkpoint_header__" not in data:
                raise ValueError("Corrupt checkpoint archive: missing header")

            header_str = str(data["__checkpoint_header__"])
            header = json.loads(header_str)

            # Restore model weights
            model_state = {}
            for key in header["param_keys"]:
                arr_key = f"model_param__{key}"
                if arr_key in data:
                    model_state[key] = data[arr_key]
            model.load_state_dict(model_state)

            # Restore optimizer if present
            if optimizer is not None and "optimizer_meta" in header:
                opt_meta = header["optimizer_meta"]
                saved_state: Dict[str, Dict[str, Any]] = {}

                # Find arrays for optimizer
                for k in data.files:
                    if k.startswith("opt_state__"):
                        parts = k.split("__")
                        param_idx = parts[1]
                        sub_k = parts[2]
                        if param_idx not in saved_state:
                            saved_state[param_idx] = {}
                        saved_state[param_idx][sub_k] = data[k]

                # Merge scalar attributes
                for k, v in opt_meta.items():
                    if k.startswith("opt_scalar__"):
                        parts = k.split("__")
                        param_idx = parts[1]
                        sub_k = parts[2]
                        if param_idx not in saved_state:
                            saved_state[param_idx] = {}
                        saved_state[param_idx][sub_k] = v

                optimizer.load_state_dict({
                    "state": saved_state,
                    "defaults": opt_meta.get("defaults", {})
                })

            return header
