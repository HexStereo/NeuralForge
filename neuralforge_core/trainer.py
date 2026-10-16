"""Trainer loop orchestrating forward, backward, metrics, and callbacks."""

import time
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from neuralforge_core.tensor import Tensor
from neuralforge_core.module import Module
from neuralforge_core.data import DataLoader
from neuralforge_core.optimizers import Optimizer
from neuralforge_core.metrics import accuracy, mean_squared_error, mean_absolute_error


class Trainer:
    """Orchestrates model training, evaluation, metrics tracking, and cancellation."""

    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_type: str = "classification",
        on_batch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
        check_cancelled: Optional[Callable[[], bool]] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.on_batch_end = on_batch_end
        self.on_epoch_end = on_epoch_end
        self.check_cancelled = check_cancelled

        self.history: List[Dict[str, Any]] = []

    def _compute_metric(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.task_type == "classification":
            return accuracy(y_pred, y_true)
        else:
            return mean_absolute_error(y_pred, y_true)

    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_trues = []
        step = 0
        total_steps = len(self.train_loader)
        epoch_start_time = time.time()

        for batch_x, batch_y in self.train_loader:
            if self.check_cancelled and self.check_cancelled():
                break

            self.optimizer.zero_grad()

            # Forward
            preds = self.model(batch_x)
            loss = self.loss_fn(preds, batch_y)

            # Backward
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Track
            loss_val = loss.item()
            total_loss += loss_val
            step += 1

            all_preds.append(preds.data)
            all_trues.append(batch_y)

            if self.on_batch_end:
                current_train_acc = None
                if self.task_type == "classification":
                    current_train_acc = accuracy(preds.data, batch_y)

                self.on_batch_end({
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "step": step,
                    "total_steps": total_steps,
                    "batch_loss": loss_val,
                    "batch_metric": current_train_acc,
                    "elapsed": time.time() - epoch_start_time
                })

        avg_loss = total_loss / max(1, step)
        concat_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        concat_trues = np.concatenate(all_trues, axis=0) if all_trues else np.array([])
        metric_val = self._compute_metric(concat_preds, concat_trues) if len(concat_preds) > 0 else 0.0

        return {"train_loss": avg_loss, "train_metric": metric_val}

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []
        step = 0

        for batch_x, batch_y in data_loader:
            preds = self.model(batch_x)
            loss = self.loss_fn(preds, batch_y)

            total_loss += loss.item()
            step += 1

            all_preds.append(preds.data)
            all_trues.append(batch_y)

        avg_loss = total_loss / max(1, step)
        concat_preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
        concat_trues = np.concatenate(all_trues, axis=0) if all_trues else np.array([])
        metric_val = self._compute_metric(concat_preds, concat_trues) if len(concat_preds) > 0 else 0.0

        return {"val_loss": avg_loss, "val_metric": metric_val}

    def fit(self, epochs: int) -> List[Dict[str, Any]]:
        self.history = []
        train_start = time.time()

        for epoch in range(1, epochs + 1):
            if self.check_cancelled and self.check_cancelled():
                break

            epoch_train_res = self.train_epoch(epoch, epochs)
            if self.check_cancelled and self.check_cancelled():
                break

            val_res = {}
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_res = self.evaluate(self.val_loader)

            elapsed = time.time() - train_start
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(epoch_train_res["train_loss"]),
                "train_metric": float(epoch_train_res["train_metric"]),
                "val_loss": float(val_res.get("val_loss", 0.0)) if val_res else None,
                "val_metric": float(val_res.get("val_metric", 0.0)) if val_res else None,
                "elapsed_seconds": round(elapsed, 2)
            }
            self.history.append(epoch_record)

            if self.on_epoch_end:
                self.on_epoch_end(epoch_record)

        return self.history
