"""Background training worker and WebSocket streaming manager."""

import os
import time
import json
import asyncio
import threading
from typing import Dict, Set, Any, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.database import SessionLocal
from backend.app.models.entities import ExperimentRecord, CheckpointRecord, DatasetRecord, ArchitectureRecord
from backend.app.services.dataset_service import DatasetService
from backend.app.services.architecture_service import ArchitectureService
from neuralforge_core import (
    Sequential, MSELoss, CrossEntropyLoss, SGD, Adam, Trainer, CheckpointManager
)


class TrainingJob:
    def __init__(self, experiment_id: str, loop: asyncio.AbstractEventLoop):
        self.experiment_id = experiment_id
        self.loop = loop
        self.cancel_event = threading.Event()
        self.status = "queued"
        self.subscribers: Set[asyncio.Queue] = set()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

    def cancel(self):
        self.cancel_event.set()
        self.status = "cancelling"

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def broadcast(self, message: Dict[str, Any]):
        """Thread-safe broadcast to all connected WebSocket queues."""
        with self.lock:
            for q in list(self.subscribers):
                try:
                    self.loop.call_soon_threadsafe(q.put_nowait, message)
                except Exception:
                    pass


class TrainingManager:
    _instance: Optional["TrainingManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}

    @classmethod
    def get_instance(cls) -> "TrainingManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_subscriber(self, experiment_id: str, queue: asyncio.Queue):
        if experiment_id in self.jobs:
            with self.jobs[experiment_id].lock:
                self.jobs[experiment_id].subscribers.add(queue)

    def unregister_subscriber(self, experiment_id: str, queue: asyncio.Queue):
        if experiment_id in self.jobs:
            with self.jobs[experiment_id].lock:
                self.jobs[experiment_id].subscribers.discard(queue)

    def cancel_job(self, experiment_id: str) -> bool:
        if experiment_id in self.jobs:
            self.jobs[experiment_id].cancel()
            return True
        return False

    def start_training_job(
        self,
        experiment_id: str,
        loop: asyncio.AbstractEventLoop
    ):
        job = TrainingJob(experiment_id, loop)
        self.jobs[experiment_id] = job

        thread = threading.Thread(
            target=self._run_training_thread,
            args=(job,),
            daemon=True,
            name=f"Trainer-{experiment_id[:8]}"
        )
        job.thread = thread
        job.status = "running"
        thread.start()

    def _run_training_thread(self, job: TrainingJob):
        db: Session = SessionLocal()
        experiment_id = job.experiment_id

        try:
            exp: Optional[ExperimentRecord] = db.query(ExperimentRecord).filter(
                ExperimentRecord.id == experiment_id
            ).first()

            if not exp:
                job.status = "failed"
                return

            exp.status = "running"
            exp.started_at = datetime.now(timezone.utc)
            db.commit()

            job.broadcast({
                "type": "status",
                "experiment_id": experiment_id,
                "status": "running",
                "message": "Training initialized."
            })

            # Load dataset and architecture
            dataset: DatasetRecord = exp.dataset
            architecture: ArchitectureRecord = exp.architecture
            hyperparams = json.loads(exp.hyperparameters_json)

            epochs = int(hyperparams.get("epochs", 20))
            batch_size = int(hyperparams.get("batch_size", 32))
            lr = float(hyperparams.get("learning_rate", 0.01))
            opt_name = str(hyperparams.get("optimizer", "adam")).lower()
            opt_params = hyperparams.get("optimizer_params", {})
            loss_name = str(hyperparams.get("loss_function", "cross_entropy")).lower()
            random_seed = int(hyperparams.get("random_seed", 42))

            # 1. Prepare data loaders
            train_loader, val_loader, test_loader, meta = DatasetService.prepare_data_loaders(
                dataset=dataset,
                batch_size=batch_size,
                seed=random_seed
            )

            # 2. Build model
            arch_cfg = json.loads(architecture.config_json)
            layer_specs = arch_cfg.get("compiled_layers", [])
            model: Sequential = ArchitectureService.build_core_model(layer_specs, seed=random_seed)

            # 3. Setup loss
            if loss_name == "mse":
                loss_fn = MSELoss(reduction="mean")
                task_type = "regression"
            else:
                loss_fn = CrossEntropyLoss(reduction="mean")
                task_type = "classification"

            # 4. Setup optimizer
            if opt_name == "sgd":
                momentum = float(opt_params.get("momentum", 0.9))
                weight_decay = float(opt_params.get("weight_decay", 0.0))
                optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            else:
                beta1 = float(opt_params.get("beta1", 0.9))
                beta2 = float(opt_params.get("beta2", 0.999))
                eps = float(opt_params.get("epsilon", 1e-8))
                weight_decay = float(opt_params.get("weight_decay", 0.0))
                optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

            # 5. Define callbacks
            start_time = time.time()
            total_steps = len(train_loader)

            def on_batch(b_info):
                step = b_info["step"]
                epoch = b_info["epoch"]
                # Stream batch progress
                job.broadcast({
                    "type": "batch_metric",
                    "experiment_id": experiment_id,
                    "status": "running",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "step": step,
                    "total_steps": total_steps,
                    "batch_loss": round(b_info["batch_loss"], 4),
                    "batch_metric": round(b_info["batch_metric"], 4) if b_info["batch_metric"] is not None else None,
                    "elapsed_seconds": round(time.time() - start_time, 2)
                })

            def on_epoch(e_info):
                current_epoch = e_info["epoch"]
                elapsed = time.time() - start_time
                epochs_done = current_epoch
                avg_time_per_epoch = elapsed / max(1, epochs_done)
                remaining_epochs = epochs - epochs_done
                eta = round(avg_time_per_epoch * remaining_epochs, 1)

                update_msg = {
                    "type": "metric",
                    "experiment_id": experiment_id,
                    "status": "running",
                    "epoch": current_epoch,
                    "total_epochs": epochs,
                    "train_loss": round(e_info["train_loss"], 4),
                    "train_acc": round(e_info["train_metric"], 4) if task_type == "classification" else None,
                    "val_loss": round(e_info["val_loss"], 4) if e_info["val_loss"] is not None else None,
                    "val_acc": round(e_info["val_metric"], 4) if task_type == "classification" and e_info["val_metric"] is not None else None,
                    "elapsed_seconds": round(elapsed, 2),
                    "eta_seconds": eta
                }
                job.broadcast(update_msg)

            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                task_type=task_type,
                on_batch_end=on_batch,
                on_epoch_end=on_epoch,
                check_cancelled=job.is_cancelled
            )

            # Execute training loop
            history = trainer.fit(epochs=epochs)

            # Check if cancelled
            if job.is_cancelled():
                exp.status = "cancelled"
                job.status = "cancelled"
            else:
                exp.status = "completed"
                job.status = "completed"

            exp.completed_at = datetime.now(timezone.utc)
            exp.history_json = json.dumps(history)

            if history:
                final_record = history[-1]
                exp.final_metrics_json = json.dumps({
                    "final_train_loss": final_record["train_loss"],
                    "final_val_loss": final_record.get("val_loss"),
                    "final_train_metric": final_record["train_metric"],
                    "final_val_metric": final_record.get("val_metric"),
                    "total_duration": round(time.time() - start_time, 2)
                })

            # Save final checkpoint
            ckpt_filename = f"{experiment_id}_final.npz"
            ckpt_path = settings.CHECKPOINTS_DIR / ckpt_filename
            saved_path = CheckpointManager.save(
                filepath=str(ckpt_path),
                model=model,
                optimizer=optimizer,
                epoch=len(history),
                metrics=history[-1] if history else {},
                metadata={"experiment_id": experiment_id, "dataset_id": dataset.id}
            )
            exp.checkpoint_path = saved_path

            # Save checkpoint record
            if history:
                ckpt_rec = CheckpointRecord(
                    id=str(uuid.uuid4()) if False else f"ckpt_{experiment_id[:8]}_{len(history)}",
                    experiment_id=experiment_id,
                    epoch=len(history),
                    step=len(history) * total_steps,
                    train_loss=history[-1]["train_loss"],
                    val_loss=history[-1].get("val_loss"),
                    metrics_json=json.dumps(history[-1]),
                    file_path=saved_path
                )
                db.add(ckpt_rec)

            db.commit()

            job.broadcast({
                "type": "status",
                "experiment_id": experiment_id,
                "status": exp.status,
                "message": f"Training {exp.status}.",
                "final_metrics": json.loads(exp.final_metrics_json) if exp.final_metrics_json else {}
            })

        except Exception as exc:
            db.rollback()
            exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
            if exp:
                exp.status = "failed"
                exp.error_message = str(exc)
                exp.completed_at = datetime.now(timezone.utc)
                db.commit()

            job.status = "failed"
            job.broadcast({
                "type": "error",
                "experiment_id": experiment_id,
                "status": "failed",
                "error": str(exc)
            })

        finally:
            db.close()
