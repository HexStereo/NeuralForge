"""Experiments management and comparison API router."""

import os
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.entities import ExperimentRecord, CheckpointRecord
from backend.app.schemas.experiment import ExperimentResponse

router = APIRouter(prefix="/experiments", tags=["Experiments"])


def _to_exp_response(record: ExperimentRecord) -> ExperimentResponse:
    hp = json.loads(record.hyperparameters_json) if record.hyperparameters_json else {}
    fm = json.loads(record.final_metrics_json) if record.final_metrics_json else {}
    hist = json.loads(record.history_json) if record.history_json else []

    return ExperimentResponse(
        id=record.id,
        project_id=record.project_id,
        name=record.name,
        architecture_id=record.architecture_id,
        dataset_id=record.dataset_id,
        status=record.status,
        hyperparameters=hp,
        final_metrics=fm,
        history=hist,
        checkpoint_path=record.checkpoint_path or "",
        error_message=record.error_message or "",
        started_at=record.started_at,
        completed_at=record.completed_at,
        created_at=record.created_at
    )


@router.get("", response_model=List[ExperimentResponse])
def list_experiments(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """List experiments, optionally filtered by project_id."""
    query = db.query(ExperimentRecord)
    if project_id:
        query = query.filter(ExperimentRecord.project_id == project_id)
    records = query.order_by(ExperimentRecord.created_at.desc()).all()
    return [_to_exp_response(r) for r in records]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Retrieve full details of an experiment."""
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return _to_exp_response(record)


@router.get("/{experiment_id}/checkpoints")
def list_checkpoints(experiment_id: str, db: Session = Depends(get_db)):
    """List all checkpoints created for an experiment."""
    ckpts = db.query(CheckpointRecord).filter(
        CheckpointRecord.experiment_id == experiment_id
    ).order_by(CheckpointRecord.epoch.asc()).all()

    return [{
        "id": c.id,
        "epoch": c.epoch,
        "step": c.step,
        "train_loss": c.train_loss,
        "val_loss": c.val_loss,
        "file_path": c.file_path,
        "created_at": c.created_at
    } for c in ckpts]


@router.post("/compare")
def compare_experiments(experiment_ids: List[str], db: Session = Depends(get_db)):
    """Compare multiple experiments side-by-side."""
    records = db.query(ExperimentRecord).filter(ExperimentRecord.id.in_(experiment_ids)).all()
    results = []

    for r in records:
        hp = json.loads(r.hyperparameters_json) if r.hyperparameters_json else {}
        fm = json.loads(r.final_metrics_json) if r.final_metrics_json else {}
        hist = json.loads(r.history_json) if r.history_json else []

        arch_name = r.architecture.name if r.architecture else "Unknown"
        ds_name = r.dataset.name if r.dataset else "Unknown"

        results.append({
            "id": r.id,
            "name": r.name,
            "architecture_name": arch_name,
            "dataset_name": ds_name,
            "status": r.status,
            "hyperparameters": hp,
            "final_metrics": fm,
            "loss_curve": [{"epoch": p.get("epoch"), "train_loss": p.get("train_loss"), "val_loss": p.get("val_loss")} for p in hist],
            "metric_curve": [{"epoch": p.get("epoch"), "train_metric": p.get("train_metric"), "val_metric": p.get("val_metric")} for p in hist],
            "created_at": r.created_at
        })

    return results


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: str, db: Session = Depends(get_db)):
    """Delete an experiment and its saved checkpoint file."""
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    if record.checkpoint_path and os.path.exists(record.checkpoint_path):
        try:
            os.remove(record.checkpoint_path)
        except Exception:
            pass

    db.delete(record)
    db.commit()
    return None
