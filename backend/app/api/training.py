"""Training API and WebSocket telemetry streaming."""

import uuid
import json
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.entities import ExperimentRecord, Project, DatasetRecord, ArchitectureRecord
from backend.app.schemas.experiment import TrainingStartRequest, ExperimentResponse
from backend.app.workers.training_worker import TrainingManager

router = APIRouter(tags=["Training"])


@router.post("/training/start", status_code=status.HTTP_201_CREATED)
async def start_training(data: TrainingStartRequest, db: Session = Depends(get_db)):
    """Start an asynchronous neural network training run."""
    project = db.query(Project).filter(Project.id == data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = db.query(DatasetRecord).filter(DatasetRecord.id == data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    architecture = db.query(ArchitectureRecord).filter(ArchitectureRecord.id == data.architecture_id).first()
    if not architecture:
        raise HTTPException(status_code=404, detail="Architecture not found")

    if not architecture.is_valid:
        raise HTTPException(status_code=400, detail="Cannot train invalid architecture")

    experiment_id = str(uuid.uuid4())
    record = ExperimentRecord(
        id=experiment_id,
        project_id=data.project_id,
        name=data.name.strip(),
        architecture_id=data.architecture_id,
        dataset_id=data.dataset_id,
        status="queued",
        hyperparameters_json=json.dumps(data.hyperparameters.model_dump())
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    # Launch in background worker thread
    loop = asyncio.get_running_loop()
    TrainingManager.get_instance().start_training_job(experiment_id, loop)

    return {
        "experiment_id": experiment_id,
        "status": "queued",
        "message": "Training job queued successfully"
    }


@router.post("/training/{experiment_id}/cancel")
def cancel_training(experiment_id: str, db: Session = Depends(get_db)):
    """Cancel an active training job."""
    exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    cancelled = TrainingManager.get_instance().cancel_job(experiment_id)
    if not cancelled and exp.status == "running":
        exp.status = "cancelled"
        db.commit()

    return {"experiment_id": experiment_id, "status": "cancelling"}


@router.get("/training/{experiment_id}/status")
def get_training_status(experiment_id: str, db: Session = Depends(get_db)):
    """Get status and current metric progress of a training run."""
    exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    final_metrics = json.loads(exp.final_metrics_json) if exp.final_metrics_json else {}
    history = json.loads(exp.history_json) if exp.history_json else []

    return {
        "id": exp.id,
        "status": exp.status,
        "error_message": exp.error_message,
        "final_metrics": final_metrics,
        "epochs_completed": len(history),
        "history": history
    }


@router.websocket("/ws/training/{experiment_id}")
async def websocket_training_endpoint(websocket: WebSocket, experiment_id: str):
    """Live WebSocket telemetry stream for a training run."""
    await websocket.accept()

    queue: asyncio.Queue = asyncio.Queue()
    mgr = TrainingManager.get_instance()
    mgr.register_subscriber(experiment_id, queue)

    try:
        # Send initial connected message
        await websocket.send_json({
            "type": "connected",
            "experiment_id": experiment_id,
            "message": "Subscribed to training telemetry stream"
        })

        while True:
            # Wait for next metric message
            message = await queue.get()
            await websocket.send_json(message)
            if message.get("type") == "status" and message.get("status") in ["completed", "failed", "cancelled"]:
                # Job has terminated
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        mgr.unregister_subscriber(experiment_id, queue)
