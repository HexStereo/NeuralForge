"""Datasets API router."""

import os
import uuid
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.database import get_db
from backend.app.models.entities import DatasetRecord, Project
from backend.app.schemas.dataset import DatasetResponse, DatasetConfigUpdate
from backend.app.services.dataset_service import DatasetService
from backend.app.services.sample_datasets import generate_sample_dataset

router = APIRouter(prefix="/datasets", tags=["Datasets"])


def _to_dataset_response(record: DatasetRecord) -> DatasetResponse:
    features = json.loads(record.feature_columns) if record.feature_columns else []
    split_cfg = json.loads(record.split_config) if record.split_config else {}
    summary = json.loads(record.summary_stats) if record.summary_stats else {}

    # Get preview rows if file exists
    preview_rows = []
    if os.path.exists(record.file_path):
        try:
            insp = DatasetService.inspect_csv(record.file_path, preview_count=10)
            preview_rows = insp.get("preview_rows", [])
        except Exception:
            pass

    return DatasetResponse(
        id=record.id,
        project_id=record.project_id,
        name=record.name,
        filename=record.filename,
        row_count=record.row_count,
        column_count=record.column_count,
        task_type=record.task_type,
        feature_columns=features,
        target_column=record.target_column,
        split_config=split_cfg,
        summary_stats=summary,
        preview_rows=preview_rows,
        created_at=record.created_at
    )


@router.get("", response_model=List[DatasetResponse])
def list_datasets(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """List datasets, optionally filtered by project_id."""
    query = db.query(DatasetRecord)
    if project_id:
        query = query.filter(DatasetRecord.project_id == project_id)
    records = query.order_by(DatasetRecord.created_at.desc()).all()
    return [_to_dataset_response(r) for r in records]


@router.post("/upload", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a CSV dataset and auto-profile columns."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dest_filename = f"{uuid.uuid4()}_{filename}"
    save_path = settings.DATA_DIR / dest_filename

    # Stream file to disk while validating size
    total_bytes = 0
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    with open(save_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024): # 1MB chunks
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                buffer.close()
                if save_path.exists():
                    save_path.unlink()
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {settings.MAX_UPLOAD_SIZE_MB}MB size limit"
                )
            buffer.write(chunk)

    try:
        inspection = DatasetService.inspect_csv(str(save_path))
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    cols = [c["name"] for c in inspection["columns"]]
    target_candidate = cols[-1] if cols else ""
    feature_candidates = cols[:-1] if len(cols) > 1 else cols

    # Detect if target is likely classification or regression
    task_type = "classification"
    if inspection["columns"]:
        target_info = next((c for c in inspection["columns"] if c["name"] == target_candidate), None)
        if target_info and target_info.get("dtype") in ["float64", "float32"] and target_info.get("unique_count", 0) > 20:
            task_type = "regression"

    record = DatasetRecord(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name=name.strip(),
        filename=filename,
        file_path=str(save_path),
        row_count=inspection["row_count"],
        column_count=inspection["column_count"],
        task_type=task_type,
        feature_columns=json.dumps(feature_candidates),
        target_column=target_candidate,
        split_config=json.dumps({"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42}),
        summary_stats=json.dumps(inspection)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_dataset_response(record)


@router.post("/sample", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
def create_sample_dataset_endpoint(project_id: str = Form(...), db: Session = Depends(get_db)):
    """Generate and register the built-in sensor telemetry sample dataset."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    sample_path = generate_sample_dataset()
    inspection = DatasetService.inspect_csv(str(sample_path))

    cols = [c["name"] for c in inspection["columns"]]
    target = "fault_condition"
    features = [c for c in cols if c != target]

    record = DatasetRecord(
        id=str(uuid.uuid4()),
        project_id=project_id,
        name="Sample Sensor Telemetry",
        filename="sample_sensor_faults.csv",
        file_path=str(sample_path),
        row_count=inspection["row_count"],
        column_count=inspection["column_count"],
        task_type="classification",
        feature_columns=json.dumps(features),
        target_column=target,
        split_config=json.dumps({"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_seed": 42}),
        summary_stats=json.dumps(inspection)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_dataset_response(record)


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Retrieve dataset details by ID."""
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return _to_dataset_response(record)


@router.put("/{dataset_id}/config", response_model=DatasetResponse)
def update_dataset_config(dataset_id: str, config: DatasetConfigUpdate, db: Session = Depends(get_db)):
    """Update dataset feature/target columns, task type, and split config."""
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    if config.name is not None:
        record.name = config.name.strip()
    if config.task_type is not None:
        record.task_type = config.task_type
    if config.feature_columns is not None:
        record.feature_columns = json.dumps(config.feature_columns)
    if config.target_column is not None:
        record.target_column = config.target_column
    if config.split_config is not None:
        record.split_config = json.dumps(config.split_config.model_dump())

    db.commit()
    db.refresh(record)
    return _to_dataset_response(record)


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset record and its file."""
    record = db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")

    if os.path.exists(record.file_path) and "sample_sensor_faults.csv" not in record.file_path:
        try:
            os.remove(record.file_path)
        except Exception:
            pass

    db.delete(record)
    db.commit()
    return None
