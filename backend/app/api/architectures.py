"""Architectures API router."""

import uuid
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.entities import ArchitectureRecord, Project
from backend.app.schemas.architecture import (
    ArchitectureCreate, ArchitectureResponse, ArchitectureConfig,
    ArchitectureValidationResult
)
from backend.app.services.architecture_service import ArchitectureService

router = APIRouter(prefix="/architectures", tags=["Architectures"])


def _to_arch_response(record: ArchitectureRecord) -> ArchitectureResponse:
    config = json.loads(record.config_json) if record.config_json else {}
    errors = json.loads(record.validation_errors) if record.validation_errors else []
    return ArchitectureResponse(
        id=record.id,
        project_id=record.project_id,
        name=record.name,
        description=record.description or "",
        config=config,
        parameter_count=record.parameter_count,
        is_valid=record.is_valid,
        validation_errors=errors,
        created_at=record.created_at
    )


@router.post("/validate", response_model=ArchitectureValidationResult)
def validate_architecture(config: ArchitectureConfig):
    """Validate graph structure, dimensions, and calculate parameter counts."""
    result, _ = ArchitectureService.validate_and_compile(config)
    return result


@router.get("", response_model=List[ArchitectureResponse])
def list_architectures(project_id: Optional[str] = None, db: Session = Depends(get_db)):
    """List architectures, optionally filtered by project_id."""
    query = db.query(ArchitectureRecord)
    if project_id:
        query = query.filter(ArchitectureRecord.project_id == project_id)
    records = query.order_by(ArchitectureRecord.created_at.desc()).all()
    return [_to_arch_response(r) for r in records]


@router.post("", response_model=ArchitectureResponse, status_code=status.HTTP_201_CREATED)
def create_architecture(data: ArchitectureCreate, db: Session = Depends(get_db)):
    """Create and persist a validated neural network architecture."""
    project = db.query(Project).filter(Project.id == data.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    val_res, layer_specs = ArchitectureService.validate_and_compile(data.config)

    config_dict = data.config.model_dump()
    config_dict["compiled_layers"] = layer_specs

    record = ArchitectureRecord(
        id=str(uuid.uuid4()),
        project_id=data.project_id,
        name=data.name.strip(),
        description=data.description.strip() if data.description else "",
        config_json=json.dumps(config_dict),
        parameter_count=val_res.total_parameters,
        is_valid=val_res.is_valid,
        validation_errors=json.dumps(val_res.errors)
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return _to_arch_response(record)


@router.get("/{arch_id}", response_model=ArchitectureResponse)
def get_architecture(arch_id: str, db: Session = Depends(get_db)):
    """Retrieve an architecture by ID."""
    record = db.query(ArchitectureRecord).filter(ArchitectureRecord.id == arch_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Architecture {arch_id} not found")
    return _to_arch_response(record)


@router.delete("/{arch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_architecture(arch_id: str, db: Session = Depends(get_db)):
    """Delete an architecture."""
    record = db.query(ArchitectureRecord).filter(ArchitectureRecord.id == arch_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Architecture {arch_id} not found")
    db.delete(record)
    db.commit()
    return None
