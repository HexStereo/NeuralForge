"""Projects API router."""

import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.entities import Project, DatasetRecord, ArchitectureRecord, ExperimentRecord
from backend.app.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(prefix="/projects", tags=["Projects"])

def _to_project_response(project: Project, db: Session) -> ProjectResponse:
    dataset_count = db.query(DatasetRecord).filter(DatasetRecord.project_id == project.id).count()
    arch_count = db.query(ArchitectureRecord).filter(ArchitectureRecord.project_id == project.id).count()
    exp_count = db.query(ExperimentRecord).filter(ExperimentRecord.project_id == project.id).count()
    
    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description or "",
        created_at=project.created_at,
        updated_at=project.updated_at,
        dataset_count=dataset_count,
        architecture_count=arch_count,
        experiment_count=exp_count
    )

@router.get("", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    """List all projects."""
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    return [_to_project_response(p, db) for p in projects]

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project."""
    new_project = Project(
        id=str(uuid.uuid4()),
        name=data.name.strip(),
        description=data.description.strip() if data.description else ""
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return _to_project_response(new_project, db)

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    """Retrieve a project by ID."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return _to_project_response(project, db)

@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: str, data: ProjectUpdate, db: Session = Depends(get_db)):
    """Update a project."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    if data.name is not None:
        project.name = data.name.strip()
    if data.description is not None:
        project.description = data.description.strip()
    db.commit()
    db.refresh(project)
    return _to_project_response(project, db)

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project and its associated artifacts."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    db.delete(project)
    db.commit()
    return None
