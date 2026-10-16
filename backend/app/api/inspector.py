"""Model Inspector API router."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.models.database import get_db
from backend.app.models.entities import ExperimentRecord
from backend.app.services.inspector_service import InspectorService

router = APIRouter(prefix="/inspector", tags=["Model Inspector"])


@router.get("/{experiment_id}")
def inspect_model(
    experiment_id: str,
    sample_size: int = Query(50, ge=10, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve weights, activation statistics, and confusion matrix for an experiment."""
    exp = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    if exp.status != "completed":
        raise HTTPException(status_code=400, detail=f"Experiment must be completed before inspection (status: {exp.status})")

    try:
        report = InspectorService.inspect_experiment(exp, sample_size=sample_size)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")
