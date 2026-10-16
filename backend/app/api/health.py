"""Health and system status API router."""

import platform
import sys
from fastapi import APIRouter
from backend.app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
def get_health():
    """Return system health status and environment details."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "storage_paths": {
                "datasets": str(settings.DATA_DIR),
                "artifacts": str(settings.ARTIFACTS_DIR),
                "checkpoints": str(settings.CHECKPOINTS_DIR)
            }
        }
    }
