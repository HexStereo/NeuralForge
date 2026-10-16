# ==============================================================================
#      _    _ ______ __   __ _____ _______ ______ _____  ______ ____  
#     | |  | |  ____|\ \ / // ____|__   __|  ____|  __ \|  ____/ __ \ 
#     | |__| | |__    \ V /| (___    | |  | |__  | |__) | |__ | |  | |
#     |  __  |  __|    > <  \___ \   | |  |  __| |  _  /|  __|| |  | |
#     | |  | | |____  / . \ ____) |  | |  | |____| | \ \| |___| |__| |
#     |_|  |_|______|/_/ \_\_____/   |_|  |______|_|  \_\______\____/ 
#
#                         NeuralForge Studio
#                        Created by HEXSTEREO
# ==============================================================================
"""FastAPI application entrypoint for NeuralForge.

Author: hexstereo (https://github.com/HexStereo)
"""


from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.models.database import engine, Base
from backend.app.api import (
    health,
    projects,
    datasets,
    architectures,
    training,
    experiments,
    inspector
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Scientific Deep Learning Training and Research Studio",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Include API routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(projects.router, prefix=settings.API_V1_STR)
app.include_router(datasets.router, prefix=settings.API_V1_STR)
app.include_router(architectures.router, prefix=settings.API_V1_STR)
app.include_router(training.router, prefix=settings.API_V1_STR)
app.include_router(experiments.router, prefix=settings.API_V1_STR)
app.include_router(inspector.router, prefix=settings.API_V1_STR)

# Direct websocket alias for frontend SPA compatibility
@app.websocket("/ws/training/{experiment_id}")
async def direct_ws_training(websocket: training.WebSocket, experiment_id: str):
    await training.websocket_training_endpoint(websocket, experiment_id)

from fastapi import Request

# Serve built frontend single-page application (SPA) if dist exists
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        fav = frontend_dist / "favicon.svg"
        if fav.exists():
            return FileResponse(fav)
        return {"detail": "Not found"}

    @app.get("/", include_in_schema=False)
    async def root_or_spa(request: Request):
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return FileResponse(frontend_dist / "index.html")
        return {
            "message": f"Welcome to {settings.PROJECT_NAME} API",
            "author": "hexstereo",
            "signature": "HEXSTEREO",
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_STR}/health"
        }

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return {"detail": "Not found"}
        target = frontend_dist / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(frontend_dist / "index.html")
else:
    @app.get("/")
    def root():
        return {
            "message": f"Welcome to {settings.PROJECT_NAME} API",
            "author": "hexstereo",
            "signature": "HEXSTEREO",
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_STR}/health"
        }



