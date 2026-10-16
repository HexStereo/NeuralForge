from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "datasets"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

class Settings:
    PROJECT_NAME: str = "NeuralForge"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api"
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    ARTIFACTS_DIR: Path = ARTIFACTS_DIR
    CHECKPOINTS_DIR: Path = CHECKPOINTS_DIR
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'neuralforge.db'}"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_CSV_EXTENSIONS: set[str] = {".csv"}
    DEFAULT_RANDOM_SEED: int = 42

settings = Settings()
