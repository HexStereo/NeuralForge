"""Pydantic schemas for experiments and training configurations."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Hyperparameters(BaseModel):
    epochs: int = Field(20, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=2048)
    learning_rate: float = Field(0.01, gt=0, le=10.0)
    optimizer: str = Field("adam", pattern="^(adam|sgd)$")
    optimizer_params: Dict[str, Any] = Field(default_factory=lambda: {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "momentum": 0.9})
    loss_function: str = Field("cross_entropy", pattern="^(cross_entropy|mse)$")
    random_seed: int = 42
    checkpoint_frequency: int = 5 # epochs

class TrainingStartRequest(BaseModel):
    project_id: str
    name: str = Field(..., min_length=1, max_length=255)
    architecture_id: str
    dataset_id: str
    hyperparameters: Hyperparameters

class MetricPoint(BaseModel):
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metric: Optional[float] = None # e.g. Accuracy or MAE
    val_metric: Optional[float] = None
    learning_rate: float
    elapsed_time: float
    eta: Optional[float] = None

class LiveMetricUpdate(BaseModel):
    type: str = "metric" # metric | status | log | error
    experiment_id: str
    status: str
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    elapsed_seconds: float
    eta_seconds: Optional[float] = None
    message: Optional[str] = None

class ExperimentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    architecture_id: Optional[str]
    dataset_id: Optional[str]
    status: str
    hyperparameters: Dict[str, Any]
    final_metrics: Dict[str, Any]
    history: List[Dict[str, Any]] = []
    checkpoint_path: str = ""
    error_message: str = ""
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True
