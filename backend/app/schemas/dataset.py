"""Pydantic schemas for datasets."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DatasetSplitConfig(BaseModel):
    train_ratio: float = Field(0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(0.15, ge=0.05, le=0.5)
    test_ratio: float = Field(0.15, ge=0.05, le=0.5)
    random_seed: int = 42
    shuffle: bool = True

class DatasetPreprocessConfig(BaseModel):
    normalize_features: bool = True
    normalization_type: str = "standard" # standard | minmax
    handle_missing: str = "mean" # mean | median | drop
    encode_labels: bool = True

class ColumnStat(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    sample_values: List[Any] = []
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None

class DatasetResponse(BaseModel):
    id: str
    project_id: str
    name: str
    filename: str
    row_count: int
    column_count: int
    task_type: str
    feature_columns: List[str]
    target_column: str
    split_config: Dict[str, Any]
    summary_stats: Dict[str, Any]
    preview_rows: List[Dict[str, Any]] = []
    created_at: datetime

    class Config:
        from_attributes = True

class DatasetConfigUpdate(BaseModel):
    name: Optional[str] = None
    task_type: Optional[str] = None # classification | regression
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    split_config: Optional[DatasetSplitConfig] = None
    preprocess_config: Optional[DatasetPreprocessConfig] = None
