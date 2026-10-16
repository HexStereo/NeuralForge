"""Pydantic schemas for neural network architectures."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class LayerParam(BaseModel):
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    activation: Optional[str] = None # relu, sigmoid, tanh, softmax, none
    drop_rate: Optional[float] = None
    bias: bool = True

class ArchitectureNode(BaseModel):
    id: str
    type: str # input, dense, relu, sigmoid, tanh, softmax, flatten, dropout, output
    data: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[Dict[str, float]] = None

class ArchitectureEdge(BaseModel):
    id: str
    source: str
    target: str

class ArchitectureConfig(BaseModel):
    input_dim: int = Field(..., gt=0)
    output_dim: int = Field(..., gt=0)
    nodes: List[ArchitectureNode] = []
    edges: List[ArchitectureEdge] = []
    layers: Optional[List[Dict[str, Any]]] = None # Sequential layer representations

class ArchitectureCreate(BaseModel):
    project_id: str
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = ""
    config: ArchitectureConfig

class LayerSummary(BaseModel):
    index: int
    name: str
    type: str
    input_shape: List[int]
    output_shape: List[int]
    param_count: int

class ArchitectureValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    total_parameters: int = 0
    layer_summaries: List[LayerSummary] = []

class ArchitectureResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str
    config: Dict[str, Any]
    parameter_count: int
    is_valid: bool
    validation_errors: List[str] = []
    created_at: datetime

    class Config:
        from_attributes = True
