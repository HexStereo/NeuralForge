"""SQLAlchemy ORM models for NeuralForge."""

from datetime import datetime, timezone
import json
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.app.models.database import Base

def utcnow():
    return datetime.now(timezone.utc)

class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    datasets = relationship("DatasetRecord", back_populates="project", cascade="all, delete-orphan")
    architectures = relationship("ArchitectureRecord", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("ExperimentRecord", back_populates="project", cascade="all, delete-orphan")

class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, index=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    task_type = Column(String(50), default="classification") # classification | regression
    feature_columns = Column(Text, default="[]") # JSON list
    target_column = Column(String(255), default="")
    split_config = Column(Text, default="{}") # JSON dict with train/val/test ratios & seed
    summary_stats = Column(Text, default="{}") # JSON summary statistics
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="datasets")
    experiments = relationship("ExperimentRecord", back_populates="dataset")

class ArchitectureRecord(Base):
    __tablename__ = "architectures"

    id = Column(String(36), primary_key=True, index=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    config_json = Column(Text, nullable=False) # JSON graph or sequential spec
    parameter_count = Column(Integer, default=0)
    is_valid = Column(Boolean, default=True)
    validation_errors = Column(Text, default="[]") # JSON list
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="architectures")
    experiments = relationship("ExperimentRecord", back_populates="architecture")

class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, index=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    architecture_id = Column(String(36), ForeignKey("architectures.id"), nullable=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    status = Column(String(50), default="queued") # queued, running, completed, cancelled, failed
    hyperparameters_json = Column(Text, default="{}") # epochs, batch_size, lr, optimizer, loss
    final_metrics_json = Column(Text, default="{}") # loss, val_loss, accuracy, etc.
    history_json = Column(Text, default="[]") # list of epoch/step metric records
    checkpoint_path = Column(Text, default="")
    error_message = Column(Text, default="")
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="experiments")
    architecture = relationship("ArchitectureRecord", back_populates="experiments")
    dataset = relationship("DatasetRecord", back_populates="experiments")
    checkpoints = relationship("CheckpointRecord", back_populates="experiment", cascade="all, delete-orphan")

class CheckpointRecord(Base):
    __tablename__ = "checkpoints"

    id = Column(String(36), primary_key=True, index=True)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    step = Column(Integer, default=0)
    train_loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)
    metrics_json = Column(Text, default="{}")
    file_path = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    experiment = relationship("ExperimentRecord", back_populates="checkpoints")
