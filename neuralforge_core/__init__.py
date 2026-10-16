"""NeuralForge Core - Deep Learning Computation & Training Engine."""

from neuralforge_core.tensor import Tensor, to_tensor
from neuralforge_core.module import Module, Parameter
from neuralforge_core.layers import (
    Dense,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Flatten,
    Dropout,
    Sequential,
)
from neuralforge_core.losses import MSELoss, CrossEntropyLoss
from neuralforge_core.optimizers import Optimizer, SGD, Adam
from neuralforge_core.data import Dataset, ArrayDataset, DataLoader
from neuralforge_core.trainer import Trainer
from neuralforge_core.checkpoint import CheckpointManager
from neuralforge_core.metrics import (
    accuracy,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "to_tensor",
    "Module",
    "Parameter",
    "Dense",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Flatten",
    "Dropout",
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "Optimizer",
    "SGD",
    "Adam",
    "Dataset",
    "ArrayDataset",
    "DataLoader",
    "Trainer",
    "CheckpointManager",
    "accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "confusion_matrix",
    "classification_report",
]
