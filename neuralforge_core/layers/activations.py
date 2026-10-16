"""Activation layers and functions."""

from typing import Optional, Tuple
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor
from neuralforge_core.module import Module
from neuralforge_core.autograd import Function


class SoftmaxOp(Function):
    """Numerically stable Softmax autograd operation."""

    def __init__(self, tensor: Tensor, axis: int = -1):
        super().__init__(tensor)
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        shifted_x = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted_x)
        probs = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        self.save_for_backward(probs)
        return probs

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        probs, = self.saved_tensors
        # Vectorized softmax vector-Jacobian product:
        # dX = S * (dOut - sum(dOut * S, axis=-1, keepdims=True))
        sum_grad_p = np.sum(grad_output * probs, axis=self.axis, keepdims=True)
        dx = probs * (grad_output - sum_grad_p)
        return (dx,)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        op = SoftmaxOp(x, axis=self.axis)
        out_data = op.forward(x.data)
        return Tensor(out_data, requires_grad=x.requires_grad, creator=op if x.requires_grad else None)

    def __repr__(self) -> str:
        return f"Softmax(axis={self.axis})"
