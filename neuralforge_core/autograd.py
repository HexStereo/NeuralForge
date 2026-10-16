"""Autograd engine and differentiable Function base for NeuralForge."""

from __future__ import annotations
from typing import List, Tuple, Sequence, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from neuralforge_core.tensor import Tensor


def unbroadcast(grad: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Sum out broadcasted dimensions in grad to match target_shape."""
    target_shape = tuple(target_shape)
    if grad.shape == target_shape:
        return grad

    grad_ndim = grad.ndim
    target_ndim = len(target_shape)

    leading_dims = grad_ndim - target_ndim
    if leading_dims > 0:
        grad = grad.sum(axis=tuple(range(leading_dims)))

    axes_to_sum = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape))
        if t_dim == 1 and g_dim > 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

    return grad.reshape(target_shape)


class Function:
    """Base class for all autograd differentiable operations."""

    def __init__(self, *inputs: "Tensor"):
        self.saved_tensors: Tuple["Tensor", ...] = ()
        self.inputs: Tuple["Tensor", ...] = inputs
        self.needs_input_grad: List[bool] = [t.requires_grad for t in inputs]

    def save_for_backward(self, *tensors: "Tensor"):
        self.saved_tensors = tensors

    def forward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError


class Add(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gx = unbroadcast(grad_output, self.inputs[0].shape) if self.needs_input_grad[0] else None
        gy = unbroadcast(grad_output, self.inputs[1].shape) if self.needs_input_grad[1] else None
        return gx, gy


class Sub(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x - y

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        gx = unbroadcast(grad_output, self.inputs[0].shape) if self.needs_input_grad[0] else None
        gy = unbroadcast(-grad_output, self.inputs[1].shape) if self.needs_input_grad[1] else None
        return gx, gy


class Mul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x * y

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y = self.inputs[0].data, self.inputs[1].data
        gx = unbroadcast(grad_output * y, self.inputs[0].shape) if self.needs_input_grad[0] else None
        gy = unbroadcast(grad_output * x, self.inputs[1].shape) if self.needs_input_grad[1] else None
        return gx, gy


class Div(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x / y

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y = self.inputs[0].data, self.inputs[1].data
        gx = unbroadcast(grad_output / y, self.inputs[0].shape) if self.needs_input_grad[0] else None
        gy = unbroadcast(-grad_output * x / (y ** 2), self.inputs[1].shape) if self.needs_input_grad[1] else None
        return gx, gy


class MatMul(Function):
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.matmul(x, y)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y = self.inputs[0].data, self.inputs[1].data
        # d/dx (X @ Y) = grad @ Y.T
        gx = np.matmul(grad_output, y.T) if self.needs_input_grad[0] else None
        # d/dy (X @ Y) = X.T @ grad
        gy = np.matmul(x.T, grad_output) if self.needs_input_grad[1] else None
        return gx, gy


class Sum(Function):
    def __init__(self, tensor: "Tensor", axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        x_shape = self.inputs[0].shape
        if not self.keepdims and self.axis is not None:
            # Expand dimensions back to broadcast correctly
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            shape_with_dims = list(x_shape)
            for ax in axes:
                norm_ax = ax if ax >= 0 else ax + len(x_shape)
                shape_with_dims[norm_ax] = 1
            grad_expanded = grad_output.reshape(shape_with_dims)
        else:
            grad_expanded = grad_output

        return (np.broadcast_to(grad_expanded, x_shape),)


class Mean(Function):
    def __init__(self, tensor: "Tensor", axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        x_shape = self.inputs[0].shape
        if self.axis is None:
            n = self.inputs[0].data.size
        elif isinstance(self.axis, int):
            n = x_shape[self.axis]
        else:
            n = 1
            for a in self.axis:
                n *= x_shape[a]

        if not self.keepdims and self.axis is not None:
            axes = (self.axis,) if isinstance(self.axis, int) else self.axis
            shape_with_dims = list(x_shape)
            for ax in axes:
                norm_ax = ax if ax >= 0 else ax + len(x_shape)
                shape_with_dims[norm_ax] = 1
            grad_expanded = grad_output.reshape(shape_with_dims)
        else:
            grad_expanded = grad_output

        return (np.broadcast_to(grad_expanded / n, x_shape),)


class Reshape(Function):
    def __init__(self, tensor: "Tensor", new_shape: Tuple[int, ...]):
        super().__init__(tensor)
        self.new_shape = new_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.new_shape)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output.reshape(self.inputs[0].shape),)


class Transpose(Function):
    def __init__(self, tensor: "Tensor", axes: Optional[Tuple[int, ...]] = None):
        super().__init__(tensor)
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.transpose(x, self.axes)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        if self.axes is None:
            return (np.transpose(grad_output),)
        else:
            # Invert permutation
            inv_axes = np.argsort(self.axes)
            return (np.transpose(grad_output, inv_axes),)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        return (-grad_output if self.needs_input_grad[0] else None,)


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.exp(x)
        self.save_for_backward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        out, = self.saved_tensors
        return (grad_output * out,)


class Log(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to prevent log(0)
        x_clipped = np.clip(x, 1e-15, None)
        self.save_for_backward(x_clipped)
        return np.log(x_clipped)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        x_clipped, = self.saved_tensors
        return (grad_output / x_clipped,)


class Maximum(Function):
    """Element-wise maximum: max(x, y)."""
    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.maximum(x, y)

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        x, y = self.inputs[0].data, self.inputs[1].data
        gx = unbroadcast(grad_output * (x >= y), self.inputs[0].shape) if self.needs_input_grad[0] else None
        gy = unbroadcast(grad_output * (y > x), self.inputs[1].shape) if self.needs_input_grad[1] else None
        return gx, gy
