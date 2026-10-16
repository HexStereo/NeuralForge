"""Core Tensor class for NeuralForge computation engine."""

from __future__ import annotations
from typing import Union, Optional, Tuple, Sequence, List, Set
import numpy as np

from neuralforge_core.autograd import (
    Function, Add, Sub, Mul, Div, MatMul, Sum, Mean,
    Reshape, Transpose, Neg, Exp, Log, Maximum
)

ArrayLike = Union[int, float, list, np.ndarray, "Tensor"]


def to_tensor(x: ArrayLike, requires_grad: bool = False, dtype=np.float64) -> "Tensor":
    """Helper to convert array-like values into a Tensor."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, requires_grad=requires_grad, dtype=dtype)


class Tensor:
    """Multidimensional array supporting automatic reverse-mode differentiation."""

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        creator: Optional[Function] = None,
        dtype=np.float64
    ):
        if isinstance(data, Tensor):
            self.data = np.array(data.data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype, copy=False) if data.dtype != dtype else data
        else:
            self.data = np.array(data, dtype=dtype)

        self.requires_grad = requires_grad
        self.creator = creator
        self.grad: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    def zero_grad(self):
        """Reset gradient to None."""
        self.grad = None

    def backward(self, grad: Optional[Union[np.ndarray, "Tensor"]] = None):
        """Compute gradients of this tensor with respect to graph leaves."""
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                self.grad = np.ones_like(self.data, dtype=self.data.dtype)
            else:
                raise RuntimeError("Grad can only be implicitly created for scalar outputs")
        else:
            if isinstance(grad, Tensor):
                self.grad = np.array(grad.data, dtype=self.data.dtype)
            else:
                self.grad = np.array(grad, dtype=self.data.dtype)

        # Build topological ordering of computation graph
        topo: List[Tensor] = []
        visited: Set[int] = set()

        def build_topo(v: Tensor):
            if id(v) not in visited:
                visited.add(id(v))
                if v.creator is not None:
                    for inp in v.creator.inputs:
                        build_topo(inp)
                topo.append(v)

        build_topo(self)

        # Backpropagate in reverse topological order
        for node in reversed(topo):
            if node.creator is not None and node.grad is not None:
                grads = node.creator.backward(node.grad)
                for inp, g in zip(node.creator.inputs, grads):
                    if inp.requires_grad and g is not None:
                        if inp.grad is None:
                            inp.grad = np.array(g, dtype=inp.data.dtype, copy=True)
                        else:
                            inp.grad = inp.grad + g

    # --- Arithmetic Operations ---

    def __add__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = Add(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = Sub(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        return other_t.__sub__(self)

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = Mul(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = Div(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        return other_t.__truediv__(self)

    def __neg__(self) -> "Tensor":
        op = Neg(self)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = MatMul(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    def matmul(self, other: "Tensor") -> "Tensor":
        return self.__matmul__(other)

    # --- Shape and Reduction Operations ---

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        op = Sum(self, axis=axis, keepdims=keepdims)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        op = Mean(self, axis=axis, keepdims=keepdims)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        op = Reshape(self, target_shape)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def transpose(self, *axes: Union[int, Tuple[int, ...]]) -> "Tensor":
        if len(axes) == 0:
            target_axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            target_axes = tuple(axes[0])
        else:
            target_axes = tuple(axes)

        op = Transpose(self, axes=target_axes)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def exp(self) -> "Tensor":
        op = Exp(self)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def log(self) -> "Tensor":
        op = Log(self)
        out_data = op.forward(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=op if self.requires_grad else None)

    def maximum(self, other: ArrayLike) -> "Tensor":
        other_t = to_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        op = Maximum(self, other_t)
        out_data = op.forward(self.data, other_t.data)
        return Tensor(out_data, requires_grad=req_grad, creator=op if req_grad else None)

    # --- Activation functions as methods ---

    def relu(self) -> "Tensor":
        return self.maximum(0.0)

    def sigmoid(self) -> "Tensor":
        # Numerically stable sigmoid: 1 / (1 + exp(-x))
        # Note: can use 1 / (1 + (-self).exp())
        return 1.0 / (1.0 + (-self).exp())

    def tanh(self) -> "Tensor":
        # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) or 2 * sigmoid(2x) - 1
        return 2.0 * (2.0 * self).sigmoid() - 1.0

    def __repr__(self) -> str:
        grad_str = f", grad_fn=<{self.creator.__class__.__name__}>" if self.creator else ""
        req_str = f", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{req_str}{grad_str})"
