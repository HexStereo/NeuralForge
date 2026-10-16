"""Sequential container, Flatten, and Dropout layers."""

from typing import List, Optional, Tuple
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor
from neuralforge_core.module import Module
from neuralforge_core.autograd import Function


class DropoutOp(Function):
    """Autograd operation for Dropout."""

    def __init__(self, tensor: Tensor, mask: np.ndarray, scale: float):
        super().__init__(tensor)
        self.mask = mask
        self.scale = scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * self.mask * self.scale

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output * self.mask * self.scale,)


class Dropout(Module):
    """Inverts dropout rate p during training, identity during eval."""

    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p = p
        self.scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
        self.rng = np.random.default_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask = (self.rng.random(size=x.shape) >= self.p).astype(x.dtype)
        op = DropoutOp(x, mask, self.scale)
        out_data = op.forward(x.data)
        return Tensor(out_data, requires_grad=x.requires_grad, creator=op if x.requires_grad else None)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Flatten(Module):
    """Flattens input dimensions preserving the batch dimension."""

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def __repr__(self) -> str:
        return "Flatten()"


class Sequential(Module):
    """Sequential container passing output of one layer to the next."""

    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = []
        for i, layer in enumerate(layers):
            self.add_module(f"layer_{i}", layer)

    def add_module(self, name: str, module: Module):
        setattr(self, name, module)
        self.layers.append(module)

    def append(self, module: Module):
        self.add_module(f"layer_{len(self.layers)}", module)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def __repr__(self) -> str:
        layer_strs = "\n  ".join([f"({i}): {repr(l)}" for i, l in enumerate(self.layers)])
        return f"Sequential(\n  {layer_strs}\n)"
