"""Dense (Fully Connected) linear layer."""

import math
from typing import Optional
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor
from neuralforge_core.module import Module, Parameter


class Dense(Module):
    """Fully connected linear transformation layer: Y = XW + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_strategy: str = "he",
        seed: Optional[int] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        rng = np.random.default_rng(seed)

        # Weight initialization
        if init_strategy == "he":
            std = math.sqrt(2.0 / in_features)
            w_data = rng.normal(0.0, std, size=(in_features, out_features))
        elif init_strategy == "xavier" or init_strategy == "glorot":
            limit = math.sqrt(6.0 / (in_features + out_features))
            w_data = rng.uniform(-limit, limit, size=(in_features, out_features))
        else: # Standard normal scaled
            w_data = rng.normal(0.0, 0.05, size=(in_features, out_features))

        self.weight = Parameter(w_data)

        if bias:
            b_data = np.zeros(out_features, dtype=np.float64)
            self.bias = Parameter(b_data)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.has_bias and self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return f"Dense(in_features={self.in_features}, out_features={self.out_features}, bias={self.has_bias})"
