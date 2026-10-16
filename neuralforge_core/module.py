"""Base Module and Parameter classes for neural network layers."""

from typing import Dict, List, Tuple, Iterator, Optional, Any
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor


class Parameter(Tensor):
    """A trainable neural network parameter."""

    def __init__(self, data, dtype=np.float64):
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """Base class for all neural network modules and layers."""

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_modules"):
                super().__setattr__("_modules", {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        """Return a list of all parameters in this module and submodules."""
        params: List[Parameter] = list(self._parameters.values())
        for submodule in self._modules.values():
            params.extend(submodule.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> List[Tuple[str, Parameter]]:
        """Return a list of (name, parameter) tuples."""
        named_params: List[Tuple[str, Parameter]] = []
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            named_params.append((full_name, param))
        for mod_name, submodule in self._modules.items():
            sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            named_params.extend(submodule.named_parameters(prefix=sub_prefix))
        return named_params

    def zero_grad(self):
        """Zero the gradients of all parameters."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True):
        """Set module and submodules into training mode."""
        self.training = mode
        for submodule in self._modules.values():
            submodule.train(mode)
        return self

    def eval(self):
        """Set module and submodules into evaluation mode."""
        return self.train(False)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Export parameters as a dictionary of numpy arrays."""
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        """Load parameters from a dictionary of numpy arrays."""
        current_params = dict(self.named_parameters())
        for name, arr in state.items():
            if name in current_params:
                current_params[name].data = np.array(arr, dtype=current_params[name].dtype)
            else:
                raise KeyError(f"Parameter '{name}' not found in model state")
