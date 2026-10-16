"""Optimizers for NeuralForge."""

from typing import List, Dict, Any, Tuple
import numpy as np
from neuralforge_core.module import Parameter


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: List[Parameter], defaults: Dict[str, Any]):
        self.params = list(params)
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self):
        """Zero out the gradients of all monitored parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self):
        """Perform a single optimization step."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Serialize optimizer state for checkpoint resumption."""
        state_repr = {}
        for i, p in enumerate(self.params):
            if id(p) in self.state:
                p_state = {}
                for k, v in self.state[id(p)].items():
                    if isinstance(v, np.ndarray):
                        p_state[k] = v.copy()
                    else:
                        p_state[k] = v
                state_repr[i] = p_state
        return {
            "state": state_repr,
            "defaults": self.defaults
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore optimizer state from serialized dictionary."""
        self.defaults.update(state_dict.get("defaults", {}))
        saved_state = state_dict.get("state", {})
        for i, p in enumerate(self.params):
            if i in saved_state or str(i) in saved_state:
                key = i if i in saved_state else str(i)
                p_state = {}
                for k, v in saved_state[key].items():
                    if isinstance(v, np.ndarray):
                        p_state[k] = np.array(v, dtype=p.dtype)
                    else:
                        p_state[k] = v
                self.state[id(p)] = p_state


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with momentum and weight decay."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = {"lr": lr, "momentum": momentum, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        lr = self.defaults["lr"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if weight_decay != 0.0:
                grad = grad + weight_decay * p.data

            if momentum != 0.0:
                p_id = id(p)
                if p_id not in self.state:
                    self.state[p_id] = {"velocity": np.zeros_like(p.data)}
                velocity = self.state[p_id]["velocity"]
                velocity = momentum * velocity + grad
                self.state[p_id]["velocity"] = velocity
                step_direction = velocity
            else:
                step_direction = grad

            p.data = p.data - lr * step_direction


class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer with bias correction."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad
            if weight_decay != 0.0:
                grad = grad + weight_decay * p.data

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    "step": 0,
                    "exp_avg": np.zeros_like(p.data),
                    "exp_avg_sq": np.zeros_like(p.data)
                }

            p_state = self.state[p_id]
            p_state["step"] += 1
            step = p_state["step"]

            m = p_state["exp_avg"]
            v = p_state["exp_avg_sq"]

            # Update biased first moment estimate
            m = beta1 * m + (1.0 - beta1) * grad
            # Update biased second raw moment estimate
            v = beta2 * v + (1.0 - beta2) * (grad ** 2)

            p_state["exp_avg"] = m
            p_state["exp_avg_sq"] = v

            # Compute bias-corrected first and second moment estimates
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)

            # Apply update
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            p.data = p.data - update
