"""Loss functions for regression and classification."""

from typing import Optional, Tuple, Union
import numpy as np
from neuralforge_core.tensor import Tensor, to_tensor
from neuralforge_core.module import Module
from neuralforge_core.autograd import Function


class CrossEntropyOp(Function):
    """Numerically stable Fused Softmax + Cross Entropy with log-sum-exp."""

    def __init__(self, logits: Tensor, targets: np.ndarray, reduction: str = "mean"):
        super().__init__(logits)
        self.reduction = reduction
        # targets can be (N,) integer class indices or (N, C) one-hot / probabilities
        self.targets = targets

    def forward(self, logits: np.ndarray) -> np.ndarray:
        # Subtract max for log-sum-exp stability
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
        log_sum_exp = max_logits + np.log(sum_exp)

        # log_probs = logits - log_sum_exp
        log_probs = shifted - np.log(sum_exp)
        probs = exp_shifted / sum_exp

        self.save_for_backward(probs)

        N = logits.shape[0]
        C = logits.shape[1]

        if self.targets.ndim == 1: # Integer class labels
            loss_per_sample = -log_probs[np.arange(N), self.targets.astype(int)]
        else: # Probability distribution / one-hot
            loss_per_sample = -np.sum(self.targets * log_probs, axis=-1)

        if self.reduction == "mean":
            return np.array(np.mean(loss_per_sample), dtype=logits.dtype)
        elif self.reduction == "sum":
            return np.array(np.sum(loss_per_sample), dtype=logits.dtype)
        else:
            return loss_per_sample

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray]]:
        if not self.needs_input_grad[0]:
            return (None,)
        probs, = self.saved_tensors
        N = probs.shape[0]
        C = probs.shape[1]

        if self.targets.ndim == 1:
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), self.targets.astype(int)] = 1.0
            grad_logits = probs - one_hot
        else:
            grad_logits = probs - self.targets

        if self.reduction == "mean":
            grad_logits = grad_logits / N

        return (grad_output * grad_logits,)


class CrossEntropyLoss(Module):
    """Numerically stable Cross-Entropy loss for multiclass or binary classification."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Union[Tensor, np.ndarray]) -> Tensor:
        target_arr = targets.data if isinstance(targets, Tensor) else np.array(targets)
        op = CrossEntropyOp(logits, target_arr, reduction=self.reduction)
        out_data = op.forward(logits.data)
        return Tensor(out_data, requires_grad=logits.requires_grad, creator=op if logits.requires_grad else None)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class MSELoss(Module):
    """Mean Squared Error loss for regression."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        target_t = to_tensor(target)
        diff = pred - target_t
        squared = diff * diff
        if self.reduction == "mean":
            return squared.mean()
        elif self.reduction == "sum":
            return squared.sum()
        else:
            return squared

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"
