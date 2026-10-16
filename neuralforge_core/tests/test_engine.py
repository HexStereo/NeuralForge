"""Numerical and gradient validation tests for NeuralForge Core."""

import numpy as np
import pytest
from neuralforge_core import (
    Tensor, Dense, ReLU, Sigmoid, Tanh, Softmax,
    MSELoss, CrossEntropyLoss, SGD, Adam, Sequential
)


def numerical_gradient(func, tensor: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Compute finite-difference numerical gradient of a scalar output w.r.t tensor."""
    x = tensor.data
    grad_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        orig_val = x[idx]

        # f(x + eps)
        x[idx] = orig_val + eps
        pos_loss = func().item()

        # f(x - eps)
        x[idx] = orig_val - eps
        neg_loss = func().item()

        # restore
        x[idx] = orig_val

        grad_num[idx] = (pos_loss - neg_loss) / (2.0 * eps)
        it.iternext()

    return grad_num


def check_gradient(analytical_grad: np.ndarray, numerical_grad: np.ndarray, tol: float = 1e-4):
    """Assert relative difference between analytical and numerical gradient is below tolerance."""
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    norm_sum = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad) + 1e-12
    rel_error = diff / norm_sum
    assert rel_error < tol, f"Gradient check failed! Rel error: {rel_error}, diff: {diff}\nAnalytical:\n{analytical_grad}\nNumerical:\n{numerical_grad}"


def test_tensor_basic_ops():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)

    c = (a * b + a).sum()
    c.backward()

    # c = a0*b0 + a1*b1 + a0 + a1
    # dc/da0 = b0 + 1 = 5, dc/da1 = b1 + 1 = 6
    # dc/db0 = a0 = 2, dc/db1 = a1 = 3
    np.testing.assert_allclose(a.grad, [5.0, 6.0])
    np.testing.assert_allclose(b.grad, [2.0, 3.0])


def test_broadcasting_gradient():
    # Matrix (3, 2) + Vector (2,)
    X = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)

    out = (X + b).sum()
    out.backward()

    # X gradient should be all ones of shape (3, 2)
    np.testing.assert_allclose(X.grad, np.ones((3, 2)))
    # b gradient should be sum along axis 0: [3.0, 3.0]
    np.testing.assert_allclose(b.grad, [3.0, 3.0])


def test_matmul_gradient_finite_difference():
    np.random.seed(42)
    X = Tensor(np.random.randn(4, 3), requires_grad=True)
    W = Tensor(np.random.randn(3, 2), requires_grad=True)

    def forward_fn():
        return (X @ W).sum()

    out = forward_fn()
    out.backward()

    num_grad_X = numerical_gradient(forward_fn, X)
    num_grad_W = numerical_gradient(forward_fn, W)

    check_gradient(X.grad, num_grad_X)
    check_gradient(W.grad, num_grad_W)


def test_dense_finite_difference():
    np.random.seed(123)
    layer = Dense(in_features=3, out_features=4, bias=True)
    x = Tensor(np.random.randn(5, 3), requires_grad=True)

    def forward_fn():
        return layer(x).sum()

    out = forward_fn()
    out.backward()

    num_grad_x = numerical_gradient(forward_fn, x)
    num_grad_w = numerical_gradient(forward_fn, layer.weight)
    num_grad_b = numerical_gradient(forward_fn, layer.bias)

    check_gradient(x.grad, num_grad_x)
    check_gradient(layer.weight.grad, num_grad_w)
    check_gradient(layer.bias.grad, num_grad_b)


def test_activations_finite_difference():
    np.random.seed(99)
    x = Tensor(np.random.randn(3, 4), requires_grad=True)

    # Sigmoid
    def forward_sig():
        return x.sigmoid().sum()
    x.zero_grad()
    forward_sig().backward()
    num_grad_sig = numerical_gradient(forward_sig, x)
    check_gradient(x.grad, num_grad_sig)

    # Tanh
    def forward_tanh():
        return x.tanh().sum()
    x.zero_grad()
    forward_tanh().backward()
    num_grad_tanh = numerical_gradient(forward_tanh, x)
    check_gradient(x.grad, num_grad_tanh)

    # Softmax
    softmax_layer = Softmax(axis=-1)
    def forward_sm():
        # Weighted sum to test non-trivial gradients
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        return (softmax_layer(x) * weights).sum()
    x.zero_grad()
    forward_sm().backward()
    num_grad_sm = numerical_gradient(forward_sm, x)
    check_gradient(x.grad, num_grad_sm)


def test_cross_entropy_loss_finite_difference():
    np.random.seed(55)
    logits = Tensor(np.random.randn(4, 3), requires_grad=True)
    targets = np.array([0, 2, 1, 2])
    criterion = CrossEntropyLoss(reduction="mean")

    def forward_fn():
        return criterion(logits, targets)

    loss = forward_fn()
    loss.backward()

    num_grad = numerical_gradient(forward_fn, logits)
    check_gradient(logits.grad, num_grad)


def test_mse_loss_finite_difference():
    np.random.seed(77)
    pred = Tensor(np.random.randn(4, 2), requires_grad=True)
    target = np.random.randn(4, 2)
    criterion = MSELoss(reduction="mean")

    def forward_fn():
        return criterion(pred, target)

    loss = forward_fn()
    loss.backward()

    num_grad = numerical_gradient(forward_fn, pred)
    check_gradient(pred.grad, num_grad)


def test_adam_optimizer_step():
    W = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    W.grad = np.array([[0.1, -0.2], [0.5, 0.0]])

    opt = Adam([W], lr=0.1)
    orig_data = W.data.copy()
    opt.step()

    # Adam should update weights in the opposite direction of gradients
    assert not np.array_equal(W.data, orig_data)
    assert W.data[0, 0] < orig_data[0, 0] # positive grad -> decrease
    assert W.data[0, 1] > orig_data[0, 1] # negative grad -> increase
