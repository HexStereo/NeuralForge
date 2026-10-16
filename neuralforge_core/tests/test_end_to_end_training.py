"""End-to-end training and checkpoint tests on synthetic datasets."""

import os
import tempfile
import numpy as np
import pytest

from neuralforge_core import (
    Sequential, Dense, ReLU, CrossEntropyLoss, Adam,
    ArrayDataset, DataLoader, Trainer, CheckpointManager, Tensor
)


def make_synthetic_moons(n_samples: int = 400, noise: float = 0.1, seed: int = 42):
    """Generate two interleaving half circles."""
    rng = np.random.default_rng(seed)
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([
        np.append(outer_circ_x, inner_circ_x),
        np.append(outer_circ_y, inner_circ_y)
    ]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=int), np.ones(n_samples_in, dtype=int)])

    if noise > 0:
        X += rng.normal(scale=noise, size=X.shape)

    # Shuffle
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def test_end_to_end_training_loss_decrease():
    # 1. Prepare data
    X, y = make_synthetic_moons(n_samples=400, noise=0.08, seed=42)
    split = int(0.8 * len(X))
    train_ds = ArrayDataset(X[:split], y[:split])
    val_ds = ArrayDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, seed=42)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Build model
    model = Sequential(
        Dense(2, 32, bias=True, seed=42),
        ReLU(),
        Dense(32, 16, bias=True, seed=43),
        ReLU(),
        Dense(16, 2, bias=True, seed=44)
    )

    loss_fn = CrossEntropyLoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=0.03)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        task_type="classification"
    )

    # 3. Train
    history = trainer.fit(epochs=35)

    assert len(history) == 35
    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    final_acc = history[-1]["train_metric"]
    final_val_acc = history[-1]["val_metric"]

    print(f"Training history: initial_loss={initial_loss:.4f}, final_loss={final_loss:.4f}, val_acc={final_val_acc:.4f}")

    # Loss must strictly decrease
    assert final_loss < initial_loss * 0.35, f"Expected loss to drop by at least 65%, got {initial_loss} -> {final_loss}"
    # Accuracy must learn the non-linear boundary
    assert final_acc > 0.90, f"Expected train accuracy > 90%, got {final_acc}"
    assert final_val_acc > 0.85, f"Expected val accuracy > 85%, got {final_val_acc}"


def test_checkpoint_save_and_load():
    X, y = make_synthetic_moons(n_samples=100, seed=123)
    model1 = Sequential(
        Dense(2, 8, bias=True, seed=1),
        ReLU(),
        Dense(8, 2, bias=True, seed=2)
    )
    optimizer = Adam(model1.parameters(), lr=0.01)

    # Run one forward pass
    test_x = Tensor(X[:5])
    out1 = model1(test_x).data.copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_checkpoint.npz")

        # Save checkpoint
        saved_path = CheckpointManager.save(
            ckpt_path,
            model=model1,
            optimizer=optimizer,
            epoch=5,
            metrics={"train_loss": 0.25}
        )
        assert os.path.exists(saved_path)

        # Create model2 with different weights
        model2 = Sequential(
            Dense(2, 8, bias=True, seed=999),
            ReLU(),
            Dense(8, 2, bias=True, seed=998)
        )
        # Restore checkpoint into model2
        header = CheckpointManager.load(saved_path, model=model2)
        assert header["epoch"] == 5
        assert header["metrics"]["train_loss"] == 0.25

        # Output from model2 must now be identical to out1
        out2 = model2(test_x).data
        np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)


def test_training_cancellation():
    X, y = make_synthetic_moons(n_samples=200, seed=42)
    train_ds = ArrayDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=16)

    model = Sequential(Dense(2, 4), ReLU(), Dense(4, 2))
    opt = Adam(model.parameters(), lr=0.01)

    # Cancel after 2 epochs
    epoch_count = [0]
    def on_epoch(info):
        epoch_count[0] += 1

    def should_cancel():
        return epoch_count[0] >= 2

    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=opt,
        train_loader=train_loader,
        on_epoch_end=on_epoch,
        check_cancelled=should_cancel
    )

    history = trainer.fit(epochs=20)
    assert len(history) <= 3, f"Expected cancellation to stop within 2-3 epochs, ran {len(history)}"
