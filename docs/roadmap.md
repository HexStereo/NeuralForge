# Roadmap & Known Limitations

## 1. Roadmap for Future Milestones

### Milestone 7.1 — Vision & Spatial Layers
- 2D Convolution (`Conv2D`) with configurable stride, padding, and dilation.
- Max Pooling (`MaxPool2D`) and Average Pooling (`AvgPool2D`).
- Batch Normalization (`BatchNorm1d`, `BatchNorm2d`) with running mean and variance tracking.

### Milestone 7.2 — Sequence Modeling & Attention
- Recurrent layers (`RNN`, `LSTM`, `GRU`) with backward-through-time (BPTT).
- Scaled Dot-Product Attention: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- Multi-Head Attention layer and Transformer Encoder blocks.

### Milestone 7.3 — Automated Hyperparameter Search & PyTorch Cross-Validation
- Grid search and Bayesian optimization across learning rates, optimizers, and layer widths.
- Automated parity check comparing numerical convergence and gradient magnitudes against `torch.nn`.

### Milestone 7.4 — Hardware Acceleration Backends
- C++ / Cython SIMD matrix multiplication acceleration using OpenBLAS / MKL.
- Optional CuPy / OpenCL backend for GPU acceleration.

---

## 2. Known Limitations & Security Assumptions

1. **Local Single-Node Deployment**:
   - The current release is architected for single-machine local workstation execution on standard CPUs.
   - Database persistence uses SQLite with concurrent reader support.

2. **Sequential Architecture in Graph Builder**:
   - Version 0.1.0 visual builder focuses on sequential feed-forward networks (Dense + Activations + Dropout + Flatten). Branching topologies (residual skip connections, inception blocks) are scheduled for Milestone 7.

3. **In-Memory Batch Execution**:
   - Datasets are loaded via pandas and transformed into NumPy arrays. Datasets should comfortably fit in workstation RAM (up to 50 MB by default limit, configurable in `backend/app/core/config.py`).

4. **Security & Untrusted Uploads**:
   - Model checkpoints are serialized strictly using `np.savez_compressed` without Python `pickle`.
   - Imported JSON architecture configurations are strictly validated by Pydantic and topological analyzers before execution. Never execute arbitrary Python code.
