# NeuralForge Architecture Specification

## 1. Architectural Philosophy
NeuralForge is designed with strict boundaries separating:
1. **Numerical Computation Engine (`neuralforge_core`)**: Independent of web frameworks, databases, or UI. Pure mathematical execution.
2. **Orchestration & Persistence Layer (`backend/app/`)**: FastAPI services, background worker threads, thread-safe WebSocket broadcast queues, and SQLite database models.
3. **Interactive Research Frontend (`frontend/`)**: React, TypeScript, React Flow node editor, and Recharts visualization.

## 2. Component Design

### 2.1 NeuralForge Core
- **Tensor**: NumPy-backed array with gradient accumulation, creator node tracking, and reverse topological backward execution.
- **Autograd Functions**:
  - `unbroadcast(grad, target_shape)`: Automatically sums out prepended dimensions and dimensions of size 1 created during broadcasting.
  - `MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Sum`, `Mean`, `Reshape`, `Transpose`, `Maximum`, `Exp`, `Log`.
- **Layers**:
  - `Module`: Base class managing parameters and child modules.
  - `Parameter`: Subclass of `Tensor` with `requires_grad=True`.
  - `Dense`: Linear transformation with He (`sqrt(2/fan_in)`) and Xavier/Glorot (`sqrt(6/(fan_in + fan_out))`) initialization.
  - `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`, `Dropout`, `Sequential`.
- **Losses**:
  - `MSELoss`: Mean squared error.
  - `CrossEntropyLoss`: Numerically stable fused Softmax and negative log-likelihood with log-sum-exp stabilization.
- **Optimizers**:
  - `SGD`: Stochastic Gradient Descent with momentum and weight decay.
  - `Adam`: First and second moment running averages with bias corrections.
- **Checkpointing**:
  - `CheckpointManager`: Uses `np.savez_compressed` without Python pickle to prevent arbitrary code execution vulnerabilities.

### 2.2 Backend Asynchronous Execution
- Long-running training loops execute inside a dedicated background worker thread (`TrainingJob` managed by `TrainingManager`).
- FastAPI's `asyncio` event loop is never blocked.
- Live telemetry is pushed into an `asyncio.Queue` using `call_soon_threadsafe`, streaming immediately to subscribed WebSocket clients at `/ws/training/{experiment_id}`.
- Cooperative cancellation is checked at each batch and epoch boundary via a thread-safe `threading.Event`.

### 2.3 Visual Graph Compilation
- Visual node graph configs submitted via React Flow are validated through Kahn's algorithm for topological sorting and cycle detection.
- Layer dimension compatibility ($W_1$ output == $W_2$ input) is strictly checked on the backend before compilation to an executable `Sequential` model.
