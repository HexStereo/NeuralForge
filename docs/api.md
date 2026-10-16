# NeuralForge API Documentation

NeuralForge provides a typed REST API and WebSocket streaming interface powered by FastAPI.

Interactive Swagger/OpenAPI documentation is hosted at:
`http://127.0.0.1:8008/docs`

---

## 1. System Health
- **`GET /api/health`**
  - **Returns**: Status (`"healthy"`), version, Python version, platform, and storage paths.

---

## 2. Projects
- **`GET /api/projects`**: List all research projects.
- **`POST /api/projects`**: Create a new project.
  - **Body**: `{"name": "string", "description": "string"}`
- **`GET /api/projects/{project_id}`**: Retrieve project metadata and artifact counts.
- **`PUT /api/projects/{project_id}`**: Update project name or description.
- **`DELETE /api/projects/{project_id}`**: Delete project and cascade-delete its datasets, architectures, and experiments.

---

## 3. Datasets
- **`GET /api/datasets?project_id={id}`**: List datasets.
- **`POST /api/datasets/upload`**: Upload CSV file with multipart form-data. Enforces 50 MB upload limit.
- **`POST /api/datasets/sample`**: Ingest the built-in 4-feature industrial sensor telemetry dataset.
- **`GET /api/datasets/{dataset_id}`**: Get dataset column profile, sample preview, and summary statistics.
- **`PUT /api/datasets/{dataset_id}/config`**: Update feature/target selection, task type (`classification` or `regression`), and train/val/test split ratios.
- **`DELETE /api/datasets/{dataset_id}`**: Delete dataset and file from disk.

---

## 4. Architectures
- **`POST /api/architectures/validate`**: Dry-run validation of visual graph configuration. Returns `is_valid`, error list, total parameters, and layer summaries.
- **`POST /api/architectures`**: Save and compile a validated architecture.
- **`GET /api/architectures?project_id={id}`**: List architectures for a project.
- **`GET /api/architectures/{arch_id}`**: Retrieve architecture details.
- **`DELETE /api/architectures/{arch_id}`**: Delete architecture.

---

## 5. Training & Telemetry
- **`POST /api/training/start`**:
  - Launches background training run.
  - **Body**:
    ```json
    {
      "project_id": "string",
      "name": "string",
      "architecture_id": "string",
      "dataset_id": "string",
      "hyperparameters": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.01,
        "optimizer": "adam",
        "loss_function": "cross_entropy",
        "random_seed": 42
      }
    }
    ```
- **`POST /api/training/{experiment_id}/cancel`**: Cooperatively cancels an active training job.
- **`GET /api/training/{experiment_id}/status`**: Returns current epoch, loss, and training state.
- **`WebSocket /ws/training/{experiment_id}`**:
  - Live metric stream.
  - **Frame Schema**:
    ```json
    {
      "type": "metric",
      "experiment_id": "uuid",
      "status": "running",
      "epoch": 5,
      "total_epochs": 20,
      "train_loss": 0.2312,
      "train_acc": 0.9250,
      "val_loss": 0.2850,
      "val_acc": 0.9000,
      "elapsed_seconds": 2.45,
      "eta_seconds": 7.3
    }
    ```

---

## 6. Model Inspector
- **`GET /api/inspector/{experiment_id}?sample_size=50`**:
  - Returns:
    - Layer-by-layer parameter statistics (mean, std, min, max, 10-bin weight histograms).
    - Real intermediate layer activations captured from test batch inference.
    - Sparsity percentage (% of inactive zero neurons).
    - Multi-class Confusion Matrix and classification report (Precision, Recall, F1, Support).

---

## 7. Experiments
- **`GET /api/experiments?project_id={id}`**: List past training runs.
- **`GET /api/experiments/{experiment_id}`**: Full experiment history and metrics.
- **`GET /api/experiments/{experiment_id}/checkpoints`**: List checkpoint files.
- **`POST /api/experiments/compare`**:
  - **Body**: `["exp-id-1", "exp-id-2"]`
  - Returns side-by-side metric table and overlayable loss convergence curves.
- **`DELETE /api/experiments/{experiment_id}`**: Delete experiment record and `.npz` checkpoint file.
