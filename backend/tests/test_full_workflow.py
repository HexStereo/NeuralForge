"""Full workflow end-to-end integration test across API and services."""

import time
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.database import Base, engine


@pytest.fixture(autouse=True)
def init_db():
    Base.metadata.create_all(bind=engine)
    yield


def test_full_system_workflow():
    client = TestClient(app)

    # 1. Create project
    proj_res = client.post("/api/projects", json={
        "name": "E2E Sensor Project",
        "description": "Integration testing full training studio"
    })
    assert proj_res.status_code == 201
    project_id = proj_res.json()["id"]

    # 2. Ingest sample dataset
    sample_res = client.post("/api/datasets/sample", data={"project_id": project_id})
    assert sample_res.status_code == 201
    dataset = sample_res.json()
    dataset_id = dataset["id"]
    assert dataset["row_count"] > 0
    assert dataset["column_count"] == 5
    assert len(dataset["feature_columns"]) == 4

    # 3. Validate and create architecture
    # Input dim: 4, Output dim: 3 (3 classes: Normal, Overheating, Mechanical_Imbalance)
    arch_config = {
        "input_dim": 4,
        "output_dim": 3,
        "nodes": [
            {"id": "node_in", "type": "input", "data": {"dim": 4}},
            {"id": "node_d1", "type": "dense", "data": {"in_features": 4, "out_features": 16, "bias": True}},
            {"id": "node_act1", "type": "relu", "data": {}},
            {"id": "node_d2", "type": "dense", "data": {"in_features": 16, "out_features": 3, "bias": True}},
            {"id": "node_out", "type": "output", "data": {}}
        ],
        "edges": [
            {"id": "e1", "source": "node_in", "target": "node_d1"},
            {"id": "e2", "source": "node_d1", "target": "node_act1"},
            {"id": "e3", "source": "node_act1", "target": "node_d2"},
            {"id": "e4", "source": "node_d2", "target": "node_out"}
        ]
    }

    val_res = client.post("/api/architectures/validate", json=arch_config)
    assert val_res.status_code == 200
    val_data = val_res.json()
    assert val_data["is_valid"] is True
    assert val_data["total_parameters"] == (4 * 16 + 16) + (16 * 3 + 3) # 80 + 51 = 131

    create_arch_res = client.post("/api/architectures", json={
        "project_id": project_id,
        "name": "Sensor MLP Classifier",
        "description": "2-layer Dense network",
        "config": arch_config
    })
    assert create_arch_res.status_code == 201
    arch_id = create_arch_res.json()["id"]

    # 4. Start training
    train_res = client.post("/api/training/start", json={
        "project_id": project_id,
        "name": "Run 1 - Sensor Faults",
        "architecture_id": arch_id,
        "dataset_id": dataset_id,
        "hyperparameters": {
            "epochs": 15,
            "batch_size": 32,
            "learning_rate": 0.05,
            "optimizer": "adam",
            "loss_function": "cross_entropy",
            "random_seed": 42
        }
    })
    assert train_res.status_code == 201
    experiment_id = train_res.json()["experiment_id"]

    # 5. Wait for background training completion (15 epochs takes ~0.5s)
    max_wait = 10
    start = time.time()
    final_status = None

    while time.time() - start < max_wait:
        status_res = client.get(f"/api/training/{experiment_id}/status")
        assert status_res.status_code == 200
        final_status = status_res.json()["status"]
        if final_status in ["completed", "failed", "cancelled"]:
            break
        time.sleep(0.2)

    assert final_status == "completed", f"Training did not complete in time, status: {final_status}"

    # 6. Verify experiment results
    exp_res = client.get(f"/api/experiments/{experiment_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["status"] == "completed"
    assert len(exp_data["history"]) == 15
    assert exp_data["final_metrics"]["final_train_loss"] < 0.5
    assert exp_data["final_metrics"]["final_train_metric"] > 0.85

    # 7. Model Inspector verification
    insp_res = client.get(f"/api/inspector/{experiment_id}")
    assert insp_res.status_code == 200
    insp_data = insp_res.json()
    assert "layers" in insp_data
    assert "activations" in insp_data
    assert "evaluation" in insp_data
    assert "confusion_matrix" in insp_data["evaluation"]
    assert insp_data["evaluation"]["accuracy"] > 0.80

    # 8. Compare experiments endpoint
    cmp_res = client.post("/api/experiments/compare", json=[experiment_id])
    assert cmp_res.status_code == 200
    assert len(cmp_res.json()) == 1
    assert cmp_res.json()[0]["id"] == experiment_id
