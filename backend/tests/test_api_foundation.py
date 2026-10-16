"""Tests for API foundation and database integration."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.models.database import Base, engine

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    # Cleanup after test if needed

def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "NeuralForge" in data["message"]

def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "NeuralForge"
    assert "environment" in data
    assert "python_version" in data["environment"]

def test_project_crud():
    client = TestClient(app)
    # 1. Create project
    create_payload = {
        "name": "Test Neural Project",
        "description": "A project for testing unit foundation"
    }
    res = client.post("/api/projects", json=create_payload)
    assert res.status_code == 201
    proj = res.json()
    proj_id = proj["id"]
    assert proj["name"] == "Test Neural Project"
    assert proj["dataset_count"] == 0

    # 2. Get project by ID
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 200
    assert res.json()["id"] == proj_id

    # 3. List projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    projects = res.json()
    assert any(p["id"] == proj_id for p in projects)

    # 4. Update project
    update_payload = {"name": "Updated Test Project"}
    res = client.put(f"/api/projects/{proj_id}", json=update_payload)
    assert res.status_code == 200
    assert res.json()["name"] == "Updated Test Project"

    # 5. Delete project
    res = client.delete(f"/api/projects/{proj_id}")
    assert res.status_code == 204

    # 6. Verify deleted
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 404
