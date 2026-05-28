import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_response_model(self):
        response = client.get("/api/v1/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_health_content_type(self):
        response = client.get("/api/v1/health")
        assert response.headers["content-type"] == "application/json"

    def test_readiness(self):
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "models_loaded" in data
        assert "models_failed" in data
        assert isinstance(data["ready"], bool)
        assert isinstance(data["models_loaded"], list)
        assert isinstance(data["models_failed"], list)

    def test_version(self):
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "app_version" in data
        assert "python_version" in data
        assert "dependencies" in data
