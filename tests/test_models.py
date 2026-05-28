import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestModels:
    def test_list_models(self):
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert len(data["models"]) >= 3
        model_names = [m["name"] for m in data["models"]]
        assert "cnn_bilstm_ctc" in model_names
        assert "dab_transformer" in model_names
        assert "wav2vec2" in model_names

    def test_get_model_by_name(self):
        response = client.get("/api/v1/models/cnn_bilstm_ctc")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "cnn_bilstm_ctc"
        assert data["display_name"] == "CNN-BiLSTM-CTC"
        assert "phoneme_set" in data
        assert "loaded" in data

    def test_get_model_not_found(self):
        response = client.get("/api/v1/models/nonexistent_model")
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    def test_model_info_has_required_fields(self):
        for model_name in ["cnn_bilstm_ctc", "dab_transformer", "wav2vec2"]:
            response = client.get(f"/api/v1/models/{model_name}")
            assert response.status_code == 200
            data = response.json()
            required = ["name", "display_name", "version", "architecture", "task", "sample_rate", "phoneme_set"]
            for field in required:
                assert field in data, f"Missing field '{field}' in model {model_name}"

    def test_labels(self):
        response = client.get("/api/v1/labels?model_name=wav2vec2")
        assert response.status_code == 200
        data = response.json()
        assert "phonemes" in data
        assert "total" in data
        assert "model_name" in data
        assert data["total"] > 0
