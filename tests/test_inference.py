import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.inference import InferenceResponse, PhonemePrediction, InferenceSummary, InferenceResult
from pydantic import ValidationError

client = TestClient(app)


class TestInferenceSchemas:
    def test_phoneme_prediction_schema(self):
        pred = PhonemePrediction(
            phoneme="AE1",
            status="incorrect",
            confidence=0.3,
            expected="AE1",
            actual="AH0",
            reason="Nham /AE1/ thanh /AH0/",
        )
        assert pred.phoneme == "AE1"
        assert pred.status == "incorrect"
        assert pred.confidence == 0.3

    def test_inference_summary_schema(self):
        summary = InferenceSummary(
            total_phonemes=10,
            correct_phonemes=7,
            incorrect_phonemes=3,
            accuracy=0.7,
        )
        assert summary.total_phonemes == 10
        assert summary.accuracy == 0.7

    def test_inference_response_schema(self):
        response = InferenceResponse(
            success=True,
            model_name="wav2vec2",
            input_file="test.wav",
            predictions=[
                PhonemePrediction(phoneme="HH", status="correct", confidence=0.95)
            ],
            result=InferenceResult(
                phoneme_sequence=["HH"],
                phoneme_string="HH",
                overall_confidence=0.95,
            ),
            summary=InferenceSummary(total_phonemes=1, correct_phonemes=1, incorrect_phonemes=0, accuracy=1.0),
            processing_time_ms=123.45,
            request_id="test123",
        )
        assert response.success is True
        assert response.model_name == "wav2vec2"
        assert len(response.predictions) == 1

    def test_invalid_confidence_range(self):
        with pytest.raises(ValidationError):
            PhonemePrediction(
                phoneme="AE",
                status="correct",
                confidence=1.5,
            )

    def test_invalid_top_k_range(self):
        from app.schemas.inference import InferenceRequest
        with pytest.raises(ValidationError):
            InferenceRequest(model_name="test", top_k=0)


class TestInferenceEndpoints:
    def test_inference_without_file_returns_422(self):
        response = client.post("/api/v1/infer")
        assert response.status_code == 422

    def test_inference_invalid_model_name(self):
        response = client.post(
            "/api/v1/infer/cnn-bilstm-ctc",
            files={"file": ("test.wav", b"fakeaudio", "audio/wav")},
        )
        assert response.status_code in (400, 422, 503)

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/v1/infer" in schema["paths"]
        assert "/api/v1/health" in schema["paths"]
        assert "/api/v1/models" in schema["paths"]

    def test_swagger_ui(self):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_ui(self):
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
