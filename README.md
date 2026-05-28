# MDD Core Application — FastAPI Mispronunciation Detection API

A production-ready FastAPI application for **Mispronunciation Detection and Diagnosis (MDD)** in English pronunciation. Supports three distinct model architectures in a unified inference API.

## Features

- **3 model architectures** unified under a single API:
  - **CNN-BiLSTM-CTC**: Convolutional + Bidirectional LSTM with CTC loss for phoneme recognition
  - **DAB-Transformer**: Dynamic Attention Bias Transformer with CTC for phoneme recognition
  - **Wav2Vec2-MDD**: Wav2Vec2-based phoneme scoring with cross-attention for error detection
- **Unified inference API** with consistent request/response schemas
- **Swagger UI** at `/docs` and **ReDoc** at `/redoc`
- **Auto model selection** — let the API pick the best available model
- **Per-phoneme diagnosis** — detailed feedback on correct/incorrect/substitution/deletion/insertion
- **ASR output** — phoneme sequence transcription from audio
- **Configurable** — choose model, threshold, top-k, sample rate
- **Clean architecture** — API / service / model / schema separation
- **Lazy loading** — models load on first inference or on demand

## Architecture

```
MDD-core-application/
├── app/                          # FastAPI application
│   ├── main.py                   # App entry point, lifespan, router registration
│   ├── api/
│   │   ├── deps.py               # Dependency injection helpers
│   │   └── routers/
│   │       ├── health.py         # /health, /ready endpoints
│   │       ├── models.py         # /models endpoint
│   │       ├── inference.py      # /infer endpoints
│   │       └── metadata.py       # /version endpoint
│   ├── core/
│   │   ├── config.py             # Pydantic settings from .env
│   │   ├── exceptions.py         # Custom exception classes
│   │   ├── logging.py            # Loguru configuration
│   │   └── middleware.py         # Request ID, CORS, error handling
│   ├── schemas/
│   │   ├── common.py             # Health, Version, Error schemas
│   │   ├── inference.py          # Inference request/response schemas
│   │   └── model.py              # Model info, Labels schemas
│   ├── services/
│   │   ├── audio_service.py      # Audio loading, resampling, validation
│   │   ├── inference_service.py  # Orchestrates inference workflow
│   │   ├── model_registry.py     # Model registry (register/get/list)
│   │   ├── preprocessors/        # Audio preprocessing per model
│   │   │   ├── base.py
│   │   │   ├── cnn_bilstm_ctc.py
│   │   │   ├── dab_transformer.py
│   │   │   └── wav2vec2.py
│   │   └── postprocessors/
│   │       └── base.py
│   ├── models/
│   │   ├── base_model.py         # Abstract base class for all models
│   │   ├── cnn_bilstm_ctc_wrapper.py
│   │   ├── dab_transformer_wrapper.py
│   │   └── wav2vec2_wrapper.py
│   └── utils/
│       ├── file_utils.py         # File validation and upload
│       ├── id_utils.py           # Request ID generation
│       └── time_utils.py         # Performance measurement
├── CNN_BiLSTM_CTC/               # Existing project (not modified)
├── DAB_Transformer_Project/      # Existing project (not modified)
├── Wav2vec2/                     # Existing project (not modified)
├── tests/
│   ├── test_health.py            # Health endpoint tests
│   ├── test_models.py            # Model listing tests
│   └── test_inference.py         # Schema validation + endpoint tests
├── docs/
│   ├── api_usage.md              # Comprehensive API usage guide
│   └── model_integration.md      # Model integration guide
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (with CUDA optional for GPU acceleration)
- ~8GB RAM minimum (16GB recommended)

### Installation

```bash
# Clone the repository
cd MDD-core-application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install additional per-project dependencies (if needed)
cd CNN_BiLSTM_CTC && pip install -r requirements.txt && cd ..
# DAB_Transformer_Project lacks a requirements.txt — install torch, torchaudio, jiwer, g2p_en
# Wav2vec2: pip install -r Wav2vec2/requirements.txt

# Copy environment config
cp .env.example .env
```

### Configuration

All configuration is via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `MDD Core Application` | Application name |
| `APP_VERSION` | `1.0.0` | Application version |
| `DEBUG` | `false` | Enable debug mode |
| `API_PREFIX` | `/api/v1` | API route prefix |
| `LOG_LEVEL` | `INFO` | Logging level |
| `MODEL_LOAD_ON_STARTUP` | `false` | Pre-load all models at startup |
| `DEFAULT_MODEL` | `wav2vec2` | Default model for auto-select |
| `DEFAULT_THRESHOLD` | `0.5` | Default confidence threshold |
| `MAX_UPLOAD_SIZE_MB` | `10` | Max audio file upload size |
| `ALLOWED_AUDIO_FORMATS` | `wav,mp3,flac,m4a,ogg` | Allowed file extensions |

Checkpoints are auto-detected from the existing project directories. To override:

```bash
export WAV2VEC2_CHECKPOINT=/path/to/best_mdd_model_v4.pt
export CNN_BILSTM_CTC_CHECKPOINT=/path/to/best.pt
export DAB_TRANSFORMER_CHECKPOINT=/path/to/model_e23.pt
```

### Running

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Verify installation

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List models
curl http://localhost:8000/api/v1/models

# Swagger UI
open http://localhost:8000/docs
```

## API Endpoints

### Health & Readiness

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Service health status |
| `GET` | `/api/v1/ready` | Model readiness check |
| `GET` | `/api/v1/version` | System version and dependency info |

### Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/models` | List all registered models |
| `GET` | `/api/v1/models/{name}` | Get model details |
| `POST` | `/api/v1/models/{name}/load` | Load a model into memory |
| `POST` | `/api/v1/models/{name}/unload` | Unload a model from memory |
| `GET` | `/api/v1/labels` | Get phoneme label set |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/infer` | Auto-select model inference |
| `POST` | `/api/v1/infer/cnn-bilstm-ctc` | CNN-BiLSTM-CTC inference |
| `POST` | `/api/v1/infer/dab-transformer` | DAB-Transformer inference |
| `POST` | `/api/v1/infer/wav2vec2` | Wav2Vec2-MDD inference |

## Usage Examples

### Inference with auto model selection

```bash
curl -X POST http://localhost:8000/api/v1/infer \
  -F "file=@sample.wav" \
  -F "model_name=auto" \
  -F "text=hello world" \
  -F "threshold=0.5"
```

### Inference with Wav2Vec2 (requires text)

```bash
curl -X POST http://localhost:8000/api/v1/infer/wav2vec2 \
  -F "file=@sample.wav" \
  -F "text=hello world" \
  -F "threshold=0.5" \
  -F "return_details=true"
```

### Inference with CNN-BiLSTM-CTC (text optional)

```bash
curl -X POST http://localhost:8000/api/v1/infer/cnn-bilstm-ctc \
  -F "file=@sample.wav" \
  -F "text=hello world" \
  -F "return_details=true"
```

### Response example

```json
{
  "success": true,
  "model_name": "wav2vec2",
  "input_file": "sample.wav",
  "predictions": [
    {
      "phoneme": "HH",
      "status": "correct",
      "confidence": 0.95,
      "start_time": null,
      "end_time": null,
      "reason": null,
      "expected": "HH",
      "actual": "HH"
    },
    {
      "phoneme": "AH0",
      "status": "incorrect",
      "confidence": 0.31,
      "start_time": null,
      "end_time": null,
      "reason": "Phát âm lỗi âm /AH0/ (score: 0.310)",
      "expected": "AH0",
      "actual": "AH0"
    }
  ],
  "result": {
    "phoneme_sequence": ["HH", "AH0", "L", "OW1"],
    "phoneme_string": "HH AH0 L OW1",
    "overall_confidence": 0.85
  },
  "summary": {
    "total_phonemes": 4,
    "correct_phonemes": 2,
    "incorrect_phonemes": 2,
    "accuracy": 0.5,
    "precision": null,
    "recall": null,
    "f1_score": null
  },
  "processing_time_ms": 245.67,
  "request_id": "a1b2c3d4e5f6g7h8"
}
```

### Error response example

```json
{
  "success": false,
  "error": {
    "code": "AUDIO_FORMAT_ERROR",
    "message": "Unsupported format 'mp4'. Allowed: wav, mp3, flac, m4a, ogg",
    "details": {}
  },
  "request_id": "a1b2c3d4e5f6g7h8"
}
```

## Model Comparison

| Aspect | CNN-BiLSTM-CTC | DAB-Transformer | Wav2Vec2-MDD |
|--------|----------------|-----------------|--------------|
| **Backbone** | CNN + BiLSTM | Conv1d + Transformer | Wav2Vec2 + Bi-GRU + Cross-Attention |
| **Task** | ASR + MDD | ASR + MDD | Phoneme scoring |
| **Requires text?** | Optional (recommended for MDD) | Optional (recommended for MDD) | **Required** |
| **Output** | Phoneme sequence + alignment | Phoneme sequence + alignment | Per-phoneme correctness score |
| **Checkpoints** | `best.pt` / `last.pt` | `model_e*.pt` (23 epochs) | `best_mdd_model_v*.pt` (v1-v4) |
| **Phoneme set** | 42 (ARPABET + stress) | 41 (ARPABET no stress) | 46 (ARPABET + special tokens) |
| **Preprocessing** | Mel spectrogram 80-band | Raw waveform truncation | Wav2Vec2 feature extraction |

## Error Codes

| HTTP Code | Error Code | Description |
|-----------|-----------|-------------|
| 400 | `AUDIO_FORMAT_ERROR` | Invalid or corrupted audio file |
| 400 | `AUDIO_TOO_LARGE` | File exceeds size limit |
| 400 | `TEXT_REQUIRED` | Model requires ground-truth text |
| 404 | `MODEL_NOT_FOUND` | Requested model not registered |
| 422 | `PREPROCESS_ERROR` | Audio preprocessing failure |
| 503 | `MODEL_NOT_LOADED` | Model failed to load or initialize |
| 500 | `INFERENCE_ERROR` | Model inference failure |

## Extending with New Models

1. Create a model wrapper in `app/models/` implementing `BaseModelWrapper`
2. Implement `load_model()`, `unload_model()`, `preprocess()`, `predict()`, `postprocess()`
3. Register it in `app/services/model_registry.py` (add to `_register_builtins()`)
4. Create a dedicated inference endpoint in `app/api/routers/inference.py`

See `docs/model_integration.md` for detailed instructions.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_health.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

## Notes

- **Wav2Vec2 model requires ground-truth text**: The scoring architecture needs the expected phoneme sequence to evaluate pronunciation correctness.
- **CNN-BiLSTM-CTC and DAB-Transformer** can run without text (ASR mode) but MDD analysis requires ground-truth text for alignment.
- **Time alignment**: None of the current models provide phoneme-level timestamps. `start_time` and `end_time` fields are reserved for future use.
- **GPU**: Models auto-detect CUDA. Set `CUDA_VISIBLE_DEVICES=""` to force CPU.
- **Checkpoints**: The system auto-detects checkpoints on startup. Override via `.env` or environment variables.
- **Audio format**: WAV is recommended. Other formats are converted internally via librosa.
- **Vietnamese error messages**: Some models return error descriptions in Vietnamese (from the original DAB-Transformer project). This can be customized in the postprocessing layer.

## License

This project integrates three existing research projects. Each may have its own license terms. See individual project directories for details.
