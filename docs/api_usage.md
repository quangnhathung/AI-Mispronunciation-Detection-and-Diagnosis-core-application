# API Usage Guide

## Authentication

This API does not include built-in authentication. In production, add an auth middleware or API gateway.

## Base URL

All endpoints are prefixed with `/api/v1`.

## Endpoint Reference

### Health Check

```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Readiness Check

```http
GET /api/v1/ready
```

**Response**:
```json
{
  "ready": true,
  "models_loaded": ["cnn_bilstm_ctc", "dab_transformer", "wav2vec2"],
  "models_failed": []
}
```

### List Models

```http
GET /api/v1/models
```

**Response**:
```json
{
  "models": [
    {
      "name": "cnn_bilstm_ctc",
      "display_name": "CNN-BiLSTM-CTC",
      "version": "1.0.0",
      "description": "CNN-BiLSTM with CTC loss for phoneme recognition...",
      "architecture": "CNN + BiLSTM + CTC",
      "task": "ASR phoneme recognition + MDD",
      "requires_text": false,
      "requires_gpu": false,
      "sample_rate": 16000,
      "phoneme_set_size": 42,
      "phoneme_set": ["<blank>", "<unk>", "AA0", ...],
      "loaded": false,
      "status": "unloaded"
    }
  ],
  "total": 3
}
```

### Get Model Details

```http
GET /api/v1/models/{model_name}
```

### Load Model

```http
POST /api/v1/models/{model_name}/load
```

### Unload Model

```http
POST /api/v1/models/{model_name}/unload
```

### Get Phoneme Labels

```http
GET /api/v1/labels?model_name=wav2vec2
```

### Inference (Auto-Select)

```http
POST /api/v1/infer
```

**Request** (multipart/form-data):
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | Audio file (wav, mp3, flac, m4a, ogg) |
| `model_name` | String | No | `auto` / `cnn_bilstm_ctc` / `dab_transformer` / `wav2vec2` |
| `text` | String | No* | Ground-truth transcript (required for wav2vec2) |
| `top_k` | Integer | No | Results count (1-100, default: 10) |
| `threshold` | Float | No | Confidence threshold (0-1, default: 0.5) |
| `return_details` | Boolean | No | Return per-phoneme details (default: true) |
| `sample_rate` | Integer | No | Override sample rate |

### Inference (Specific Model)

```http
POST /api/v1/infer/cnn-bilstm-ctc
POST /api/v1/infer/dab-transformer
POST /api/v1/infer/wav2vec2
```

Same form fields as above, except `model_name` is fixed.

### Version Info

```http
GET /api/v1/version
```

## Postman Collection

Import the OpenAPI schema:

```bash
curl http://localhost:8000/openapi.json > openapi.json
```

Then import `openapi.json` into Postman.

## cURL Examples

### Minimal (auto model, no text)

```bash
curl -X POST http://localhost:8000/api/v1/infer \
  -F "file=@sample.wav"
```

### Full options

```bash
curl -X POST http://localhost:8000/api/v1/infer/wav2vec2 \
  -F "file=@sample.wav" \
  -F "text=hello world" \
  -F "threshold=0.6" \
  -F "return_details=true" \
  -F "top_k=50"
```

### Python requests

```python
import requests

url = "http://localhost:8000/api/v1/infer/wav2vec2"
files = {"file": open("sample.wav", "rb")}
data = {
    "text": "hello world",
    "threshold": 0.5,
    "return_details": True,
}
resp = requests.post(url, files=files, data=data)
print(resp.json())
```

### JavaScript fetch

```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('text', 'hello world');
formData.append('threshold', '0.5');

fetch('http://localhost:8000/api/v1/infer/wav2vec2', {
  method: 'POST',
  body: formData,
})
.then(r => r.json())
.then(console.log);
```
