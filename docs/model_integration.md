# Model Integration Guide

## Overview

Each model is wrapped in an adapter class that inherits from `BaseModelWrapper` (defined in `app/models/base_model.py`). The adapter standardizes the interface so the API layer can treat all models uniformly.

## Base Model Interface

```python
class BaseModelWrapper(ABC):
    name: str                         # Unique identifier
    display_name: str                 # Human-readable name
    version: str                      # Model version
    description: str                  # Description
    architecture: str                 # Architecture type
    task: str                         # Task description
    requires_text: bool               # Whether GT text is required
    requires_gpu: bool                # Whether GPU is required
    sample_rate: int                  # Expected input sample rate
    phoneme_set_size: int             # Vocabulary size
    phoneme_set: list[str]            # Phoneme symbols

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @abstractmethod
    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any: ...

    @abstractmethod
    def predict(self, features: Any, **kwargs) -> Any: ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]: ...
```

## Adding a New Model

### Step 1: Create the wrapper

Create `app/models/my_model_wrapper.py`:

```python
from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary

class MyModel(BaseModelWrapper):
    name = "my_model"
    display_name = "My Custom Model"
    version = "1.0.0"
    # ... set all class attributes ...
    requires_text = False
    sample_rate = 16000

    def load_model(self) -> None:
        # Load checkpoint, initialize model, move to device
        self._loaded = True

    def unload_model(self) -> None:
        self.model = None
        self._loaded = False

    def preprocess(self, audio, sample_rate, **kwargs):
        # Convert audio to model input features
        return features

    def predict(self, features, **kwargs):
        # Run model inference
        return raw_output

    def postprocess(self, raw_output, **kwargs):
        # Convert raw output to standard predictions
        predictions = [...]
        result = InferenceResult(...)
        summary = InferenceSummary(...)
        return predictions, result, summary
```

### Step 2: Register in ModelRegistry

Edit `app/services/model_registry.py`:

```python
from app.models.my_model_wrapper import MyModel

class ModelRegistry:
    def _register_builtins(self) -> None:
        # ... existing registrations ...
        self.register(MyModel(checkpoint_path=settings.my_model_checkpoint))
```

### Step 3: Add config

Add to `app/core/config.py`:

```python
my_model_checkpoint: Optional[str] = None
```

Add auto-detection in `auto_detect_checkpoints()`.

### Step 4: Add inference endpoint

Add to `app/api/routers/inference.py`:

```python
@router.post("/infer/my-model", ...)
async def infer_my_model(...):
    # Similar to other inference endpoints
```

## Preprocessing Notes

### CNN-BiLSTM-CTC
- Input: 16kHz mono WAV
- Feature: Mel spectrogram (80 bands, n_fft=512, win_length=400, hop_length=160)
- Normalization: mean-std per sample
- Max duration: 30s

### DAB-Transformer
- Input: 16kHz mono WAV
- Feature: Raw waveform, truncated/padded to 160k samples (10s)
- No spectrogram extraction

### Wav2Vec2-MDD
- Input: 16kHz mono WAV
- Feature: Wav2Vec2 feature extractor (normalization + CNN features)
- Requires: ground-truth text → g2p_en → ARPABET phoneme IDs

## Postprocessing Notes

### ASR Models (CNN-BiLSTM-CTC, DAB-Transformer)
- Greedy CTC decoding → phoneme sequence
- With GT text: alignment via jiwer → per-phoneme feedback
- Without GT text: raw phoneme sequence only

### Scoring Models (Wav2Vec2-MDD)
- Sigmoid → per-phoneme correctness probability
- Threshold-based classification (default: 0.5)
- No phoneme recognition (requires GT phonemes)

## Error Handling Conventions

All model wrappers should raise:
- `ModelNotLoadedError` if `load_model()` hasn't been called
- `PreprocessError` if input processing fails
- `InferenceError` if model forward pass fails
- `TextRequiredError` if model requires but doesn't receive text

The API layer catches these and returns consistent JSON error responses.
