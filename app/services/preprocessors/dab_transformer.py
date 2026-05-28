import numpy as np
from app.services.preprocessors.base import BasePreprocessor
from app.core.exceptions import PreprocessError


class DABTransformerPreprocessor(BasePreprocessor):
    def __init__(self, sample_rate: int = 16000, max_samples: int = 160000):
        self.sample_rate = sample_rate
        self.max_samples = max_samples

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        try:
            if sample_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            return audio
        except Exception as e:
            raise PreprocessError(f"DAB-Transformer preprocessing failed: {e}") from e
