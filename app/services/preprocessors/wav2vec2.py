import numpy as np
from app.services.preprocessors.base import BasePreprocessor
from app.core.exceptions import PreprocessError


class Wav2Vec2Preprocessor(BasePreprocessor):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> np.ndarray:
        try:
            if sample_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            return audio
        except Exception as e:
            raise PreprocessError(f"Wav2Vec2 preprocessing failed: {e}") from e
