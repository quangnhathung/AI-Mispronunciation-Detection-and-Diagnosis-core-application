from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger
from app.core.exceptions import AudioFormatError


class AudioService:
    @staticmethod
    def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        ext = path.suffix.lower()
        try:
            if ext == ".wav":
                import soundfile as sf
                audio, sr = sf.read(str(path))
            else:
                import librosa
                audio, sr = librosa.load(str(path), sr=None, mono=True)
        except Exception as e:
            raise AudioFormatError(f"Cannot load audio file: {e}") from e

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return audio, sr

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    @staticmethod
    def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        import librosa
        audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio

    @staticmethod
    def validate_audio(audio: np.ndarray, sample_rate: int, max_duration: float = 30.0) -> None:
        duration = len(audio) / sample_rate
        if duration < 0.1:
            raise AudioFormatError(f"Audio too short: {duration:.2f}s (min 0.1s)")
        if duration > max_duration:
            raise AudioFormatError(f"Audio too long: {duration:.2f}s (max {max_duration}s)")
