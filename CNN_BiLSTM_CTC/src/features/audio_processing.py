from __future__ import annotations

from typing import Optional, Tuple

import torch
import torchaudio


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        trim_silence: bool = True,
        trim_top_db: float = 20.0,
        normalize: bool = True,
        target_dB: float = -3.0,
    ):
        self.sample_rate = sample_rate
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.normalize = normalize
        self.target_dB = target_dB

    def load(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        return self.process(waveform, sr)

    def process(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if self.trim_silence:
            waveform = self._trim_silence(waveform)
        if self.normalize:
            waveform = self._normalize(waveform)
        return waveform

    def _trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform_np = waveform.squeeze(0).numpy()
        try:
            import librosa
            waveform_trimmed, _ = librosa.effects.trim(
                waveform_np, top_db=self.trim_top_db
            )
            return torch.from_numpy(waveform_trimmed).unsqueeze(0)
        except ImportError:
            return waveform

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.abs().max() > 0:
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > 0:
                target_rms = 10 ** (self.target_dB / 20)
                waveform = waveform * (target_rms / rms)
            peak = waveform.abs().max()
            if peak > 0.99:
                waveform = waveform * (0.99 / peak)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            return resampler(waveform)
        return waveform
