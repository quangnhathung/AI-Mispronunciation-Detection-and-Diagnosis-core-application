from __future__ import annotations

from typing import Optional

import torch
import torchaudio


class MelFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
        power: float = 2.0,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = "slaney",
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            center=center,
            pad_mode=pad_mode,
            norm=norm,
            mel_scale=mel_scale,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(waveform)
        mel_db = self.amplitude_to_db(mel)
        return mel_db

    def extra_repr(self) -> str:
        return (
            f"sample_rate={self.sample_rate}, n_fft={self.n_fft}, "
            f"win_length={self.win_length}, hop_length={self.hop_length}, "
            f"n_mels={self.n_mels}"
        )
