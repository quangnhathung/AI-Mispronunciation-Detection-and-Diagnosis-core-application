from __future__ import annotations

from typing import Optional

import torch
import torchaudio
from torchaudio import transforms as T


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        p: float = 0.5,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        if torch.rand(1).item() > self.p:
            return spec
        x = spec
        for _ in range(self.n_freq_masks):
            x = self.freq_masking(x)
        for _ in range(self.n_time_masks):
            x = self.time_masking(x)
        return x


class TimeStretchAugment(torch.nn.Module):
    def __init__(self, min_rate: float = 0.9, max_rate: float = 1.1, p: float = 0.3):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return waveform
        rate = self.min_rate + torch.rand(1).item() * (self.max_rate - self.min_rate)
        try:
            import librosa
            wav_np = waveform.squeeze(0).numpy()
            stretched = librosa.effects.time_stretch(wav_np, rate=rate)
            return torch.from_numpy(stretched).unsqueeze(0)
        except ImportError:
            return waveform


class NoiseInjector(torch.nn.Module):
    def __init__(self, noise_level: float = 0.005, p: float = 0.3):
        super().__init__()
        self.noise_level = noise_level
        self.p = p

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return waveform
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise


class AudioAugmentationPipeline(torch.nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        noise_level: float = 0.005,
        time_stretch: bool = False,
        spec_augment_p: float = 0.5,
        noise_p: float = 0.3,
    ):
        super().__init__()
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            p=spec_augment_p,
        )
        self.noise = NoiseInjector(noise_level=noise_level, p=noise_p)
        self.time_stretch = time_stretch
        if time_stretch:
            self.stretch = TimeStretchAugment(p=0.3)

    def forward(self, spec: torch.Tensor, waveform: Optional[torch.Tensor] = None) -> torch.Tensor:
        if waveform is not None and self.time_stretch:
            waveform = self.stretch(waveform)
        if waveform is not None:
            waveform = self.noise(waveform)
        x = spec
        x = self.spec_augment(x)
        return x
