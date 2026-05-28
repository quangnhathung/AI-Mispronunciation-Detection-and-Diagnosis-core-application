from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
from loguru import logger

from src.datasets.tokenizer import PhonemeTokenizer


class L2ArcticDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: PhonemeTokenizer,
        sample_rate: int = 16000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
        feature_fn: Optional[Callable] = None,
        augmentation_fn: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.feature_fn = feature_fn
        self.augmentation_fn = augmentation_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()]

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loaded {len(self.data)} utterances from {manifest_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _get_cache_path(self, idx: int) -> Optional[Path]:
        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / f"utt_{idx}.pt"
        return cache_path if cache_path.exists() else None

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_path = self._get_cache_path(idx)
        if cache_path:
            return torch.load(cache_path, weights_only=False)

        item = self.data[idx]
        wav_path = item["wav_path"]
        phonemes = item["phonemes"]

        waveform = self._load_audio(wav_path)

        audio_length = waveform.size(1) / self.sample_rate
        if audio_length > self.max_audio_length:
            max_samples = int(self.max_audio_length * self.sample_rate)
            waveform = waveform[:, :max_samples]
        if audio_length < self.min_audio_length:
            pad_samples = int(self.min_audio_length * self.sample_rate) - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_samples))

        phoneme_ids = self.tokenizer.encode(phonemes)
        audio_length = waveform.size(1)
        phoneme_length = len(phoneme_ids)

        features = waveform
        if self.feature_fn:
            features = self.feature_fn(waveform)

        if self.augmentation_fn and self.augmentation_fn.training:
            features = self.augmentation_fn(features, waveform)

        feature_length = features.size(-1)

        sample = {
            "audio": features,
            "phonemes": phoneme_ids,
            "audio_length": audio_length,
            "feature_length": feature_length,
            "phoneme_length": phoneme_length,
            "speaker": item.get("speaker", ""),
            "utterance_id": item.get("utterance_id", ""),
            "wav_path": wav_path,
        }

        if self.use_cache and self.cache_dir and not cache_path:
            torch.save(sample, self.cache_dir / f"utt_{idx}.pt")

        return sample

    def get_phoneme_sequence(self, idx: int) -> List[str]:
        return self.data[idx].get("phonemes", [])

    def get_transcript(self, idx: int) -> str:
        return self.data[idx].get("transcript", "")

    def get_speaker(self, idx: int) -> str:
        return self.data[idx].get("speaker", "")
