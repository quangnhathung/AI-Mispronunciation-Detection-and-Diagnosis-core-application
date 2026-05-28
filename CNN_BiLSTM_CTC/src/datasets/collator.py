from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class Collator:
    def __init__(
        self,
        pad_token_id: int = 0,
        audio_pad_value: float = 0.0,
        max_audio_length: int = 480000,
        max_phoneme_length: int = 500,
    ):
        self.pad_token_id = pad_token_id
        self.audio_pad_value = audio_pad_value
        self.max_audio_length = max_audio_length
        self.max_phoneme_length = max_phoneme_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        feat_is_2d = batch[0]["audio"].dim() == 2
        if feat_is_2d:
            audios = [item["audio"].t() for item in batch]
        else:
            audios = [item["audio"].squeeze(0).t() for item in batch]

        phonemes = [item["phonemes"] for item in batch]
        feature_lengths = [item.get("feature_length", item["audio"].size(-1)) for item in batch]
        phoneme_lengths = [item["phoneme_length"] for item in batch]

        audio_padded = pad_sequence(audios, batch_first=True, padding_value=self.audio_pad_value)
        audio_padded = audio_padded.permute(0, 2, 1).contiguous()

        phoneme_padded = pad_sequence(phonemes, batch_first=True, padding_value=self.pad_token_id)

        feature_lengths_tensor = torch.tensor(feature_lengths, dtype=torch.long)
        phoneme_lengths_tensor = torch.tensor(phoneme_lengths, dtype=torch.long)

        return {
            "audio": audio_padded,
            "phonemes": phoneme_padded,
            "audio_lengths": feature_lengths_tensor,
            "phoneme_lengths": phoneme_lengths_tensor,
            "speakers": [item.get("speaker", "") for item in batch],
            "utterance_ids": [item.get("utterance_id", "") for item in batch],
            "wav_paths": [item.get("wav_path", "") for item in batch],
        }

    def collate_inference(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        audios = [item["audio"].squeeze(0) for item in batch]
        audio_lengths = [item["audio_length"] for item in batch]
        audio_padded = pad_sequence(audios, batch_first=True, padding_value=self.audio_pad_value)

        return {
            "audio": audio_padded,
            "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
            "speakers": [item.get("speaker", "") for item in batch],
            "utterance_ids": [item.get("utterance_id", "") for item in batch],
            "wav_paths": [item.get("wav_path", "") for item in batch],
        }
