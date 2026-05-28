from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
from loguru import logger

from src.datasets.l2arctic_parser import L2ArcticParser
from src.features.mel_spec import MelFeatureExtractor
from src.preprocessing.manifest import ManifestBuilder


class L2ArcticPreprocessor:
    def __init__(
        self,
        data_dir: str,
        processed_dir: str,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.parser = L2ArcticParser(str(self.data_dir))
        self.feature_extractor = MelFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def preprocess_speaker(
        self,
        speaker: str,
        force_recompute: bool = False,
    ) -> int:
        speaker_out_dir = self.processed_dir / speaker
        speaker_out_dir.mkdir(parents=True, exist_ok=True)

        utterances = self.parser.load_all_utterances(speaker)
        processed_count = 0

        for utt in utterances:
            feat_path = speaker_out_dir / f"{utt['utterance_id']}.pt"
            if feat_path.exists() and not force_recompute:
                continue

            try:
                waveform, sr = torchaudio.load(utt["wav_path"])
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                features = self.feature_extractor(waveform)

                phoneme_ids = self._phonemes_to_ids(utt["phonemes"])

                torch.save(
                    {
                        "features": features,
                        "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long),
                        "audio_length": waveform.size(1),
                        "phoneme_length": len(phoneme_ids),
                        "transcript": utt["transcript"],
                        "speaker": speaker,
                        "utterance_id": utt["utterance_id"],
                    },
                    feat_path,
                )
                processed_count += 1
            except Exception as e:
                logger.warning(f"Failed to process {utt['utterance_id']}: {e}")

        logger.info(
            f"Preprocessed {speaker}: {processed_count}/{len(utterances)} utterances"
        )
        return processed_count

    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        vocab = {
            "SIL": 0, "AA": 1, "AE": 2, "AH": 3, "AO": 4, "AW": 5,
            "AY": 6, "B": 7, "CH": 8, "D": 9, "DH": 10, "EH": 11,
            "ER": 12, "EY": 13, "F": 14, "G": 15, "HH": 16, "IH": 17,
            "IY": 18, "JH": 19, "K": 20, "L": 21, "M": 22, "N": 23,
            "NG": 24, "OW": 25, "OY": 26, "P": 27, "R": 28, "S": 29,
            "SH": 30, "T": 31, "TH": 32, "UH": 33, "UW": 34, "V": 35,
            "W": 36, "Y": 37, "Z": 38, "ZH": 39, "SP": 40,
        }
        return [vocab.get(p.upper(), 0) for p in phonemes]

    def preprocess_all(self, force_recompute: bool = False) -> Dict[str, int]:
        results = {}
        for speaker in self.parser.get_speakers():
            count = self.preprocess_speaker(speaker, force_recompute)
            results[speaker] = count
        return results
