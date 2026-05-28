from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from src.decoders.greedy import GreedyDecoder
from src.decoders.beam_search import BeamSearchDecoder
from src.features.mel_spec import MelFeatureExtractor
from src.mdd.detector import MispronunciationDetector


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: Any,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.feature_extractor = MelFeatureExtractor(
            sample_rate=config.data.sample_rate,
            n_fft=config.data.n_fft,
            win_length=config.data.win_length,
            hop_length=config.data.hop_length,
            n_mels=config.data.n_mels,
        ).to(device)

        if getattr(config.inference, "use_beam_search", False):
            self.decoder = BeamSearchDecoder(
                blank_id=tokenizer.blank_id,
                beam_width=getattr(config.inference, "beam_width", 10),
            )
        else:
            self.decoder = GreedyDecoder(blank_id=tokenizer.blank_id)

        self.mdd = MispronunciationDetector(tokenizer)

        self.model.eval()

    def predict_audio(self, waveform: torch.Tensor) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            waveform = waveform.to(self.device)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            features = self.feature_extractor(waveform)
            log_probs = self.model(features)
            input_lengths = torch.tensor([log_probs.size(1)], device=self.device)

            decoded = self.decoder.decode_with_confidence(log_probs, input_lengths)
            pred_ids, confidence = decoded[0]
            pred_phonemes = self.tokenizer.decode(pred_ids)

        return {
            "predicted_ids": pred_ids,
            "predicted_phonemes": pred_phonemes,
            "confidence": confidence,
            "log_probs": log_probs.cpu(),
        }

    def predict_with_gt(
        self,
        waveform: torch.Tensor,
        target_phonemes: List[str],
        utterance_id: str = "",
        speaker: str = "",
    ) -> Dict[str, Any]:
        pred_result = self.predict_audio(waveform)
        target_ids = self.tokenizer.encode(target_phonemes).tolist()
        feedback = self.mdd.detect(
            predicted_ids=pred_result["predicted_ids"],
            target_ids=target_ids,
            utterance_id=utterance_id,
            speaker=speaker,
        )

        return {
            "predicted_phonemes": pred_result["predicted_phonemes"],
            "target_phonemes": target_phonemes,
            "confidence": pred_result["confidence"],
            "feedback": feedback,
        }

    def predict_file(self, audio_path: str) -> Dict[str, Any]:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.config.data.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.data.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return self.predict_audio(waveform)

    def predict_batch(
        self, waveforms: List[torch.Tensor]
    ) -> List[Dict[str, Any]]:
        results = []
        for wav in waveforms:
            result = self.predict_audio(wav)
            results.append(result)
        return results

    def predict_microphone(self, duration: float = 5.0, sample_rate: int = 16000) -> Dict[str, Any]:
        try:
            import sounddevice as sd
            logger.info(f"Recording for {duration} seconds...")
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            logger.info("Recording finished")
            waveform = torch.from_numpy(audio.T)
            return self.predict_audio(waveform)
        except ImportError:
            logger.error("sounddevice not installed. Install with: pip install sounddevice")
            return {"error": "sounddevice not available"}

    def get_forced_alignment(
        self,
        waveform: torch.Tensor,
        target_phonemes: List[str],
    ) -> List[Dict[str, Any]]:
        pred_result = self.predict_audio(waveform)
        target_ids = self.tokenizer.encode(target_phonemes)
        feedback = self.mdd.detect(
            predicted_ids=pred_result["predicted_ids"],
            target_ids=target_ids.tolist(),
        )

        alignment = []
        phoneme_dur = 1.0 / max(len(target_phonemes), 1)
        for i, (ph, is_correct) in enumerate(
            zip(target_phonemes, [
                ph not in [s[0] for s in feedback.substitutions]
                and ph not in feedback.deletions
                for ph in target_phonemes
            ])
        ):
            alignment.append({
                "phoneme": ph,
                "start": i * phoneme_dur,
                "end": (i + 1) * phoneme_dur,
                "correct": is_correct,
            })

        return alignment
