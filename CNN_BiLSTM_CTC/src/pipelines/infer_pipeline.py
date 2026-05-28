from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from src.datasets.tokenizer import PhonemeTokenizer
from src.inference.predictor import Predictor
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
from src.utils.config import Config
from src.utils.helpers import load_checkpoint


class InferPipeline:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            getattr(config.inference, "device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.tokenizer = PhonemeTokenizer(include_stress=True)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        self.model = CNNBiLSTMCTC(
            input_dim=self.config.model.input_dim,
            vocab_size=self.tokenizer.vocab_size,
            cnn_channels=self.config.model.cnn_channels,
            cnn_kernel_sizes=self.config.model.cnn_kernel_sizes,
            cnn_strides=self.config.model.cnn_strides,
            cnn_dropout=self.config.model.cnn_dropout,
            rnn_hidden_size=self.config.model.rnn_hidden_size,
            rnn_num_layers=self.config.model.rnn_num_layers,
            rnn_dropout=self.config.model.rnn_dropout,
            rnn_bidirectional=self.config.model.rnn_bidirectional,
        ).to(self.device)

        load_checkpoint(self.checkpoint_path, self.model, device=self.device)
        logger.info(f"Loaded checkpoint from {self.checkpoint_path}")

        self.predictor = Predictor(
            model=self.model,
            tokenizer=self.tokenizer,
            config=config,
            device=self.device,
        )

    def infer_file(self, audio_path: str) -> Dict[str, Any]:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        result = self.predictor.predict_file(audio_path)
        phonemes = self.tokenizer.decode_to_string(result["predicted_ids"])
        logger.info(f"Predicted phonemes: {phonemes}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        return {
            "phoneme_sequence": result["predicted_phonemes"],
            "phoneme_string": phonemes,
            "confidence": result["confidence"],
        }

    def infer_with_ground_truth(
        self,
        audio_path: str,
        target_phonemes: List[str],
        utterance_id: str = "",
        speaker: str = "",
    ) -> Dict[str, Any]:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.config.data.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.data.sample_rate)
            waveform = resampler(waveform)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        result = self.predictor.predict_with_gt(
            waveform=waveform,
            target_phonemes=target_phonemes,
            utterance_id=utterance_id,
            speaker=speaker,
        )

        print("\n" + result["feedback"].colored_output())
        return result

    def infer_microphone(self, duration: float = 5.0) -> Dict[str, Any]:
        result = self.predictor.predict_microphone(duration=duration)
        if "error" in result:
            logger.error(result["error"])
            return result
        phonemes = self.tokenizer.decode_to_string(result["predicted_ids"])
        logger.info(f"Predicted phonemes: {phonemes}")
        return {
            "phoneme_sequence": result["predicted_phonemes"],
            "phoneme_string": phonemes,
            "confidence": result["confidence"],
        }

    def batch_infer(self, audio_dir: str) -> List[Dict[str, Any]]:
        audio_dir_path = Path(audio_dir)
        audio_files = list(audio_dir_path.glob("*.wav")) + list(audio_dir_path.glob("*.mp3"))
        results = []
        for audio_file in audio_files:
            try:
                result = self.infer_file(str(audio_file))
                result["file"] = str(audio_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to infer {audio_file}: {e}")
        return results
