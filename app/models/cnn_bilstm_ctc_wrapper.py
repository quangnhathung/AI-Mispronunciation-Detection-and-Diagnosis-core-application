import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError


class CNNBiLSTMCTCModel(BaseModelWrapper):
    name = "cnn_bilstm_ctc"
    display_name = "CNN-BiLSTM-CTC"
    version = "1.0.0"
    description = "CNN-BiLSTM with CTC loss for phoneme recognition and mispronunciation detection"
    architecture = "CNN + BiLSTM + CTC"
    task = "ASR phoneme recognition + MDD"
    requires_text = False
    requires_gpu = False
    sample_rate = 16000
    phoneme_set_size = 42
    phoneme_set = [
        "<blank>", "<unk>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2",
        "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2",
        "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0", "EH1", "EH2",
        "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH",
        "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
    ]

    def __init__(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
        super().__init__(checkpoint_path)
        self.config_path = Path(config_path) if config_path else None
        self.model = None
        self.device = None
        self.processor = None
        self.tokenizer = None
        self.decoder = None
        self.audio_processor = None

    def load_model(self) -> None:
        try:
            import torch
            cnn_root = Path(__file__).parent.parent.parent / "CNN_BiLSTM_CTC"
            if cnn_root not in [Path(p) for p in sys.path]:
                sys.path.insert(0, str(cnn_root))

            from src.features.audio_processing import AudioProcessor
            from src.features.mel_spec import MelFeatureExtractor
            from src.datasets.tokenizer import PhonemeTokenizer
            from src.decoders.greedy import GreedyDecoder
            from src.utils.config import load_config

            from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
            from src.utils.helpers import load_checkpoint

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            cfg = None
            if self.config_path and self.config_path.exists():
                cfg = load_config(self.config_path)
            else:
                config_file = cnn_root / "configs" / "config.yaml"
                if config_file.exists():
                    cfg = load_config(config_file)

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            self.tokenizer = PhonemeTokenizer()
            vocab_size = len(self.tokenizer)

            model_cfg = cfg.model if cfg else {}
            self.model = CNNBiLSTMCTC(
                input_size=model_cfg.get("input_size", 80),
                cnn_channels=model_cfg.get("cnn_channels", [64, 128, 256]),
                rnn_hidden=model_cfg.get("rnn_hidden", 256),
                rnn_layers=model_cfg.get("rnn_layers", 4),
                vocab_size=vocab_size,
                dropout=model_cfg.get("dropout", 0.2),
            )

            checkpoint = load_checkpoint(str(self.checkpoint_path), self.model, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded epoch: {checkpoint.get('epoch', 'unknown')}")

            self.audio_processor = AudioProcessor(target_sr=self.sample_rate)
            self.processor = MelFeatureExtractor(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=512,
                win_length=400,
                hop_length=160,
            )
            self.decoder = GreedyDecoder(tokenizer=self.tokenizer)
            self._loaded = True
            logger.info(f"CNN-BiLSTM-CTC model loaded: {self.checkpoint_path}")
        except Exception as e:
            self._loaded = False
            raise ModelNotLoadedError(f"Failed to load CNN-BiLSTM-CTC model: {e}") from e

    def unload_model(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.decoder = None
        self.audio_processor = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("CNN-BiLSTM-CTC model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        if self.audio_processor is None or self.processor is None:
            raise PreprocessError("Model not loaded. Call load_model() first.")
        try:
            audio_processed = self.audio_processor(audio, sample_rate)
            mel_spec = self.processor(audio_processed)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
            return mel_tensor.to(self.device)
        except Exception as e:
            raise PreprocessError(f"CNN preprocessing failed: {e}") from e

    def predict(self, features: Any, **kwargs) -> Any:
        import torch
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            with torch.no_grad():
                logits = self.model(features)
            return logits
        except Exception as e:
            raise InferenceError(f"CNN inference failed: {e}") from e

    def postprocess(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]:
        if self.decoder is None or self.tokenizer is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            import torch
            logits = raw_output
            text = kwargs.get("text")

            if isinstance(logits, torch.Tensor):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = logits

            decoded_seq, conf = self.decoder.decode(logits_np)
            phoneme_string = " ".join(decoded_seq)

            result = InferenceResult(
                phoneme_sequence=decoded_seq,
                phoneme_string=phoneme_string,
                overall_confidence=float(conf),
            )

            predictions: list[PhonemePrediction] = []
            for ph in decoded_seq:
                predictions.append(
                    PhonemePrediction(
                        phoneme=ph,
                        status="unknown",
                        confidence=0.0,
                        reason="ASR output, no alignment available without ground truth",
                    )
                )

            summary = InferenceSummary(
                total_phonemes=len(decoded_seq),
                correct_phonemes=0,
                incorrect_phonemes=0,
                accuracy=0.0,
            )

            if text:
                from src.mdd.detector import MispronunciationDetector
                from src.datasets.tokenizer import PhonemeTokenizer
                tokenizer = self.tokenizer
                target_ids = tokenizer.encode(text)
                target_phonemes = [tokenizer.idx_to_phoneme[idx] for idx in target_ids if idx > 0]

                detector = MispronunciationDetector(tokenizer=tokenizer)
                feedback = detector.detect(target_phonemes, decoded_seq, logits_np)

                predictions = []
                for item in feedback.to_dict():
                    predictions.append(
                        PhonemePrediction(
                            phoneme=item.get("target", ""),
                            status=item.get("type", "unknown"),
                            confidence=item.get("confidence", 0.0),
                            expected=item.get("target"),
                            actual=item.get("predicted"),
                            reason=item.get("message"),
                        )
                    )

                summary = InferenceSummary(
                    total_phonemes=len(predictions),
                    correct_phonemes=sum(1 for p in predictions if p.status == "correct"),
                    incorrect_phonemes=sum(1 for p in predictions if p.status != "correct"),
                    accuracy=feedback.accuracy if hasattr(feedback, "accuracy") else 0.0,
                )

            return predictions, result, summary
        except Exception as e:
            raise InferenceError(f"CNN postprocessing failed: {e}") from e
