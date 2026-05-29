import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError
from app.models.normalize import safe_tensor_to_list, safe_item, safe_index

_cnn_root = Path(__file__).parent.parent.parent / "CNN_BiLSTM_CTC"
_CNN_ROOT_STR = str(_cnn_root.resolve())
_PROJECT_ROOT = str(_cnn_root.parent.resolve())

logger.debug(f"[CNN] CNN root: {_CNN_ROOT_STR}")
logger.debug(f"[CNN] Project root: {_PROJECT_ROOT}")

for _p in [_PROJECT_ROOT, _CNN_ROOT_STR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _key in list(sys.modules.keys()):
    if _key.startswith("src.") or _key == "src":
        sys.modules.pop(_key, None)


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
    phoneme_set_size = 73
    phoneme_set = [
        "<blank>", "<unk>",
        "AA0", "AA1", "AA2", "AE0", "AE1", "AE2",
        "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
        "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
        "B", "CH", "D", "DH",
        "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
        "EY0", "EY1", "EY2",
        "F", "G",
        "HH",
        "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
        "JH",
        "K", "L", "M", "N", "NG",
        "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
        "P", "R", "S", "SH",
        "SIL", "SP",
        "T", "TH",
        "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
        "V", "W", "Y", "Z", "ZH",
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
        self.g2p = None

    def _ensure_path(self):
        root_str = str(_cnn_root.resolve())
        project_str = str(_cnn_root.parent.resolve())
        other_roots = [p for p in sys.path if p != root_str and p != project_str and Path(p).name in ("Wav2vec2", "DAB_Transformer_Project")]
        for p in other_roots:
            sys.path.remove(p)
        for _p in [project_str, root_str]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        for _k in list(sys.modules.keys()):
            if _k.startswith("CNN_BiLSTM_CTC.src"):
                sys.modules.pop(_k, None)
        logger.debug(f"[CNN] _ensure_path: root={root_str} project={project_str}")

    def load_model(self) -> None:
        try:
            import torch
            self._ensure_path()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            from CNN_BiLSTM_CTC.src.features.audio_processing import AudioProcessor
            from CNN_BiLSTM_CTC.src.features.mel_spec import MelFeatureExtractor
            from CNN_BiLSTM_CTC.src.datasets.tokenizer import PhonemeTokenizer
            from CNN_BiLSTM_CTC.src.decoders.greedy import GreedyDecoder
            from CNN_BiLSTM_CTC.src.utils.config import get_config
            from CNN_BiLSTM_CTC.src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
            from CNN_BiLSTM_CTC.src.utils.helpers import load_checkpoint

            logger.debug("[CNN] All imports resolved successfully")

            cfg = None
            if self.config_path and self.config_path.exists():
                cfg = get_config(str(self.config_path))
            else:
                config_file = _cnn_root / "configs" / "config.yaml"
                if config_file.exists():
                    cfg = get_config(str(config_file))

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            self.tokenizer = PhonemeTokenizer()
            vocab_size = len(self.tokenizer)
            blank_id = self.tokenizer.blank_id

            model_cfg = cfg.model if cfg and hasattr(cfg, 'model') else {}
            def _mc(key: str, default):
                if isinstance(model_cfg, dict):
                    return model_cfg.get(key, default)
                return getattr(model_cfg, key, default)
            self.model = CNNBiLSTMCTC(
                input_dim=_mc('input_dim', 80),
                cnn_channels=_mc('cnn_channels', [64, 128, 256]),
                cnn_kernel_sizes=_mc('cnn_kernel_sizes', [3, 3, 3]),
                cnn_strides=_mc('cnn_strides', [2, 2, 2]),
                cnn_dropout=_mc('cnn_dropout', 0.2),
                rnn_hidden_size=_mc('rnn_hidden_size', 256),
                rnn_num_layers=_mc('rnn_num_layers', 4),
                rnn_dropout=_mc('rnn_dropout', 0.3),
                rnn_bidirectional=_mc('rnn_bidirectional', True),
                vocab_size=vocab_size,
            )

            checkpoint = load_checkpoint(str(self.checkpoint_path), self.model, device=self.device)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded epoch: {checkpoint.get('epoch', 'unknown')}")

            self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)
            self.processor = MelFeatureExtractor(
                sample_rate=self.sample_rate,
                n_mels=80,
                n_fft=512,
                win_length=400,
                hop_length=160,
            )
            self.decoder = GreedyDecoder(blank_id=blank_id)

            try:
                from g2p_en import G2p
                self.g2p = G2p()
            except ImportError:
                logger.warning("g2p_en not installed, text-to-phoneme conversion disabled")
                self.g2p = None

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
        self.g2p = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("CNN-BiLSTM-CTC model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        if self.audio_processor is None or self.processor is None:
            raise PreprocessError("Model not loaded. Call load_model() first.")
        try:
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            processed = self.audio_processor.process(audio_tensor, sample_rate)
            mel_spec = self.processor(processed)
            batch, n_mels, time = mel_spec.shape
            return mel_spec.to(self.device)
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
            logger.error(f"CNN predict failed | features shape: {getattr(features, 'shape', 'N/A')} | error type: {type(e).__name__} | {e}", exc_info=True)
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

            logits_tensor = torch.from_numpy(logits_np).float()
            decoded_with_conf = self.decoder.decode_with_confidence(logits_tensor)
            if decoded_with_conf:
                decoded_seq = decoded_with_conf[0][0]
                decoded_confs = decoded_with_conf[0][1]
                conf = decoded_with_conf[0][2]
            else:
                decoded_seq = []
                decoded_confs = []
                conf = 0.0
            _id2ph = self.tokenizer._id_to_phoneme
            decoded_seq_str = [_id2ph[idx] for idx in decoded_seq]
            phoneme_string = " ".join(decoded_seq_str)

            result = InferenceResult(
                phoneme_sequence=decoded_seq_str,
                phoneme_string=phoneme_string,
                overall_confidence=float(conf),
            )

            predictions: list[PhonemePrediction] = []
            for ph, cnf in zip(decoded_seq_str, decoded_confs):
                predictions.append(
                    PhonemePrediction(
                        phoneme=ph,
                        status="unknown",
                        confidence=float(cnf),
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
                tokenizer = self.tokenizer

                if self.g2p:
                    raw_phones = self.g2p(text)
                    phoneme_tokens = [p.strip().upper() for p in raw_phones if p.strip()]
                else:
                    phoneme_tokens = text.upper().split()

                valid_tokens = []
                for ph in phoneme_tokens:
                    key = ph.upper() if not (ph.startswith("<") and ph.endswith(">")) else ph
                    if key in tokenizer._phoneme_to_id and tokenizer._phoneme_to_id[key] != tokenizer.unk_id:
                        valid_tokens.append(ph)

                if valid_tokens:
                    phoneme_tokens = valid_tokens
                else:
                    phoneme_tokens = phoneme_tokens[:1]

                target_ids = safe_tensor_to_list(tokenizer.encode(phoneme_tokens), desc="target_ids")
                logger.debug(f"[CNN] text='{text}' -> g2p={phoneme_tokens} -> ids={target_ids}")
                _id2ph = tokenizer._id_to_phoneme
                target_phonemes = [_id2ph[safe_index(idx)] for idx in target_ids if safe_index(idx) not in (tokenizer.blank_id, tokenizer.unk_id)]
                logger.debug(f"[CNN] target_phonemes={target_phonemes}")

                from CNN_BiLSTM_CTC.src.mdd.detector import MispronunciationDetector
                detector = MispronunciationDetector(tokenizer=tokenizer)
                feedback = detector.detect(
                    predicted_ids=decoded_seq,
                    target_ids=target_ids,
                    confidences=decoded_confs,
                )
                logger.debug(f"[CNN] feedback type={type(feedback).__name__} accuracy={feedback.accuracy}")

                predictions = []
                for ph, cnf in zip(feedback.correct_phonemes, feedback.correct_confidences):
                    predictions.append(PhonemePrediction(
                        phoneme=ph, status="correct", confidence=float(cnf),
                        expected=ph, actual=ph,
                    ))
                for (exp, pred, pos), cnf in zip(feedback.substitutions, feedback.substitution_confidences):
                    predictions.append(PhonemePrediction(
                        phoneme=exp, status="substitution", confidence=float(cnf),
                        expected=exp, actual=pred,
                        reason=f"Nhầm /{exp}/ thành /{pred}/",
                    ))
                for ph, cnf in zip(feedback.deletions, feedback.deletions_confidences):
                    predictions.append(PhonemePrediction(
                        phoneme=ph, status="deletion", confidence=float(cnf),
                        expected=ph, actual=None,
                        reason=f"Bạn bị nuốt âm /{ph}/",
                    ))
                for ph, cnf in zip(feedback.insertions, feedback.insertions_confidences):
                    predictions.append(PhonemePrediction(
                        phoneme="-", status="insertion", confidence=float(cnf),
                        expected=None, actual=ph,
                        reason=f"Phát âm thừa âm /{ph}/",
                    ))

                summary = InferenceSummary(
                    total_phonemes=feedback.total_phonemes,
                    correct_phonemes=len(feedback.correct_phonemes),
                    incorrect_phonemes=feedback.error_count,
                    accuracy=max(0.0, min(1.0, feedback.accuracy)),
                )

            return predictions, result, summary
        except Exception as e:
            raise InferenceError(f"CNN postprocessing failed: {e}") from e
