import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError


class DABTransformerModel(BaseModelWrapper):
    name = "dab_transformer"
    display_name = "DAB-Transformer"
    version = "1.0.0"
    description = "Dynamic Attention Bias Transformer with CTC for phoneme recognition"
    architecture = "Conv1d + Transformer + CTC"
    task = "ASR phoneme recognition + MDD"
    requires_text = False
    requires_gpu = False
    sample_rate = 16000
    phoneme_set_size = 41
    phoneme_set = [
        "<blank>", "<space>", "AA", "AE", "AH", "AO", "AW", "AY",
        "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
        "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
        "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W",
        "Y", "Z", "ZH",
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path)
        self.model = None
        self.device = None
        self.config = None
        self.text_processor = None

    def load_model(self) -> None:
        try:
            import torch
            dab_root = Path(__file__).parent.parent.parent / "DAB_Transformer_Project"
            if dab_root not in [Path(p) for p in sys.path]:
                sys.path.insert(0, str(dab_root))

            from config import Config
            from model import DAB_Transformer
            from utils import TextProcess

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.config = Config()

            vocab_size = self.config.OUTPUT_SIZE
            self.model = DAB_Transformer(
                input_dim=self.config.INPUT_DIM,
                d_model=self.config.D_MODEL,
                nhead=self.config.NHEAD,
                num_layers=self.config.NUM_LAYERS,
                dim_feedforward=self.config.DIM_FF,
                dropout=self.config.DROPOUT,
                vocab_size=vocab_size,
                max_len=self.config.MAX_LEN,
            )

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"DAB-Transformer loaded epoch: {checkpoint.get('epoch', 'unknown')}")

            self.text_processor = TextProcess()
            self._loaded = True
            logger.info(f"DAB-Transformer model loaded: {self.checkpoint_path}")
        except Exception as e:
            self._loaded = False
            raise ModelNotLoadedError(f"Failed to load DAB-Transformer model: {e}") from e

    def unload_model(self) -> None:
        self.model = None
        self.text_processor = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("DAB-Transformer model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        if self.config is None:
            raise PreprocessError("Model not loaded")
        try:
            if sample_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            max_samples = self.config.MAX_SAMPLES
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)))
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            return audio_tensor.to(self.device)
        except Exception as e:
            raise PreprocessError(f"DAB preprocessing failed: {e}") from e

    def predict(self, features: Any, **kwargs) -> Any:
        import torch
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            with torch.no_grad():
                logits = self.model(features)
            return logits
        except Exception as e:
            raise InferenceError(f"DAB inference failed: {e}") from e

    def postprocess(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]:
        if self.text_processor is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            import torch
            logits = raw_output
            text = kwargs.get("text")
            threshold = kwargs.get("threshold", 0.5)

            if isinstance(logits, torch.Tensor):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = logits

            from utils import greedy_decoder
            decoded_ids = greedy_decoder(logits_np, blank=self.config.BLANK_ID)
            phoneme_list = [self.config.ID_PHONEME[idx] for idx in decoded_ids if idx not in (self.config.BLANK_ID, self.config.PAD_ID)]
            phoneme_string = " ".join(phoneme_list)

            confidences = []
            for i, idx in enumerate(decoded_ids):
                if len(logits_np.shape) == 3:
                    step_probs = logits_np[0, i, :]
                else:
                    step_probs = logits_np[i, :]
                confidences.append(float(np.max(step_probs)))

            overall_conf = float(np.mean(confidences)) if confidences else 0.0
            result = InferenceResult(
                phoneme_sequence=phoneme_list,
                phoneme_string=phoneme_string,
                overall_confidence=overall_conf,
            )

            predictions: list[PhonemePrediction] = []
            if text and self.text_processor:
                from jiwer import process_words
                target_phonemes = self.text_processor.text_to_phoneme(text)
                target_str = " ".join(target_phonemes)
                pred_str = phoneme_string
                alignment = process_words(target_str, pred_str)

                ref_ph = target_str.split()
                hyp_ph = pred_str.split() if pred_str else []

                ref_idx = 0
                hyp_idx = 0
                for hit in alignment.hits:
                    predictions.append(PhonemePrediction(phoneme=ref_ph[ref_idx], status="correct", confidence=1.0, expected=ref_ph[ref_idx], actual=ref_ph[ref_idx]))
                    ref_idx += 1
                    hyp_idx += 1
                for sub in alignment.substitutions:
                    predictions.append(PhonemePrediction(phoneme=ref_ph[ref_idx], status="substitution", confidence=0.0, expected=ref_ph[ref_idx], actual=hyp_ph[hyp_idx], reason=f"Nhầm /{ref_ph[ref_idx]}/ thành /{hyp_ph[hyp_idx]}/"))
                    ref_idx += 1
                    hyp_idx += 1
                for ins in alignment.insertions:
                    predictions.append(PhonemePrediction(phoneme=ins, status="insertion", confidence=0.0, expected=None, actual=ins, reason=f"Phát âm thừa âm /{ins}/"))
                    hyp_idx += 1
                for d_idx in range(len(alignment.deletions)):
                    predictions.append(PhonemePrediction(phoneme=ref_ph[ref_idx], status="deletion", confidence=0.0, expected=ref_ph[ref_idx], actual=None, reason=f"Bạn bị nuốt âm /{ref_ph[ref_idx]}/"))
                    ref_idx += 1

                correct = sum(1 for p in predictions if p.status == "correct")
                summary = InferenceSummary(
                    total_phonemes=len(predictions),
                    correct_phonemes=correct,
                    incorrect_phonemes=len(predictions) - correct,
                    accuracy=correct / len(predictions) if predictions else 0.0,
                )
            else:
                summary = InferenceSummary(
                    total_phonemes=len(phoneme_list),
                    correct_phonemes=0,
                    incorrect_phonemes=0,
                    accuracy=0.0,
                )

            return predictions, result, summary
        except Exception as e:
            raise InferenceError(f"DAB postprocessing failed: {e}") from e
