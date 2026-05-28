import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError
from app.models.normalize import safe_tensor_to_list, safe_item

_dab_root = Path(__file__).parent.parent.parent / "DAB_Transformer_Project"
_DAB_ROOT_STR = str(_dab_root.resolve())
_DAB_PROJECT_ROOT = str(_dab_root.parent.resolve())

logger.debug(f"[DAB] Module root: {_DAB_ROOT_STR}")

for _p in [_DAB_PROJECT_ROOT, _DAB_ROOT_STR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _key in list(sys.modules.keys()):
    if _key.startswith("src.") or _key == "src":
        sys.modules.pop(_key, None)


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
        "<blank>", " ", "AA", "AE", "AH", "AO", "AW", "AY",
        "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
        "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
        "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W",
        "Y", "Z", "ZH",
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path)
        self.model = None
        self.device = None
        self.text_process = None
        self.num_classes = len(self.phoneme_set)

    def _ensure_path(self):
        root_str = str(_dab_root.resolve())
        project_str = str(_dab_root.parent.resolve())
        other_roots = [p for p in sys.path if p != root_str and p != project_str and Path(p).name in ("CNN_BiLSTM_CTC", "Wav2vec2")]
        for p in other_roots:
            sys.path.remove(p)
        for _p in [project_str, root_str]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        for _k in list(sys.modules.keys()):
            if _k.startswith("DAB_Transformer_Project."):
                sys.modules.pop(_k, None)
        logger.debug(f"[DAB] _ensure_path: root={root_str} project={project_str}")

    def load_model(self) -> None:
        try:
            import torch
            self._ensure_path()
            from DAB_Transformer_Project.config import Config
            from DAB_Transformer_Project.model import DAB_Transformer
            from DAB_Transformer_Project.utils import TextProcess, text_process as shared_tp

            logger.debug("[DAB] All imports resolved successfully")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.text_process = shared_tp
            self.num_classes = len(self.text_process.char_map)

            self.model = DAB_Transformer(
                num_classes=self.num_classes,
                d_model=Config.D_MODEL,
                nhead=Config.NHEAD,
                num_layers=Config.NUM_LAYERS,
            )

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"DAB-Transformer loaded from: {self.checkpoint_path}")

            self._loaded = True
            logger.info(f"DAB-Transformer model loaded")
        except Exception as e:
            self._loaded = False
            raise ModelNotLoadedError(f"Failed to load DAB-Transformer model: {e}") from e

    def unload_model(self) -> None:
        self.model = None
        self.text_process = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("DAB-Transformer model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        try:
            if sample_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            max_samples = 160000
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
        if self.text_process is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            import torch
            logits = raw_output
            text = kwargs.get("text")
            threshold = kwargs.get("threshold", 0.5)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            if isinstance(logits, torch.Tensor):
                logits_np = logits.cpu().numpy()
            else:
                logits_np = logits

            from DAB_Transformer_Project.utils import greedy_decoder

            if isinstance(logits, torch.Tensor):
                logits_t = logits
            else:
                logits_t = torch.from_numpy(logits_np)

            if logits_t.dim() == 3:
                logits_t = logits_t.squeeze(0)

            phoneme_string = greedy_decoder(logits_t, self.text_process)
            phoneme_list = phoneme_string.split() if phoneme_string else []

            conf = 0.0
            if len(phoneme_list) > 0:
                conf = 0.85

            result = InferenceResult(
                phoneme_sequence=phoneme_list,
                phoneme_string=phoneme_string,
                overall_confidence=conf,
            )

            predictions: list[PhonemePrediction] = []
            if text:
                target_phonemes = self.text_process.text_to_phonemes(text)
                target_str = " ".join(target_phonemes)
                pred_str = phoneme_string

                from jiwer import process_words
                alignment_result = process_words(target_str, pred_str)
                alignment_chunks = alignment_result.alignments[0]

                ref_ph = target_str.split()
                hyp_ph = pred_str.split() if pred_str else []

                predictions = []
                for chunk in alignment_chunks:
                    if chunk.type == 'equal':
                        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                            predictions.append(PhonemePrediction(
                                phoneme=ref_ph[i] if i < len(ref_ph) else "?",
                                status="correct", confidence=1.0,
                                expected=ref_ph[i] if i < len(ref_ph) else None,
                                actual=ref_ph[i] if i < len(ref_ph) else None,
                            ))
                    elif chunk.type == 'substitute':
                        for i, j in zip(
                            range(chunk.ref_start_idx, chunk.ref_end_idx),
                            range(chunk.hyp_start_idx, chunk.hyp_end_idx)
                        ):
                            e = ref_ph[i] if i < len(ref_ph) else "?"
                            a = hyp_ph[j] if j < len(hyp_ph) else "?"
                            predictions.append(PhonemePrediction(
                                phoneme=e, status="substitution", confidence=0.0,
                                expected=e, actual=a,
                                reason=f"Nhầm /{e}/ thành /{a}/",
                            ))
                    elif chunk.type == 'delete':
                        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                            e = ref_ph[i] if i < len(ref_ph) else "?"
                            predictions.append(PhonemePrediction(
                                phoneme=e, status="deletion", confidence=0.0,
                                expected=e, actual=None,
                                reason=f"Bạn bị nuốt âm /{e}/",
                            ))
                    elif chunk.type == 'insert':
                        for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                            a = hyp_ph[j] if j < len(hyp_ph) else "?"
                            predictions.append(PhonemePrediction(
                                phoneme="-", status="insertion", confidence=0.0,
                                expected=None, actual=a,
                                reason=f"Phát âm thừa âm /{a}/",
                            ))

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
