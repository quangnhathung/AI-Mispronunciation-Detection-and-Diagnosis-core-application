import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError, TextRequiredError
from app.models.normalize import safe_tensor_to_list, safe_item

_w2v2_root = Path(__file__).parent.parent.parent / "Wav2vec2"
_W2V2_ROOT_STR = str(_w2v2_root.resolve())
_W2V2_PROJECT_ROOT = str(_w2v2_root.parent.resolve())

logger.debug(f"[W2V2] Module root: {_W2V2_ROOT_STR}")

for _p in [_W2V2_PROJECT_ROOT, _W2V2_ROOT_STR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _key in list(sys.modules.keys()):
    if _key.startswith("src.") or _key == "src":
        sys.modules.pop(_key, None)


class Wav2Vec2Model(BaseModelWrapper):
    name = "wav2vec2"
    display_name = "Wav2Vec2-MDD"
    version = "4.0.0"
    description = "Wav2Vec2-based mispronunciation detection with cross-attention scoring"
    architecture = "Wav2Vec2 + Bi-GRU + Cross-Attention"
    task = "Phoneme scoring / error detection"
    requires_text = True
    requires_gpu = False
    sample_rate = 16000
    phoneme_set_size = 42
    phoneme_set = [
        "PAD", "UNK", "SIL", "SP",
        "AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER",
        "EY", "IH", "IY", "OW", "OY", "UH", "UW",
        "B", "CH", "D", "DH", "DX", "F", "G", "HH", "JH",
        "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
        "T", "TH", "V", "W", "Y", "Z", "ZH",
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path)
        self.model = None
        self.device = None
        self.feature_extractor = None
        self.g2p = None
        self.phoneme_to_id = None
        self.id_to_phoneme = None
        self.vocab_size = 0
        self.unk_id = 1

    def _ensure_path(self):
        root_str = str(_w2v2_root.resolve())
        project_str = str(_w2v2_root.parent.resolve())
        other_roots = [p for p in sys.path if p != root_str and p != project_str and Path(p).name in ("CNN_BiLSTM_CTC", "DAB_Transformer_Project")]
        for p in other_roots:
            sys.path.remove(p)
        for _p in [project_str, root_str]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        for _k in list(sys.modules.keys()):
            if _k.startswith("Wav2vec2.src") or _k.startswith("src."):
                sys.modules.pop(_k, None)
        logger.debug(f"[W2V2] _ensure_path: root={root_str} project={project_str}")

    def load_model(self) -> None:
        try:
            import torch
            self._ensure_path()
            from Wav2vec2.src.model.mmd_model_v2 import MDDModelV2
            from Wav2vec2.src.data.dictionary import ARPABET_PHONEMES, PHONEME_TO_ID, ID_TO_PHONEME, get_phoneme_id
            from transformers import Wav2Vec2FeatureExtractor

            logger.debug("[W2V2] All imports resolved successfully")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.phoneme_to_id = PHONEME_TO_ID
            self.id_to_phoneme = ID_TO_PHONEME
            self.vocab_size = len(ARPABET_PHONEMES)
            self.unk_id = PHONEME_TO_ID["UNK"]

            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

            self.model = MDDModelV2(vocab_size=self.vocab_size)

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Wav2Vec2-MDD model loaded: {self.checkpoint_path}")

            try:
                from g2p_en import G2p
                self.g2p = G2p()
            except ImportError:
                logger.warning("g2p_en not installed, phoneme conversion will be limited")
                self.g2p = None

            self._loaded = True
        except Exception as e:
            self._loaded = False
            raise ModelNotLoadedError(f"Failed to load Wav2Vec2 model: {e}") from e

    def unload_model(self) -> None:
        self.model = None
        self.feature_extractor = None
        self.g2p = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("Wav2Vec2-MDD model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        if self.feature_extractor is None or self.phoneme_to_id is None:
            raise PreprocessError("Model not loaded")
        text = kwargs.get("text")
        if not text:
            raise TextRequiredError("Wav2Vec2 model requires ground-truth text for inference")
        try:
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            inputs = self.feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            audio_features = inputs.input_values.to(self.device)
            attn_mask = inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None
            logger.debug(f"[Wav2Vec2] audio_features shape={audio_features.shape} attn_mask={attn_mask.shape if attn_mask is not None else None}")

            if self.g2p:
                phonemes_raw = self.g2p(text)
                logger.debug(f"[Wav2Vec2] g2p({text!r}) -> {phonemes_raw}")
                phonemes = []
                for ph in phonemes_raw:
                    ph_clean = ph.replace(" ", "").upper()
                    if ph_clean and ph_clean not in ("", " "):
                        from Wav2vec2.src.data.dictionary import normalize_phoneme
                        phonemes.append(normalize_phoneme(ph_clean))
                logger.debug(f"[Wav2Vec2] normalized phonemes={phonemes}")
            else:
                phonemes = text.upper().split()
                logger.debug(f"[Wav2Vec2] raw text split phonemes={phonemes}")

            phoneme_ids = []
            for ph in phonemes:
                pid = self.phoneme_to_id.get(ph, self.unk_id)
                phoneme_ids.append(pid)

            if not phoneme_ids:
                raise PreprocessError("Could not convert text to phonemes")

            phoneme_tensor = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
            logger.debug(f"[Wav2Vec2] phoneme_tensor shape={phoneme_tensor.shape} ids={phoneme_ids}")
            return {"audio": audio_features, "attention_mask": attn_mask, "phonemes": phoneme_tensor, "phoneme_list": phonemes}
        except TextRequiredError:
            raise
        except Exception as e:
            raise PreprocessError(f"Wav2Vec2 preprocessing failed: {e}") from e

    def predict(self, features: Any, **kwargs) -> Any:
        import torch
        if self.model is None:
            raise ModelNotLoadedError("Model not loaded")
        try:
            audio = features["audio"]
            phonemes = features["phonemes"]
            attention_mask = features.get("attention_mask")
            logger.debug(f"[Wav2Vec2] predict input: audio shape={audio.shape} attention_mask={attention_mask.shape if attention_mask is not None else None} phonemes shape={phonemes.shape}")
            with torch.no_grad():
                logits, attn_weights = self.model(input_values=audio, attention_mask=attention_mask, canonical_ids=phonemes)
            logger.debug(f"[Wav2Vec2] predict output: logits shape={logits.shape} attn_weights shape={attn_weights.shape if isinstance(attn_weights, torch.Tensor) else 'N/A'}")
            return {
                "logits": logits,
                "phoneme_list": features["phoneme_list"],
            }
        except Exception as e:
            raise InferenceError(f"Wav2Vec2 inference failed: {e}") from e

    def postprocess(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]:
        try:
            import torch
            logits = raw_output["logits"]
            phoneme_list = raw_output["phoneme_list"]
            threshold = kwargs.get("threshold", 0.5)

            if isinstance(logits, torch.Tensor):
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = logits

            if len(probs.shape) > 1 and probs.shape[0] == 1:
                probs = probs[0]
            if len(probs.shape) > 1:
                probs = probs.flatten()

            predictions: list[PhonemePrediction] = []
            correct_count = 0
            for i, ph in enumerate(phoneme_list):
                score = float(probs[i]) if i < len(probs) else 0.0
                is_correct = score >= threshold
                status = "correct" if is_correct else "incorrect"
                if is_correct:
                    correct_count += 1
                predictions.append(
                    PhonemePrediction(
                        phoneme=ph,
                        status=status,
                        confidence=score,
                        expected=ph,
                        actual=ph,
                        reason=None if is_correct else f"Phát âm lỗi âm /{ph}/ (score: {score:.3f})",
                    )
                )

            total = len(predictions)
            summary = InferenceSummary(
                total_phonemes=total,
                correct_phonemes=correct_count,
                incorrect_phonemes=total - correct_count,
                accuracy=correct_count / total if total > 0 else 0.0,
            )

            result = InferenceResult(
                phoneme_sequence=phoneme_list,
                phoneme_string=" ".join(phoneme_list),
                overall_confidence=correct_count / total if total > 0 else 0.0,
            )

            return predictions, result, summary
        except Exception as e:
            raise InferenceError(f"Wav2Vec2 postprocessing failed: {e}") from e
