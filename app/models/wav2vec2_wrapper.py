import sys
from pathlib import Path
from typing import Any, Optional
import numpy as np
from loguru import logger

from app.models.base_model import BaseModelWrapper
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
from app.core.exceptions import ModelNotLoadedError, InferenceError, PreprocessError, TextRequiredError


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
    phoneme_set_size = 46
    phoneme_set = [
        "<pad>", "<unk>", "<sil>", "<sp>",
        "AA", "AE", "AH", "AO", "AW", "AY",
        "B", "CH", "D", "DH", "EH", "ER", "EY",
        "F", "G", "HH", "IH", "IY", "JH", "K",
        "L", "M", "N", "NG", "OW", "OY", "P",
        "R", "S", "SH", "T", "TH", "UH", "UW",
        "V", "W", "Y", "Z", "ZH",
    ]

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(checkpoint_path)
        self.model = None
        self.device = None
        self.feature_extractor = None
        self.dictionary = None
        self.g2p = None

    def load_model(self) -> None:
        try:
            import torch
            w2v2_root = Path(__file__).parent.parent.parent / "Wav2vec2"
            if w2v2_root not in [Path(p) for p in sys.path]:
                sys.path.insert(0, str(w2v2_root))

            from src.model.mmd_model_v2 import MDDModelV2
            from src.data.dictionary import ARPABETDictionary
            from transformers import Wav2Vec2FeatureExtractor

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.dictionary = ARPABETDictionary()
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

            vocab_size = len(self.dictionary)
            self.model = MDDModelV2(vocab_size=vocab_size)

            if not self.checkpoint_path or not self.checkpoint_path.exists():
                raise ModelNotLoadedError(f"Checkpoint not found: {self.checkpoint_path}")

            checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
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
        self.dictionary = None
        self.g2p = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        logger.info("Wav2Vec2-MDD model unloaded")

    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        import torch
        if self.feature_extractor is None or self.dictionary is None:
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

            if self.g2p:
                phonemes_raw = self.g2p(text)
                phonemes = []
                for ph, _ in phonemes_raw:
                    ph_clean = ph.replace(" ", "").upper()
                    if ph_clean and ph_clean not in ("", " "):
                        phonemes.append(ph_clean)
            else:
                phonemes = text.upper().split()

            phoneme_ids = []
            for ph in phonemes:
                try:
                    pid = self.dictionary.phoneme_to_id(ph)
                    phoneme_ids.append(pid)
                except KeyError:
                    phoneme_ids.append(self.dictionary.unk_id)

            if not phoneme_ids:
                raise PreprocessError("Could not convert text to phonemes")

            phoneme_tensor = torch.tensor([phoneme_ids], dtype=torch.long, device=self.device)
            return {"audio": audio_features, "phonemes": phoneme_tensor, "phoneme_list": phonemes}
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
            with torch.no_grad():
                logits = self.model(audio, phonemes)
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
