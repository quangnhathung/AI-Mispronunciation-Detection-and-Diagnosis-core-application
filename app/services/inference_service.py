from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger

from app.services.model_registry import ModelRegistry
from app.services.audio_service import AudioService
from app.schemas.inference import InferenceResponse, PhonemePrediction, InferenceResult, InferenceSummary
from app.core.exceptions import InferenceError, PreprocessError, ModelNotLoadedError, TextRequiredError
from app.utils.time_utils import measure_time
from app.utils.id_utils import generate_request_id


class InferenceService:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.audio_service = AudioService()

    async def infer(
        self,
        audio_path: Path,
        original_filename: str = "unknown",
        model_name: str = "auto",
        text: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        return_details: bool = True,
        sample_rate_override: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> InferenceResponse:
        request_id = request_id or generate_request_id()
        resolved_name = None
        predictions = []
        result = None
        summary = InferenceSummary()

        with measure_time() as timer:
            try:
                resolved_name = self.registry.resolve_model(model_name)
                model = self.registry.get(resolved_name)

                if not model.loaded:
                    self.registry.load_model(resolved_name)

                target_sr = sample_rate_override or model.sample_rate
                audio, sr = self.audio_service.load_audio(audio_path, target_sr=target_sr)
                self.audio_service.validate_audio(audio, sr)
                audio = self.audio_service.normalize_audio(audio)

                if model.requires_text and not text:
                    raise TextRequiredError(f"Model '{resolved_name}' requires ground-truth text")

                predictions, result, summary = model.infer(
                    audio=audio,
                    sample_rate=sr,
                    text=text,
                    top_k=top_k,
                    threshold=threshold,
                )

                if not return_details:
                    predictions = []

            except TextRequiredError:
                raise
            except (InferenceError, PreprocessError, ModelNotLoadedError):
                raise
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                raise InferenceError(f"Inference failed: {e}") from e

        return InferenceResponse(
            success=True,
            model_name=resolved_name or "unknown",
            input_file=original_filename,
            predictions=predictions,
            result=result,
            summary=summary,
            processing_time_ms=timer["elapsed_ms"],
            request_id=request_id,
        )
