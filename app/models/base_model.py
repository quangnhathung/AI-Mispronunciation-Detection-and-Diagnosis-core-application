from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from app.schemas.model import ModelInfo
from app.schemas.inference import InferenceResult, PhonemePrediction, InferenceSummary
import numpy as np


class BaseModelWrapper(ABC):
    name: str = ""
    display_name: str = ""
    version: str = ""
    description: str = ""
    architecture: str = ""
    task: str = ""
    requires_text: bool = False
    requires_gpu: bool = False
    sample_rate: int = 16000
    phoneme_set_size: int = 0
    phoneme_set: list[str] = []
    checkpoint_path: Optional[Path] = None
    _loaded: bool = False

    def __init__(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def unload_model(self) -> None:
        ...

    @abstractmethod
    def preprocess(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        ...

    @abstractmethod
    def predict(self, features: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]:
        ...

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            display_name=self.display_name,
            version=self.version,
            description=self.description,
            architecture=self.architecture,
            task=self.task,
            requires_text=self.requires_text,
            requires_gpu=self.requires_gpu,
            sample_rate=self.sample_rate,
            phoneme_set_size=self.phoneme_set_size,
            phoneme_set=self.phoneme_set,
            checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            loaded=self._loaded,
            status="loaded" if self._loaded else "unloaded",
        )

    async def infer(
        self,
        audio: np.ndarray,
        sample_rate: int,
        text: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        **kwargs,
    ) -> tuple[list[PhonemePrediction], Optional[InferenceResult], InferenceSummary]:
        features = self.preprocess(audio, sample_rate, text=text)
        raw_output = self.predict(features, text=text, threshold=threshold)
        predictions, result, summary = self.postprocess(raw_output, text=text, top_k=top_k, threshold=threshold)
        return predictions, result, summary
