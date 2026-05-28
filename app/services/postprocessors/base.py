from abc import ABC, abstractmethod
from typing import Any
from app.schemas.inference import PhonemePrediction, InferenceResult, InferenceSummary


class BasePostprocessor(ABC):
    @abstractmethod
    def process(self, raw_output: Any, **kwargs) -> tuple[list[PhonemePrediction], InferenceResult, InferenceSummary]:
        ...
