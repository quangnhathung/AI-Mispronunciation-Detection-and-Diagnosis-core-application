from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BasePreprocessor(ABC):
    @abstractmethod
    def process(self, audio: np.ndarray, sample_rate: int, **kwargs) -> Any:
        ...
