from typing import Optional
from loguru import logger
from app.models.base_model import BaseModelWrapper
from app.models.cnn_bilstm_ctc_wrapper import CNNBiLSTMCTCModel
from app.models.dab_transformer_wrapper import DABTransformerModel
from app.models.wav2vec2_wrapper import Wav2Vec2Model
from app.core.config import settings
from app.core.exceptions import ModelNotFoundError, ModelNotLoadedError
from app.schemas.model import ModelInfo


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, BaseModelWrapper] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register(
            CNNBiLSTMCTCModel(
                checkpoint_path=settings.cnn_bilstm_ctc_checkpoint,
                config_path=settings.cnn_bilstm_ctc_config,
            )
        )
        self.register(
            DABTransformerModel(
                checkpoint_path=settings.dab_transformer_checkpoint,
            )
        )
        self.register(
            Wav2Vec2Model(
                checkpoint_path=settings.wav2vec2_checkpoint,
            )
        )

    def register(self, model: BaseModelWrapper) -> None:
        self._models[model.name] = model
        logger.info(f"Registered model: {model.name}")

    def get(self, name: str) -> BaseModelWrapper:
        model = self._models.get(name)
        if model is None:
            raise ModelNotFoundError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return model

    def get_all(self) -> list[BaseModelWrapper]:
        return list(self._models.values())

    def load_model(self, name: str) -> None:
        model = self.get(name)
        if not model.loaded:
            try:
                model.load_model()
                logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                raise ModelNotLoadedError(f"Failed to load model '{name}': {e}") from e

    def unload_model(self, name: str) -> None:
        model = self.get(name)
        if model.loaded:
            model.unload_model()
            logger.info(f"Unloaded model: {name}")

    def load_all(self) -> None:
        for name in self._models:
            try:
                self.load_model(name)
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    def get_info(self, name: str) -> ModelInfo:
        return self.get(name).get_info()

    def get_all_info(self) -> list[ModelInfo]:
        return [m.get_info() for m in self._models.values()]

    def is_loaded(self, name: str) -> bool:
        model = self._models.get(name)
        return model is not None and model.loaded

    def get_loaded_info(self) -> list[ModelInfo]:
        return [m.get_info() for m in self._models.values() if m.loaded]

    def resolve_model(self, model_name: str) -> str:
        if model_name == "auto":
            for preferred in [settings.default_model, "wav2vec2", "dab_transformer", "cnn_bilstm_ctc"]:
                try:
                    model = self.get(preferred)
                    if model.loaded:
                        return preferred
                except ModelNotFoundError:
                    continue
            available = [n for n in self._models.keys()]
            if available:
                return available[0]
            raise ModelNotFoundError("No models available")
        return model_name
