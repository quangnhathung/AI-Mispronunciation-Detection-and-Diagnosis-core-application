from typing import Any, Optional


class MDDException(Exception):
    status_code: int = 500
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred"
    details: Optional[dict[str, Any]] = None

    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        if message:
            self.message = message
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details or {},
        }


class AudioFormatError(MDDException):
    status_code = 400
    code = "AUDIO_FORMAT_ERROR"
    message = "Invalid audio format or corrupted file"


class AudioTooLargeError(MDDException):
    status_code = 400
    code = "AUDIO_TOO_LARGE"
    message = "Audio file exceeds maximum allowed size"


class ModelNotLoadedError(MDDException):
    status_code = 503
    code = "MODEL_NOT_LOADED"
    message = "Model is not loaded or failed to initialize"


class ModelNotFoundError(MDDException):
    status_code = 404
    code = "MODEL_NOT_FOUND"
    message = "Requested model is not registered"


class InferenceError(MDDException):
    status_code = 500
    code = "INFERENCE_ERROR"
    message = "Model inference failed"


class PreprocessError(MDDException):
    status_code = 422
    code = "PREPROCESS_ERROR"
    message = "Failed to preprocess input data"


class PostprocessError(MDDException):
    status_code = 500
    code = "POSTPROCESS_ERROR"
    message = "Failed to postprocess model output"


class TextRequiredError(MDDException):
    status_code = 400
    code = "TEXT_REQUIRED"
    message = "This model requires ground-truth text transcript for inference"


class ConfigError(MDDException):
    status_code = 500
    code = "CONFIG_ERROR"
    message = "Invalid configuration"
