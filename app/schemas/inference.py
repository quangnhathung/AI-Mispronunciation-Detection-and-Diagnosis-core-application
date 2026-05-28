from pydantic import BaseModel, Field
from typing import Any, Optional


class PhonemePrediction(BaseModel):
    phoneme: str = Field(..., description="Predicted/canonical phoneme symbol")
    status: str = Field(..., description="Status: correct, incorrect, substitution, deletion, insertion, unknown")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    start_time: Optional[float] = Field(None, description="Start time in seconds (null if unavailable)")
    end_time: Optional[float] = Field(None, description="End time in seconds (null if unavailable)")
    reason: Optional[str] = Field(None, description="Error reason or description")
    expected: Optional[str] = Field(None, description="Expected phoneme (for substitutions)")
    actual: Optional[str] = Field(None, description="Actual predicted phoneme (for substitutions/deletions/insertions)")


class InferenceSummary(BaseModel):
    total_phonemes: int = Field(0, description="Total number of phonemes evaluated")
    correct_phonemes: int = Field(0, description="Number of correctly pronounced phonemes")
    incorrect_phonemes: int = Field(0, description="Number of incorrectly pronounced phonemes")
    accuracy: float = Field(0.0, ge=0.0, le=1.0, description="Overall accuracy")
    precision: Optional[float] = Field(None, description="Precision score (if available)")
    recall: Optional[float] = Field(None, description="Recall score (if available)")
    f1_score: Optional[float] = Field(None, description="F1 score (if available)")


class InferenceResult(BaseModel):
    phoneme_sequence: list[str] = Field(default_factory=list, description="Predicted phoneme sequence (ASR models)")
    phoneme_string: str = Field("", description="Space-separated phoneme sequence")
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence score")


class InferenceRequest(BaseModel):
    model_name: str = Field("auto", description="Model to use: cnn_bilstm_ctc, dab_transformer, wav2vec2, or auto")
    text: Optional[str] = Field(None, description="Ground-truth text transcript (required for wav2vec2)")
    top_k: int = Field(10, ge=1, le=100, description="Number of top results to return")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for error classification")
    return_details: bool = Field(True, description="Return detailed per-phoneme predictions")
    language: str = Field("en", description="Language code")
    sample_rate: Optional[int] = Field(None, description="Override sample rate")


class InferenceResponse(BaseModel):
    success: bool = Field(True, description="Whether inference succeeded")
    model_name: str = Field(..., description="Model used for inference")
    input_file: str = Field(..., description="Original input filename")
    predictions: list[PhonemePrediction] = Field(default_factory=list, description="Per-phoneme predictions")
    result: Optional[InferenceResult] = Field(None, description="ASR recognition result")
    summary: InferenceSummary = Field(default_factory=InferenceSummary, description="Summary statistics")
    processing_time_ms: float = Field(0.0, description="Total processing time in milliseconds")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class BatchInferenceRequest(BaseModel):
    requests: list[InferenceRequest] = Field(..., description="List of inference requests", min_length=1, max_length=32)


class BatchInferenceResponse(BaseModel):
    success: bool = Field(True, description="Whether all inferences succeeded")
    results: list[InferenceResponse] = Field(..., description="List of individual inference results")
    total_processing_time_ms: float = Field(0.0, description="Total batch processing time")
