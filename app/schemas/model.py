from pydantic import BaseModel, Field
from typing import Any, Optional


class ModelInfo(BaseModel):
    name: str = Field(..., description="Unique model identifier")
    display_name: str = Field(..., description="Human-readable model name")
    version: str = Field(..., description="Model version or checkpoint identifier")
    description: str = Field("", description="Model description")
    architecture: str = Field("", description="Model architecture type")
    task: str = Field("", description="Task type (ASR, scoring, etc.)")
    requires_text: bool = Field(False, description="Whether model requires ground-truth text")
    requires_gpu: bool = Field(False, description="Whether model requires GPU")
    sample_rate: int = Field(16000, description="Expected input sample rate")
    phoneme_set_size: int = Field(0, description="Number of phonemes in vocabulary")
    phoneme_set: list[str] = Field(default_factory=list, description="List of phoneme symbols")
    checkpoint_path: Optional[str] = Field(None, description="Path to model checkpoint")
    loaded: bool = Field(False, description="Whether model is currently loaded in memory")
    status: str = Field("unloaded", description="Model status: loaded, unloaded, error")


class ModelListResponse(BaseModel):
    models: list[ModelInfo] = Field(..., description="List of registered models")
    total: int = Field(..., description="Total number of models")


class LabelInfo(BaseModel):
    phoneme: str = Field(..., description="Phoneme symbol")
    name: str = Field("", description="Full name of the phoneme")
    arpabet: str = Field("", description="ARPABET representation")
    ipa: str = Field("", description="IPA representation")
    category: str = Field("", description="Phoneme category (vowel/consonant)")


class LabelsResponse(BaseModel):
    model_name: str = Field(..., description="Model these labels belong to")
    phonemes: list[LabelInfo] = Field(..., description="List of phoneme labels")
    total: int = Field(..., description="Total number of phonemes")
