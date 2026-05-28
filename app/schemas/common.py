from pydantic import BaseModel, Field
from typing import Any, Optional


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    version: str = Field(..., description="Application version")


class ReadyResponse(BaseModel):
    ready: bool = Field(..., description="Whether all models are ready")
    models_loaded: list[str] = Field(default_factory=list, description="List of loaded model names")
    models_failed: list[str] = Field(default_factory=list, description="List of failed model names")


class VersionResponse(BaseModel):
    app_name: str = Field(..., description="Application name")
    app_version: str = Field(..., description="Application version")
    python_version: str = Field(..., description="Python runtime version")
    dependencies: dict[str, str] = Field(default_factory=dict, description="Key dependency versions")


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Error code string")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error context")


class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
