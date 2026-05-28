from fastapi import APIRouter, Depends
from app.schemas.common import HealthResponse, ReadyResponse
from app.api.deps import get_model_registry
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
    description="Returns the service health status and application version.",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    from app.core.config import settings
    return HealthResponse(status="ok", version=settings.app_version)


@router.get(
    "/ready",
    summary="Readiness check",
    description="Checks if all registered models are loaded and ready for inference.",
    response_model=ReadyResponse,
)
async def readiness_check(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ReadyResponse:
    loaded = []
    failed = []
    for info in registry.get_all_info():
        if info.loaded:
            loaded.append(info.name)
        else:
            failed.append(info.name)
    return ReadyResponse(
        ready=len(failed) == 0,
        models_loaded=loaded,
        models_failed=failed,
    )
