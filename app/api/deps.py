from typing import Optional
from fastapi import Request, Depends
from app.services.model_registry import ModelRegistry


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def get_model_registry() -> ModelRegistry:
    from app.main import model_registry as registry
    return registry


async def get_registry(request: Request) -> ModelRegistry:
    registry: Optional[ModelRegistry] = request.app.state.model_registry
    if registry is None:
        registry = get_model_registry()
    return registry
