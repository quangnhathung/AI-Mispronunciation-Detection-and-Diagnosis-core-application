from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.middleware import register_middleware
from app.services.model_registry import ModelRegistry
from app.api.routers import health, models, inference, metadata

model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    app.state.model_registry = model_registry

    for info in model_registry.get_all_info():
        logger.info(f"  Registered: {info.name} ({info.display_name})")
        if info.checkpoint_path:
            logger.info(f"    Checkpoint: {info.checkpoint_path}")

    if settings.model_load_on_startup:
        logger.info("Loading models on startup...")
        for m in model_registry.get_all():
            try:
                m.load_model()
                logger.info(f"  Loaded: {m.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {m.name}: {e}")

    logger.info(f"{settings.app_name} is ready")
    yield
    logger.info("Shutting down...")
    for m in model_registry.get_all():
        if m.loaded:
            m.unload_model()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
app.include_router(inference.router, prefix=settings.api_prefix)
app.include_router(metadata.router, prefix=settings.api_prefix)

register_middleware(app)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        routes=app.routes,
    )
    schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", include_in_schema=False)
async def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_prefix}/docs",
        "redoc": f"{settings.api_prefix}/redoc",
    }
