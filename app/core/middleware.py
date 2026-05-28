import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import settings
from app.core.exceptions import MDDException
from app.utils.id_utils import generate_request_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", generate_request_id())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 2))
        return response


def mdd_exception_handler(request: Request, exc: MDDException) -> JSONResponse:
    logger.warning(f"MDDException: {exc.code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.to_dict(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc) if settings.debug else "Internal server error",
                "details": {},
            },
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(MDDException, mdd_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    logger.debug("Middleware and exception handlers registered")
