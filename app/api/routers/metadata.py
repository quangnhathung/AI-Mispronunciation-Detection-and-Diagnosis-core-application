import sys
import platform
from fastapi import APIRouter
from app.schemas.common import VersionResponse
from app.core.config import settings

router = APIRouter(tags=["Metadata"])


@router.get(
    "/version",
    summary="System version info",
    description="Returns application version, Python version, and key dependency versions.",
    response_model=VersionResponse,
)
async def get_version() -> VersionResponse:
    deps = {}
    try:
        import torch
        deps["torch"] = torch.__version__
    except ImportError:
        deps["torch"] = "not installed"
    try:
        import torchaudio
        deps["torchaudio"] = torchaudio.__version__
    except ImportError:
        deps["torchaudio"] = "not installed"
    try:
        import transformers
        deps["transformers"] = transformers.__version__
    except ImportError:
        deps["transformers"] = "not installed"
    try:
        import librosa
        deps["librosa"] = librosa.__version__
    except ImportError:
        deps["librosa"] = "not installed"
    try:
        import fastapi
        deps["fastapi"] = fastapi.__version__
    except ImportError:
        deps["fastapi"] = "not installed"

    return VersionResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        python_version=sys.version,
        dependencies=deps,
    )
