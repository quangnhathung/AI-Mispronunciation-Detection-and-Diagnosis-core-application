from fastapi import APIRouter, Depends, Path
from app.schemas.model import ModelListResponse, ModelInfo, LabelsResponse, LabelInfo
from app.api.deps import get_model_registry
from app.services.model_registry import ModelRegistry
from app.core.exceptions import ModelNotFoundError

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    summary="List all models",
    description="Returns information about all registered models in the system.",
    response_model=ModelListResponse,
)
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelListResponse:
    models = registry.get_all_info()
    return ModelListResponse(models=models, total=len(models))


@router.get(
    "/models/{model_name}",
    summary="Get model details",
    description="Returns detailed information about a specific model.",
    response_model=ModelInfo,
)
async def get_model(
    model_name: str = Path(..., description="Model name identifier"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelInfo:
    try:
        return registry.get_info(model_name)
    except ModelNotFoundError:
        raise


@router.post(
    "/models/{model_name}/load",
    summary="Load a model",
    description="Load a specific model into memory.",
    response_model=ModelInfo,
)
async def load_model(
    model_name: str = Path(..., description="Model name to load"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelInfo:
    registry.load_model(model_name)
    return registry.get_info(model_name)


@router.post(
    "/models/{model_name}/unload",
    summary="Unload a model",
    description="Unload a specific model from memory to free resources.",
    response_model=ModelInfo,
)
async def unload_model(
    model_name: str = Path(..., description="Model name to unload"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelInfo:
    registry.unload_model(model_name)
    return registry.get_info(model_name)


@router.get(
    "/labels",
    summary="List phoneme labels",
    description="Returns the phoneme label set for all or a specific model.",
    response_model=LabelsResponse,
)
async def get_labels(
    model_name: str = "wav2vec2",
    registry: ModelRegistry = Depends(get_model_registry),
) -> LabelsResponse:
    info = registry.get_info(model_name)
    phonemes = [
        LabelInfo(phoneme=ph, arpabet=ph)
        for ph in info.phoneme_set
    ]
    return LabelsResponse(
        model_name=model_name,
        phonemes=phonemes,
        total=len(phonemes),
    )
