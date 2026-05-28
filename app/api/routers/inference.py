from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, Query
from typing import Optional
from pathlib import Path
from loguru import logger

from app.schemas.inference import InferenceResponse, BatchInferenceRequest, BatchInferenceResponse
from app.api.deps import get_model_registry, get_request_id
from app.services.model_registry import ModelRegistry
from app.services.inference_service import InferenceService
from app.core.exceptions import MDDException, TextRequiredError
from app.utils.file_utils import validate_audio_file, save_upload, cleanup_upload

router = APIRouter(tags=["Inference"])


def get_inference_service(registry: ModelRegistry = Depends(get_model_registry)) -> InferenceService:
    return InferenceService(registry)


@router.post(
    "/infer",
    summary="Run inference (auto-select model)",
    description="Upload an audio file and run mispronunciation detection. "
    "The system auto-selects the best available model, or you can specify one. "
    "For Wav2Vec2 model, ground-truth text is required.",
    response_model=InferenceResponse,
)
async def infer(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)"),
    model_name: str = Form("auto", description="Model to use: auto, cnn_bilstm_ctc, dab_transformer, wav2vec2"),
    text: Optional[str] = Form(None, description="Ground-truth text transcript (required for wav2vec2)"),
    top_k: int = Form(10, description="Number of top results", ge=1, le=100),
    threshold: float = Form(0.5, description="Confidence threshold for error detection", ge=0.0, le=1.0),
    return_details: bool = Form(True, description="Return detailed per-phoneme predictions"),
    language: str = Form("en", description="Language code"),
    sample_rate: Optional[int] = Form(None, description="Override audio sample rate"),
    request_id: str = Depends(get_request_id),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    saved_path = None
    try:
        ext = validate_audio_file(file)
        saved_path = Path(await save_upload(file, ext))
        logger.info(f"Inference request: model={model_name}, file={file.filename}, text={text}")

        response = await service.infer(
            audio_path=saved_path,
            original_filename=file.filename or "unknown",
            model_name=model_name,
            text=text,
            top_k=top_k,
            threshold=threshold,
            return_details=return_details,
            sample_rate_override=sample_rate,
            request_id=request_id,
        )
        return response
    except MDDException:
        raise
    except Exception as e:
        logger.error(f"Inference endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if saved_path and saved_path.exists():
            cleanup_upload(saved_path)


@router.post(
    "/infer/cnn-bilstm-ctc",
    summary="Run inference with CNN-BiLSTM-CTC",
    description="Run mispronunciation detection using the CNN-BiLSTM-CTC model. "
    "This is an ASR-based model that predicts phoneme sequences from audio. "
    "Optionally provide ground-truth text for detailed MDD analysis.",
    response_model=InferenceResponse,
)
async def infer_cnn_bilstm_ctc(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)"),
    text: Optional[str] = Form(None, description="Ground-truth text (optional, enables detailed MDD)"),
    top_k: int = Form(10, description="Number of top results", ge=1, le=100),
    threshold: float = Form(0.5, description="Confidence threshold", ge=0.0, le=1.0),
    return_details: bool = Form(True, description="Return detailed predictions"),
    sample_rate: Optional[int] = Form(None, description="Override sample rate"),
    request_id: str = Depends(get_request_id),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    saved_path = None
    try:
        ext = validate_audio_file(file)
        saved_path = Path(await save_upload(file, ext))
        response = await service.infer(
            audio_path=saved_path,
            original_filename=file.filename or "unknown",
            model_name="cnn_bilstm_ctc",
            text=text,
            top_k=top_k,
            threshold=threshold,
            return_details=return_details,
            sample_rate_override=sample_rate,
            request_id=request_id,
        )
        return response
    except MDDException:
        raise
    except Exception as e:
        logger.error(f"CNN-BiLSTM-CTC inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if saved_path and saved_path.exists():
            cleanup_upload(saved_path)


@router.post(
    "/infer/dab-transformer",
    summary="Run inference with DAB-Transformer",
    description="Run mispronunciation detection using the DAB-Transformer model. "
    "This is an ASR-based model. Provide ground-truth text for detailed alignment-based MDD.",
    response_model=InferenceResponse,
)
async def infer_dab_transformer(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)"),
    text: Optional[str] = Form(None, description="Ground-truth text (recommended for detailed MDD)"),
    top_k: int = Form(10, description="Number of top results", ge=1, le=100),
    threshold: float = Form(0.5, description="Confidence threshold", ge=0.0, le=1.0),
    return_details: bool = Form(True, description="Return detailed predictions"),
    sample_rate: Optional[int] = Form(None, description="Override sample rate"),
    request_id: str = Depends(get_request_id),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    saved_path = None
    try:
        ext = validate_audio_file(file)
        saved_path = Path(await save_upload(file, ext))
        response = await service.infer(
            audio_path=saved_path,
            original_filename=file.filename or "unknown",
            model_name="dab_transformer",
            text=text,
            top_k=top_k,
            threshold=threshold,
            return_details=return_details,
            sample_rate_override=sample_rate,
            request_id=request_id,
        )
        return response
    except MDDException:
        raise
    except Exception as e:
        logger.error(f"DAB-Transformer inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if saved_path and saved_path.exists():
            cleanup_upload(saved_path)


@router.post(
    "/infer/wav2vec2",
    summary="Run inference with Wav2Vec2-MDD",
    description="Run mispronunciation detection using the Wav2Vec2-MDD model. "
    "REQUIRES ground-truth text. This model scores each phoneme's pronunciation correctness.",
    response_model=InferenceResponse,
)
async def infer_wav2vec2(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A, OGG)"),
    text: str = Form(..., description="Ground-truth text transcript (REQUIRED for this model)"),
    top_k: int = Form(10, description="Number of top results", ge=1, le=100),
    threshold: float = Form(0.5, description="Confidence threshold for correct/incorrect classification", ge=0.0, le=1.0),
    return_details: bool = Form(True, description="Return detailed per-phoneme predictions"),
    sample_rate: Optional[int] = Form(None, description="Override sample rate"),
    request_id: str = Depends(get_request_id),
    service: InferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    saved_path = None
    try:
        ext = validate_audio_file(file)
        saved_path = Path(await save_upload(file, ext))
        response = await service.infer(
            audio_path=saved_path,
            original_filename=file.filename or "unknown",
            model_name="wav2vec2",
            text=text,
            top_k=top_k,
            threshold=threshold,
            return_details=return_details,
            sample_rate_override=sample_rate,
            request_id=request_id,
        )
        return response
    except TextRequiredError:
        raise
    except MDDException:
        raise
    except Exception as e:
        logger.error(f"Wav2Vec2 inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if saved_path and saved_path.exists():
            cleanup_upload(saved_path)
