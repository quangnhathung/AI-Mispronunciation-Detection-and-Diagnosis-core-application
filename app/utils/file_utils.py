import os
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import AudioFormatError, AudioTooLargeError

ALLOWED_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
}


def validate_audio_file(file: UploadFile) -> str:
    ext = ""
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
    if not ext:
        raise AudioFormatError("File has no extension")
    if ext not in settings.allowed_audio_formats:
        raise AudioFormatError(
            f"Unsupported format '{ext}'. Allowed: {settings.allowed_audio_formats}"
        )
    return ext


async def save_upload(file: UploadFile, ext: str) -> Path:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise AudioTooLargeError(
            f"File exceeds {settings.max_upload_size_mb}MB limit"
        )
    filename = f"{uuid.uuid4().hex}.{ext}"
    dest = settings.upload_dir / filename
    dest.write_bytes(content)
    return dest


def cleanup_upload(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except OSError:
        pass
