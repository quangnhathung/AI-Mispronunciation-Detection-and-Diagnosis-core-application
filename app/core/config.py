import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "MDD Core Application"
    app_version: str = "1.0.0"
    app_description: str = (
        "FastAPI-based API for Mispronunciation Detection and Diagnosis (MDD) "
        "in English pronunciation. Supports CNN-BiLSTM-CTC, DAB-Transformer, "
        "and Wav2Vec2-based models."
    )
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    project_root: Path = Path(__file__).parent.parent.parent.resolve()
    upload_dir: Path = Path(project_root, "data", "uploads")
    log_dir: Path = Path(project_root, "logs")

    max_upload_size_mb: int = 10
    allowed_audio_formats: list[str] = ["wav", "mp3", "flac", "m4a", "ogg"]
    default_sample_rate: int = 16000

    model_cache_enabled: bool = True
    model_load_on_startup: bool = False

    cnn_bilstm_ctc_checkpoint: Optional[str] = None
    dab_transformer_checkpoint: Optional[str] = None
    wav2vec2_checkpoint: Optional[str] = None

    cnn_bilstm_ctc_config: Optional[str] = None

    default_model: str = "wav2vec2"
    default_top_k: int = 10
    default_threshold: float = 0.5
    default_return_details: bool = True

    log_level: str = "INFO"
    log_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": False}

    def auto_detect_checkpoints(self):
        root = self.project_root
        cnn_dir = root / "CNN_BiLSTM_CTC" / "checkpoints"
        dab_dir = root / "DAB_Transformer_Project" / "checkpoints_phoneme_8vram"
        w2v2_dir = root / "Wav2vec2" / "checkpoints"
        if not self.cnn_bilstm_ctc_checkpoint:
            candidates = list(cnn_dir.glob("best.pt")) + list(cnn_dir.glob("last.pt"))
            if candidates:
                self.cnn_bilstm_ctc_checkpoint = str(candidates[0])
        if not self.dab_transformer_checkpoint:
            candidates = sorted(dab_dir.glob("model_e*.pt"))
            if candidates:
                self.dab_transformer_checkpoint = str(candidates[-1])
        if not self.wav2vec2_checkpoint:
            candidates = sorted(w2v2_dir.glob("best_mdd_model*.pt"))
            if candidates:
                self.wav2vec2_checkpoint = str(candidates[-1])
        if not self.cnn_bilstm_ctc_config:
            cfg = root / "CNN_BiLSTM_CTC" / "configs" / "config.yaml"
            if cfg.exists():
                self.cnn_bilstm_ctc_config = str(cfg)


settings = Settings()
settings.auto_detect_checkpoints()

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
