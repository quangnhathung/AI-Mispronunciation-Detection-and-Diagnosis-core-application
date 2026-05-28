from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ModelConfig:
    name: str = "cnn_bilstm_ctc"
    input_dim: int = 80
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    cnn_dropout: float = 0.2
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 4
    rnn_dropout: float = 0.3
    rnn_bidirectional: bool = True
    vocab_size: int = 42


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: str = "cosine_annealing"
    T_max: int = 100
    eta_min: float = 1e-6
    warmup_steps: int = 0
    warmup_ratio: float = 0.1


@dataclass
class DataConfig:
    dataset: str = "l2arctic"
    data_dir: str = "./data/raw/L2_ARCTIC"
    cache_dir: str = "./data/cache"
    manifest_dir: str = "./data/manifests"
    processed_dir: str = "./data/processed"
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 25
    max_audio_length: float = 30.0
    min_audio_length: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    gradient_clip: float = 5.0
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    early_stopping_patience: int = 10
    save_top_k: int = 3
    log_interval: int = 10
    eval_interval: int = 1
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    profiler: bool = False


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    tensorboard: bool = True
    csv_log: bool = True
    log_level: str = "INFO"


@dataclass
class InferenceConfig:
    checkpoint_path: str = "./checkpoints/best.pt"
    beam_width: int = 5
    use_beam_search: bool = False
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> Config:
        cfg = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "optimizer" in raw:
            for k, v in raw["optimizer"].items():
                if hasattr(cfg.optimizer, k):
                    setattr(cfg.optimizer, k, v)
        if "scheduler" in raw:
            for k, v in raw["scheduler"].items():
                if hasattr(cfg.scheduler, k):
                    setattr(cfg.scheduler, k, v)
        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(cfg.data, k):
                    setattr(cfg.data, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(cfg.training, k):
                    setattr(cfg.training, k, v)
        if "logging" in raw:
            for k, v in raw["logging"].items():
                if hasattr(cfg.logging, k):
                    setattr(cfg.logging, k, v)
        if "inference" in raw:
            for k, v in raw["inference"].items():
                if hasattr(cfg.inference, k):
                    setattr(cfg.inference, k, v)
        return cfg

    def update_from_cli(self, args: Any) -> None:
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "inference": self.inference.__dict__,
        }


def get_config(config_path: str = "./configs/config.yaml") -> Config:
    if os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config()
