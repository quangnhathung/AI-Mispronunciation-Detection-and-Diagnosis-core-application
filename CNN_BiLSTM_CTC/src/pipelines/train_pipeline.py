from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.callbacks.model_checkpoint import ModelCheckpoint
from src.datasets.collator import Collator
from src.datasets.l2arctic_dataset import L2ArcticDataset
from src.datasets.tokenizer import PhonemeTokenizer
from src.features.augmentation import AudioAugmentationPipeline
from src.features.mel_spec import MelFeatureExtractor
from src.losses.ctc_loss import CTCLossWrapper
from src.models.cnn_bilstm_ctc import CNNBiLSTMCTC
from src.trainers.trainer import Trainer
from src.utils.config import Config
from src.utils.seed import set_seed



class TrainPipeline:
    def __init__(self, config: Config):
        self.config = config
        set_seed(getattr(config.training, "seed", 42))
        self.device = torch.device(
            getattr(config.training, "device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    def _build_tokenizer(self) -> PhonemeTokenizer:
        tokenizer = PhonemeTokenizer(include_stress=True)
        logger.info(f"Tokenizer: {tokenizer}")
        return tokenizer

    def _build_feature_extractor(self) -> MelFeatureExtractor:
        return MelFeatureExtractor(
            sample_rate=self.config.data.sample_rate,
            n_fft=self.config.data.n_fft,
            win_length=self.config.data.win_length,
            hop_length=self.config.data.hop_length,
            n_mels=self.config.data.n_mels,
        )

    def _build_augmentation(self) -> Optional[AudioAugmentationPipeline]:
        if self.config.data.spec_augment:
            return AudioAugmentationPipeline(
                freq_mask_param=self.config.data.freq_mask_param,
                time_mask_param=self.config.data.time_mask_param,
                spec_augment_p=0.5,
            )
        return None

    def _build_datasets(
        self,
        tokenizer: PhonemeTokenizer,
        feature_fn: Any,
        augmentation_fn: Any,
    ) -> tuple:
        manifest_dir = Path(self.config.data.manifest_dir)
        train_manifest = manifest_dir / "train.jsonl"
        val_manifest = manifest_dir / "val.jsonl"

        if not train_manifest.exists() or not val_manifest.exists():
            from src.preprocessing.manifest import ManifestBuilder
            builder = ManifestBuilder(
                data_dir=self.config.data.data_dir,
                manifest_dir=str(manifest_dir),
                train_ratio=self.config.data.train_ratio,
                val_ratio=self.config.data.val_ratio,
                test_ratio=self.config.data.test_ratio,
            )
            builder.build_all()

        train_dataset = L2ArcticDataset(
            manifest_path=str(train_manifest),
            tokenizer=tokenizer,
            sample_rate=self.config.data.sample_rate,
            max_audio_length=self.config.data.max_audio_length,
            min_audio_length=self.config.data.min_audio_length,
            feature_fn=feature_fn,
            augmentation_fn=augmentation_fn,
            cache_dir=self.config.data.cache_dir,
            use_cache=False,
        )

        val_dataset = L2ArcticDataset(
            manifest_path=str(val_manifest),
            tokenizer=tokenizer,
            sample_rate=self.config.data.sample_rate,
            max_audio_length=self.config.data.max_audio_length,
            min_audio_length=self.config.data.min_audio_length,
            feature_fn=feature_fn,
            augmentation_fn=None,
            cache_dir=self.config.data.cache_dir,
            use_cache=False,
        )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
        return train_dataset, val_dataset

    def _build_dataloaders(
        self, train_dataset, val_dataset, tokenizer: PhonemeTokenizer
    ) -> tuple:
        collator = Collator(pad_token_id=tokenizer.blank_id)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            collate_fn=collator,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.persistent_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            collate_fn=collator,
            pin_memory=self.config.training.pin_memory,
            persistent_workers=self.config.training.persistent_workers,
        )

        return train_loader, val_loader

    def _build_model(self, tokenizer: PhonemeTokenizer) -> CNNBiLSTMCTC:
        model = CNNBiLSTMCTC(
            input_dim=self.config.model.input_dim,
            vocab_size=tokenizer.vocab_size,
            cnn_channels=self.config.model.cnn_channels,
            cnn_kernel_sizes=self.config.model.cnn_kernel_sizes,
            cnn_strides=self.config.model.cnn_strides,
            cnn_dropout=self.config.model.cnn_dropout,
            rnn_hidden_size=self.config.model.rnn_hidden_size,
            rnn_num_layers=self.config.model.rnn_num_layers,
            rnn_dropout=self.config.model.rnn_dropout,
            rnn_bidirectional=self.config.model.rnn_bidirectional,
        )
        model.init_weights()
        model = model.to(self.device)
        logger.info(f"Model built: {model.__class__.__name__}")
        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        if self.config.optimizer.name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=self.config.optimizer.betas,
            )
        elif self.config.optimizer.name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.config.optimizer.lr,
                momentum=0.9,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")

    def _build_scheduler(self, optimizer, train_loader) -> Any:
        if self.config.scheduler.name == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler.T_max,
                eta_min=self.config.scheduler.eta_min,
            )
        elif self.config.scheduler.name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        elif self.config.scheduler.name == "warmup":
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.scheduler.warmup_steps,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler.T_max,
                eta_min=self.config.scheduler.eta_min,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.config.scheduler.warmup_steps],
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.scheduler.T_max
            )

    def run(self) -> Dict[str, Any]:
        tokenizer = self._build_tokenizer()
        self.config.model.vocab_size = tokenizer.vocab_size
        feature_fn = self._build_feature_extractor()
        augmentation_fn = self._build_augmentation()

        train_dataset, val_dataset = self._build_datasets(
            tokenizer, feature_fn, augmentation_fn
        )
        train_loader, val_loader = self._build_dataloaders(train_dataset, val_dataset, tokenizer)
        model = self._build_model(tokenizer)

        loss_fn = CTCLossWrapper(
            blank_id=tokenizer.blank_id,
            reduction="mean",
            zero_infinity=True,
        )

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer, train_loader)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            config=self.config,
            tokenizer=tokenizer,
            device=self.device,
            resume_from=getattr(self.config.training, "resume_from", None),
        )

        history = trainer.fit(num_epochs=self.config.training.epochs)

        logger.info("Training plots saved by Trainer (live + comprehensive report)")

        self.config.training.save_dir = str(
            Path(getattr(self.config.training, "save_dir", "./checkpoints"))
        )

        return {"history": history, "config": self.config}
