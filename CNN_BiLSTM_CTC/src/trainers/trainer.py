from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.decoders.greedy import GreedyDecoder
from src.losses.ctc_loss import CTCLossWrapper
from src.metrics.per import PERMetric
from src.metrics.f1 import F1Metric
from src.utils.helpers import count_parameters, to_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: CTCLossWrapper,
        config: Any,
        tokenizer: Any,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        save_dir = getattr(config.training, "save_dir", "./checkpoints")
        self.monitor_metric = getattr(config.training, "monitor_metric", "val_f1_macro")
        monitor_mode = "max" if "f1" in self.monitor_metric else "min"

        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=save_dir,
            monitor=self.monitor_metric,
            mode=monitor_mode,
            save_top_k=getattr(config.training, "save_top_k", 3),
        )

        self.early_stopping = EarlyStopping(
            patience=getattr(config.training, "early_stopping_patience", 10),
            mode=monitor_mode,
        )

        self.greedy_decoder = GreedyDecoder(blank_id=tokenizer.blank_id)
        self.per_metric = PERMetric(blank_id=tokenizer.blank_id)
        self.f1_metric = F1Metric(
            blank_id=tokenizer.blank_id,
            num_classes=tokenizer.vocab_size,
        )

        self.writer: Optional[SummaryWriter] = None
        if getattr(config.logging, "tensorboard", False):
            log_dir = Path(getattr(config.logging, "log_dir", "./logs")) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))

        self.csv_logger: Optional[Any] = None
        if getattr(config.logging, "csv_log", False):
            log_dir = Path(getattr(config.logging, "log_dir", "./logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = log_dir / "training_log.csv"
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                [
                    "epoch",
                    "step",
                    "train_loss",
                    "val_loss",
                    "train_per",
                    "val_per",
                    "val_f1_macro",
                    "val_f1_micro",
                    "lr",
                    "gpu_memory_mb",
                    "elapsed_time",
                ]
            )
        else:
            self.csv_path = None
            self.csv_file = None
            self.csv_writer = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp())

        logger.info(
            f"Model parameters: {count_parameters(model)}"
        )
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {self._use_amp()}")

    def _use_amp(self) -> bool:
        return (
            self.device.type == "cuda"
            and getattr(self.config.training, "mixed_precision", False)
        )

    def _get_lr(self) -> float:
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def _log_metrics(
        self,
        tag: str,
        metrics: Dict[str, float],
        step: int,
    ) -> None:
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{tag}/{key}", value, step)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_predictions: List[List[int]] = []
        all_targets: List[List[int]] = []

        pbar = self.train_loader
        for batch_idx, batch in enumerate(pbar):
            batch = to_device(batch, self.device)
            audios = batch["audio"]
            phonemes = batch["phonemes"]
            audio_lengths = batch["audio_lengths"]
            phoneme_lengths = batch["phoneme_lengths"]

            with torch.cuda.amp.autocast(enabled=self._use_amp()):
                log_probs = self.model(audios, audio_lengths)
                input_lengths = self.model.get_feat_lengths(audio_lengths)
                loss = self.loss_fn(log_probs, phonemes, input_lengths, phoneme_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss at batch {batch_idx}, skipping")
                continue

            loss = loss / getattr(self.config.training, "gradient_accumulation", 1)

            if self._use_amp():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % getattr(self.config.training, "gradient_accumulation", 1) == 0:
                grad_clip = getattr(self.config.training, "gradient_clip", 5.0)
                if self._use_amp():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * getattr(self.config.training, "gradient_accumulation", 1)

            with torch.no_grad():
                preds = self.greedy_decoder.decode(log_probs, input_lengths)
                for pred, phon, plen in zip(preds, phonemes, phoneme_lengths):
                    all_predictions.append(pred)
                    all_targets.append(phon[:plen].tolist())

            if batch_idx % getattr(self.config.training, "log_interval", 10) == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {self.current_epoch} | Batch {batch_idx}/{len(self.train_loader)} "
                    f"| Loss: {avg_loss:.4f} | LR: {self._get_lr():.2e}"
                )

        avg_loss = total_loss / len(self.train_loader)
        self.per_metric.reset()
        self.per_metric.update(all_predictions, all_targets)
        avg_per = self.per_metric.compute()

        metrics = {"loss": avg_loss, "per": avg_per}
        self._log_metrics("train", metrics, self.current_epoch)
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_predictions: List[List[int]] = []
        all_targets: List[List[int]] = []

        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            audios = batch["audio"]
            phonemes = batch["phonemes"]
            audio_lengths = batch["audio_lengths"]
            phoneme_lengths = batch["phoneme_lengths"]

            log_probs = self.model(audios, audio_lengths)
            input_lengths = self.model.get_feat_lengths(audio_lengths)
            loss = self.loss_fn(log_probs, phonemes, input_lengths, phoneme_lengths)

            total_loss += loss.item()

            preds = self.greedy_decoder.decode(log_probs, input_lengths)
            for pred, phon, plen in zip(preds, phonemes, phoneme_lengths):
                all_predictions.append(pred)
                all_targets.append(phon[:plen].tolist())

        avg_loss = total_loss / len(self.val_loader)
        self.per_metric.reset()
        self.per_metric.update(all_predictions, all_targets)
        avg_per = self.per_metric.compute()

        self.f1_metric.reset()
        self.f1_metric.update(all_predictions, all_targets)
        f1_scores = self.f1_metric.compute()

        metrics = {
            "loss": avg_loss,
            "per": avg_per,
            "f1_macro": f1_scores["f1_macro"],
            "f1_micro": f1_scores["f1_micro"],
            "precision_macro": f1_scores["precision_macro"],
            "recall_macro": f1_scores["recall_macro"],
        }
        self._log_metrics("val", metrics, self.current_epoch)
        return metrics

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        logger.info("Starting training")
        start_time = time.time()
        history = {"train_loss": [], "val_loss": [], "train_per": [], "val_per": [], "val_f1_macro": [], "val_f1_micro": [], "lr": []}

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_metrics = self.train_epoch()

            if epoch % getattr(self.config.training, "eval_interval", 1) == 0:
                val_metrics = self.validate()
            else:
                val_metrics = {"loss": 0.0, "per": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    plateau_metric = val_metrics.get(self.monitor_metric.replace("val_", ""), val_metrics["loss"])
                    self.scheduler.step(plateau_metric)
                else:
                    self.scheduler.step()

            current_lr = self._get_lr()
            epoch_time = time.time() - epoch_start

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_per"].append(train_metrics["per"])
            history["val_per"].append(val_metrics["per"])
            history["val_f1_macro"].append(val_metrics["f1_macro"])
            history["val_f1_micro"].append(val_metrics["f1_micro"])
            history["lr"].append(current_lr)

            gpu_memory = 0
            if self.device.type == "cuda":
                gpu_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

            if self.csv_writer:
                self.csv_writer.writerow(
                    [
                        epoch,
                        self.global_step,
                        f"{train_metrics['loss']:.4f}",
                        f"{val_metrics['loss']:.4f}",
                        f"{train_metrics['per']:.4f}",
                        f"{val_metrics['per']:.4f}",
                        f"{val_metrics['f1_macro']:.4f}",
                        f"{val_metrics['f1_micro']:.4f}",
                        f"{current_lr:.2e}",
                        f"{gpu_memory:.1f}",
                        f"{epoch_time:.1f}",
                    ]
                )
                self.csv_file.flush()

            logger.info(
                f"Epoch {epoch}/{num_epochs} "
                f"| Train Loss: {train_metrics['loss']:.4f} "
                f"| Train PER: {train_metrics['per']:.4f} "
                f"| Val Loss: {val_metrics['loss']:.4f} "
                f"| Val PER: {val_metrics['per']:.4f} "
                f"| Val F1 Macro: {val_metrics['f1_macro']:.4f} "
                f"| Val F1 Micro: {val_metrics['f1_micro']:.4f} "
                f"| LR: {current_lr:.2e} "
                f"| Time: {epoch_time:.1f}s"
            )

            ckpt_metrics = {
                "val_loss": val_metrics["loss"],
                "val_per": val_metrics["per"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_micro": val_metrics["f1_micro"],
            }
            self.checkpoint_callback(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                ckpt_metrics,
                self.config,
            )

            monitor_value = val_metrics.get(self.monitor_metric.replace("val_", ""), val_metrics["loss"])
            if self.early_stopping(monitor_value):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")

        if self.writer:
            self.writer.close()
        if self.csv_file:
            self.csv_file.close()

        return history
