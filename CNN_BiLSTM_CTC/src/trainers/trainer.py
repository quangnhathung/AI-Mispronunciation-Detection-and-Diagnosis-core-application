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

import numpy as np

from src.callbacks.early_stopping import EarlyStopping
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.decoders.greedy import GreedyDecoder
from src.losses.ctc_loss import CTCLossWrapper
from src.metrics.per import PERMetric
from src.metrics.f1 import F1Metric
from src.utils.helpers import count_parameters, to_device
from src.visualization.plots import TrainingPlotter, ConfusionPlotter, PredictionSamplePlotter


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
        resume_from: Optional[str] = None,
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

        self._resume_from_checkpoint(resume_from)

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
            csv_mode = "a" if self.current_epoch > 0 else "w"
            self.csv_file = open(self.csv_path, csv_mode, newline="")
            self.csv_writer = csv.writer(self.csv_file)
            if self.current_epoch == 0:
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

        self.plotter = TrainingPlotter(
            output_dir=str(Path(getattr(config.training, "save_dir", "./checkpoints")).parent / "plots"),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp())

        self._log_resume_info()

        logger.info(
            f"Model parameters: {count_parameters(model)}"
        )
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Device: {device}")
        logger.info(f"Mixed precision: {self._use_amp()}")

    def _resume_from_checkpoint(self, path: Optional[str]) -> None:
        if not path:
            return
        path = str(path)
        if not os.path.exists(path):
            logger.warning(f"Resume checkpoint not found: {path}")
            return
        logger.info(f"Loading resume checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] and self.scheduler:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")
        if "epoch" in ckpt:
            self.current_epoch = ckpt["epoch"]
            logger.info(f"Resuming from epoch {self.current_epoch}")
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
        if "scaler_state_dict" in ckpt and self._use_amp():
            try:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore scaler state: {e}")
        if "metrics" in ckpt:
            metrics = ckpt["metrics"]
            score = metrics.get(self.monitor_metric.replace("val_", ""), metrics.get("loss"))
            if score is not None:
                self.checkpoint_callback.best_score = score
                self.checkpoint_callback.best_epoch = self.current_epoch
                self.early_stopping.best_score = score
                logger.info(f"Restored best {self.monitor_metric}: {score:.4f} (epoch {self.current_epoch})")
        logger.info(f"Resume complete — continuing from epoch {self.current_epoch}, step {self.global_step}")

    def _log_resume_info(self) -> None:
        if self.current_epoch > 0:
            msg = f"Resumed session — starting from epoch {self.current_epoch + 1}, step {self.global_step}"
            if self.csv_file and self.csv_path and os.path.exists(self.csv_path):
                msg += f", log appended to {self.csv_path}"
            logger.info(msg)

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

        # Accumulate per-phoneme confusion matrix
        conf_matrix = getattr(self, "_confusion_matrix", None)
        if conf_matrix is None:
            n = self.tokenizer.vocab_size
            self._confusion_matrix = np.zeros((n, n), dtype=np.int64)
            self._sample_predictions = []

        for pred, targ in zip(all_predictions, all_targets):
            aligned_pred, aligned_target = self._align_ids(pred, targ)
            for p, t in zip(aligned_pred, aligned_target):
                if p >= 0 and t >= 0 and p < self.tokenizer.vocab_size and t < self.tokenizer.vocab_size:
                    self._confusion_matrix[t, p] += 1

        if self.current_epoch % max(1, getattr(self.config.training, "eval_interval", 1) * 5) == 0:
            for pred, targ in zip(all_predictions[:10], all_targets[:10]):
                self._sample_predictions.append({
                    "utterance_id": f"epoch_{self.current_epoch}",
                    "target_phonemes": self.tokenizer.decode(targ),
                    "predicted_phonemes": self.tokenizer.decode(pred),
                    "per": self.per_metric.compute() if hasattr(self, 'per_metric') else 0,
                })

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

    def _align_ids(
        self, pred: List[int], target: List[int]
    ) -> tuple[list[int], list[int]]:
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i - 1] == target[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        aligned_pred, aligned_target = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i - 1] == target[j - 1]:
                aligned_pred.append(pred[i - 1])
                aligned_target.append(target[j - 1])
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                aligned_pred.append(pred[i - 1])
                aligned_target.append(target[j - 1])
                i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                aligned_pred.append(pred[i - 1])
                aligned_target.append(-1)
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                aligned_pred.append(-1)
                aligned_target.append(target[j - 1])
                j -= 1
            else:
                if i > 0:
                    aligned_pred.append(pred[i - 1]); aligned_target.append(-1); i -= 1
                else:
                    aligned_pred.append(-1); aligned_target.append(target[j - 1]); j -= 1
        aligned_pred.reverse()
        aligned_target.reverse()
        return aligned_pred, aligned_target

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        start_epoch = self.current_epoch + 1
        target_epochs = num_epochs
        if self.current_epoch > 0:
            logger.info(f"Resuming — already trained {self.current_epoch} epochs, running {num_epochs} more epochs")
            target_epochs = self.current_epoch + num_epochs
        logger.info(f"Starting training — epochs {start_epoch} to {target_epochs}")
        start_time = time.time()
        history = {"train_loss": [], "val_loss": [], "train_per": [], "val_per": [], "val_f1_macro": [], "val_f1_micro": [], "lr": []}

        interrupted = False
        last_metrics: Dict[str, float] = {}
        try:
            for epoch in range(start_epoch, target_epochs + 1):
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
                    f"Epoch {epoch}/{target_epochs} "
                    f"| Train Loss: {train_metrics['loss']:.4f} "
                    f"| Train PER: {train_metrics['per']:.4f} "
                    f"| Val Loss: {val_metrics['loss']:.4f} "
                    f"| Val PER: {val_metrics['per']:.4f} "
                    f"| Val F1 Macro: {val_metrics['f1_macro']:.4f} "
                    f"| Val F1 Micro: {val_metrics['f1_micro']:.4f} "
                    f"| LR: {current_lr:.2e} "
                    f"| Time: {epoch_time:.1f}s"
                )

                last_metrics = {
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
                    last_metrics,
                    self.config,
                )

                # Live plot every epoch
                try:
                    self.plotter.plot_live(history, epoch)
                except Exception as e:
                    logger.warning(f"Live plot failed at epoch {epoch}: {e}")

                monitor_value = val_metrics.get(self.monitor_metric.replace("val_", ""), val_metrics["loss"])
                if self.early_stopping(monitor_value):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (Ctrl+C)")
            interrupted = True
            self._save_emergency_checkpoint(last_metrics)
            logger.info("Emergency checkpoint saved — you can resume later with --resume ./checkpoints/resume.pt")

        total_time = time.time() - start_time
        if interrupted:
            logger.info(f"Training interrupted after {total_time:.1f}s (epoch {self.current_epoch})")
        else:
            logger.info(f"Training completed in {total_time:.1f}s")

        # ── Comprehensive evaluation (skip if interrupted) ──
        if not interrupted:
            try:
                phoneme_stats = self._compute_phoneme_stats()
                self.plotter.plot_comprehensive_report(
                    history=history,
                    phoneme_stats=phoneme_stats,
                    confusion_matrix=self._confusion_matrix if hasattr(self, "_confusion_matrix") else None,
                    id_to_phoneme=self.tokenizer._id_to_phoneme,
                    best_epoch=self.checkpoint_callback.best_epoch,
                )
                sample_plotter = PredictionSamplePlotter(
                    str(Path(getattr(self.config.training, "save_dir", "./checkpoints")).parent / "plots")
                )
                if hasattr(self, "_sample_predictions") and self._sample_predictions:
                    sample_plotter.plot_samples(self._sample_predictions)
                    sample_plotter.save_text_summary(self._sample_predictions)
            except Exception as e:
                logger.warning(f"Comprehensive evaluation plots failed: {e}")

        if self.writer:
            self.writer.close()
        if self.csv_file:
            self.csv_file.close()

        return history

    def _save_emergency_checkpoint(self, metrics: Optional[Dict[str, float]]) -> None:
        save_dir = Path(getattr(self.config.training, "save_dir", "./checkpoints"))
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "resume.pt"
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics or {},
            "config": self.config,
        }
        if self._use_amp():
            state["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(state, str(path))
        logger.info(f"Emergency checkpoint saved: {path}")

    def _compute_phoneme_stats(self) -> Optional[Dict[str, Dict[str, float]]]:
        cm = getattr(self, "_confusion_matrix", None)
        if cm is None or cm.sum() == 0:
            return None
        stats = {}
        n = cm.shape[0]
        for i in range(n):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            count = cm[i, :].sum()
            if count == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            ph = self.tokenizer._id_to_phoneme.get(i, f"ID{i}")
            if ph in ("<blank>", "<unk>"):
                continue
            stats[ph] = {
                "accuracy": float(tp / count) if count > 0 else 0.0,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "count": int(count),
            }
        return stats
