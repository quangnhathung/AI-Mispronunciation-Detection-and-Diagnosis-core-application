from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger


class ModelCheckpoint:
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        self.best_score: float = float("inf") if mode == "min" else -float("inf")
        self.best_epoch: int = 0
        self.best_checkpoints: List[Dict[str, Any]] = []
        self.last_checkpoint_path: Optional[Path] = None

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_score
        return current > self.best_score

    def _save_checkpoint(
        self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: Any, epoch: int, metrics: Dict[str, float], config: Any,
    ) -> None:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
        }, path)

    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, float],
        config: Any,
    ) -> Optional[str]:
        current_score = metrics.get(self.monitor, 0.0)

        ckpt_path = str(
            self.checkpoint_dir / f"epoch_{epoch:03d}_{self.monitor}_{current_score:.4f}.pt"
        )
        self._save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, metrics, config)
        self.last_checkpoint_path = Path(ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        if self._is_better(current_score):
            self.best_score = current_score
            self.best_epoch = epoch

            best_path = str(self.checkpoint_dir / "best.pt")
            self._save_checkpoint(best_path, model, optimizer, scheduler, epoch, metrics, config)
            logger.info(f"New best model saved: {best_path} (score={current_score:.4f})")

            topk_path = str(
                self.checkpoint_dir / f"topk_epoch_{epoch:03d}_{self.monitor}_{current_score:.4f}.pt"
            )
            self._save_checkpoint(topk_path, model, optimizer, scheduler, epoch, metrics, config)

            self.best_checkpoints.append(
                {"path": topk_path, "score": current_score, "epoch": epoch}
            )
            self.best_checkpoints.sort(
                key=lambda x: x["score"], reverse=(self.mode == "max")
            )
            while len(self.best_checkpoints) > self.save_top_k:
                old = self.best_checkpoints.pop(-1)
                if os.path.exists(old["path"]):
                    os.remove(old["path"])

        if self.save_last:
            last_path = str(self.checkpoint_dir / "last.pt")
            self._save_checkpoint(last_path, model, optimizer, scheduler, epoch, metrics, config)

        return ckpt_path
