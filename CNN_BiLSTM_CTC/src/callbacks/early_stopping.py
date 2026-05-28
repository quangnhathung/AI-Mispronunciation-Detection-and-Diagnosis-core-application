from __future__ import annotations

from loguru import logger


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: float = float("inf") if mode == "min" else -float("inf")
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, current_score: float) -> bool:
        if self.mode == "min":
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(
                f"Early stopping triggered after {self.counter} epochs without improvement"
            )

        return self.early_stop

    def reset(self) -> None:
        self.best_score = float("inf") if self.mode == "min" else -float("inf")
        self.counter = 0
        self.early_stop = False
