from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TrainingPlotter:
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Metrics",
    ) -> str:
        if not HAS_MATPLOTLIB:
            return ""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        if "train_loss" in history and "val_loss" in history:
            axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker=".")
            axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker=".")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss")
            axes[0].legend()
            axes[0].grid(True)

        if "train_per" in history and "val_per" in history:
            axes[1].plot(epochs, history["train_per"], label="Train PER", marker=".")
            axes[1].plot(epochs, history["val_per"], label="Val PER", marker=".")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("PER")
            axes[1].set_title("Phoneme Error Rate")
            axes[1].legend()
            axes[1].grid(True)

        if "val_f1_macro" in history:
            axes[2].plot(epochs, history["val_f1_macro"], label="Val F1 Macro", marker=".", color="purple")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("F1")
            axes[2].set_title("F1 Macro Score")
            axes[2].legend()
            axes[2].grid(True)

        if "lr" in history:
            axes[3].plot(epochs, history["lr"], label="Learning Rate", marker=".", color="green")
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("LR")
            axes[3].set_title("Learning Rate")
            axes[3].set_yscale("log")
            axes[3].grid(True)

        fig.suptitle(title)
        plt.tight_layout()
        save_path = self.output_dir / "training_metrics.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)


class SpectrogramPlotter:
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, spectrogram: torch.Tensor, title: str = "Spectrogram") -> str:
        if not HAS_MATPLOTLIB:
            return ""
        spec = spectrogram.squeeze().cpu().numpy()
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time Frame")
        ax.set_ylabel("Mel Bin")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "_")
        save_path = self.output_dir / f"{safe_title}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)


class ConfusionPlotter:
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot(
        self,
        matrix: np.ndarray,
        id_to_phoneme: Optional[Dict[int, str]] = None,
        top_k: int = 30,
        title: str = "Confusion Matrix",
    ) -> str:
        if not HAS_MATPLOTLIB:
            return ""
        matrix = np.asarray(matrix, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return ""
        if matrix.shape[0] > top_k:
            diag_scores = np.diag(matrix)
            top_indices = np.argsort(diag_scores)[-top_k:]
            matrix = matrix[np.ix_(top_indices, top_indices)]
        else:
            top_indices = list(range(matrix.shape[0]))

        labels = (
            [id_to_phoneme.get(i, str(i)) for i in top_indices]
            if id_to_phoneme
            else [str(i) for i in top_indices]
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(matrix, aspect="auto", cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
