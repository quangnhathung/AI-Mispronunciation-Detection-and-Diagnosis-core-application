from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _ensure_has_ph(fig) -> None:
    fig.tight_layout()


def _safe_save(fig, path: Path, dpi: int = 150) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


class TrainingPlotter:
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = Path(output_dir)
        self.live_dir = self.output_dir / "live"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)

    def plot_metrics(
        self,
        history: Dict[str, List[float]],
        title: str = "Training Metrics",
    ) -> str:
        if not HAS_MATPLOTLIB:
            return ""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        epochs = range(1, len(history.get("train_loss", [])) + 1)

        pairs = [
            (("train_loss", "val_loss"), "Loss", 0, None),
            (("train_per", "val_per"), "Phoneme Error Rate", 1, None),
            (("val_f1_macro",), "F1 Macro", 2, None),
            (("val_f1_micro",), "F1 Micro", 3, None),
            (("val_precision_macro", "val_recall_macro"), "Precision / Recall", 4, None),
            (("lr",), "Learning Rate", 5, "log"),
        ]
        for (keys, label, idx, scale) in pairs:
            ax = axes[idx]
            has_data = False
            colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0"]
            for ki, key in enumerate(keys):
                if key in history and history[key]:
                    vals = history[key]
                    x = range(1, len(vals) + 1)
                    offset = len(epochs) - len(vals)
                    if offset > 0:
                        x = range(offset + 1, offset + 1 + len(vals))
                    ax.plot(x, vals, label=key.replace("val_", "val ").replace("train_", "train ").replace("_", " ").title(), marker=".", color=colors[ki % len(colors)])
                    has_data = True
            if has_data:
                ax.set_xlabel("Epoch")
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if scale:
                    ax.set_yscale(scale)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = self.output_dir / "training_metrics.png"
        return _safe_save(fig, save_path)

    def plot_live(
        self,
        history: Dict[str, List[float]],
        epoch: int,
    ) -> str:
        if not HAS_MATPLOTLIB:
            return ""
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        epochs = range(1, epoch + 1)

        pairs = [
            (("train_loss", "val_loss"), "Loss", None),
            (("train_per", "val_per"), "PER", None),
            (("val_f1_macro",), "F1 Macro", None),
            (("val_f1_micro",), "F1 Micro", None),
            (("val_precision_macro", "val_recall_macro"), "Precision / Recall", None),
            (("lr",), "Learning Rate", "log"),
        ]
        for ki, (keys, label, scale) in enumerate(pairs):
            ax = axes[ki]
            has_data = False
            colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]
            for ci, key in enumerate(keys):
                if key in history and history[key]:
                    vals = history[key]
                    x = range(1, len(vals) + 1)
                    offset = epoch - len(vals)
                    if offset > 0:
                        x = range(offset + 1, offset + 1 + len(vals))
                    ax.plot(x, vals, label=key.replace("val_", "val ").replace("train_", "train ").replace("_", " ").title(), marker=".", color=colors[ci % len(colors)], linewidth=1.5)
                    has_data = True
            if has_data:
                ax.set_xlabel("Epoch")
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if scale:
                    ax.set_yscale(scale)

        fig.suptitle(f"Training Progress — Epoch {epoch}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = self.live_dir / f"epoch_{epoch:04d}.png"
        return _safe_save(fig, save_path)

    def plot_per_phoneme(
        self,
        phoneme_stats: Dict[str, Dict[str, float]],
        top_k: int = 15,
        title: str = "Per-Phoneme Accuracy",
    ) -> str:
        if not HAS_MATPLOTLIB or not phoneme_stats:
            return ""
        sorted_ph = sorted(phoneme_stats.items(), key=lambda x: x[1].get("accuracy", 0))
        worst = sorted_ph[:top_k]
        best = sorted_ph[-top_k:]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, top_k * 0.4)))

        for ax, data, label in [(ax1, worst, "Worst"), (ax2, best, "Best")]:
            phs = [d[0] for d in data]
            accs = [d[1].get("accuracy", 0) for d in data]
            precs = [d[1].get("precision", 0) for d in data]
            recs = [d[1].get("recall", 0) for d in data]
            f1s = [d[1].get("f1", 0) for d in data]

            y = np.arange(len(phs))
            height = 0.2
            ax.barh(y - 1.5 * height, accs, height, label="Accuracy", color="#4CAF50")
            ax.barh(y - 0.5 * height, precs, height, label="Precision", color="#2196F3")
            ax.barh(y + 0.5 * height, recs, height, label="Recall", color="#FF9800")
            ax.barh(y + 1.5 * height, f1s, height, label="F1", color="#9C27B0")

            ax.set_yticks(y)
            ax.set_yticklabels(phs, fontsize=9)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Score")
            ax.set_title(f"{label} {top_k} Phonemes")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3, axis="x")
            ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = self.output_dir / "per_phoneme_accuracy.png"
        return _safe_save(fig, save_path)

    def plot_comprehensive_report(
        self,
        history: Dict[str, List[float]],
        phoneme_stats: Optional[Dict[str, Dict[str, float]]] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        id_to_phoneme: Optional[Dict[int, str]] = None,
        best_epoch: int = 0,
    ) -> List[str]:
        if not HAS_MATPLOTLIB:
            return []
        saved = []

        saved.append(self.plot_metrics(history, title="Training Metrics (Comprehensive)"))

        if phoneme_stats:
            saved.append(self.plot_per_phoneme(phoneme_stats, top_k=15))

        if confusion_matrix is not None and confusion_matrix.shape[0] > 1:
            plotter = ConfusionPlotter(str(self.output_dir))
            saved.append(plotter.plot(confusion_matrix, id_to_phoneme, top_k=35))
            saved.append(plotter.plot(confusion_matrix, id_to_phoneme, top_k=len(id_to_phoneme) if id_to_phoneme else confusion_matrix.shape[0], title="Confusion Matrix (Full)"))

        self._save_summary_table(history, phoneme_stats, best_epoch)

        return saved

    def _save_summary_table(
        self,
        history: Dict[str, List[float]],
        phoneme_stats: Optional[Dict[str, Dict[str, float]]] = None,
        best_epoch: int = 0,
    ) -> str:
        path = self.output_dir / "evaluation_summary.txt"
        lines = []
        lines.append("=" * 70)
        lines.append("TRAINING EVALUATION SUMMARY")
        lines.append("=" * 70)

        if best_epoch > 0:
            lines.append(f"\nBest epoch: {best_epoch}")
        lines.append(f"Total epochs trained: {len(history.get('train_loss', []))}")

        for key in ["val_loss", "val_per", "val_f1_macro", "val_f1_micro"]:
            if key in history and history[key]:
                vals = history[key]
                best = min(vals) if "loss" in key or "per" in key else max(vals)
                last = vals[-1]
                best_idx = vals.index(best) + 1
                label = key.replace("val_", "").replace("_", " ").title()
                arrow = "min" if "loss" in key or "per" in key else "max"
                lines.append(f"  {label:20s}: best={best:.4f} (epoch {best_idx}), last={last:.4f} ({arrow})")

        if phoneme_stats:
            lines.append(f"\nPer-Phoneme Summary ({len(phoneme_stats)} phonemes):")
            sorted_ph = sorted(phoneme_stats.items(), key=lambda x: x[1].get("accuracy", 0))
            lines.append(f"  Worst 5:")
            for ph, st in sorted_ph[:5]:
                lines.append(f"    {ph:>8}: acc={st.get('accuracy',0):.2%}, prec={st.get('precision',0):.2%}, rec={st.get('recall',0):.2%}, f1={st.get('f1',0):.2%}, count={st.get('count',0)}")
            lines.append(f"  Best 5:")
            for ph, st in sorted_ph[-5:]:
                lines.append(f"    {ph:>8}: acc={st.get('accuracy',0):.2%}, prec={st.get('precision',0):.2%}, rec={st.get('recall',0):.2%}, f1={st.get('f1',0):.2%}, count={st.get('count',0)}")
            avg_acc = np.mean([s.get("accuracy", 0) for s in phoneme_stats.values()])
            avg_f1 = np.mean([s.get("f1", 0) for s in phoneme_stats.values()])
            lines.append(f"  Average accuracy: {avg_acc:.2%}")
            lines.append(f"  Average F1:       {avg_f1:.2%}")

        lines.append("=" * 70)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(path)


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
        fig.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "_")
        save_path = self.output_dir / f"{safe_title}.png"
        return _safe_save(fig, save_path)


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
        matrix = np.asarray(matrix, dtype=np.float64)
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

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        norm = matrix / row_sums

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        for ax, mat, fmt, cmap, lbl in [
            (ax1, matrix, "d", "Blues", "Counts"),
            (ax2, norm, ".2f", "RdYlBu_r", "Normalized"),
        ]:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0 if lbl == "Counts" else 0, vmax=None if lbl == "Counts" else 1)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Truth")
            ax.set_title(f"{title} ({lbl})")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticklabels(labels, fontsize=5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "_")
        save_path = self.output_dir / f"{safe_title}_top{top_k}.png"
        return _safe_save(fig, save_path)


class PredictionSamplePlotter:
    def __init__(self, output_dir: str = "./outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_samples(
        self,
        samples: List[Dict[str, Any]],
        max_samples: int = 10,
    ) -> str:
        if not HAS_MATPLOTLIB:
            return ""
        samples = samples[:max_samples]
        n = len(samples)
        fig, axes = plt.subplots(n, 1, figsize=(14, max(4, n * 1.5)))
        if n == 1:
            axes = [axes]

        for i, s in enumerate(samples):
            ax = axes[i]
            target = s.get("target_phonemes", [])
            predicted = s.get("predicted_phonemes", [])

            max_len = max(len(target), len(predicted))
            if max_len == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            x = np.arange(max_len)
            t_padded = target + [""] * (max_len - len(target))
            p_padded = predicted + [""] * (max_len - len(predicted))

            colors = []
            for t, p in zip(t_padded, p_padded):
                if not t or not p:
                    colors.append("#E0E0E0")
                elif t == p:
                    colors.append("#4CAF50")
                else:
                    colors.append("#F44336")

            ax.bar(x, np.ones(max_len), color=colors, alpha=0.7, width=0.8)
            ax.set_xticks(x)
            labels = [f"T:{t}\nP:{p}" for t, p in zip(t_padded, p_padded)]
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.set_ylim(0, 1.5)
            ax.set_yticks([])
            ax.set_title(f"{s.get('utterance_id', 'utt')} — PER: {s.get('per', 0):.2%}", fontsize=10)
            for j, (t, p) in enumerate(zip(t_padded, p_padded)):
                if t and p and t != p:
                    ax.annotate("✗", (j, 0.5), ha="center", va="center", fontsize=8, color="white",
                                fontweight="bold", bbox=dict(boxstyle="circle", facecolor="red", alpha=0.7))

        fig.tight_layout()
        save_path = self.output_dir / "prediction_samples.png"
        return _safe_save(fig, save_path)

    def save_text_summary(
        self,
        samples: List[Dict[str, Any]],
        max_samples: int = 20,
    ) -> str:
        path = self.output_dir / "prediction_samples.txt"
        lines = ["Prediction Samples", "=" * 70]
        for s in samples[:max_samples]:
            t = " ".join(s.get("target_phonemes", []))
            p = " ".join(s.get("predicted_phonemes", []))
            per = s.get("per", 0)
            uid = s.get("utterance_id", "?")
            mark = "✓" if per < 0.3 else "✗" if per > 0.7 else "~"
            lines.append(f"\n[{mark}] {uid}  (PER: {per:.2%})")
            lines.append(f"  Target:    {t}")
            lines.append(f"  Predicted: {p}")
        lines.append("=" * 70)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return str(path)
