from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.decoders.greedy import GreedyDecoder
from src.metrics.per import PERMetric
from src.metrics.cer import CERMetric
from src.metrics.confusion import ConfusionMatrix
from src.metrics.f1 import F1Metric
from src.mdd.detector import MispronunciationDetector
from src.utils.helpers import to_device


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        tokenizer: Any,
        config: Any,
        device: torch.device,
    ):
        self.model = model
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        self.decoder = GreedyDecoder(blank_id=tokenizer.blank_id)
        self.per_metric = PERMetric(blank_id=tokenizer.blank_id)
        self.cer_metric = CERMetric(blank_id=tokenizer.blank_id)
        self.confusion = ConfusionMatrix(
            vocab_size=tokenizer.vocab_size,
            blank_id=tokenizer.blank_id,
        )
        self.f1_metric = F1Metric(
            blank_id=tokenizer.blank_id,
            num_classes=tokenizer.vocab_size,
        )
        self.mdd = MispronunciationDetector(tokenizer)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        self.model.eval()
        all_feedbacks: List[Dict] = []
        all_predictions: List[List[int]] = []
        all_targets: List[List[int]] = []

        for batch_idx, batch in enumerate(self.test_loader):
            batch = to_device(batch, self.device)
            audios = batch["audio"]
            phonemes = batch["phonemes"]
            audio_lengths = batch["audio_lengths"]
            phoneme_lengths = batch["phoneme_lengths"]

            log_probs = self.model(audios, audio_lengths)
            input_lengths = self.model.get_feat_lengths(audio_lengths)

            preds = self.decoder.decode(log_probs, input_lengths)
            for pred, phon, plen in zip(preds, phonemes, phoneme_lengths):
                all_predictions.append(pred)
                all_targets.append(phon[:plen].tolist())

            for i, (pred, phon, plen) in enumerate(zip(preds, phonemes, phoneme_lengths)):
                target = phon[:plen].tolist()
                feedback = self.mdd.detect(
                    predicted_ids=pred,
                    target_ids=target,
                    utterance_id=batch["utterance_ids"][i] if "utterance_ids" in batch else f"batch_{batch_idx}_utt_{i}",
                    speaker=batch["speakers"][i] if "speakers" in batch else "",
                )
                all_feedbacks.append(feedback.to_dict())

        self.per_metric.reset()
        self.per_metric.update(all_predictions, all_targets)
        per = self.per_metric.compute()

        self.confusion.reset()
        self.confusion.update(all_predictions, all_targets)
        confusion_acc = self.confusion.get_accuracy()

        pred_strings = [
            self.tokenizer.decode_to_string(p) for p in all_predictions
        ]
        target_strings = [
            self.tokenizer.decode_to_string(t) for t in all_targets
        ]
        self.cer_metric.reset()
        self.cer_metric.update(pred_strings, target_strings)
        cer = self.cer_metric.compute()

        self.f1_metric.reset()
        self.f1_metric.update(all_predictions, all_targets)
        f1_scores = self.f1_metric.compute()
        per_class_f1 = self.f1_metric.compute_per_class(
            idx_to_phoneme={i: p for i, p in enumerate(self.tokenizer.vocab)}
        )

        results = {
            "per": per,
            "cer": cer,
            "confusion_accuracy": confusion_acc,
            "f1_macro": f1_scores["f1_macro"],
            "f1_micro": f1_scores["f1_micro"],
            "precision_macro": f1_scores["precision_macro"],
            "recall_macro": f1_scores["recall_macro"],
            "per_class_f1": per_class_f1,
            "num_samples": len(all_feedbacks),
            "feedbacks": all_feedbacks,
            "confusion_matrix": self.confusion.matrix.tolist(),
        }

        return results

    def export_predictions(self, output_path: str, results: Dict[str, Any]) -> None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "utterance_id",
                    "speaker",
                    "per",
                    "accuracy",
                    "total_phonemes",
                    "error_count",
                    "error_details",
                ]
            )
            for fb in results["feedbacks"]:
                writer.writerow(
                    [
                        fb["utterance_id"],
                        fb["speaker"],
                        f"{fb['per']:.4f}",
                        f"{fb['accuracy']:.4f}",
                        fb["total_phonemes"],
                        fb["error_count"],
                        json.dumps(fb),
                    ]
                )

        logger.info(f"Predictions exported to {output_path}")

    def export_confusion_report(
        self, output_path: str, results: Dict[str, Any], top_k: int = 20
    ) -> None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        id_to_phoneme = {i: p for i, p in enumerate(self.tokenizer.vocab)}
        confused = self.confusion.get_most_confused(
            top_k=top_k, id_to_phoneme=id_to_phoneme
        )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["truth", "predicted", "count"])
            for t, p, c in confused:
                writer.writerow([t, p, c])

        logger.info(f"Confusion report exported to {output_path}")
