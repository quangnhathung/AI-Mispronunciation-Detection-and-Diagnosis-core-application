from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class F1Metric:
    def __init__(self, blank_id: int = 0, num_classes: int = 42):
        self.blank_id = blank_id
        self.num_classes = num_classes
        self.confusion: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(
        self,
        predictions: List[List[int]],
        targets: List[List[int]],
    ) -> None:
        for pred, target in zip(predictions, targets):
            pred_clean = [p for p in pred if p != self.blank_id]
            target_clean = [t for t in target if t != self.blank_id]
            aligned_pred, aligned_target = self._align(pred_clean, target_clean)
            for p, t in zip(aligned_pred, aligned_target):
                if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                    self.confusion[t, p] += 1

    def _align(
        self, pred: List[int], target: List[int]
    ) -> Tuple[List[int], List[int]]:
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
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                aligned_pred.append(pred[i - 1])
                aligned_target.append(target[j - 1])
                i -= 1
                j -= 1
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
                    aligned_pred.append(pred[i - 1])
                    aligned_target.append(-1)
                    i -= 1
                else:
                    aligned_pred.append(-1)
                    aligned_target.append(target[j - 1])
                    j -= 1

        aligned_pred.reverse()
        aligned_target.reverse()
        return aligned_pred, aligned_target

    def compute(self) -> Dict[str, float]:
        if self.confusion is None:
            return {"f1_macro": 0.0, "f1_micro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}

        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=float),
            where=(precision + recall) > 0,
        )

        macro_f1 = float(f1.mean())
        macro_precision = float(precision.mean())
        macro_recall = float(recall.mean())

        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        micro_precision = float(np.divide(total_tp, total_tp + total_fp, out=np.zeros(1), where=(total_tp + total_fp) > 0)[0])
        micro_recall = float(np.divide(total_tp, total_tp + total_fn, out=np.zeros(1), where=(total_tp + total_fn) > 0)[0])
        micro_f1 = float(
            np.divide(
                2 * micro_precision * micro_recall,
                micro_precision + micro_recall,
                out=np.zeros(1),
                where=(micro_precision + micro_recall) > 0,
            )[0]
        )

        return {
            "f1_macro": macro_f1,
            "f1_micro": micro_f1,
            "precision_macro": macro_precision,
            "recall_macro": macro_recall,
            "precision_micro": micro_precision,
            "recall_micro": micro_recall,
        }

    def compute_per_class(self, idx_to_phoneme: Optional[Dict[int, str]] = None) -> List[Dict]:
        if self.confusion is None:
            return []

        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp

        results = []
        for i in range(self.num_classes):
            p = float(np.divide(tp[i], tp[i] + fp[i], out=np.zeros(1), where=(tp[i] + fp[i]) > 0)[0])
            r = float(np.divide(tp[i], tp[i] + fn[i], out=np.zeros(1), where=(tp[i] + fn[i]) > 0)[0])
            f = float(np.divide(2 * p * r, p + r, out=np.zeros(1), where=(p + r) > 0)[0])
            label = idx_to_phoneme.get(i, str(i)) if idx_to_phoneme else str(i)
            if tp[i] > 0 or fp[i] > 0 or fn[i] > 0:
                results.append({"phoneme": label, "precision": p, "recall": r, "f1": f, "support": int(tp[i] + fn[i])})

        results.sort(key=lambda x: x["f1"])
        return results
