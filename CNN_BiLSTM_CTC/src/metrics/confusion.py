from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, vocab_size: int, blank_id: int = 0):
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.matrix = np.zeros((vocab_size, vocab_size), dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self.matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.int64)

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
                if p < self.vocab_size and t < self.vocab_size:
                    self.matrix[t, p] += 1

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

    def get_most_confused(
        self, top_k: int = 10, id_to_phoneme: Optional[Dict[int, str]] = None
    ) -> List[Tuple[str, str, int]]:
        confused = []
        for t in range(self.vocab_size):
            for p in range(self.vocab_size):
                if t != p and self.matrix[t, p] > 0:
                    t_label = id_to_phoneme.get(t, str(t)) if id_to_phoneme else str(t)
                    p_label = id_to_phoneme.get(p, str(p)) if id_to_phoneme else str(p)
                    confused.append((t_label, p_label, int(self.matrix[t, p])))
        confused.sort(key=lambda x: x[2], reverse=True)
        return confused[:top_k]

    def get_accuracy(self) -> float:
        correct = np.trace(self.matrix)
        total = self.matrix.sum()
        return correct / max(total, 1)
