from __future__ import annotations

from typing import List, Optional

import torch


class PERMetric:
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id
        self.total_errors = 0
        self.total_phonemes = 0
        self.total_correct = 0

    def reset(self) -> None:
        self.total_errors = 0
        self.total_phonemes = 0
        self.total_correct = 0

    def update(
        self,
        predictions: List[List[int]],
        targets: List[List[int]],
    ) -> None:
        for pred, target in zip(predictions, targets):
            if not target:
                continue
            pred_clean = [p for p in pred if p != self.blank_id]
            target_clean = [t for t in target if t != self.blank_id]
            errors = self._levenshtein_distance(pred_clean, target_clean)
            self.total_errors += errors
            self.total_phonemes += len(target_clean)
            if pred_clean == target_clean:
                self.total_correct += 1

    def _levenshtein_distance(self, a: List[int], b: List[int]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def compute(self) -> float:
        if self.total_phonemes == 0:
            return 0.0
        return self.total_errors / self.total_phonemes

    def compute_accuracy(self) -> float:
        total = self.total_correct + self.total_errors
        if total == 0:
            return 0.0
        return self.total_correct / total
