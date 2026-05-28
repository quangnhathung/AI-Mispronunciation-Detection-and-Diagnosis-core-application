from __future__ import annotations

from typing import List


class CERMetric:
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id
        self.total_errors = 0
        total_phonemes = 0

    def reset(self) -> None:
        self.total_errors = 0
        self.total_chars = 0

    def update(
        self,
        predictions: List[str],
        targets: List[str],
    ) -> None:
        for pred, target in zip(predictions, targets):
            if not target:
                continue
            pred_clean = pred.replace(" ", "").replace("<blank>", "")
            target_clean = target.replace(" ", "").replace("<blank>", "")
            errors = self._levenshtein_distance(pred_clean, target_clean)
            self.total_errors += errors
            self.total_chars += len(target_clean)

    def _levenshtein_distance(self, a: str, b: str) -> int:
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
        if self.total_chars == 0:
            return 0.0
        return self.total_errors / self.total_chars
