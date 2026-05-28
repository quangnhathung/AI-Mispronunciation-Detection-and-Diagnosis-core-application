from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PronunciationFeedback:
    utterance_id: str
    speaker: str
    correct_phonemes: List[str] = field(default_factory=list)
    incorrect_phonemes: List[str] = field(default_factory=list)
    substitutions: List[Tuple[str, str, int]] = field(default_factory=list)
    deletions: List[str] = field(default_factory=list)
    insertions: List[str] = field(default_factory=list)
    per: float = 0.0
    accuracy: float = 0.0
    total_phonemes: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "utterance_id": self.utterance_id,
            "speaker": self.speaker,
            "correct_phonemes": self.correct_phonemes,
            "incorrect_phonemes": self.incorrect_phonemes,
            "substitutions": [
                {"expected": s[0], "predicted": s[1], "position": s[2]}
                for s in self.substitutions
            ],
            "deletions": self.deletions,
            "insertions": self.insertions,
            "per": self.per,
            "accuracy": self.accuracy,
            "total_phonemes": self.total_phonemes,
            "error_count": self.error_count,
        }

    def colored_output(self) -> str:
        lines = []
        lines.append(f"Utterance: {self.utterance_id}")
        lines.append(f"Speaker: {self.speaker}")
        lines.append(f"PER: {self.per:.2%} | Accuracy: {self.accuracy:.2%}")

        if self.correct_phonemes:
            correct_str = " ".join(
                [f"\033[32m{p}\033[0m" for p in self.correct_phonemes]
            )
            lines.append(f"Correct: {correct_str}")

        for exp, pred, pos in self.substitutions:
            lines.append(
                f"  Position {pos}: \033[31m'{exp}' -> '{pred}'\033[0m (Substitution)"
            )
        for d in self.deletions:
            lines.append(f"  \033[33m'{d}' missing (Deletion)\033[0m")
        for ins in self.insertions:
            lines.append(f"  \033[34mExtra '{ins}' (Insertion)\033[0m")

        return "\n".join(lines)


class MispronunciationDetector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def detect(
        self,
        predicted_ids: List[int],
        target_ids: List[int],
        utterance_id: str = "",
        speaker: str = "",
    ) -> PronunciationFeedback:
        pred_clean = [p for p in predicted_ids]
        targ_clean = [t for t in target_ids]

        pred_phonemes = self.tokenizer.decode(pred_clean)
        targ_phonemes = self.tokenizer.decode(targ_clean)

        aligned_pred, aligned_target = self._align_ids(pred_clean, targ_clean)

        substitutions: List[Tuple[str, str, int]] = []
        deletions: List[str] = []
        insertions: List[str] = []
        correct: List[str] = []
        incorrect: List[str] = []

        for pos, (p, t) in enumerate(zip(aligned_pred, aligned_target)):
            if p == -1 and t != -1:
                t_ph = self.tokenizer.decode([t])[0]
                deletions.append(t_ph)
                incorrect.append(t_ph)
            elif p != -1 and t == -1:
                p_ph = self.tokenizer.decode([p])[0]
                insertions.append(p_ph)
                incorrect.append(p_ph)
            elif p == t:
                t_ph = self.tokenizer.decode([t])[0]
                correct.append(t_ph)
            else:
                t_ph = self.tokenizer.decode([t])[0]
                p_ph = self.tokenizer.decode([p])[0]
                substitutions.append((t_ph, p_ph, pos))
                incorrect.append(t_ph)

        total = len(targ_clean)
        errors = len(substitutions) + len(deletions) + len(insertions)
        per = errors / max(total, 1)
        accuracy = 1.0 - per

        return PronunciationFeedback(
            utterance_id=utterance_id,
            speaker=speaker,
            correct_phonemes=correct,
            incorrect_phonemes=incorrect,
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            per=per,
            accuracy=accuracy,
            total_phonemes=total,
            error_count=errors,
        )

    def detect_from_tensors(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        utterance_ids: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
    ) -> List[PronunciationFeedback]:
        from src.decoders.greedy import GreedyDecoder

        decoder = GreedyDecoder(blank_id=self.tokenizer.blank_id)
        predictions = decoder.decode(log_probs, input_lengths)

        results = []
        for i, (pred, targ) in enumerate(zip(predictions, targets)):
            targ_len = target_lengths[i].item()
            targ_clean = targ[:targ_len].tolist()
            uid = utterance_ids[i] if utterance_ids else f"utt_{i}"
            spk = speakers[i] if speakers else ""
            feedback = self.detect(pred, targ_clean, uid, spk)
            results.append(feedback)

        return results

    def _align_ids(
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
