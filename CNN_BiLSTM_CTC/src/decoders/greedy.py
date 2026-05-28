from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class GreedyDecoder:
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def decode(self, log_probs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        B, T, V = log_probs.shape
        predictions = log_probs.argmax(dim=-1)
        results = []
        for b in range(B):
            seq_len = lengths[b].item() if lengths is not None else T
            pred = predictions[b, :seq_len].tolist()
            decoded = self._collapse(pred)
            results.append(decoded)
        return results

    def decode_with_confidence(
        self, log_probs: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[List[int], float]]:
        B, T, V = log_probs.shape
        probs = torch.softmax(log_probs, dim=-1)
        predictions = log_probs.argmax(dim=-1)

        results = []
        for b in range(B):
            seq_len = lengths[b].item() if lengths is not None else T
            pred = predictions[b, :seq_len]
            prob = probs[b, :seq_len]

            collapsed_ids = []
            collapsed_conf = []
            prev = self.blank_id
            for t in range(seq_len):
                token = pred[t].item()
                if token != self.blank_id and token != prev:
                    collapsed_ids.append(token)
                    collapsed_conf.append(prob[t, token].item())
                prev = token

            avg_conf = sum(collapsed_conf) / len(collapsed_conf) if collapsed_conf else 0.0
            results.append((collapsed_ids, avg_conf))
        return results

    def _collapse(self, sequence: List[int]) -> List[int]:
        collapsed = []
        prev = self.blank_id
        for token in sequence:
            if token != self.blank_id and token != prev:
                collapsed.append(token)
            prev = token
        return collapsed

    def __repr__(self) -> str:
        return f"GreedyDecoder(blank_id={self.blank_id})"
