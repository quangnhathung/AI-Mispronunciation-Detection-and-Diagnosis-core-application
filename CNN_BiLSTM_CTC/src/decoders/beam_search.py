from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class BeamEntry:
    tokens: List[int] = field(default_factory=list)
    score: float = 0.0
    text: str = ""


_CTC_BLANK = 0


class BeamSearchDecoder:
    def __init__(
        self,
        blank_id: int = 0,
        beam_width: int = 10,
        vocabulary: Optional[List[str]] = None,
        lm_weight: float = 0.0,
        lm_model: Optional[object] = None,
    ):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.vocabulary = vocabulary or []
        self.lm_weight = lm_weight
        self.lm_model = lm_model

    def _merge_beam_entries(self, entries: List[BeamEntry]) -> List[BeamEntry]:
        merged = {}
        for entry in entries:
            key = tuple(entry.tokens)
            if key in merged:
                merged[key] = BeamEntry(
                    tokens=entry.tokens,
                    score=self._log_add(merged[key].score, entry.score),
                )
            else:
                merged[key] = BeamEntry(tokens=entry.tokens, score=entry.score)
        return list(merged.values())

    def _log_add(self, log_a: float, log_b: float) -> float:
        if log_a <= -math.inf:
            return log_b
        if log_b <= -math.inf:
            return log_a
        if log_a > log_b:
            return log_a + math.log(1.0 + math.exp(log_b - log_a))
        return log_b + math.log(1.0 + math.exp(log_a - log_b))

    def _apply_lm(self, tokens: List[int]) -> float:
        if self.lm_model is None or self.lm_weight == 0.0:
            return 0.0
        return 0.0

    def decode(self, log_probs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        B, T, V = log_probs.shape
        results = []

        for b in range(B):
            seq_len = lengths[b].item() if lengths is not None else T
            probs = torch.softmax(log_probs[b, :seq_len], dim=-1).cpu().numpy()

            beams = {(): 0.0}

            for t in range(seq_len):
                new_beams: Dict[Tuple[int, ...], float] = {}
                for tokens, score in beams.items():
                    for v in range(V):
                        log_prob = math.log(probs[t, v] + 1e-10)
                        if v == self.blank_id:
                            new_tokens = tokens
                        elif tokens and tokens[-1] == v:
                            new_tokens = tokens + (v,)
                        else:
                            new_tokens = tokens + (v,)

                        new_score = score + log_prob
                        key = new_tokens
                        if key in new_beams:
                            new_beams[key] = self._log_add(new_beams[key], new_score)
                        else:
                            new_beams[key] = new_score

                sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
                beams = dict(sorted_beams[: self.beam_width])

            if beams:
                best_tokens = max(beams.items(), key=lambda x: x[1])[0]
                results.append(list(best_tokens))
            else:
                results.append([])

        return results

    def decode_with_confidence(
        self, log_probs: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[Tuple[List[int], float]]:
        sequences = self.decode(log_probs, lengths)
        B, T, V = log_probs.shape
        results = []

        for b, seq in enumerate(sequences):
            seq_len = lengths[b].item() if lengths is not None else T
            probs = torch.softmax(log_probs[b, :seq_len], dim=-1)

            if seq:
                confs = []
                t_idx = 0
                for token in seq:
                    while t_idx < seq_len:
                        frame_prob = probs[t_idx, token].item()
                        confs.append(frame_prob)
                        t_idx += 1
                        break
                avg_conf = sum(confs) / len(confs) if confs else 0.0
            else:
                avg_conf = 0.0

            results.append((seq, avg_conf))
        return results

    def __repr__(self) -> str:
        return f"BeamSearchDecoder(beam_width={self.beam_width}, blank_id={self.blank_id})"
