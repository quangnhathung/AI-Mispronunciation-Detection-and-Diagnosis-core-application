from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CTCLossWrapper(nn.Module):
    def __init__(
        self,
        blank_id: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction=reduction,
            zero_infinity=zero_infinity,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = log_probs.permute(1, 0, 2)
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss

    def __repr__(self) -> str:
        return (
            f"CTCLossWrapper(blank_id={self.blank_id}, "
            f"reduction={self.reduction})"
        )
