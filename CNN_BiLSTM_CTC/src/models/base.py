from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_log_probs(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def count_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "conv" in name or "projection" in name:
                    nn.init.kaiming_uniform_(param, mode="fan_out", nonlinearity="relu")
                elif "rnn" in name:
                    nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "norm.weight" in name:
                nn.init.ones_(param)

    def load_pretrained(self, checkpoint_path: str, strict: bool = True, device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=strict)

    def export_onnx(
        self,
        output_path: str,
        input_dim: int = 80,
        max_length: int = 2000,
        device: str = "cpu",
    ) -> None:
        self.eval()
        dummy_input = torch.randn(1, input_dim, max_length, device=device)
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "time"},
                "output": {0: "batch", 1: "time"},
            },
            opset_version=14,
        )

    def export_torchscript(self, output_path: str, device: str = "cpu") -> None:
        self.eval()
        scripted = torch.jit.script(self)
        scripted.save(output_path)
