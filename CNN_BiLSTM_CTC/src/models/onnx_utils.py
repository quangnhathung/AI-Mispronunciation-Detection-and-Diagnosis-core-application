from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_dim: int = 80,
    max_length: int = 2000,
    batch_size: int = 1,
    device: str = "cpu",
    verbose: bool = False,
) -> None:
    model.eval()
    dummy_input = torch.randn(batch_size, input_dim, max_length, device=device)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["audio"],
        output_names=["log_probs"],
        dynamic_axes={
            "audio": {0: "batch_size", 2: "time"},
            "log_probs": {0: "batch_size", 1: "time"},
        },
        opset_version=14,
        verbose=verbose,
    )


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    device: str = "cpu",
) -> None:
    model.eval()
    scripted = torch.jit.script(model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(output_path)


def verify_onnx(path: str, input_dim: int = 80, max_length: int = 500) -> bool:
    try:
        import onnx
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return True
    except Exception:
        return False
