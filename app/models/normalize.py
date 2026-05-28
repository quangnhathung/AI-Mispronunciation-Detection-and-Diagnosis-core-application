from typing import Any, Optional
import torch
import numpy as np
from loguru import logger


def safe_tensor_to_list(
    value: Any,
    desc: str = "value",
) -> list:
    if value is None:
        logger.warning(f"[Normalize] {desc} is None, returning []")
        return []
    if isinstance(value, torch.Tensor):
        logger.debug(f"[Normalize] {desc} is torch.Tensor shape={value.shape} dtype={value.dtype}")
        if value.dim() == 0:
            result = [value.item()]
        elif value.dim() == 1:
            result = value.tolist()
        else:
            result = value.flatten().tolist()
        logger.debug(f"[Normalize] {desc} -> list[int] len={len(result)}")
        return result
    if isinstance(value, np.ndarray):
        logger.debug(f"[Normalize] {desc} is np.ndarray shape={value.shape}")
        if value.ndim == 0:
            return [value.item()]
        return value.flatten().tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if isinstance(value[0], (int, float, str)):
            return list(value)
        if isinstance(value[0], torch.Tensor):
            return [v.item() if v.dim() == 0 else int(v) for v in value]
        return [int(v) for v in value]
    if isinstance(value, (int, float)):
        logger.debug(f"[Normalize] {desc} is scalar {type(value).__name__}={value}")
        return [int(value)]
    logger.warning(f"[Normalize] {desc} unexpected type={type(value).__name__}, returning [int(value)]")
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def safe_item(value: Any, desc: str = "value") -> float:
    if value is None:
        logger.warning(f"[Normalize] {desc} is None, returning 0.0")
        return 0.0
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return float(value.item())
        logger.warning(f"[Normalize] {desc} is tensor with shape={value.shape}, "
                        f"taking .mean().item()")
        return float(value.mean().item())
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.ndim == 0 else float(value.mean())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(np.mean([float(v) for v in value]))
    return float(value)


def safe_to_tensor(
    value: Any,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    desc: str = "value",
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(dtype=dtype)
        if device:
            t = t.to(device)
        return t
    if isinstance(value, np.ndarray):
        t = torch.from_numpy(value).to(dtype=dtype)
        if device:
            t = t.to(device)
        return t
    if isinstance(value, (list, tuple)):
        t = torch.tensor(value, dtype=dtype)
        if device:
            t = t.to(device)
        return t
    if isinstance(value, (int, float)):
        t = torch.tensor([value], dtype=dtype)
        if device:
            t = t.to(device)
        return t
    logger.warning(f"[Normalize] {desc} unexpected type={type(value).__name__}, "
                    f"creating empty tensor")
    return torch.tensor([], dtype=dtype, device=device)


def safe_index(index: Any, desc: str = "index") -> int:
    if isinstance(index, torch.Tensor):
        return int(index.item())
    if isinstance(index, np.integer):
        return int(index)
    return int(index)
