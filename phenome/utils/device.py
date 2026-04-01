"""
Device and determinism utilities for the phenotyping pipeline.
"""

import os
import random

import numpy as np
import torch

_DEFAULT_DEVICE: torch.device | None = None


def set_default_device(device: torch.device | str | None) -> None:
    """
    Set the default device for phenotyping operations.

    Args:
        device: Target device (torch.device, string like 'cuda:0', or None to reset).
    """
    global _DEFAULT_DEVICE
    _DEFAULT_DEVICE = None if device is None else torch.device(device)


def get_default_device() -> torch.device:
    """
    Return a default device for phenotyping operations.

    If set_default_device() was called, returns that device.
    Otherwise, uses a single GPU (cuda:0) when available, else CPU.
    When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the first visible GPU.

    Returns:
        torch.device: cuda:0 if CUDA is available, else cpu.
    """
    if _DEFAULT_DEVICE is not None:
        return _DEFAULT_DEVICE
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_determinism(seed: int) -> None:
    """
    Set random seeds for reproducibility across all frameworks.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
