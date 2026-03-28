"""
Device and determinism utilities for the phenotyping pipeline.
"""

import os
import random

import numpy as np
import torch


def get_default_device() -> torch.device:
    """
    Return a default device for phenotyping operations.

    Uses a single GPU (cuda:0) when available, otherwise CPU.
    When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the first visible GPU.

    Returns:
        torch.device: cuda:0 if CUDA is available, else cpu.
    """
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
