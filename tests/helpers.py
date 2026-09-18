"""Shared helpers for PhenoMe smoke tests (not pytest fixtures)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from phenome.utils.model_wrapper import ModelWrapper


class DummyModelWrapper(ModelWrapper):
    """Deterministic (B, D) embeddings from batch intensity — no torch.hub."""

    def __init__(self, dim: int = 8, device: torch.device | None = None) -> None:
        super().__init__(model=object(), device=device)
        self.dim = dim

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return (B, D) embeddings from per-image mean intensity plus a fixed offset."""
        batch = tensor.shape[0]
        scale = tensor.reshape(batch, -1).mean(dim=1, keepdim=True).to(dtype=torch.float32)
        offsets = torch.arange(self.dim, device=tensor.device, dtype=torch.float32)
        return scale + offsets.unsqueeze(0)


def write_rgb_png(path: Path, value: int) -> None:
    """Write a solid RGB PNG via OpenCV (BGR on disk, matching ``read_image``)."""
    rgb = np.full((16, 16, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok, f"cv2.imwrite failed for {path}"
