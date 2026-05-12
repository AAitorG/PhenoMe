"""
Transforms and transform builder for the phenotyping pipeline.

Includes both NumPy-based image utilities (ensure_hwc, scale_minmax,
quantile_normalize, normalize_by_dtype_max) and PyTorch transforms (TypeMaxNorm,
PadToSize, TransformBuilder).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# =============================================================================
# NumPy-based image utilities
# =============================================================================


def scale_minmax(img: np.ndarray) -> np.ndarray:
    """Apply min-max scaling to each channel independently.

    Args:
        img: np.ndarray, shape (..., C) with channels as last dimension. Any dtype.

    Returns:
        np.ndarray: Same shape as input, dtype float. Values in [0, 1] per channel.
    """
    img = img.copy()  # Avoid modifying input in place
    # Scale each channel independently to [0, 1] (handles multi-channel images)
    for i in range(img.shape[-1]):
        c = img[..., i]
        if c.max() > c.min():
            img[..., i] = (c - c.min()) / (c.max() - c.min())
    return img


def quantile_normalize(img: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Apply quantile normalization to each channel independently.

    Clips values at the specified quantile threshold, then rescales to [0, 1].

    Args:
        img: np.ndarray, shape (..., C) with channels as last dimension. Any dtype.
        quantile: float, quantile threshold for clipping (default: 0.99).

    Returns:
        np.ndarray: Same shape as input, dtype float64. Values in [0, 1] per channel.
    """
    img = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
    # Scale each channel independently to [0, 1] (handles multi-channel images)
    for i in range(img.shape[-1]):
        c = img[..., i]
        q_val = np.quantile(c, quantile)
        c_min = c.min()
        if q_val > c_min:
            img[..., i] = np.clip((c - c_min) / (q_val - c_min), 0, 1)
        else:
            img[..., i] = np.zeros_like(c)
    return img


def normalize_by_dtype_max(img: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] based on dtype-inferred maximum.

    Images already in [0, 1] (max <= 1.0) are returned unchanged.
    8-bit images (max <= 255) are divided by 255; 16-bit (max <= 65535) by 65535.

    Args:
        img: np.ndarray, any shape. Value-based inference: max<=1 unchanged;
            max<=255 (uint8-like), max<=65535 (uint16-like), else divide by max.

    Returns:
        np.ndarray: Same shape, dtype float64. Values in [0, 1].
    """
    out = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
    mx = out.max()
    if mx <= 1.0:
        return out
    if mx <= 255.0:
        return out / 255.0
    if mx <= 65535.0:
        return out / 65535.0
    return out / mx  # fallback: generic min-max via max


# =============================================================================
# PyTorch transforms
# =============================================================================


class TypeMaxNorm:
    """Normalize image by its dtype-inferred maximum (255 or 65535).

    PyTorch transform for normalizing images based on their maximum value.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor.

        Args:
            img: torch.Tensor, shape (C, H, W) or (B, C, H, W). Values in [0, 255] or [0, 65535].

        Returns:
            torch.Tensor: Same shape. Values in [0, 1].

        Raises:
            ValueError: If image max value is not supported.
        """
        mx = img.max()
        if mx <= 1:
            return img
        if mx <= 255:
            return img / 255.0
        if mx <= 65535:
            return img / 65535.0
        raise ValueError(f"Unsupported data type max: {mx}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PadToSize:
    """Pad image (C, H, W) equally on all sides to pad_size.

    PyTorch transform for padding images to a minimum size.
    """

    def __init__(self, pad_size: int = 112):
        """Initialize pad transform.

        Args:
            pad_size: Minimum size for height and width after padding.
        """
        self.pad_size = pad_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Pad image tensor if needed.

        Args:
            img: torch.Tensor, shape (C, H, W). Padded with zeros on all sides.

        Returns:
            torch.Tensor: Shape (C, max(H, pad_size), max(W, pad_size)).
        """
        _c, h, w = img.shape
        if h < self.pad_size or w < self.pad_size:
            # Pad equally on all sides (top/bottom, left/right)
            ph, pw = max(0, self.pad_size - h), max(0, self.pad_size - w)
            pt, pl = ph // 2, pw // 2
            img = F.pad(img, (pl, pw - pl, pt, ph - pt), "constant", 0)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pad_size={self.pad_size})"


class TransformBuilder:
    """@section Transforms
    @order 60

    Builds torchvision transform pipelines for image preprocessing.

    Uses ImageNet normalization by default.

    Args:
        mean: Normalization mean per channel (default: ImageNet).
        std: Normalization std per channel (default: ImageNet).
    """

    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std = std

    def build(
        self,
        resize_size: int | None = None,
        pad_size: int | None = None,
    ) -> transforms.Compose:
        """Build torchvision transform pipeline for image preprocessing.

        Pipeline steps: ToTensor -> TypeMaxNorm -> [PadToSize] -> [Resize] -> Normalize.
        When both pad_size and resize_size are given, padding is applied first, then resize
        (even if resize_size is smaller than pad_size).

        Args:
            resize_size: Target spatial size (H, W) for final resize. If None, no resize.
            pad_size: Minimum size for H and W before resize. If None, no padding.

        Returns:
            transforms.Compose: Pipeline for (H, W, C) numpy image -> (C, H', W') tensor.
        """
        t = [transforms.ToTensor(), TypeMaxNorm()]
        if pad_size is not None:
            t.append(PadToSize(pad_size=pad_size))
        if resize_size is not None:
            t.append(transforms.Resize((resize_size, resize_size), antialias=True))
        t.append(transforms.Normalize(mean=self.mean, std=self.std))
        return transforms.Compose(t)
