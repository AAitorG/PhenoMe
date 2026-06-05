"""
Transforms and transform builder for the phenotyping pipeline.

Includes both NumPy-based image utilities (ensure_hwc, scale_minmax,
quantile_normalize, resolve_intensity_scale, normalize_by_dtype_max) and PyTorch
transforms (TypeMaxNorm, PadToSize, TransformBuilder).
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
        img: np.ndarray, shape (H, W) or (..., C) with channels as last dimension. Any dtype.

    Returns:
        np.ndarray: Same shape as input, dtype float. Values in [0, 1] per channel.
    """
    if img.ndim == 2:
        img = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
        c = img
        if c.max() > c.min():
            return (c - c.min()) / (c.max() - c.min())
        return np.zeros_like(c)

    img = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
    for i in range(img.shape[-1]):
        c = img[..., i]
        if c.max() > c.min():
            img[..., i] = (c - c.min()) / (c.max() - c.min())
        else:
            img[..., i] = 0.0
    return img


def quantile_normalize(img: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Apply quantile normalization to each channel independently.

    Clips values at the specified quantile threshold, then rescales to [0, 1].

    Args:
        img: np.ndarray, shape (H, W) or (..., C) with channels as last dimension. Any dtype.
        quantile: float, quantile threshold for clipping (default: 0.99).

    Returns:
        np.ndarray: Same shape as input, dtype float64. Values in [0, 1] per channel.
    """
    if img.ndim == 2:
        img = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
        c = img
        q_val = np.quantile(c, quantile)
        c_min = c.min()
        if q_val > c_min:
            return np.clip((c - c_min) / (q_val - c_min), 0, 1)
        return np.zeros_like(c)

    img = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
    for i in range(img.shape[-1]):
        c = img[..., i]
        q_val = np.quantile(c, quantile)
        c_min = c.min()
        if q_val > c_min:
            img[..., i] = np.clip((c - c_min) / (q_val - c_min), 0, 1)
        else:
            img[..., i] = np.zeros_like(c)
    return img


STANDARD_INT_SCALES = (255.0, 65535.0, 4_294_967_295.0)


def _numpy_dtype(dtype: np.dtype | torch.dtype) -> np.dtype:
    """Map NumPy or PyTorch dtype to NumPy dtype for scale lookup."""
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, torch.dtype):
        torch_to_numpy = {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float32,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bfloat16: np.float32,
        }
        np_type = torch_to_numpy.get(dtype)
        if np_type is not None:
            return np.dtype(np_type)
    return np.dtype(np.float64)


def _dtype_scale_max(dtype: np.dtype | torch.dtype) -> float | None:
    """Return nominal full-scale max for integer dtypes, or None for floats."""
    np_dtype = _numpy_dtype(dtype)
    if np_dtype.kind in ("u", "i"):
        return float(np.iinfo(np_dtype).max)
    return None


def resolve_intensity_scale(
    data_max: float,
    data_min: float,
    dtype: np.dtype | torch.dtype,
) -> tuple[str, float]:
    """Infer how to normalize image intensities to [0, 1].

    Uses observed value range first (handles misplaced dtypes and float32 loads),
    then falls back to nominal dtype scale or data max.

    Args:
        data_max: Maximum pixel value in the image.
        data_min: Minimum pixel value in the image.
        dtype: Stored array or tensor dtype.

    Returns:
        Tuple of (mode, scale) where mode is ``identity``, ``minmax``, or ``divide``.
        For ``minmax``, scale is unused; for ``divide``, scale is the divisor.
    """
    if data_max <= 1.0:
        return "identity", 1.0
    if data_min < 0:
        return "minmax", 0.0
    for scale in STANDARD_INT_SCALES:
        if data_max <= scale:
            return "divide", scale
    dtype_scale = _dtype_scale_max(dtype)
    if dtype_scale is not None:
        return "divide", dtype_scale
    return "divide", data_max


def normalize_by_dtype_max(img: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] using effective intensity scale inference.

    Images already in [0, 1] (max <= 1.0) are returned unchanged.
    Non-negative images use the smallest standard bit-depth scale (255, 65535, …)
    that fits the data max, so misplaced dtypes (e.g. uint8 stored as int16) scale
    correctly. Signed images with negative values use min-max scaling.

    Args:
        img: np.ndarray, any shape.

    Returns:
        np.ndarray: Same shape, dtype float64. Values in [0, 1].
    """
    out = img.astype(np.float64) if img.dtype != np.float64 else img.copy()
    mx, mn = float(out.max()), float(out.min())
    mode, scale = resolve_intensity_scale(mx, mn, img.dtype)
    if mode == "identity":
        return out
    if mode == "minmax":
        if mx > mn:
            return (out - mn) / (mx - mn)
        return np.zeros_like(out)
    return out / scale


# =============================================================================
# PyTorch transforms
# =============================================================================


class TypeMaxNorm:
    """Normalize image by effective intensity scale (shared with normalize_by_dtype_max).

    PyTorch transform for normalizing images to [0, 1] using value-range inference
    with dtype fallback, matching the NumPy normalization helper.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor.

        Args:
            img: torch.Tensor, shape (C, H, W) or (B, C, H, W).

        Returns:
            torch.Tensor: Same shape. Values in [0, 1].
        """
        mx, mn = img.max().item(), img.min().item()
        mode, scale = resolve_intensity_scale(mx, mn, img.dtype)
        if mode == "identity":
            return img
        if mode == "minmax":
            if mx > mn:
                return (img - mn) / (mx - mn)
            return torch.zeros_like(img)
        return img / scale

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
        n_channels: int = 3,
    ) -> transforms.Compose:
        """Build torchvision transform pipeline for image preprocessing.

        Pipeline steps: ToTensor -> TypeMaxNorm -> [PadToSize] -> [Resize] -> Normalize.
        When both pad_size and resize_size are given, padding is applied first, then resize
        (even if resize_size is smaller than pad_size).

        Args:
            resize_size: Target spatial size (H, W) for final resize. If None, no resize.
            pad_size: Minimum size for H and W before resize. If None, no padding.
            n_channels: Number of input channels for Normalize mean/std (default 3).

        Returns:
            transforms.Compose: Pipeline for (H, W, C) numpy image -> (C, H', W') tensor.
        """
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        mean = (
            self.mean[:n_channels] if len(self.mean) >= n_channels else (self.mean[0],) * n_channels
        )
        std = self.std[:n_channels] if len(self.std) >= n_channels else (self.std[0],) * n_channels

        t = [transforms.ToTensor(), TypeMaxNorm()]
        if pad_size is not None:
            t.append(PadToSize(pad_size=pad_size))
        if resize_size is not None:
            t.append(transforms.Resize((resize_size, resize_size), antialias=True))
        t.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(t)
