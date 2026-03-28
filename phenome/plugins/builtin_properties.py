"""
Built-in property functions for the plugin registry.

Optional property computators (e.g. blob detection). Register with
register_property() or import and pass to compute_properties.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from skimage import feature

from ..utils.transforms import normalize_by_dtype_max
from .registry import register_property


def my_custom_max_intensity(
    image2d: np.ndarray | None,
    mask2d: np.ndarray | None,
) -> dict[str, float]:
    """Example: return the max intensity of the image (no mask).

    Use as a template for creating custom property functions.

    Args:
        image2d: 2D image array (H, W).
        mask2d: Not used.

    Returns:
        Dictionary with 'my_max_intensity' key.
    """
    if image2d is None:
        return {"my_max_intensity": np.nan}
    return {"my_max_intensity": float(np.max(image2d))}


def _nan_result() -> dict[str, float]:
    """Return NaN-filled blob result for invalid inputs."""
    return {
        "blob_count": np.nan,
        "blob_area_mean": np.nan,
        "blob_area_std": np.nan,
        "blob_intensity_mean": np.nan,
        "blob_intensity_std": np.nan,
    }


def _zero_result(n_blobs: int = 0) -> dict[str, float]:
    """Return zero-filled blob result when no valid blobs."""
    return {
        "blob_count": float(n_blobs),
        "blob_area_mean": 0.0,
        "blob_area_std": 0.0,
        "blob_intensity_mean": 0.0,
        "blob_intensity_std": 0.0,
    }


def get_blob_properties(
    image2d: np.ndarray | None,
    mask2d: np.ndarray | None,
) -> dict[str, float]:
    """Extract blob properties using Difference of Gaussians (DoG) blob detection.

    Detects blobs in the image and computes:
    - Number of blobs
    - Mean and standard deviation of blob areas (in pixels²)
    - Mean and standard deviation of blob intensities at blob centers

    Blob area uses the skimage convention: radius ≈ sqrt(2)*sigma for 2D images,
    hence area = 2*pi*sigma² (not pi*sigma²).

    Args:
        image2d: 2D grayscale image array (H, W). Blobs assumed light on dark.
        mask2d: Not used.

    Returns:
        Dictionary with keys:
        - 'blob_count': Number of detected blobs
        - 'blob_area_mean': Mean blob area (in pixels²)
        - 'blob_area_std': Standard deviation of blob areas
        - 'blob_intensity_mean': Mean intensity at blob centers (subpixel interpolated)
        - 'blob_intensity_std': Standard deviation of blob intensities

        All values are np.nan if image is None or invalid.
    """
    if image2d is None:
        return _nan_result()

    try:
        if image2d.ndim != 2:
            return _nan_result()

        # Reject non-finite input to avoid undefined behavior
        if not np.isfinite(image2d).all():
            return _nan_result()

        g_img = normalize_by_dtype_max(image2d)
        h, w = g_img.shape

        # Reject degenerate images
        if h < 3 or w < 3:
            return _zero_result(0)

        # Reject constant images (DoG would be all zeros)
        if np.ptp(g_img) <= 0:
            return _zero_result(0)

        # Scale sigma range with image size for robustness across resolutions.
        # Typical cell blobs: ~3-50 px radius -> sigma ~2-35 (radius ~ sqrt(2)*sigma).
        min_sigma = 2.0
        max_sigma = max(5.0, min(50.0, min(h, w) / 12.0))

        blob_kwargs: dict = {
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "sigma_ratio": 1.6,
            "overlap": 0.3,
            "threshold": 0.06,
            "threshold_rel": 0.1,
            "exclude_border": 2,  # Avoid partial blobs at image edges
        }
        blobs = feature.blob_dog(g_img, **blob_kwargs)

        if len(blobs) == 0:
            return _zero_result(0)

        n_blobs = len(blobs)
        sigmas = blobs[:, 2]
        # Per skimage: radius ≈ sqrt(2)*sigma for 2D -> area = 2*pi*sigma²
        blob_areas = 2.0 * np.pi * sigmas**2

        blob_y = blobs[:, 0]
        blob_x = blobs[:, 1]

        # Valid blobs: centers inside image (for subpixel sampling)
        valid_mask = (blob_y >= 0) & (blob_y < h - 1e-6) & (blob_x >= 0) & (blob_x < w - 1e-6)

        if not np.any(valid_mask):
            return _zero_result(n_blobs)

        valid_y = blob_y[valid_mask]
        valid_x = blob_x[valid_mask]
        valid_areas = blob_areas[valid_mask]

        # Subpixel intensity via bilinear interpolation (more accurate than truncation)
        coords = np.stack([valid_y, valid_x], axis=0)
        blob_intensities = map_coordinates(g_img, coords, order=1, mode="nearest", cval=np.nan)
        # Drop any NaN from out-of-bounds (should not occur with valid_mask)
        finite = np.isfinite(blob_intensities)
        if not np.any(finite):
            return _zero_result(n_blobs)
        blob_intensities = blob_intensities[finite]
        valid_areas = valid_areas[finite]

        blob_area_mean = float(np.mean(valid_areas))
        blob_area_std = float(np.std(valid_areas)) if len(valid_areas) > 1 else 0.0
        blob_intensity_mean = float(np.mean(blob_intensities))
        blob_intensity_std = float(np.std(blob_intensities)) if len(blob_intensities) > 1 else 0.0

        return {
            "blob_count": float(n_blobs),
            "blob_area_mean": blob_area_mean,
            "blob_area_std": blob_area_std,
            "blob_intensity_mean": blob_intensity_mean,
            "blob_intensity_std": blob_intensity_std,
        }

    except Exception:
        return _nan_result()


# Register built-in properties on import
register_property("blob", get_blob_properties)
register_property("my_max_intensity", my_custom_max_intensity)
