"""
Factory functions for property computation in phenotyping analysis.

Each factory creates a callable that computes properties from 2D image and/or
mask arrays. The returned functions are designed for use with the
compute_properties method of PhenoMe.

Function signature:
    Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]]

Input arrays (when provided):
    - image2d: np.ndarray, shape (H, W) or (H, W, C). Intensity image.
    - mask2d: np.ndarray, shape (H, W). Binary mask; values > 0.5 are foreground.

Output:
    Dict[str, float]: Property names mapped to computed values. NaN for invalid/missing inputs.

Note: Functions that work without masks (mask-free intensity properties) can use
requirement type "image" and simply ignore the mask2d parameter.

"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

from .transforms import normalize_by_dtype_max

# ============================================================================
# Public API
# ============================================================================

_PRESET_NAMES = ("basic", "regionprops", "intensity", "full", "full_extended")

# Regionprops preset lists (reused in get_preset_property_functions)
REGIONPROPS_BASIC = ("area", "perimeter", "eccentricity", "solidity")
REGIONPROPS_FULL = (
    "area",
    "axis_major_length",
    "axis_minor_length",
    "perimeter",
    "eccentricity",
    "solidity",
)
REGIONPROPS_EXTENDED = (
    "area",
    "axis_major_length",
    "axis_minor_length",
    "perimeter",
    "eccentricity",
    "solidity",
    "orientation",
    "extent",
    "equivalent_diameter_area",
    "euler_number",
    "circularity",
)


def create_regionprops_function(
    property_names: list[str], derived_properties: dict[str, Callable] | None = None
) -> Callable:
    """@section Property factory functions
    @order 10

    Create property function that extracts skimage regionprops from a mask.

    Args:
        property_names: Names to extract (e.g. 'area', 'perimeter', 'eccentricity',
            'solidity', 'axis_major_length', 'circularity').
        derived_properties: Optional dict mapping names to (region) -> float functions.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
    """
    # skimage.measure.regionprops attributes (skimage 0.26+ names)
    valid_regionprops = {
        "area",
        "axis_major_length",
        "axis_minor_length",
        "perimeter",
        "eccentricity",
        "solidity",
        "convex_area",
        "orientation",
        "euler_number",
        "extent",
        "equivalent_diameter_area",
    }

    # Derived properties: computed from regionprops (e.g. aspect_ratio = major/minor)
    # Note: compactness = 1/circularity, so it is omitted to avoid redundancy.
    derived_defaults = {
        "aspect_ratio": lambda r: (
            maj / mn
            if (maj := getattr(r, "axis_major_length", None))
            and (mn := getattr(r, "axis_minor_length", None))
            and mn > 0
            else np.nan
        ),
        "circularity": lambda r: (
            4 * np.pi * r.area / (r.perimeter**2) if r.perimeter > 0 else np.nan
        ),
        "roundness": lambda r: (
            4 * r.area / (np.pi * maj**2)
            if (maj := getattr(r, "axis_major_length", None)) and maj > 0
            else np.nan
        ),
    }

    # Merge user-provided derived properties
    if derived_properties:
        derived_defaults.update(derived_properties)

    # Validate property names
    invalid = set(property_names) - valid_regionprops - set(derived_defaults.keys())
    if invalid:
        raise ValueError(f"Invalid property names: {invalid}")

    def property_function(
        image2d: np.ndarray | None, mask2d: np.ndarray | None
    ) -> dict[str, float]:
        """Generated property function."""
        if mask2d is None:
            return dict.fromkeys(property_names, np.nan)

        # Use largest connected component by area; ignore smaller fragments
        region = _get_largest_region(mask2d)
        if region is None:
            return dict.fromkeys(property_names, np.nan)

        result = {}
        for name in property_names:
            if name in valid_regionprops:
                value = getattr(region, name, None)
                result[name] = float(value) if value is not None else np.nan
            elif name in derived_defaults:
                try:
                    value = derived_defaults[name](region)
                    result[name] = float(value) if not np.isnan(value) else np.nan
                except Exception:
                    result[name] = np.nan

        return result

    return property_function


# ============================================================================
# Intensity-based Property Functions
# ============================================================================


def create_masked_intensity_function(
    stat_name: str, stat_func: Callable[[np.ndarray], float]
) -> Callable:
    """Create intensity property function over masked pixels only (mask > 0.5).

    Args:
        stat_name: Statistic name (e.g. 'mean', 'max', 'std').
        stat_func: Function that computes the statistic from a 1D array.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output key: intensity_{stat_name}_masked.
    """

    def intensity_function(
        image2d: np.ndarray | None, mask2d: np.ndarray | None
    ) -> dict[str, float]:
        """Generated intensity function."""
        if image2d is None or mask2d is None:
            return {f"intensity_{stat_name}_masked": np.nan}

        masked_intensity = _get_masked_intensity(image2d, mask2d)
        if masked_intensity is None or len(masked_intensity) == 0:
            return {f"intensity_{stat_name}_masked": np.nan}

        # Special handling for variance/std (need at least 2 pixels)
        if stat_name in ("variance", "std") and len(masked_intensity) < 2:
            return {f"intensity_{stat_name}_masked": np.nan}

        try:
            value = stat_func(masked_intensity)
            return {f"intensity_{stat_name}_masked": float(value)}
        except Exception:
            return {f"intensity_{stat_name}_masked": np.nan}

    return intensity_function


# ============================================================================
# Mask-Free Intensity Properties
# ============================================================================


def create_intensity_function(stat_name: str, stat_func: Callable[[np.ndarray], float]) -> Callable:
    """Create intensity property function over entire image (no mask).

    Args:
        stat_name: Statistic name (e.g. 'mean', 'max', 'std').
        stat_func: Function that computes the statistic from a 2D array.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output key: intensity_{stat_name}.
    """

    def intensity_function(
        image2d: np.ndarray | None, mask2d: np.ndarray | None
    ) -> dict[str, float]:
        """Generated intensity function (no mask)."""
        if image2d is None:
            return {f"intensity_{stat_name}": np.nan}

        # Special handling for variance/std (need at least 2 pixels)
        if stat_name in ("variance", "std") and image2d.size < 2:
            return {f"intensity_{stat_name}": np.nan}

        try:
            value = stat_func(image2d)
            return {f"intensity_{stat_name}": float(value)}
        except Exception:
            return {f"intensity_{stat_name}": np.nan}

    return intensity_function


def create_blur_effect_function() -> Callable:
    """Create property function that computes blur strength (Laplacian variance).

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output key: blur_effect.
    """

    def blur_function(image2d: np.ndarray | None, mask2d: np.ndarray | None) -> dict[str, float]:
        if image2d is None:
            return {"blur_effect": np.nan}
        try:
            value = measure.blur_effect(image2d)
            scalar = float(np.max(value)) if isinstance(value, (list, np.ndarray)) else float(value)
            return {"blur_effect": scalar}
        except Exception:
            return {"blur_effect": np.nan}

    return blur_function


def create_entropy_function() -> Callable:
    """Create property function that computes Shannon entropy of intensity distribution.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output key: intensity_entropy.
    """

    def entropy_function(image2d: np.ndarray | None, mask2d: np.ndarray | None) -> dict[str, float]:
        if image2d is None:
            return {"intensity_entropy": np.nan}
        try:
            value = measure.shannon_entropy(image2d)
            return {"intensity_entropy": float(value)}
        except Exception:
            return {"intensity_entropy": np.nan}

    return entropy_function


# ============================================================================
# Concentric Ring Property Function
# ============================================================================


def compute_concentric_ring_mask(mask2d: np.ndarray, num_rings: int) -> np.ndarray:
    """
    Compute a mask of concentric rings based on distance from object boundary.

    Rings are numbered from 1 (outermost) to num_rings (innermost).
    Background is 0.

    Args:
        mask2d: np.ndarray, shape (H, W). Binary mask; values > 0.5 are foreground.
        num_rings: Number of concentric rings.

    Returns:
        np.ndarray, shape (H, W), dtype int. Pixel values 0 (background) to num_rings (innermost).
    """
    # Ensure boolean mask; values > 0.5 are foreground
    binary_mask = mask2d > 0.5

    if not np.any(binary_mask):
        return np.zeros(mask2d.shape, dtype=int)

    # Distance transform: each pixel = distance to nearest background (boundary)
    # This naturally handles C-shapes etc.
    dists = np.asarray(distance_transform_edt(binary_mask))
    max_dist = float(np.max(dists))

    if max_dist == 0:
        return np.zeros(mask2d.shape, dtype=int)

    # Split distance range into num_rings bins; ring 1 = outermost, num_rings = innermost
    edges = np.linspace(0, max_dist, num_rings + 1)
    ring_mask = np.zeros(mask2d.shape, dtype=int)

    for i in range(num_rings):
        lower = edges[i]
        upper = edges[i + 1]

        # Pixels in this ring: [lower, upper) except innermost includes upper.
        # Restrict to foreground (dists=0 is background).
        if i == num_rings - 1:
            mask_ring = (dists >= lower) & (dists <= upper + 1e-9) & binary_mask
        else:
            mask_ring = (dists >= lower) & (dists < upper) & binary_mask

        ring_mask[mask_ring] = i + 1

    return ring_mask


def create_concentric_ring_function(num_rings: int, stats: list[str] | None = None) -> Callable:
    """Create property function that computes stats per concentric ring (edge to core).

    Args:
        num_rings: Number of rings.
        stats: Statistics per ring: 'mean', 'std', 'max', 'min', 'median'.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output keys: ring_{1..N}_{stat}.
    """
    if stats is None:
        stats = ["mean", "std"]
    valid_stats = {
        "mean": np.mean,
        "std": np.std,
        "max": np.max,
        "min": np.min,
        "median": np.median,
    }

    # Validate stats
    for stat in stats:
        if stat not in valid_stats:
            raise ValueError(f"Invalid stat: {stat}. Valid options: {list(valid_stats.keys())}")

    def ring_function(image2d: np.ndarray | None, mask2d: np.ndarray | None) -> dict[str, float]:
        """Generated concentric ring function."""
        # Initialize result with NaNs
        result = {}
        for i in range(num_rings):
            for stat in stats:
                result[f"ring_{i + 1}_{stat}"] = np.nan

        if image2d is None or mask2d is None:
            return result

        if image2d.shape != mask2d.shape:
            return result

        # Normalize image
        image2d = normalize_by_dtype_max(image2d)

        # Compute ring mask
        ring_mask = compute_concentric_ring_mask(mask2d, num_rings)

        if not np.any(ring_mask > 0):
            return result

        for i in range(num_rings):
            ring_idx = i + 1
            # Extract intensities for current ring
            in_ring = ring_mask == ring_idx
            ring_vals = image2d[in_ring]

            if len(ring_vals) == 0:
                continue

            for stat in stats:
                try:
                    if stat == "std" and len(ring_vals) < 2:
                        val = 0.0
                    else:
                        val = valid_stats[stat](ring_vals)
                    result[f"ring_{i + 1}_{stat}"] = float(val)
                except Exception:
                    pass  # Keep NaN values

        return result

    return ring_function


# ============================================================================
# Texture Property Function (GLCM)
# ============================================================================


def create_texture_function(
    properties: list[str] | None = None,
) -> Callable:
    """Create property function that computes GLCM texture features within the mask.

    Args:
        properties: GLCM properties. Default: contrast, dissimilarity, homogeneity,
            energy, correlation. Valid: 'contrast', 'dissimilarity', 'homogeneity',
            'energy', 'correlation', 'ASM'.

    Returns:
        Callable[[Optional[np.ndarray], Optional[np.ndarray]], Dict[str, float]].
        Output keys: texture_{prop}.
    """
    valid_props = {"contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"}
    props = properties or ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    invalid = set(props) - valid_props
    if invalid:
        raise ValueError(f"Invalid texture properties: {invalid}. Valid: {list(valid_props)}")

    def texture_function(image2d: np.ndarray | None, mask2d: np.ndarray | None) -> dict[str, float]:
        """Generated texture function."""
        result = {f"texture_{p}": np.nan for p in props}
        if image2d is None or mask2d is None:
            return result
        if image2d.shape != mask2d.shape:
            return result

        region = _get_largest_region(mask2d)
        if region is None:
            return result

        try:
            from skimage.feature import graycomatrix, graycoprops

            min_row, min_col, max_row, max_col = region.bbox
            crop_img = image2d[min_row:max_row, min_col:max_col].astype(np.float64)
            crop_mask = mask2d[min_row:max_row, min_col:max_col] > 0.5

            if not np.any(crop_mask) or crop_img.size < 4:
                return result

            # Normalize and mask: set background to 0
            crop_norm = normalize_by_dtype_max(crop_img)
            crop_norm[~crop_mask] = 0
            img_uint8 = (crop_norm * 255).astype(np.uint8)

            glcm = graycomatrix(
                img_uint8,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True,
            )

            for p in props:
                vals = graycoprops(glcm, p)
                if vals.size > 0:
                    result[f"texture_{p}"] = float(np.mean(vals))
        except Exception:
            pass

        return result

    return texture_function


def get_preset_property_functions(
    preset: str,
) -> dict[str, list[Callable]]:
    """Return a preset dict of property functions for use with compute_properties.

    Args:
        preset: "basic", "regionprops", "intensity", "full", or "full_extended".

    Returns:
        Dict with keys 'image', 'mask', 'both' mapping to lists of property functions.
    """
    if preset not in _PRESET_NAMES:
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {_PRESET_NAMES}")

    regionprops_basic = create_regionprops_function(list(REGIONPROPS_BASIC))
    regionprops_full = create_regionprops_function(list(REGIONPROPS_FULL))
    regionprops_extended = create_regionprops_function(list(REGIONPROPS_EXTENDED))
    intensity_mean = create_intensity_function("mean", np.mean)
    intensity_std = create_intensity_function("std", np.std)
    intensity_min = create_intensity_function("min", np.min)
    intensity_max = create_intensity_function("max", np.max)
    masked_mean = create_masked_intensity_function("mean", np.mean)
    masked_std = create_masked_intensity_function("std", np.std)
    ring_func = create_concentric_ring_function(3, ["mean", "std"])
    texture_func = create_texture_function()
    masked_min = create_masked_intensity_function("min", np.min)
    masked_max = create_masked_intensity_function("max", np.max)
    blur_func = create_blur_effect_function()
    entropy_func = create_entropy_function()

    if preset == "basic":
        return {
            "image": [intensity_mean, intensity_std],
            "mask": [regionprops_basic],
            "both": [masked_mean, masked_std],
        }
    if preset == "regionprops":
        return {"mask": [regionprops_full]}
    if preset == "intensity":
        return {
            "image": [intensity_mean, intensity_std, intensity_min, intensity_max],
        }
    if preset == "full":
        return {
            "image": [intensity_mean, intensity_std, intensity_min, intensity_max],
            "mask": [regionprops_full],
            "both": [masked_mean, masked_std, ring_func],
        }
    if preset == "full_extended":
        return {
            "image": [
                intensity_mean,
                intensity_std,
                intensity_min,
                intensity_max,
                blur_func,
                entropy_func,
            ],
            "mask": [regionprops_extended],
            "both": [
                masked_mean,
                masked_std,
                masked_min,
                masked_max,
                ring_func,
                texture_func,
            ],
        }

    raise ValueError(f"Unknown preset: {preset}")


# ============================================================================
# Private helpers
# ============================================================================


def _get_largest_region(mask2d: np.ndarray) -> Any:
    """Get the largest region from a binary mask using regionprops.

    Args:
        mask2d: 2D binary mask array (H, W). Values > 0.5 are considered foreground.

    Returns:
        RegionProperties object for the largest region, or None if no valid region.
    """
    binary_mask = (mask2d > 0.5).astype(bool)
    if not np.any(binary_mask):
        return None

    props = measure.regionprops(binary_mask.astype(int))
    if len(props) == 0:
        return None

    # Return region with maximum area
    return max(props, key=lambda x: x.area)


def _get_masked_intensity(image2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray | None:
    """Extract intensity values from image within mask region.

    Args:
        image2d: 2D image array (H, W).
        mask2d: 2D binary mask array (H, W). Values > 0.5 are considered foreground.

    Returns:
        np.ndarray shape (n_pixels,) or None if invalid (shape mismatch, empty mask).
    """
    if image2d.shape != mask2d.shape:
        return None

    binary_mask = (mask2d > 0.5).astype(bool)
    if not np.any(binary_mask):
        return None

    # Extract intensities where mask is True
    return image2d[binary_mask]
