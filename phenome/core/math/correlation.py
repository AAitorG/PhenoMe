"""
Correlation algorithms.

Pure functions for Pearson, Spearman, distance correlation, and mutual information.
Pearson and histogram entropy use PyTorch on CPU; scipy for Spearman; dcor for
distance correlation; sklearn mutual_info_regression for mutual information (KNN-based).
"""

import contextlib

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from ..._logging import get_logger
from ...utils.device import get_default_device

logger = get_logger(__name__)

try:
    import dcor

    _DCOR_AVAILABLE = True
except ImportError:
    _DCOR_AVAILABLE = False


def clean_correlation_inputs(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 3,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Clean correlation inputs by removing NaN/inf values.

    Args:
        x: First array, shape (n_samples,) or (n_samples, n_features).
        y: Second array, shape (n_samples,).
        min_samples: Minimum number of valid samples required.

    Returns:
        Tuple of (x_clean, y_clean, valid_mask) or (None, None, valid_mask) if insufficient samples:
        - x_clean: np.ndarray or None. Same shape as x (minus dropped rows).
        - y_clean: np.ndarray or None. Shape (n_valid,).
        - valid_mask: np.ndarray, shape (n_samples,), dtype bool. True where both x and y are finite.
    """
    # Drop rows where either x or y has NaN/inf; require min_samples valid pairs
    y_ok = np.isfinite(y)
    if x.ndim == 1:
        ok = np.isfinite(x) & y_ok
        n_dropped = y.shape[0] - ok.sum()
        if n_dropped > 0:
            logger.warning(f"Dropped {n_dropped} sample(s) due to NaN/inf values in inputs.")

        if ok.sum() < min_samples:
            return None, None, ok
        return x[ok], y[ok], ok
    else:
        # For 2D x: drop entire row if any feature is non-finite
        x_ok = np.isfinite(x).all(axis=1)
        ok = x_ok & y_ok
        n_dropped = y.shape[0] - ok.sum()
        if n_dropped > 0:
            logger.warning(
                f"Dropped {n_dropped} sample(s) due to NaN/inf values (listwise deletion). "
                f"Consider handling missing values beforehand to preserve statistical power."
            )

        if ok.sum() < min_samples:
            return None, None, ok
        return x[ok, :], y[ok], ok


def compute_pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute Pearson correlation between x and y using torch.corrcoef on GPU/CPU.

    Args:
        x: First array (n_samples,) or (n_samples, n_features).
        y: Second array (n_samples,).
        device: Torch device to run computation on. Defaults to utils.device.get_default_device().

    Returns:
        np.ndarray: Correlation coefficient(s), dtype float64. If x is 1D, shape (1,).
            If x is 2D, shape (n_features,). NaN where insufficient valid samples or constant columns.
    """
    x_clean, y_clean, _ = clean_correlation_inputs(x, y, min_samples=3)
    if x_clean is None:
        return np.array([np.nan]) if x.ndim == 1 else np.full(x.shape[1], np.nan)

    # Pearson correlation undefined when y is constant (zero variance)
    if _is_constant(y_clean):
        return np.array([np.nan]) if x.ndim == 1 else np.full(x.shape[1], np.nan)

    result = _pearson_torch(x_clean, y_clean, device=device)
    if x.ndim == 1:
        r = (
            float(result)
            if isinstance(result, (float, np.floating))
            else float(np.asarray(result).item())
        )
        return np.array([np.nan if not np.isfinite(r) else r], dtype=np.float64)
    out = np.asarray(result, dtype=np.float64)
    for i in range(x_clean.shape[1]):
        if _is_constant(x_clean[:, i]):
            out[i] = np.nan
    return out


def compute_spearman_correlation(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute Spearman rank correlation between x and y using scipy.

    Spearman is Pearson correlation applied to ranks. Robust to outliers and
    non-normality; captures monotonic (not just linear) relationships.

    Args:
        x: First array (n_samples,) or (n_samples, n_features).
        y: Second array (n_samples,).
        device: Ignored; Spearman uses scipy on CPU (kept for API parity with other metrics).

    Returns:
        np.ndarray: Correlation coefficient(s), dtype float64. If x is 1D, scalar (as length-1 array).
            If x is 2D, shape (n_features,). NaN where insufficient valid samples.
    """
    x_clean, y_clean, _ = clean_correlation_inputs(x, y, min_samples=3)
    if x_clean is None:
        return np.array([np.nan]) if x.ndim == 1 else np.full(x.shape[1], np.nan)

    if x.ndim == 1:
        if _is_constant(x_clean) or _is_constant(y_clean):
            return np.array([np.nan])
        try:
            r, _ = spearmanr(x_clean, y_clean)
            return np.array([np.nan if not np.isfinite(r) else r])
        except (ValueError, TypeError):
            return np.array([np.nan])

    results = np.full(x.shape[1], np.nan)
    y_constant = _is_constant(y_clean)
    for i in range(x.shape[1]):
        if y_constant or _is_constant(x_clean[:, i]):
            results[i] = np.nan
            continue
        try:
            r, _ = spearmanr(x_clean[:, i], y_clean)
            results[i] = np.nan if not np.isfinite(r) else r
        except (ValueError, TypeError):
            pass
    return results


def compute_distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute distance correlation between x and y.

    Args:
        x: First array (n_samples,) or (n_samples, n_features).
        y: Second array (n_samples,).
        device: Torch device (included for API consistency; dcor uses CPU).

    Returns:
        np.ndarray: Distance correlation coefficient(s), dtype float64. If x is 1D, scalar (as array).
            If x is 2D, shape (n_features,). NaN where insufficient valid samples.

    Raises:
        ImportError: If the ``dcor`` library is not installed.
    """
    if not _DCOR_AVAILABLE:
        raise ImportError(
            "dcor library is required for distance correlation. Install it with: pip install dcor"
        )

    x_clean, y_clean, _ = clean_correlation_inputs(x, y, min_samples=4)
    if x_clean is None or y_clean is None:
        return np.nan if x.ndim == 1 else np.full(x.shape[1], np.nan)

    # dcor requires float64 for numerical stability
    x_clean = x_clean.astype(np.float64)
    y_clean = y_clean.astype(np.float64)

    if x.ndim == 1:
        try:
            return np.array(dcor.distance_correlation(x_clean, y_clean))
        except Exception:
            return np.array(np.nan)
    results = np.full(x.shape[1], np.nan)
    for i in range(x.shape[1]):
        try:
            results[i] = dcor.distance_correlation(x_clean[:, i], y_clean)
        except Exception:
            results[i] = np.nan
    return results


def compute_entropy(
    data: np.ndarray,
    bins: int = 10,
    device: torch.device | str | None = None,
) -> float | np.ndarray:
    """Compute entropy of data using histogram (PyTorch on GPU/CPU).

    Args:
        data: np.ndarray, shape (n_samples,) or (n_samples, n_features). If 2D, entropy per column.
        bins: Number of histogram bins.
        device: Torch device to run computation on. Defaults to utils.device.get_default_device().

    Returns:
        float or np.ndarray: Shannon entropy in nats. Float for 1D; shape (n_features,) for 2D, dtype float64.
    """
    if data.size == 0:
        return 0.0 if data.ndim == 1 else np.array([], dtype=np.float64)

    if data.ndim == 1:
        return _entropy_1d(data, bins, device=device)

    # 2D case: Vectorized entropy per column using torch
    dev = device if device is not None else get_default_device()
    data_t = torch.from_numpy(data.astype(np.float64)).to(dev)

    n_samples, n_features = data_t.shape
    n_bins = max(1, min(bins, n_samples))

    # Vectorized histogram-based entropy
    # Find min/max per column to match np.histogram behavior
    d_min = data_t.min(dim=0).values
    d_max = data_t.max(dim=0).values

    ranges = d_max - d_min
    # Avoid division by zero for constant columns
    ranges[ranges < 1e-12] = 1.0

    # Map data to bin indices [0, n_bins - 1]
    # We use a small epsilon to ensure d_max maps to n_bins - 1
    scaled = (data_t - d_min) / ranges * (n_bins - 1e-7)
    indices = scaled.long().clamp(0, n_bins - 1)

    # Count occurrences: (n_bins, n_features)
    counts = torch.zeros((n_bins, n_features), device=dev, dtype=torch.float64)
    ones = torch.ones_like(data_t)
    counts.scatter_add_(0, indices, ones)

    probs = counts / n_samples
    # H = -sum(p * log(p))
    log_probs = torch.zeros_like(probs)
    mask = probs > 0
    log_probs[mask] = torch.log(probs[mask])

    entropy = -torch.sum(probs * log_probs, dim=0)
    return entropy.cpu().numpy()


def compute_mutual_info(
    x: np.ndarray,
    y: np.ndarray,
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Compute MI-derived correlation coefficient between x and y.

    Uses sklearn mutual_info_regression (KNN-based estimator) mapped to a
    correlation-like [0, 1] value via the Gaussian bivariate transform
    ``sqrt(1 - exp(-2 * MI))``.  This is **not** the standard information-theoretic
    Normalized Mutual Information (``I(X;Y) / sqrt(H(X)*H(Y))``); rather, it
    converts raw MI into a scale comparable to |Pearson r| under a joint-Gaussian
    assumption.

    Args:
        x: First array (n_samples,) or (n_samples, n_features).
        y: Second array (n_samples,).
        seed: Random seed for mutual_info_regression.
        device: Torch device (included for API consistency; sklearn uses CPU).

    Returns:
        np.ndarray: MI-derived correlation coefficient in [0, 1], dtype float64.
            If x is 1D, scalar (as array).  If x is 2D, shape (n_features,).
            NaN where insufficient valid samples.
    """
    # sklearn default n_neighbors=3 requires at least 4 valid samples
    x_clean, y_clean, _ = clean_correlation_inputs(x, y, min_samples=4)
    if x_clean is None:
        return np.nan if x.ndim == 1 else np.full(x.shape[1], np.nan)

    if x.ndim == 1:
        try:
            mi = mutual_info_regression(
                x_clean.reshape(-1, 1),
                y_clean,
                random_state=seed,
            )[0]
            mi_normalized = np.sqrt(1.0 - np.exp(-2.0 * mi))
            return np.array(float(np.clip(mi_normalized, 0.0, 1.0)))
        except Exception:
            return np.array(np.nan)

    n_features = x_clean.shape[1]
    try:
        mi_values = mutual_info_regression(
            x_clean,
            y_clean,
            random_state=seed,
        )
    except Exception:
        mi_values = np.full(n_features, np.nan)
        for i in range(n_features):
            with contextlib.suppress(Exception):
                mi_values[i] = mutual_info_regression(
                    x_clean[:, i].reshape(-1, 1),
                    y_clean,
                    random_state=seed,
                )[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        mi_normalized = np.sqrt(1.0 - np.exp(-2.0 * mi_values))
    return np.clip(np.where(np.isfinite(mi_normalized), mi_normalized, np.nan), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _is_constant(arr: np.ndarray) -> bool:
    """Return True if array has no variance (all values equal).

    Used to avoid ConstantInputWarning from scipy.stats.spearmanr when
    correlation is undefined for constant inputs.
    """
    if arr.size == 0:
        return True
    return bool(np.ptp(arr) == 0)


def _pearson_torch(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device | str | None = None,
) -> float | np.ndarray:
    """Compute Pearson correlation using torch on GPU/CPU.

    Args:
        x: First array (n_samples,) or (n_samples, n_features).
        y: Second array (n_samples,).
        device: Torch device to run computation on. Defaults to utils.device.get_default_device().

    Returns:
        Correlation coefficient (float for 1D x) or array of shape (n_features,)
        for 2D x.
    """
    dev = device if device is not None else get_default_device()
    x_t = torch.from_numpy(x.astype(np.float64)).to(dev)
    y_t = torch.from_numpy(y.astype(np.float64)).to(dev)

    with torch.no_grad():
        # Stack for corrcoef: rows = variables, cols = samples
        if x.ndim == 1:
            stacked = torch.stack([x_t, y_t], dim=0)  # (2, n_samples)
            corr_mat = torch.corrcoef(stacked)
            r = corr_mat[0, 1].item()
            return np.nan if not np.isfinite(r) else r
        else:
            # O(MN) dot-product Pearson computation instead of O(M^2) full covariance
            y_c = y_t - y_t.mean()
            x_c = x_t - x_t.mean(dim=0)

            y_norm = torch.norm(y_c)
            x_norm = torch.norm(x_c, dim=0)

            # Avoid division by zero
            x_norm = torch.where(
                x_norm == 0, torch.tensor(float("inf"), device=dev, dtype=torch.float64), x_norm
            )
            if y_norm == 0:
                y_norm = torch.tensor(float("inf"), device=dev, dtype=torch.float64)

            cov = torch.mv(x_c.t(), y_c)
            r = (cov / (x_norm * y_norm)).cpu().numpy()
            r[~np.isfinite(r)] = np.nan
            return r


def _entropy_1d(
    data: np.ndarray,
    bins: int,
    device: torch.device | str | None = None,
) -> float:
    """Compute Shannon entropy H(X) = -sum(p*log(p)) in nats (natural log).

    Uses torch.histogram for consistency and speed.

    Args:
        data: np.ndarray, shape (n_samples,).
        bins: Number of histogram bins.
        device: Torch device to run computation on. Defaults to utils.device.get_default_device().

    Returns:
        float: Entropy in nats.
    """
    n = len(data)
    if n == 0:
        return 0.0
    n_bins = max(1, min(bins, n))

    dev = device if device is not None else get_default_device()
    # Use torch for histogram calculation
    data_t = torch.from_numpy(data.astype(np.float64)).to(dev)
    # torch.histogram returns a namedtuple (hist, bin_edges)
    hist = torch.histogram(data_t, bins=n_bins).hist

    probs = hist[hist > 0] / n
    entropy = -torch.sum(probs * torch.log(probs))
    return float(entropy.item())
