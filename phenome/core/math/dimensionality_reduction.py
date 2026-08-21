"""
Dimensionality reduction algorithms.

Provides pure matrix-based reduction and pipeline-aware wrapper.
Uses torchdr for GPU acceleration when available and device is CUDA.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

try:
    import umap
except ImportError:
    umap = None

try:
    from torchdr import TSNE as TORCHDR_TSNE
    from torchdr import UMAP as TORCHDR_UMAP
    from torchdr import ExactIncrementalPCA as TorchdrExactIncrementalPCA

    _TORCHDR_AVAILABLE = True
except ImportError:
    _TORCHDR_AVAILABLE = False

try:
    from torchdr.utils import pykeops

    _KEOPS_AVAILABLE = bool(pykeops)
except ImportError:
    _KEOPS_AVAILABLE = False

from ..._logging import get_logger
from ..results_export import prepare_embedding_dataframe
from ..results_metadata import filter_indices
from .combined_features import build_combined_features

logger = get_logger(__name__)


class TSNEInsufficientSamplesError(ValueError):
    """Raised when t-SNE perplexity is not smaller than the number of samples."""


def _torchdr_random_state(seed: int | None) -> int | None:
    """Integer random_state for TorchDR (sklearn-style); None if no seed."""
    if seed is None:
        return None
    return int(seed)


def _should_fallback_torchdr_to_cpu(exc: BaseException) -> bool:
    """True for likely transient GPU / backend failures worth retrying on CPU."""
    if isinstance(exc, (MemoryError, ImportError)):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg:
            return True
        if "cublas" in msg or "cudnn" in msg or "keops" in msg:
            return True
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return oom_cls is not None and isinstance(exc, oom_cls)


def _to_batches(matrix: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Split matrix into batches for memory-efficient processing."""
    return [matrix[i : i + batch_size] for i in range(0, len(matrix), batch_size)]


def _validate_tsne_sample_count(n_samples: int, perplexity: float) -> None:
    """Ensure n_samples is large enough for the chosen perplexity (sklearn rule: perplexity < n_samples)."""
    if perplexity >= n_samples:
        hint_default = (
            " With the usual default perplexity of 30, you need at least 31 samples."
            if perplexity == 30.0
            else ""
        )
        raise TSNEInsufficientSamplesError(
            "t-SNE needs more samples than the perplexity setting. "
            f"Right now there are {n_samples} sample(s) and perplexity is {perplexity:g}.{hint_default} "
        )


def run_dimensionality_reduction_matrix(
    matrix: np.ndarray,
    method: Literal["pca", "tsne", "umap"],
    n_components: int,
    seed: int | None = None,
    device: str | torch.device | None = None,
    use_gpu: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, Any, list[str]]:
    """Run dimensionality reduction on a raw matrix (pure, no pipeline).

    Args:
        matrix: Data matrix, shape (n_samples, n_features), dtype float.
        method: 'pca', 'tsne', or 'umap'.
        n_components: Number of components. PCA and UMAP support any value.
            For t-SNE with sklearn (CPU): when n_components >= 4, uses method='exact'
            (slower, O(n²)); otherwise uses Barnes-Hut (faster). TorchDR t-SNE supports
            any n_components natively.
        seed: Random seed for reproducibility (t-SNE, UMAP).
        device: Optional torch.device. When CUDA and torchdr available, uses GPU.
        use_gpu: If True (default), use TorchDR + GPU when device is CUDA and torchdr
            is available. If False, always use sklearn/umap-learn on CPU.
        **kwargs: Passed to the DR method (e.g. perplexity for t-SNE, n_neighbors for UMAP,
            pca_batch_size for PCA batch size, default 4096).

    Returns:
        Tuple of:
        - transformed: np.ndarray, shape (n_samples, n_components), dtype float64.
        - dr_object: Fitted PCA/TSNE/UMAP object (sklearn or torchdr).
        - component_names: List[str], e.g. ['Component 1', 'Component 2', ...].

    Raises:
        ValueError: If method is unknown.
        ImportError: If UMAP is requested but neither TorchDR GPU UMAP nor umap-learn works.
    """
    use_torchdr = _use_torchdr(device, use_gpu=use_gpu)
    try:
        return _run_dimensionality_reduction_matrix_impl(
            matrix,
            method,
            n_components,
            seed=seed,
            device=device,
            use_torchdr=use_torchdr,
            **kwargs,
        )
    except TSNEInsufficientSamplesError:
        raise
    except ValueError:
        raise
    except Exception as e:
        if use_torchdr and _should_fallback_torchdr_to_cpu(e):
            logger.warning(
                "TorchDR GPU path failed (%s). Falling back to CPU (sklearn / umap-learn).",
                e,
            )
            return _run_dimensionality_reduction_matrix_impl(
                matrix,
                method,
                n_components,
                seed=seed,
                device=device,
                use_torchdr=False,
                **kwargs,
            )
        raise


def _run_dimensionality_reduction_matrix_impl(
    matrix: np.ndarray,
    method: Literal["pca", "tsne", "umap"],
    n_components: int,
    seed: int | None = None,
    device: str | torch.device | None = None,
    *,
    use_torchdr: bool,
    **kwargs: Any,
) -> tuple[np.ndarray, Any, list[str]]:
    """Core DR implementation; ``use_torchdr`` is explicit for CPU retry after GPU failure."""
    dev_str = str(device) if device is not None else "cpu"

    # t-SNE: sklearn's Barnes-Hut approximation supports only n_components < 4.
    # When n_components >= 4, use method='exact' so the user can choose any n_components.
    tsne_method: str | None = None
    if method == "tsne" and n_components >= 4 and not use_torchdr:
        tsne_method = "exact"

    if method == "pca":
        if use_torchdr:
            # Center explicitly so explained variance matches sklearn (TorchDR
            # may center internally, but we ensure consistency across versions).
            matrix_centered = matrix - np.mean(matrix, axis=0, keepdims=True)
            pca_batch_size = kwargs.get("pca_batch_size", 4096)
            pca_kwargs = {k: v for k, v in kwargs.items() if k != "pca_batch_size"}
            batches = _to_batches(matrix_centered, pca_batch_size)
            dr_obj = TorchdrExactIncrementalPCA(
                n_components=n_components,
                device=dev_str,
                random_state=_torchdr_random_state(seed),
                **pca_kwargs,
            )
            dr_obj.fit(batches)
            # Transform in batches to avoid GPU OOM
            transformed_parts = []
            for batch in batches:
                out_batch = dr_obj.transform(batch)
                transformed_parts.append(
                    out_batch.cpu().numpy() if hasattr(out_batch, "cpu") else np.asarray(out_batch)
                )
            transformed = np.concatenate(transformed_parts, axis=0)
            total_var = float(np.var(matrix_centered, axis=0, ddof=1).sum())
            dr_obj.explained_variance_ratio_ = _compute_explained_variance_ratio(
                transformed, total_var
            )
        else:
            from sklearn.decomposition import PCA

            dr_obj = PCA(n_components=n_components, **kwargs)
            transformed = dr_obj.fit_transform(matrix)
        component_names = [f"Component {i + 1}" for i in range(n_components)]
    elif method == "tsne":
        # Perplexity controls local vs global structure; must be < n_samples (sklearn constraint)
        perplexity = kwargs.get("perplexity", 30.0)
        _validate_tsne_sample_count(matrix.shape[0], float(perplexity))
        tsne_kwargs = {k: v for k, v in kwargs.items() if k != "perplexity"}
        if use_torchdr:
            # TorchDR TSNE: use KeOps backend for linear memory (avoids O(n^2) OOM)
            tsne_backends = ["keops", None] if _KEOPS_AVAILABLE else [None]
            dr_obj = None
            last_error = None
            for backend in tsne_backends:
                try:
                    dr_obj = TORCHDR_TSNE(
                        perplexity=perplexity,
                        n_components=n_components,
                        device=dev_str,
                        random_state=_torchdr_random_state(seed),
                        backend=backend,
                        **tsne_kwargs,
                    )
                    out = dr_obj.fit_transform(matrix)
                    transformed = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
                    break
                except Exception as e:
                    last_error = e
                    if backend == "keops":
                        logger.warning(
                            "KeOps backend failed (%s). Using default backend (higher memory). "
                            "Install pykeops for memory-efficient t-SNE: pip install pykeops",
                            e,
                        )
                    else:
                        raise
            if dr_obj is None:
                raise RuntimeError("TorchDR t-SNE failed with all backends") from last_error
        else:
            from sklearn.manifold import TSNE

            # sklearn: Barnes-Hut supports n_components < 4; use method='exact' for n_components >= 4.
            # When n_components < 4, do not modify kwargs (user's method passes through).
            sk_tsne_kwargs = dict(tsne_kwargs)
            if tsne_method is not None:
                sk_tsne_kwargs["method"] = tsne_method
            dr_obj = TSNE(
                n_components=n_components,
                random_state=seed,
                perplexity=perplexity,
                **sk_tsne_kwargs,
            )
            transformed = dr_obj.fit_transform(matrix)
        component_names = [f"Component {i + 1}" for i in range(n_components)]
    elif method == "umap":
        # sklearn/umap-learn defaults for plot consistency
        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        umap_kwargs = {k: v for k, v in kwargs.items() if k not in ("n_neighbors", "min_dist")}
        if use_torchdr:
            dr_obj = _torchdr_umap_fit(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                device=dev_str,
                seed=seed,
                umap_kwargs=umap_kwargs,
            )
            out = dr_obj.fit_transform(matrix)
            transformed = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
        else:
            if umap is None:
                raise ImportError(
                    "UMAP is not available: install umap-learn (pip install umap-learn) "
                    "or fix the TorchDR GPU UMAP path."
                )
            # n_jobs=1 when seed set for reproducibility; -1 otherwise for parallel speed
            n_jobs = 1 if seed is not None else -1
            dr_obj = umap.UMAP(
                n_components=n_components,
                random_state=seed,
                n_jobs=n_jobs,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                **umap_kwargs,
            )
            transformed = dr_obj.fit_transform(matrix)
        component_names = [f"Component {i + 1}" for i in range(n_components)]
    else:
        raise ValueError(f"Unknown method: {method}")

    return transformed, dr_obj, component_names


def run_dimensionality_reduction(
    pipeline: Any,
    method: Literal["pca", "tsne", "umap"],
    n_components: int,
    source: str = "embeddings",
    property_keys: list[str] | None = None,
    filters: dict[str, Any | list[Any]] | None = None,
    exclude: dict[str, Any | list[Any]] | None = None,
    normalize: bool = True,
    device: str | torch.device | None = None,
    use_gpu: bool = True,
    sample_size: int | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame | None, str | None, Any | None]:
    """Run dimensionality reduction using pipeline data and return DataFrame.

    Pipeline-aware: fetches embeddings or properties from pipeline, applies
    filters, normalizes, runs reduction, builds DataFrame. Uses pipeline.device
    when available for GPU acceleration (torchdr) when device is CUDA and use_gpu=True.

    Args:
        pipeline: PhenoMe instance with attributes: results, _get_embedding_data,
            _get_property_matrix, _normalize_embeddings_l2, seed, device.
        method: 'pca', 'tsne', or 'umap'.
        n_components: Number of components. PCA and UMAP support any value.
            t-SNE supports any value (uses method='exact' when n_components >= 4 with sklearn).
        source: 'embeddings', 'properties', or 'combined'.
        property_keys: Property subset when source='properties'.
        filters: Optional metadata filters.
        exclude: Optional metadata exclusions (same structure as filters).
        normalize: Whether to normalize before reduction.
        device: Optional torch.device. Pass pipeline.device for consistency.
        use_gpu: If True (default), use TorchDR + GPU when device is CUDA and torchdr
            is available. If False, always use sklearn/umap-learn on CPU.
        sample_size: If set and more filtered indices remain than this cap, a
            seeded subset is taken *before* loading embeddings or running DR.
        **kwargs: Passed to the DR method. May include ``seed``; if omitted, uses
            ``pipeline.seed`` when present.

    Returns:
        Tuple of (dataframe, data_type_str, dr_object) or (None, None, None) if no data:
        - dataframe: pd.DataFrame with coordinate columns, Image, Index, metadata, properties.
        - data_type_str: 'Embeddings' or 'Properties'.
        - dr_object: Fitted PCA/TSNE/UMAP object.
    """
    # Step 1: Apply metadata filters to get subset of image indices
    indices = filter_indices(pipeline.results, filters, exclude)
    if not indices:
        logger.warning("No images to plot after filtering.")
        return None, None, None

    if sample_size is not None and len(indices) > sample_size:
        seed = kwargs.get("seed")
        if seed is None:
            seed = getattr(pipeline, "seed", None)
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(indices), size=int(sample_size), replace=False))
        indices = [indices[int(i)] for i in pick]

    # Step 2: Fetch data matrix (embeddings, properties, or combined) for filtered indices
    if source == "embeddings":
        # Load ONLY the filtered rows — lazy if checkpoint-backed
        matrix = pipeline.get_embeddings(indices)
        if matrix is None or len(matrix) == 0:
            logger.warning("No embedding data available for DR.")
            return None, None, None
        data_type = "Embeddings"
        valid_indices = indices
        if normalize:
            matrix = pipeline._normalize_embeddings_l2(matrix)
    elif source == "properties":
        matrix, valid_indices, _ = pipeline._get_property_matrix(
            indices=indices,
            property_keys=property_keys,
            normalize=normalize,
            handle_nans="filter",
        )
        data_type = "Properties"
        if len(valid_indices) == 0:
            logger.error("No valid samples after filtering NaNs.")
            return None, None, None
    elif source == "combined":
        try:
            matrix, valid_indices = build_combined_features(
                pipeline, indices, property_keys=property_keys, normalize=normalize
            )
        except ValueError as e:
            logger.error("%s", e)
            return None, None, None
        data_type = "Combined"
    else:
        logger.error(
            "Unknown source '%s', expected 'embeddings', 'properties', or 'combined'.", source
        )
        return None, None, None

    # Step 3: Run reduction and build DataFrame with coordinates + metadata + properties
    # Pop seed from kwargs so we never pass it twice to run_dimensionality_reduction_matrix.
    seed = kwargs.pop("seed", None)
    if seed is None:
        seed = getattr(pipeline, "seed", None)
    dev = device if device is not None else getattr(pipeline, "device", None)
    transformed, dr_obj, component_names = run_dimensionality_reduction_matrix(
        matrix, method, n_components, seed=seed, device=dev, use_gpu=use_gpu, **kwargs
    )

    df = prepare_embedding_dataframe(pipeline.results, transformed, valid_indices, component_names)
    return df, data_type, dr_obj


def _use_torchdr(
    device: str | torch.device | None,
    use_gpu: bool = True,
) -> bool:
    """Return True if torchdr is available, device indicates CUDA, and use_gpu is True."""
    if not use_gpu:
        return False
    if not _TORCHDR_AVAILABLE:
        return False
    if device is None:
        return False
    return str(device).startswith("cuda")


def _torchdr_umap_fit(
    n_neighbors: int,
    n_components: int,
    min_dist: float,
    device: str,
    seed: int | None,
    umap_kwargs: dict[str, Any],
) -> Any:
    """Create TorchDR UMAP with memory-efficient backend, falling back as needed.

    When KeOps is available: tries ``keops``, then ``faiss``, then raw PyTorch.
    When KeOps is unavailable: tries ``faiss``, then raw PyTorch.
    """
    backends = ["keops", "faiss", None] if _KEOPS_AVAILABLE else ["faiss", None]
    last_error = None
    for backend in backends:
        try:
            dr_obj = TORCHDR_UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                device=device,
                random_state=_torchdr_random_state(seed),
                backend=backend,
                **umap_kwargs,
            )
            return dr_obj
        except (AttributeError, ValueError, ImportError, RuntimeError) as e:
            last_error = e
            if backend == "keops":
                logger.warning(
                    "KeOps backend failed (%s). Trying FAISS. "
                    "Install pykeops for memory-efficient UMAP: pip install pykeops",
                    e,
                )
            elif backend == "faiss":
                logger.warning(
                    "FAISS backend failed (%s). Falling back to PyTorch backend. "
                    "Install faiss-gpu via conda for faster UMAP: "
                    "conda install -c pytorch -c nvidia faiss-gpu",
                    e,
                )
            continue
    raise RuntimeError(
        "TorchDR UMAP failed with all backends (keops, faiss, PyTorch)"
    ) from last_error


def _compute_explained_variance_ratio(transformed: np.ndarray, total_var: float) -> np.ndarray:
    """Compute explained variance ratio from PCA-transformed data (centered).

    Formula: var_i / sum(var_original_j) where var_i = variance of component i.
    Uses ddof=1 (sample variance) for consistency with sklearn's PCA, which
    is based on sample covariance. Ratios match sklearn for large n; small
    n may show minor numerical differences.

    Args:
        transformed: np.ndarray, shape (n_samples, n_components), PCA-transformed data.
        total_var: float, total variance of the original data.

    Returns:
        np.ndarray, shape (n_components,), proportion of variance per component.
    """
    n = transformed.shape[0]
    if n < 2:
        return np.full(transformed.shape[1], np.nan)
    var = np.var(transformed, axis=0, ddof=1)
    return (var / total_var) if total_var > 0 else np.zeros_like(var)
