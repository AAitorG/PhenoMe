"""
Analysis for the phenotyping pipeline.

Provides clustering, outlier detection, enrichment analysis, prototype finding,
and correlation analysis between embeddings and extracted properties.
"""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .._logging import get_logger
from ..core import (
    build_combined_features,
    build_metadata_columns,
    compute_distance_correlation,
    compute_lasso_interpretability,
    compute_mutual_info,
    compute_pearson_correlation,
    compute_rf_interpretability,
    compute_spearman_correlation,
    filter_indices,
    get_metadata_value_from_dict,
    run_dimensionality_reduction,
    run_dimensionality_reduction_matrix,
)
from ..core.pipeline_results import PhenoMeResults
from ..core.protocols import PhenoMeProtocol
from . import _helpers
from .visualization._distance_plots import _plot_property_correlations_plotly
from .visualization._interpretability_plots import _display_multivariate_interpretability

logger = get_logger(__name__)


# Registry mapping correlation method names to PhenoMeAnalysis instance method names.
# Simplifies dispatch and makes adding new methods straightforward.
_CORRELATION_METHOD_REGISTRY: dict[str, str] = {
    "pearson": "_compute_pearson_correlation",
    "spearman": "_compute_spearman_correlation",
    "distance_correlation": "_compute_distance_correlation",
    "mutual_info": "_compute_mutual_info",
}


class PhenoMeAnalysis:
    """
    @section Analysis
    @order 5

    Pipeline providing analysis methods for PhenoMe.

    Implements PhenoMeProtocol. Expected attributes from the parent:
        - self.results: PhenoMeResults
        - self.device: torch.device (always set, defaults to CPU if not provided)
        - self._get_embedding_data(...)  (from pipeline)
        - self._get_property_matrix(...)  (from PhenoMeProperties)
        - self._get_embedding_or_property_data(...)  (from PhenoMeDistances)
        - self.get_available_property_keys()
        - self.get_image_info(idx)
        - self._build_properties_dataframe()
    """

    results: PhenoMeResults
    device: torch.device

    # Interdependencies:
    # This mixin depends on visualization helpers for:
    # - _plot_property_correlations_plotly (from .visualization._distance_plots)
    # - _display_multivariate_interpretability (from .visualization._interpretability_plots)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def compute_clustering(
        self,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        n_clusters: int = 5,
        clustering_method: Literal["kmeans", "dbscan", "gmm"] = "kmeans",
        property_keys: list[str] | None = None,
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        random_state: int | None = None,
        normalize: bool = True,
        reduce_dim: int | None = 100,
        reduce_method: Literal["pca", "tsne", "umap"] = "pca",
        return_silhouette: bool = False,
        dbscan_eps: float | None = None,
        dbscan_min_samples: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, float | None]:
        """Perform clustering and store labels in ``metadata[i]['cluster']`` for each image.

        Args:
            source: ``'embeddings'``, ``'properties'``, or ``'combined'``.
            n_clusters: Number of clusters (k-means, GMM). Ignored for ``clustering_method='dbscan'``.
            clustering_method: ``'kmeans'`` (default), ``'dbscan'``, or ``'gmm'``.
            property_keys: Property subset when *source='properties'*.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            random_state: Seed for k-means/GMM. If None, uses pipeline seed when available.
            normalize: Whether to normalize data before clustering (default: True).
                For embeddings, uses L2 normalization. For properties, uses StandardScaler.
            reduce_dim: If set, apply dimensionality reduction before clustering
                (default: 100). Only applied when data has more features than *reduce_dim*.
                Set to None to disable dimensionality reduction.
            reduce_method: Method for dimensionality reduction when *reduce_dim* is set:
                ``'pca'`` (fast, linear), ``'tsne'`` (non-linear, slower), ``'umap'``
                (non-linear, preserves structure). Default: ``'pca'``.
            return_silhouette: If True, compute and return the silhouette score (default: False).
                Returns ``(labels, score)``; score is None if it could not be computed.
                Silhouette requires at least 2 clusters and 2 samples per cluster.
            dbscan_eps: Maximum distance between two samples for DBSCAN (only when
                clustering_method='dbscan'). If None, uses 0.5.
            dbscan_min_samples: Minimum samples in a neighborhood for DBSCAN (only when
                clustering_method='dbscan'). If None, uses max(2, n_samples // 20).

        Returns:
            If return_silhouette=False: np.ndarray of shape (n_total,) with cluster labels.
            If return_silhouette=True: Tuple of (labels, silhouette_score). Score is None
                if it could not be computed. Labels: 0..K-1; NaN for excluded/DBSCAN noise.
        """
        n_total = len(self.results.img_path)

        try:
            matrix, valid_indices = self._get_data_matrix(
                source,
                filters,
                exclude,
                property_keys,
                caller="clustering",
                normalize=normalize,
            )
        except ValueError as exc:
            logger.warning("Clustering aborted: %s", exc)
            empty_labels = np.full(n_total, np.nan, dtype=np.float32)
            if return_silhouette:
                return empty_labels, None
            return empty_labels

        if len(matrix) == 0:
            logger.warning("No data available for clustering.")
            empty_labels = np.full(n_total, np.nan, dtype=np.float32)
            if return_silhouette:
                return empty_labels, None
            return empty_labels

        # Step 2: Apply dimensionality reduction
        # We extract components to avoid the curse of dimensionality and ill-conditioned fits.
        dr_object_after_reduce: Any | None = None
        if reduce_dim is not None and reduce_dim > 0:
            n_comp = min(reduce_dim, matrix.shape[0] - 1, matrix.shape[1])
            if n_comp > 0:
                orig_dim = matrix.shape[1]
                n_samples = matrix.shape[0]
                seed = random_state if random_state is not None else getattr(self, "seed", None)
                dev = getattr(self, "device", None)
                use_gpu_dr = getattr(self, "use_gpu_for_dr", True)
                dr_kwargs: dict[str, Any] = {}
                if reduce_method == "tsne":
                    # Perplexity must be < n_samples (sklearn constraint)
                    dr_kwargs["perplexity"] = min(30.0, n_samples - 1)
                elif reduce_method == "umap":
                    # n_neighbors must be <= n_samples
                    dr_kwargs["n_neighbors"] = min(15, n_samples - 1) if n_samples > 1 else 1

                matrix, dr_object_after_reduce, _ = run_dimensionality_reduction_matrix(
                    matrix,
                    method=reduce_method,
                    n_components=n_comp,
                    seed=seed,
                    device=dev,
                    use_gpu=use_gpu_dr,
                    **dr_kwargs,
                )
                logger.info(
                    "Clustering: %s extracted %d components from %d dimensions before %s.",
                    reduce_method.upper(),
                    n_comp,
                    orig_dim,
                    clustering_method.upper(),
                )

        # Fitted DR object from the optional pre-clustering step (None if no reduction).
        self._last_clustering_dr_object = dr_object_after_reduce

        # Step 3: Apply the clustering only over the extracted components
        effective_seed = random_state if random_state is not None else getattr(self, "seed", None)

        if clustering_method == "gmm":
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=effective_seed,
                covariance_type="diag",
                reg_covar=1e-6,
                max_iter=100,
            )
            # Use float64 for fitting to avoid precision issues with small variances
            labels = gmm.fit_predict(matrix.astype(np.float64)).astype(np.float32)
        elif clustering_method == "kmeans":
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=effective_seed,
            )
            labels = kmeans.fit_predict(matrix).astype(np.float32)
        elif clustering_method == "dbscan":
            from sklearn.cluster import DBSCAN

            eps = dbscan_eps if dbscan_eps is not None else 0.5
            min_samples = (
                dbscan_min_samples if dbscan_min_samples is not None else max(2, len(matrix) // 20)
            )
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            raw_labels = clusterer.fit_predict(matrix).astype(np.float32)
            # Map noise (-1) to NaN so metadata stores None
            labels = np.where(raw_labels >= 0, raw_labels, np.nan)
        else:
            raise ValueError(
                f"clustering_method must be 'kmeans', 'dbscan', or 'gmm'; got {clustering_method!r}"
            )

        # Map cluster labels back to full image set (NaN for filtered-out images)
        full_labels = _helpers.map_to_full(labels, valid_indices, n_total)

        n_actual_clusters = int(np.nanmax(labels)) + 1 if np.any(~np.isnan(labels)) else 0
        silhouette_score_val: float | None = None
        if return_silhouette and n_actual_clusters >= 2 and len(matrix) >= n_actual_clusters:
            # For DBSCAN, exclude noise (NaN) for silhouette
            mask = ~np.isnan(labels)
            if clustering_method == "dbscan" and not np.all(mask):
                sil_labels = labels[mask]
                sil_matrix = matrix[mask]
            else:
                sil_labels = labels
                sil_matrix = matrix
            unique_labels = np.unique(sil_labels[~np.isnan(sil_labels)])
            if len(unique_labels) >= 2 and len(unique_labels) <= len(sil_matrix) - 1:
                if all(np.sum(sil_labels == u) >= 2 for u in unique_labels):
                    try:
                        from sklearn.metrics import silhouette_score

                        silhouette_score_val = float(
                            silhouette_score(sil_matrix, sil_labels.astype(int))
                        )
                    except ValueError as exc:
                        logger.warning("Silhouette score failed: %s", exc)
                else:
                    logger.warning("Silhouette score skipped: need at least 2 samples per cluster.")
            else:
                logger.warning(
                    "Silhouette score skipped: need 2 <= n_clusters <= n_samples-1 (got %d clusters, %d samples).",
                    len(unique_labels),
                    len(sil_matrix),
                )

        # Store cluster labels in results['metadata'] for downstream use
        metadata_list = self.results.metadata
        while len(metadata_list) < n_total:
            metadata_list.append({})
        for i in range(n_total):
            if not isinstance(metadata_list[i], dict):
                metadata_list[i] = {}
            metadata_list[i]["cluster"] = (
                int(full_labels[i]) if not np.isnan(full_labels[i]) else None
            )
        self.results.metadata = metadata_list

        n_clusters_log = n_actual_clusters if clustering_method == "dbscan" else n_clusters
        logger.info(
            "Clustering (%s): %d clusters on %d images (source=%s).",
            clustering_method,
            n_clusters_log,
            len(valid_indices),
            source,
        )
        if return_silhouette:
            return (full_labels, silhouette_score_val)
        return full_labels

    # ------------------------------------------------------------------
    # Multivariate Interpretability
    # ------------------------------------------------------------------

    def compute_multivariate_interpretability(
        self,
        method: Literal["pca", "tsne", "umap"] = "tsne",
        component: int = 1,
        model_type: Literal["lasso", "random_forest"] = "lasso",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        normalize: bool = True,
        cv: int = 5,
        rf_n_estimators: int = 100,
        seed: int | None = None,
        plot: bool = True,
        return_fig: bool = False,
        top_k: int = 10,
        figsize: tuple[int, int] = (10, 8),
    ) -> dict[str, Any]:
        """Explain a dimensionality reduction component using LASSO or Random Forest.

        Calculates which phenotypic properties (features) best explain the variability
        seen in a deep learning embedding dimension (the target, usually t-SNE 1 or 2).

        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap').
            component: Which component to explain (1, 2, ...).
            model_type: The regression model to use ('lasso' or 'random_forest').
                'lasso' uses L1 regularization for linear, sparse explanations.
                'random_forest' captures non-linear relationships.
            property_keys: Subset of properties to use as features.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions.
            normalize: Whether to normalize features before regression (default: True).
                Uses StandardScaler for properties to ensure comparable coefficients.
            cv: Number of cross-validation folds (only for 'lasso').
            rf_n_estimators: Number of trees (only for 'random_forest').
            seed: Random seed for reproducibility. If None, uses the pipeline's ``seed`` when set.
            plot: If True (default), show an interactive Plotly bar chart of top drivers.
                If False, log a plain-text summary via the package logger instead.
            return_fig: If True, attach a Plotly figure under ``interpretability_fig`` in the
                returned dict. When ``return_fig`` is True, ``fig.show()`` is not called; use
                ``plot=True`` with ``return_fig=False`` for the default interactive display.
            top_k: Number of top drivers to show in the plot or text summary.
            figsize: Figure size ``(width, height)`` passed through to Plotly layout; each
                value is multiplied by 100 to set width and height in layout pixels.

        Returns:
            Dict with:
                - r2: Explainability Score (R^2).
                - drivers: Ranked list of properties with weights/importances.
                - method: The DR method used.
                - model_type: The regression model type used.
                - target_component: The component name explained.
                - n_samples: Number of samples used.
                - n_features: Number of properties considered.
                - interpretability_fig: Present when ``return_fig`` is True and a figure was built.
        """
        effective_seed = seed if seed is not None else getattr(self, "seed", None)

        # Step 1: Run dimensionality reduction to get the target (y)
        # We reuse the existing run_dimensionality_reduction logic
        dr_results, _, dr_obj = run_dimensionality_reduction(
            self,
            method=method,
            n_components=max(2, component),
            source="embeddings",
            filters=filters,
            exclude=exclude,
            normalize=normalize,
            device=self.device,
            use_gpu=getattr(self, "use_gpu_for_dr", True),
            seed=effective_seed,
        )

        if dr_results is None or dr_results.empty:
            logger.warning("No data available for multivariate interpretability.")
            return {}

        comp_col = f"Component {component}"
        if comp_col not in dr_results.columns:
            logger.warning("Target component '%s' not found in DR results.", comp_col)
            return {}

        y = dr_results[comp_col].values
        valid_indices = dr_results["Index"].values.tolist()

        # Step 2: Fetch the property matrix (x) for the same valid samples
        # We use the provided normalize argument (default True) to ensure comparable features
        matrix, prop_valid_indices, keys = self._get_property_matrix(  # type: ignore[attr-defined]
            indices=valid_indices,
            property_keys=property_keys,
            normalize=normalize,
            handle_nans="filter",
        )

        if len(matrix) == 0:
            logger.warning("No valid samples with properties for interpretability.")
            return {}

        # Align y with X if NaN-filtering in _get_property_matrix changed samples
        if len(matrix) < len(y):
            # Find the positions in the original valid_indices that were kept
            idx_to_pos = {idx: i for i, idx in enumerate(valid_indices)}
            # Find which positions in y correspond to the rows in matrix
            y_aligned_indices = [idx_to_pos[idx] for idx in prop_valid_indices]
            y = y[y_aligned_indices]

        # Step 3: Run regression model
        if model_type == "lasso":
            results = compute_lasso_interpretability(
                x=matrix,
                y=y,
                feature_names=keys,
                cv=cv,
                seed=effective_seed,
            )
        elif model_type == "random_forest":
            results = compute_rf_interpretability(
                x=matrix,
                y=y,
                feature_names=keys,
                n_estimators=rf_n_estimators,
                seed=effective_seed,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        results.update(
            {
                "method": method,
                "model_type": model_type,
                "target_component": comp_col,
                "dr_object": dr_obj,
            }
        )

        logger.info(
            "Multivariate Interpretability (%s, %s, %s): R^2=%.2f, %d drivers found.",
            method.upper(),
            comp_col,
            model_type,
            results["r2"],
            len(results["drivers"]),
        )

        results.pop("interpretability_fig", None)
        fig = _display_multivariate_interpretability(
            results,
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
        )
        if return_fig and fig is not None:
            results["interpretability_fig"] = fig

        return results

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        method: Literal["z-score", "iqr"] = "z-score",
        threshold: float = 3.0,
        group_by: str | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        normalize: bool = True,
        drop_outliers: bool = False,
        plot: bool = True,
    ) -> dict[str, Any]:
        """Detect outliers based on distance to centroid.

        Args:
            method: ``'z-score'`` or ``'iqr'``.
            threshold: Threshold multiplier.
            group_by: Optional metadata key for per-group centroids.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            source: Feature space for outlier detection:
                - ``'embeddings'``: model embeddings
                - ``'properties'``: scalar phenotypic properties
                - ``'combined'``: normalized concatenation of embeddings and properties.
            property_keys: Property subset when *source* is ``'properties'`` or ``'combined'``.
            normalize: Whether to normalize data before outlier detection (default: True).
                For embeddings, uses L2 normalization. For properties, uses StandardScaler.
            drop_outliers: If True, remove detected outliers from self.results (default: False).
            plot: If True (default), shows one matplotlib figure per group: subplots for
                detected outliers and a figure title naming the group (``Group: …``).
                Images are shown without pipeline ``image_transforms`` (same as
                ``apply_transforms=False`` for display).

        Returns:
            Dict with keys:
            - outlier_indices: List[int]. Global image indices of detected outliers.
            - distances: np.ndarray dtype float32. When drop_outliers=False: shape (n_total,),
              distance to centroid per image; NaN for filtered-out. When drop_outliers=True:
              shape (n_kept,) with distances for the remaining (non-outlier) images only.
            - thresholds: Dict[Any, float]. Per-group threshold values (when group_by set).
            - summary: pd.DataFrame. Columns: group (if group_by), n_outliers, threshold, etc.
        """
        empty = {
            "outlier_indices": [],
            "distances": [],
            "thresholds": {},
            "summary": pd.DataFrame(),
        }

        # drop_outliers mutates results in-place which is incompatible with HDF5-backed mode
        if drop_outliers and getattr(self, "_db", None) is not None:
            raise RuntimeError(
                "drop_outliers=True is not supported when results are backed by an HDF5 "
                "database (checkpoint_path was set or load_results() was used). "
                "Use the returned outlier_indices with filters= in subsequent analysis "
                "calls to exclude outliers without mutating the database."
            )

        try:
            data, valid_indices = self._get_data_matrix(
                source=source,
                filters=filters,
                exclude=exclude,
                property_keys=property_keys,
                caller="detect_outliers",
                normalize=normalize,
            )
        except ValueError as exc:
            logger.warning("Outlier detection aborted: %s", exc)
            return empty

        if data is None or len(data) == 0:
            return empty

        data = _helpers.numpy_for_torch(data)

        n_samples = len(data)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        distances = np.zeros(n_samples, dtype=np.float32)

        device = self.device

        # Convert data to GPU once (avoid repeated conversions in loop)
        data_t = torch.from_numpy(data).to(device) if device.type != "cpu" else None

        def _outliers_for(
            vectors: np.ndarray, vectors_t: torch.Tensor | None = None
        ) -> tuple[np.ndarray, np.ndarray, Any]:
            # Use pre-converted tensor if available, otherwise convert
            if vectors_t is not None:
                vec_t = vectors_t
            else:
                # CPU and small group slices: vectors may be a read-only view of ``data``
                vec_t = torch.from_numpy(_helpers.numpy_for_torch(vectors)).to(device)

            with torch.no_grad():
                # Compute centroid on GPU
                centroid = torch.mean(vec_t, dim=0)

                # Compute distances on GPU
                dists_t = torch.norm(vec_t - centroid.unsqueeze(0), dim=1)
                dists = dists_t.cpu().numpy() if device.type != "cpu" else dists_t.numpy()

            if method == "z-score":
                mu, sigma = np.mean(dists), np.std(dists, ddof=1)
                if sigma == 0:
                    return dists, np.zeros_like(dists, dtype=bool), mu
                tv = mu + threshold * sigma
                return dists, dists > tv, tv
            if method == "iqr":
                q1, q3 = np.percentile(dists, 25), np.percentile(dists, 75)
                iqr = q3 - q1
                if iqr == 0:
                    return dists, np.zeros_like(dists, dtype=bool), q3
                tv = q3 + threshold * iqr
                return dists, dists > tv, tv
            raise ValueError(f"Unknown method: {method}")

        # Build groups
        groups: dict[Any, np.ndarray] = {}
        if group_by:
            df = pd.DataFrame(
                build_metadata_columns(self.results, indices=valid_indices, keys=[group_by])
            )
            group_col = next(
                (c for c in df.columns if str(c).lower() == str(group_by).lower()),
                None,
            )
            if group_col is None:
                logger.warning("Grouping key '%s' not found. Treating as single group.", group_by)
                groups["all"] = np.arange(n_samples)
            else:
                for gv, gdf in df.groupby(group_col):
                    groups[gv] = gdf.index.to_numpy()
        else:
            groups["all"] = np.arange(n_samples)

        thresholds: dict[Any, float] = {}
        for gname, didx in groups.items():
            if len(didx) < 2:
                continue
            didx = _helpers.index_array_for_torch(didx)
            # Pass pre-converted tensor slice if available
            vec_t_slice = data_t[didx] if data_t is not None else None
            d, is_out, tv = _outliers_for(data[didx], vec_t_slice)
            distances[didx] = d
            outlier_mask[didx] = is_out
            thresholds[gname] = tv

        rel_out = np.where(outlier_mask)[0]
        outlier_indices = [valid_indices[i] for i in rel_out]
        logger.info(
            "Detected %d outliers out of %d images.", len(outlier_indices), len(valid_indices)
        )

        summary_rows = []
        for i, idx in enumerate(outlier_indices):
            info = self.get_image_info(idx)  # type: ignore[attr-defined]
            row: dict[str, Any] = {
                "idx": idx,
                "distance_to_centroid": distances[rel_out[i]],
                "file_path": info.get("img_path", ""),
            }
            if group_by:
                group_val = get_metadata_value_from_dict(info, group_by)
                row["group"] = "N/A" if group_val is None or group_val == "" else group_val
                row["threshold"] = thresholds.get(row["group"], 0)
            else:
                row["threshold"] = thresholds.get("all", 0)
            summary_rows.append(row)

        if plot and outlier_indices:
            import matplotlib.pyplot as plt

            # Group outliers by group name
            outliers_by_group: dict[str, list[int]] = {}
            for row in summary_rows:
                gname = str(row.get("group", "All"))
                if gname not in outliers_by_group:
                    outliers_by_group[gname] = []
                outliers_by_group[gname].append(row["idx"])

            for group_name, indices in outliers_by_group.items():
                if not indices:
                    continue
                n = len(indices)
                ncols = min(5, n)
                nrows = (n + ncols - 1) // ncols
                fig_w = min(4.0 * ncols, 22)
                fig_h = max(3.4 * nrows + 1.0, 3.8)
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(fig_w, fig_h),
                    gridspec_kw={"wspace": 0.04, "hspace": 0.10},
                )
                if nrows == 1 and ncols == 1:
                    ax_flat = np.array([axes])
                else:
                    ax_flat = np.asarray(axes).ravel()

                for j, idx in enumerate(indices):
                    self.plot_image_by_index(  # type: ignore[attr-defined]
                        idx,
                        ax=ax_flat[j],
                        apply_transforms=False,
                        downsample=720,
                        show_extra_info=False,
                    )

                for j in range(n, len(ax_flat)):
                    ax_flat[j].set_visible(False)

                fig.suptitle(f"Group: {group_name} outliers", fontsize=13, fontweight="bold")
                fig.tight_layout(pad=0.15, rect=[0, 0, 1, 0.95])
                try:
                    from IPython import get_ipython
                    from IPython.display import display as ipy_display_fig
                except ImportError:
                    ipython_shell = None
                else:
                    ipython_shell = get_ipython()
                if ipython_shell is not None:
                    ipy_display_fig(fig)
                else:
                    plt.show()
                plt.close(fig)

        n_total = len(self.results.img_path)
        full_dists = _helpers.map_to_full(distances, valid_indices, n_total)

        # Remove outliers from results if requested
        if drop_outliers and outlier_indices:
            # Sort indices in descending order to remove from end to start
            # This prevents index shifting issues
            sorted_outlier_indices = sorted(outlier_indices, reverse=True)

            # Remove from all result arrays/lists
            for idx in sorted_outlier_indices:
                # Remove from embeddings (always np.ndarray or None)
                if isinstance(self.results.embeddings, np.ndarray):
                    self.results.embeddings = np.delete(self.results.embeddings, idx, axis=0)

                # Remove from img_path
                if idx < len(self.results.img_path):
                    self.results.img_path.pop(idx)

                # Remove from metadata
                if idx < len(self.results.metadata):
                    self.results.metadata.pop(idx)

                # Remove from properties
                if idx < len(self.results.properties):
                    self.results.properties.pop(idx)

            self._property_norm_cache = None

            # Return distances aligned with new results (exclude outlier positions)
            kept_mask = np.ones(n_total, dtype=bool)
            kept_mask[np.asarray(outlier_indices)] = False
            full_dists = full_dists[kept_mask]

            logger.info(
                "Removed %d outliers from results. Remaining: %d images.",
                len(outlier_indices),
                len(self.results.img_path),
            )

        return {
            "outlier_indices": list(outlier_indices),
            "distances": full_dists,
            "thresholds": thresholds,
            "summary": pd.DataFrame(summary_rows),
        }

    # ------------------------------------------------------------------
    # Component correlation
    # ------------------------------------------------------------------

    def compute_component_correlation(
        self,
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        top_k: int | None = None,
        normalize: bool = True,
        correlation_method: Literal[
            "pearson", "spearman", "distance_correlation", "mutual_info"
        ] = "pearson",
        plot: bool = True,
        return_fig: bool = False,
        figsize: tuple[int, int] = (10, 6),
    ) -> dict[str, Any]:
        """Correlate dim-reduction components with phenotypic properties.

        Args:
            method: Dimensionality reduction method.
            n_components: Number of components to compute.
            source: ``'embeddings'``, ``'properties'``, or ``'combined'``.
            property_keys: Property subset when *source* is ``'properties'`` or ``'combined'``.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            top_k: Number of top correlations to return in summary and per facet in the plot.
            normalize: Whether to normalize data before computing correlations (default: True).
                - Source data (embeddings/properties): Normalized before dimensionality reduction
                  (L2 normalization for embeddings, StandardScaler for properties)
                - Properties: Normalized using StandardScaler before correlation computation
            correlation_method: Correlation method to use. Options:
                - ``'pearson'``: Pearson correlation coefficient (default, linear relationships)
                - ``'spearman'``: Spearman rank correlation (monotonic relationships, robust to outliers)
                - ``'distance_correlation'``: Distance correlation (detects non-linear relationships,
                  requires ``dcor`` library: ``pip install dcor``)
                - ``'mutual_info'``: Normalized mutual information (detects any dependency,
                  normalized to [0, 1] range)
            plot: If True (default), show an interactive Plotly faceted bar chart (one row per
                component). If False, log a plain-text summary via the package logger instead
                (unless *return_fig* requests a figure).
            return_fig: If True, attach a Plotly figure under ``component_correlation_fig`` in the
                returned dict. When ``return_fig`` is True, ``fig.show()`` is not called; use
                ``plot=True`` with ``return_fig=False`` for the default interactive display.
            figsize: Figure size ``(width, height)`` in inches for the Plotly layout.

        Returns:
            Dict with keys:
            - correlation_df: pd.DataFrame. Rows=properties, cols=components. Correlation values.
            - summary: pd.DataFrame. Columns: Component, Property, Correlation, AbsCorrelation.
            - component_names: List[str]. Column names (Component 1, Component 2, ...) for all methods.
            - component_correlation_fig: Present when ``return_fig`` is True and a figure was built.

        Examples:
            Text-only summary (e.g. scripts / logging)::

                result = pheno.compute_component_correlation(plot=False)

            Save the figure without an interactive window::

                result = pheno.compute_component_correlation(return_fig=True)
                fig = result["component_correlation_fig"]
                fig.write_html("component_corr.html")
        """
        df, _, _ = run_dimensionality_reduction(
            self,
            method=method,
            n_components=n_components,
            source=source,
            property_keys=property_keys if source in ("properties", "combined") else None,
            filters=filters,
            exclude=exclude,
            normalize=normalize,
            device=self.device,
            use_gpu=getattr(self, "use_gpu_for_dr", True),
        )
        if df is None or df.empty:
            logger.warning("Dimensionality reduction returned empty result.")
            return {}

        # Component columns use common naming: Component 1, Component 2, ... (all methods)
        comp_cols = [f"Component {i + 1}" for i in range(n_components)]
        comp_cols = [c for c in comp_cols if c in df.columns]
        if not comp_cols:
            return {}

        avail = set(self.get_available_property_keys())  # type: ignore[attr-defined]
        avail |= {k.capitalize() for k in avail}
        prop_cols = [c for c in df.columns if c in avail and pd.api.types.is_numeric_dtype(df[c])]
        if not prop_cols:
            logger.warning("No numeric properties found to correlate.")
            return {}

        # Normalize properties using the unified PhenoMeProperties helper if requested.
        # This ensures consistency and uses the internal normalization cache.
        if normalize and prop_cols:
            raw_keys = self.get_available_property_keys()  # type: ignore[attr-defined]
            # Map capitalized names from DataFrame back to raw keys for _get_property_matrix
            cap_to_raw = {k.capitalize(): k for k in raw_keys}
            fetch_keys = [cap_to_raw.get(c, c) for c in prop_cols]

            # We use the existing indices from the DataFrame to ensure alignment.
            indices = df["Index"].tolist()
            # Normalize=True here uses the pipeline's internal StandardScaler cache.
            # handle_nans='warn' because run_dimensionality_reduction already handled filters/NaNs.
            norm_matrix, _, _ = self._get_property_matrix(  # type: ignore[attr-defined]
                indices=indices,
                property_keys=fetch_keys,
                normalize=True,
                handle_nans="warn",
            )
            # Update DataFrame with normalized values
            for idx, col in enumerate(prop_cols):
                df[col] = norm_matrix[:, idx]

        # Compute correlation matrix based on method
        if correlation_method == "pearson":
            # Use pandas for Pearson (fast and handles NaNs well)
            corr = df[prop_cols + comp_cols].corr(method="pearson").loc[prop_cols, comp_cols]
        else:
            # For spearman, distance correlation, mutual info: compute via core (torch/scipy)
            use_progress = correlation_method in ("distance_correlation", "mutual_info")
            comp_iter = (
                tqdm(
                    comp_cols,
                    desc=f"Computing {correlation_method} correlations",
                    disable=not use_progress,
                    leave=False,
                )
                if use_progress
                else comp_cols
            )

            corr_data: dict[str, list[float]] = {}
            for comp in comp_iter:
                if comp not in df.columns:
                    continue
                comp_values = df[comp].values
                corr_data[comp] = []
                for prop in prop_cols:
                    prop_values = df[prop].values
                    r = self._compute_correlation(correlation_method, prop_values, comp_values)
                    r_arr = np.atleast_1d(np.asarray(r))
                    corr_data[comp].append(float(r_arr[0]) if r_arr.size > 0 else np.nan)

            if not corr_data:
                return {}

            corr = pd.DataFrame(corr_data, index=prop_cols)

        if corr.empty:
            return {}

        rows = []
        for comp in comp_cols:
            if comp not in corr.columns:
                continue
            top = corr[comp].abs().sort_values(ascending=False).head(top_k or len(corr))
            for prop, _ in top.items():
                r = corr.loc[prop, comp]
                rows.append(
                    {
                        "Component": comp,
                        "Property": prop,
                        "Correlation": r,
                        "AbsCorrelation": abs(r),
                    }
                )

        summary_df = pd.DataFrame(rows)
        out: dict[str, Any] = {
            "correlation_df": corr,
            "summary": summary_df,
            "component_names": comp_cols,
        }
        title = f"Component-property correlations ({method.upper()}, {correlation_method})"
        fig = self._plot_component_correlation(  # type: ignore[attr-defined]
            correlation_df=corr,
            summary=summary_df,
            component_names=comp_cols,
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
            title=title,
        )
        if return_fig and fig is not None:
            out["component_correlation_fig"] = fig
        return out

    # ------------------------------------------------------------------
    # Group enrichment
    # ------------------------------------------------------------------

    def analyze_group_enrichment(
        self,
        group_col: str = "cluster",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        plot: bool = True,
        return_fig: bool = False,
        top_k: int | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
    ) -> dict[str, Any]:
        """Compute z-score enrichment of properties per group (e.g. cluster labels).

        Uses sample mean and sample standard deviation (ddof=1) for the
        population statistics when computing z-scores.

        Args:
            group_col: Property/metadata key holding group labels (e.g. ``"cluster"``).
            property_keys: Properties to analyse (all numeric if *None*).
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            plot: If True (default), show an interactive Plotly faceted bar chart (one row per
                group). If False, log a plain-text summary via the package logger instead
                (unless *return_fig* requests a figure).
            return_fig: If True, include a Plotly figure under ``group_enrichment_fig`` in the
                returned dict. When *return_fig* is True, ``fig.show()`` is not called; use
                ``plot=True`` with ``return_fig=False`` for the default interactive display.
            top_k: Max properties per group in the figure and in the text summary (``None`` = all).
            figsize: Figure size ``(width, height)`` in inches for the Plotly layout.
            title: Optional figure title.

        Returns:
            Dict with keys:
            - enrichment: pd.DataFrame. Columns: Group, Property, Score, Mean_Group, Mean_Pop,
              AbsScore. Z-score enrichment per group-property pair.
            - group_enrichment_fig: Present when ``return_fig`` is True and a figure was built.
        """
        working_df = self._build_properties_dataframe()  # type: ignore[attr-defined]
        empty_cols = ["Group", "Property", "Score", "Mean_Group", "Mean_Pop", "AbsScore"]
        if working_df.empty:
            out_empty: dict[str, Any] = {"enrichment": pd.DataFrame(columns=empty_cols)}
            self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=out_empty["enrichment"],
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return out_empty

        if filters or exclude:
            indices = filter_indices(self.results, filters, exclude)
            working_df = working_df.iloc[indices]

        # Resolve group column (case-insensitive)
        cc = group_col
        if cc not in working_df.columns:
            for variant in (cc.capitalize(), cc.lower()):
                if variant in working_df.columns:
                    cc = variant
                    break
            else:
                raise ValueError(
                    f"Group column '{group_col}' not found. Columns: {list(working_df.columns)}"
                )

        if property_keys is None:
            exclude_cols = {"img_name", "img_path", "Index", "Image", cc}
            exclude_cols.update(c for c in working_df.columns if c.startswith("Component "))
            property_keys = [
                c
                for c in working_df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(working_df[c])
            ]
        if not property_keys:
            logger.warning("No numeric properties to analyse.")
            out_np: dict[str, Any] = {"enrichment": pd.DataFrame(columns=empty_cols)}
            self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=out_np["enrichment"],
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return out_np

        working_df = working_df.dropna(subset=[cc])
        if working_df.empty:
            logger.warning("No samples with valid group labels.")
            out_ng: dict[str, Any] = {"enrichment": pd.DataFrame(columns=empty_cols)}
            self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=out_ng["enrichment"],
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return out_ng

        pop_mean = working_df[property_keys].mean()
        pop_std = working_df[property_keys].std()
        # Replace near-zero variance to avoid huge z-scores (exact 0 and fp noise)
        pop_std = pop_std.where(pop_std >= 1e-10, np.nan)

        all_groups = sorted(working_df[cc].unique())
        skipped_small = sum(1 for g in all_groups if len(working_df[working_df[cc] == g]) < 3)
        if skipped_small > 0:
            logger.info(
                "Skipped %d group(s) with < 3 samples; z-score enrichment requires at least 3.",
                skipped_small,
            )

        rows = []
        for grp in all_groups:
            gdf = working_df[working_df[cc] == grp]
            if len(gdf) < 3:
                continue
            gmean = gdf[property_keys].mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                zs = (gmean - pop_mean) / pop_std
            for prop in property_keys:
                s = zs[prop]
                if pd.notna(s):
                    rows.append(
                        {
                            "Group": grp,
                            "Property": prop,
                            "Score": float(s),
                            "Mean_Group": float(gmean[prop]),
                            "Mean_Pop": float(pop_mean[prop]),
                        }
                    )

        if not rows:
            edf = pd.DataFrame(columns=empty_cols)
        else:
            edf = pd.DataFrame(rows)
            edf["AbsScore"] = edf["Score"].abs()
            edf = edf.sort_values(["Group", "AbsScore"], ascending=[True, False])

        out: dict[str, Any] = {"enrichment": edf}
        fig_title = title or "Group enrichment (Z-scores)"
        fig = self._plot_group_enrichment(  # type: ignore[attr-defined]
            enrichment_df=edf,
            group_names=None,
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
            title=fig_title,
        )
        if return_fig and fig is not None:
            out["group_enrichment_fig"] = fig
        return out

    # ------------------------------------------------------------------
    # Prototypes
    # ------------------------------------------------------------------

    def find_prototypes(
        self,
        cluster_col: str | None = "cluster",
        n_prototypes: int = 5,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        metric: str = "euclidean",
        normalize: bool = True,
        plot: bool = True,
    ) -> dict[str, list[int]]:
        """Find images closest to each group centroid.

        Args:
            cluster_col: Property/metadata key for grouping (one group if *None*).
            n_prototypes: How many prototypes per group.
            source: ``'embeddings'``, ``'properties'``, or ``'combined'``.
            property_keys: Property subset when *source='properties'*.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            metric: ``'euclidean'`` or ``'cosine'``.
            normalize: Whether to normalize data before finding prototypes (default: True).
                For embeddings, uses L2 normalization. For properties, uses StandardScaler.
            plot: If True (default), shows one matplotlib figure per group: subplots for
                that group's prototypes and a figure title naming the group (``Group: …``).
                Images are shown without pipeline ``image_transforms`` (same as
                ``apply_transforms=False`` for display).

        Returns:
            Dict[str, List[int]]: Group name (or "All") -> list of global image indices (prototypes).
        """
        try:
            matrix, valid_indices = self._get_data_matrix(
                source,
                filters,
                exclude,
                property_keys,
                caller="prototypes",
                normalize=normalize,
            )
        except ValueError as exc:
            logger.warning("find_prototypes aborted: %s", exc)
            return {}

        # Build group labels
        n_total = len(self.results.img_path)
        group_labels: list[Any] = [None] * n_total
        actual_cc: str | None = None

        if cluster_col:
            for store_key in ("properties", "metadata"):
                store = getattr(self.results, store_key, None) or []
                if not store or not isinstance(store[0], dict):
                    continue
                for k in store[0]:
                    if k.lower() == cluster_col.lower():
                        actual_cc = k
                        for i, entry in enumerate(store):
                            if i < n_total and isinstance(entry, dict):
                                group_labels[i] = entry.get(k)
                        break
                if actual_cc:
                    break
            if actual_cc is None:
                logger.warning("Column '%s' not found. Treating as one group.", cluster_col)

        valid_gl = [group_labels[i] for i in valid_indices]
        unique_groups: list
        if actual_cc:
            has_none = None in valid_gl
            others = sorted({v for v in valid_gl if v is not None})
            unique_groups = ([None] if has_none else []) + others
        else:
            unique_groups = [None]

        device = self.device

        matrix = _helpers.numpy_for_torch(matrix)

        # Convert entire matrix to GPU once (avoid repeated conversions per group)
        matrix_t = torch.from_numpy(matrix).to(device) if device.type != "cpu" else None

        out: dict[str, list[int]] = {}
        for grp in unique_groups:
            mask = [lab == grp for lab in valid_gl]
            mask_arr = np.array(mask)
            gi = [valid_indices[i] for i, m in enumerate(mask) if m]
            if mask_arr.sum() == 0:
                continue

            # Use pre-converted tensor slice if available, otherwise convert
            if matrix_t is not None:
                gm_t = matrix_t[mask_arr]
            else:
                gm = matrix[mask_arr]
                gm_t = torch.from_numpy(_helpers.numpy_for_torch(gm)).to(device)

            with torch.no_grad():
                # Compute centroid on GPU
                centroid_t = torch.mean(gm_t, dim=0)

                if metric == "euclidean":
                    # Compute distances on GPU
                    dists_t = torch.norm(gm_t - centroid_t.unsqueeze(0), dim=1)
                    dists = dists_t.cpu().numpy() if device.type != "cpu" else dists_t.numpy()
                    top = np.argsort(dists)[:n_prototypes]
                elif metric == "cosine":
                    # Normalize vectors on GPU
                    gm_norm = torch.nn.functional.normalize(gm_t, dim=-1)
                    centroid_norm = torch.nn.functional.normalize(centroid_t.unsqueeze(0), dim=-1)
                    # Compute cosine similarity on GPU
                    sims_t = torch.matmul(gm_norm, centroid_norm.T).squeeze()
                    sims = sims_t.cpu().numpy() if device.type != "cpu" else sims_t.numpy()
                    top = np.argsort(sims)[::-1][:n_prototypes]
                else:
                    raise ValueError(f"Unknown metric '{metric}'.")

            name = str(grp) if grp is not None else "All"
            out[name] = [gi[i] for i in top]
        if plot:
            import matplotlib.pyplot as plt

            for group_name, indices in out.items():
                if not indices:
                    continue
                n = len(indices)
                ncols = min(5, n)
                nrows = (n + ncols - 1) // ncols
                fig_w = min(4.0 * ncols, 22)
                fig_h = max(3.4 * nrows + 1.0, 3.8)
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(fig_w, fig_h),
                    gridspec_kw={"wspace": 0.04, "hspace": 0.10},
                )
                if nrows == 1 and ncols == 1:
                    ax_flat = np.array([axes])
                else:
                    ax_flat = np.asarray(axes).ravel()

                for j, idx in enumerate(indices):
                    self.plot_image_by_index(  # type: ignore[attr-defined]
                        idx,
                        ax=ax_flat[j],
                        apply_transforms=False,
                        downsample=720,
                        show_extra_info=False,
                    )

                for j in range(n, len(ax_flat)):
                    ax_flat[j].set_visible(False)

                fig.suptitle(f"Cluster {group_name} prototypes", fontsize=13, fontweight="bold")
                fig.tight_layout(pad=0.15, rect=[0, 0, 1, 0.95])
                try:
                    from IPython import get_ipython
                    from IPython.display import display as ipy_display_fig
                except ImportError:
                    ipython_shell = None
                else:
                    ipython_shell = get_ipython()
                if ipython_shell is not None:
                    ipy_display_fig(fig)
                else:
                    plt.show()
                plt.close(fig)

        return out

    # ------------------------------------------------------------------
    # Embedding-property correlation
    # ------------------------------------------------------------------

    def compute_embedding_property_correlations(
        self,
        property_keys: list[str] | None = None,
        normalize: bool = True,
        method: Literal["pearson", "spearman", "distance_correlation", "mutual_info"] = "pearson",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Compute correlations between embedding dimensions and phenotypic scalar properties.

        This is the time-consuming step that computes raw correlations for each property.

        Args:
            property_keys: Property names (all if *None*).
            normalize: Whether to normalize data before computing correlations (default: True).
                - Embeddings: L2 normalization (unit norm)
                - Properties: StandardScaler normalization (zero mean, unit variance)
            method: Correlation method to use. Options:
                - ``'pearson'``: Pearson correlation coefficient (default, linear relationships)
                - ``'spearman'``: Spearman rank correlation (monotonic relationships, robust to outliers)
                - ``'distance_correlation'``: Distance correlation (detects non-linear relationships,
                  requires ``dcor`` library: ``pip install dcor``)
                - ``'mutual_info'``: Normalized mutual information (detects any dependency,
                  normalized to [0, 1] range)

        Returns:
            Dict with:
                - ``correlations``: Dict mapping property names to correlation arrays (n_dims,)
                - ``embedding_shape``: Shape of embeddings (n_samples, n_dims)
                - ``n_properties``: Number of properties processed
                - ``correlation_method``: Method used for correlation computation
        """
        embeddings = self.get_embeddings()  # type: ignore[attr-defined]
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings not available. Run process_images() first.")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got shape {embeddings.shape}")

        # Apply L2 normalization if requested
        if normalize:
            embeddings = self._normalize_embeddings_l2(embeddings)

        n_samples, n_dims = embeddings.shape

        if property_keys is None:
            property_keys = self.get_available_property_keys()  # type: ignore[attr-defined]
        if not property_keys:
            raise ValueError("No properties found. Run compute_properties() first.")

        props_list = self.results.properties
        valid_props: dict[str, np.ndarray] = {}
        for key in property_keys:
            arr = np.array([p.get(key, np.nan) for p in props_list], dtype=np.float32)
            if len(arr) != n_samples:
                continue

            # Require at least 3 finite values and non-zero variance; skip constant properties.
            finite_mask = np.isfinite(arr)
            if finite_mask.sum() < 3:
                continue
            finite_vals = arr[finite_mask]
            if np.ptp(finite_vals) == 0:
                logger.info(
                    "Skipping property '%s': constant values (no variance).",
                    key,
                )
                continue

            valid_props[key] = arr
        if not valid_props:
            raise ValueError(f"No valid properties of length {n_samples} with finite values.")

        # Normalize properties using the unified PhenoMeProperties helper if requested.
        # This ensures consistency and uses the internal normalization cache.
        if normalize:
            valid_prop_keys = list(valid_props.keys())
            # We use all indices (None) to ensure we get the full normalized matrix
            # matching the original properties_list length.
            norm_matrix, _, _ = self._get_property_matrix(  # type: ignore[attr-defined]
                indices=None,
                property_keys=valid_prop_keys,
                normalize=True,
                handle_nans="warn",
            )
            # Update valid_props with normalized values
            for idx, key in enumerate(valid_prop_keys):
                valid_props[key] = norm_matrix[:, idx]

        corrs: dict[str, np.ndarray] = {}

        # Show progress for slower methods
        use_progress = method in ("distance_correlation", "mutual_info")

        def _worker(pname: str, parr: np.ndarray) -> tuple[str, np.ndarray | None]:
            ok = np.isfinite(parr)
            if ok.sum() < 3:
                return pname, None

            r = self._compute_correlation(method, embeddings[ok], parr[ok])

            # Ensure r is the right shape (n_dims,)
            if np.isscalar(r):
                r = np.full(n_dims, r)
            elif len(r) != n_dims:
                # Pad or truncate if needed
                if len(r) < n_dims:
                    r_padded = np.full(n_dims, np.nan)
                    r_padded[: len(r)] = r
                    r = r_padded
                else:
                    r = r[:n_dims]
            return pname, r

        if n_jobs == 1:
            prop_iter = (
                tqdm(
                    valid_props.items(),
                    desc=f"Computing {method} correlations",
                    disable=not use_progress,
                    leave=False,
                )
                if use_progress
                else valid_props.items()
            )
            for pname, parr in prop_iter:
                pname_res, r = _worker(pname, parr)
                if r is not None:
                    corrs[pname_res] = r
        else:
            # Parallel execution
            items = list(valid_props.items())
            results = Parallel(n_jobs=n_jobs)(
                delayed(_worker)(pname, parr)
                for pname, parr in tqdm(
                    items,
                    desc=f"Computing {method} correlations",
                    disable=not use_progress,
                    leave=False,
                )
            )
            for pname, r in results:
                if r is not None:
                    corrs[pname] = r

        logger.info(
            "Computed embedding-property correlations: %d properties, method=%s",
            len(valid_props),
            method,
        )

        return {
            "correlations": corrs,
            "embedding_shape": embeddings.shape,
            "n_properties": len(valid_props),
            "correlation_method": method,
        }

    @staticmethod
    def _log_embedding_property_correlation_summary(
        summary: pd.DataFrame, order_by: str, top_k: int | None
    ) -> None:
        """Log a plain-text table of the top property correlation rows."""
        want = ("property", "sign", "mean_abs", "std", "max_abs", "min_abs")
        columns = [c for c in want if c in summary.columns]
        if not columns or "property" not in columns:
            logger.info("Property correlation summary is empty; nothing to log.")
            return
        view = summary[columns].copy()
        if top_k is not None:
            view = view.head(int(top_k))
        n = len(view)
        top_note = f" (showing {n} of {len(summary)} properties)" if n < len(summary) else ""
        sep = "─" * 88
        num_cols = [c for c in columns if c not in ("property", "sign")]

        def _fmt_num_cell(v: object) -> str:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return "—"
            if not np.isfinite(x):
                return "—"
            return f"{x:.4f}"

        out = view.copy()
        for c in num_cols:
            out[c] = out[c].map(_fmt_num_cell)
        if "property" in out.columns and "sign" in out.columns:
            # Pad property (first column) so there is a bit more space before sign (second column).
            prop_w = max((len(str(x)) for x in out["property"]), default=0) + 6
            prop_w = max(prop_w, len("property") + 4)
            out = out.copy()
            out["property"] = out["property"].map(lambda s: f"{s!s:<{prop_w}}")
        block = out.to_string(index=False, col_space=2)
        logger.info(
            "%s\nEmbedding-property correlations  (ordered by %s)%s\n%s\n%s\n%s",
            sep,
            order_by,
            top_note,
            sep,
            block,
            sep,
        )

    def summarize_embedding_property_correlations(
        self,
        correlation_results: dict[str, Any],
        order_by: str = "mean_abs",
        top_k: int | None = 20,
        plot: bool = True,
        return_fig: bool = False,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Property Correlations with Embeddings",
    ) -> dict[str, Any]:
        """Summarize embedding-property correlations, optionally plot, and/or return a Plotly figure.

        Computes per-property ``mean_abs``, ``std`` (across dimensions), ``max_abs``, ``min_abs``,
        ``mean``, ``sign`` (of mean raw *r*), orders rows by ``order_by``, and either shows a
        horizontal violin plot of the distribution of |r| across dimensions per property or logs
        a plain-text table when ``plot`` is *False*.

        Args:
            correlation_results: Output from :meth:`compute_embedding_property_correlations`
                (must include ``correlations`` and ``correlation_method``).
            order_by: Metric used to sort properties (descending):
                ``mean_abs`` | ``max_abs`` | ``mean`` | ``std``.
            top_k: Number of top properties shown in the plot and listed in ``top_properties``;
                full sorted table is always in ``summary`` (*None* = all).
            plot: If *True* (default), show a Plotly violin plot of |r| per dimension for the top
                ``top_k`` properties by ``order_by`` (*None* = all). Highest metric at the **top**
                of the y-axis.
            return_fig: If *True*, include the :class:`plotly.graph_objects.Figure` in the
                result under key ``"figure"``.
            figsize: Figure size in inches, converted to pixels for Plotly layout.
            title: Chart title.

        Returns:
            Dict with:
                - ``correlations``: Original per-property correlation vectors
                - ``summary``: Sorted DataFrame with ``property``, ``mean_abs``, ``std``,
                  ``max_abs``, ``min_abs``, ``mean``, ``sign``, etc.
                - ``top_properties``: Top-``k`` property names
                - ``metrics``: Per-metric dicts (including ``min_abs``) keyed by property name
                - ``order_by`` — metric used for sorting
                - ``correlation_method``, ``embedding_shape``, ``n_properties``
                - ``figure``: Plotly figure if ``return_fig`` is *True*; otherwise *None*.
        """
        if order_by not in _helpers.ORDER_METRICS:
            raise ValueError(
                f"Unknown order_by: {order_by!r}; expected one of {_helpers.ORDER_METRICS}"
            )
        if not correlation_results or "correlations" not in correlation_results:
            raise KeyError(
                "correlation_results must contain 'correlations' (output of "
                "compute_embedding_property_correlations)."
            )

        corrs: dict[str, np.ndarray] = correlation_results["correlations"]
        embedding_shape = correlation_results.get(
            "embedding_shape", correlation_results.get("cls_shape")
        )
        n_properties = correlation_results.get(
            "n_properties", correlation_results.get("n_features")
        )
        correlation_method = str(correlation_results["correlation_method"])

        per_prop: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {m: {} for m in _helpers.ORDER_METRICS}
        metrics["min_abs"] = {}

        for pname, r in corrs.items():
            vc = r[np.isfinite(r)]
            if len(vc) == 0:
                row = {
                    "property": pname,
                    "mean": np.nan,
                    "std": np.nan,
                    "mean_abs": np.nan,
                    "max_abs": np.nan,
                    "min_abs": np.nan,
                    "sign": "—",
                }
            else:
                mean_v = float(np.mean(vc))
                std_v = float(np.std(vc, ddof=0))
                mean_abs_v = float(np.mean(np.abs(vc)))
                max_abs_v = float(np.max(np.abs(vc)))
                min_abs_v = float(np.min(np.abs(vc)))
                row = {
                    "property": pname,
                    "mean": mean_v,
                    "std": std_v,
                    "mean_abs": mean_abs_v,
                    "max_abs": max_abs_v,
                    "min_abs": min_abs_v,
                    "sign": _helpers.mean_correlation_sign(mean_v),
                }
            per_prop.append(row)
            for name in _helpers.ORDER_METRICS:
                metrics[name][pname] = float(row[name])
            metrics["min_abs"][pname] = float(row["min_abs"])

        summary = pd.DataFrame(per_prop)
        summary = summary.sort_values(order_by, ascending=False, na_position="last")
        summary = summary.reset_index(drop=True)

        top_properties: list[str] = (
            summary.head(top_k)["property"].astype(str).tolist()
            if top_k is not None
            else summary["property"].astype(str).tolist()
        )

        out_fig: Any = None
        need_fig = bool(plot) or bool(return_fig)
        if need_fig:
            display_k = top_k if top_k is not None else len(summary)
            plot_df = summary.head(int(display_k)).copy()
            # Plotly reverses categorical ``category_orders`` when mapping to y-axis positions for
            # horizontal violins; sort descending here so strongest ``order_by`` ends up at the top.
            plot_df = plot_df.sort_values(order_by, ascending=False, na_position="last")
            out_fig = _plot_property_correlations_plotly(
                plot_df,
                correlations=corrs,
                order_by=order_by,
                title=title,
                figsize=figsize,
                correlation_method=correlation_method,
            )
            if plot and out_fig is not None:
                out_fig.show()

        if not plot:
            self._log_embedding_property_correlation_summary(summary, order_by, top_k)

        figure_out: Any = out_fig if (return_fig and need_fig) else None

        return {
            "correlations": corrs,
            "summary": summary,
            "top_properties": top_properties,
            "metrics": metrics,
            "embedding_shape": embedding_shape,
            "n_properties": n_properties,
            "order_by": order_by,
            "correlation_method": correlation_method,
            "figure": figure_out,
        }

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_embeddings_l2(self, data: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to embedding vectors (unit vectors).

        Uses GPU acceleration when device is available.

        Args:
            data: Embedding matrix (n_samples, n_features).

        Returns:
            L2-normalized embeddings (unit vectors).
        """
        device = self.device
        data_t = torch.from_numpy(_helpers.numpy_for_torch(data)).to(device)

        with torch.no_grad():
            norms = torch.norm(data_t, dim=1, keepdim=True)
            # Avoid division by zero
            norms = torch.where(norms == 0, torch.tensor(1.0, device=device), norms)
            normalized = data_t / norms

        return normalized.cpu().numpy()

    # ------------------------------------------------------------------
    # Correlation helper functions
    # ------------------------------------------------------------------

    def _compute_pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation between x and y (delegates to core)."""
        return compute_pearson_correlation(x, y, device=self.device)

    def _compute_spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Spearman rank correlation between x and y (delegates to core)."""
        return compute_spearman_correlation(x, y, device=self.device)

    def _compute_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute distance correlation between x and y (delegates to core)."""
        return compute_distance_correlation(x, y, device=self.device)

    def _compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute normalized mutual information (delegates to core)."""
        return compute_mutual_info(x, y, seed=getattr(self, "seed", None), device=self.device)

    def _compute_correlation(self, method: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Dispatch to the correlation function for the given method."""
        fn_name = _CORRELATION_METHOD_REGISTRY.get(method)
        if fn_name is None:
            raise ValueError(f"Unknown correlation method: {method}")
        fn = getattr(self, fn_name)
        return fn(x, y)

    # ------------------------------------------------------------------
    # Shared helper: DRY filter -> get data -> valid_indices
    # ------------------------------------------------------------------

    def _get_data_matrix(
        self,
        source: Literal["embeddings", "properties", "combined"],
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        property_keys: list[str] | None = None,
        caller: str = "",
        normalize: bool = True,
    ) -> tuple[np.ndarray, list[int]]:
        """Get data matrix and valid indices for requested source.

        Centralizes the repeated filter-then-fetch pattern used by clustering,
        outlier detection, and prototype finding.

        Args:
            source: 'embeddings' for model embeddings, 'properties' for scalar properties,
                or 'combined' for the normalized concatenation of both.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            property_keys: Property subset (only used when *source* is ``'properties'`` or ``'combined'``).
            caller: Human-readable method name for error messages.
            normalize: Whether to normalize the data. For embeddings, uses L2 normalization.
                For properties, uses StandardScaler (handled by _get_property_matrix). For
                'combined', both parts are normalized before concatenation.

        Returns:
            Tuple of (matrix, valid_indices):
            - matrix: np.ndarray shape (n_valid, n_features).
              For embeddings: (n_valid, D); for properties: (n_valid, n_props);
              for combined: (n_valid, D + n_props).
            - valid_indices: List[int]. Global image indices corresponding to matrix rows.

        Raises:
            ValueError: When no data is available or source is unknown.
        """
        indices = filter_indices(self.results, filters, exclude) if (filters or exclude) else None

        if source == "embeddings":
            # Determine candidate indices (filter or all)
            if indices is None:
                n_samples = len(self.results.img_path)
                indices = list(range(n_samples))

            # Load ONLY the requested rows — lazy if checkpoint-backed
            data = self.get_embeddings(indices)  # type: ignore[attr-defined]
            if data is None or len(data) == 0:
                raise ValueError(f"No embeddings available ({caller}). Run process_images() first.")

            if normalize:
                data = self._normalize_embeddings_l2(data)

            return data, indices

        if source == "properties":
            matrix, valid_indices, _ = self._get_property_matrix(  # type: ignore[attr-defined]
                indices=indices,
                property_keys=property_keys,
                normalize=normalize,
                handle_nans="filter",
            )
            if len(valid_indices) == 0:
                raise ValueError(f"No valid samples after NaN filtering ({caller}).")
            return matrix, valid_indices

        if source == "combined":
            if indices is None:
                n_samples = len(self.results.img_path)
                indices = list(range(n_samples))
            try:
                combined, prop_valid_indices = build_combined_features(
                    cast(PhenoMeProtocol, self),
                    indices,
                    property_keys=property_keys,
                    normalize=normalize,
                )
            except ValueError as e:
                raise ValueError(f"{e} ({caller})") from e
            return combined, prop_valid_indices

        raise ValueError(
            f"Unknown source '{source}', expected 'embeddings', 'properties', or 'combined'."
        )
