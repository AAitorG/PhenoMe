"""
Analysis for the phenotyping pipeline.

Provides clustering, outlier detection, enrichment analysis, prototype finding,
and correlation analysis between embeddings and extracted properties.
"""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
from joblib import delayed

from .._logging import get_logger
from ..core import (
    build_combined_features,
    compute_distance_correlation,
    compute_lasso_interpretability,
    compute_mutual_info,
    compute_pearson_correlation,
    compute_rf_interpretability,
    compute_spearman_correlation,
    filter_indices,
    run_dimensionality_reduction,
    run_dimensionality_reduction_matrix,
    validate_results,
)
from ..core.dataframe_contract import (
    IMAGE_INDEX,
    IMAGE_PATH,
    TIER_A_EXCLUDE_FROM_PROPERTIES,
    pack_df_meta_fig,
    per_image_dataframe,
)
from ..core.math.correlation import (
    compute_component_property_correlation_matrix,
    filter_valid_property_arrays,
)
from ..core.math.interpretability import prepare_interpretability_inputs
from ..core.pipeline_results import PhenoMeResults
from ..core.protocols import PhenoMeProtocol
from ..core.results_metadata import (
    _is_missing_store_value,
    build_composite_group_series,
    build_index_group_labels,
    build_subset_groups,
    collect_metadata_labels,
    get_result_value,
    normalize_group_value,
    resolve_result_keys,
)
from ..core.run_log import record_step
from ..utils.display_names import (
    format_clustering_method,
    format_correlation_method,
    format_dr_method,
)
from ..utils.path_utils import path_basename
from ..utils.progress import log_computing, parallel_with_progress, track, track_steps
from . import _helpers
from .properties.grouping import (
    compute_leave_one_out_zscore_enrichment,
    resolve_dataframe_columns,
)
from .visualization._distance_plots import (
    _log_embedding_property_correlation_summary,
    _plot_property_correlations_plotly,
)
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
        include_metadata: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, float | None]:
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
                Returns ``(df, score)``; score is None if it could not be computed.
                Silhouette requires at least 2 clusters and 2 samples per cluster.
            dbscan_eps: Maximum distance between two samples for DBSCAN (only when
                clustering_method='dbscan'). If None, uses 0.5.
            dbscan_min_samples: Minimum samples in a neighborhood for DBSCAN (only when
                clustering_method='dbscan'). If None, uses max(2, n_samples // 20).
            include_metadata: If True, add metadata columns to the returned DataFrame.

        Returns:
            If return_silhouette=False: Tier-A DataFrame with image_index, image_path,
            image_name, cluster, and optional metadata.
            If return_silhouette=True: Tuple of (df, silhouette_score). Score is None
                if it could not be computed. Cluster labels are 0..K-1; None for excluded/noise.
        """
        record_step(
            self,
            "compute_clustering",
            {
                "source": source,
                "n_clusters": n_clusters,
                "clustering_method": clustering_method,
                "property_keys": property_keys,
                "filters": filters,
                "exclude": exclude,
                "random_state": random_state,
                "normalize": normalize,
                "reduce_dim": reduce_dim,
                "reduce_method": reduce_method,
                "return_silhouette": return_silhouette,
                "dbscan_eps": dbscan_eps,
                "dbscan_min_samples": dbscan_min_samples,
                "include_metadata": include_metadata,
            },
        )
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
            raise ValueError(f"Clustering failed: {exc}") from exc

        if len(matrix) == 0:
            logger.warning("No data available for clustering.")
            df = per_image_dataframe(
                self.results,
                include_metadata=include_metadata,
                extra_columns={"cluster": [None] * n_total},
            )
            if return_silhouette:
                return df, None
            return df

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
                if reduce_method in ("tsne", "umap"):
                    logger.warning(
                        "Clustering in %s space: non-linear embeddings are not isometries; "
                        "cluster boundaries may not reflect true data structure. "
                        "Consider using 'pca' for reduce_method instead.",
                        format_dr_method(reduce_method),
                    )
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
                    format_dr_method(reduce_method),
                    n_comp,
                    orig_dim,
                    format_clustering_method(clustering_method),
                )

        # Fitted DR object from the optional pre-clustering step (None if no reduction).
        self._last_clustering_dr_object = dr_object_after_reduce

        # Step 3: Apply the clustering only over the extracted components
        effective_seed = random_state if random_state is not None else getattr(self, "seed", None)
        log_computing(
            f"Computing {format_clustering_method(clustering_method)} clustering "
            f"on {len(matrix)} images..."
        )

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
        if return_silhouette:
            silhouette_score_val = _helpers.compute_silhouette_if_valid(
                matrix, labels, clustering_method=clustering_method
            )

        # Store cluster labels in results.metadata for downstream use
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
        validate_results(self.results)

        n_clusters_log = n_actual_clusters if clustering_method == "dbscan" else n_clusters
        logger.info(
            "Clustering (%s): %d clusters on %d images (source=%s).",
            clustering_method,
            n_clusters_log,
            len(valid_indices),
            source,
        )

        df = per_image_dataframe(
            self.results,
            include_metadata=include_metadata,
            extra_columns={
                "cluster": [int(x) if not np.isnan(x) else None for x in full_labels],
            },
        )

        if return_silhouette:
            return (df, silhouette_score_val)
        return df

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
        stratify_by: str | None = None,
        rf_n_estimators: int = 100,
        seed: int | None = None,
        plot: bool = True,
        return_fig: bool = False,
        return_meta: bool = False,
        top_k: int = 10,
        figsize: tuple[int, int] = (10, 8),
    ) -> (
        pd.DataFrame
        | tuple[pd.DataFrame, dict[str, Any]]
        | tuple[pd.DataFrame, Any]
        | tuple[pd.DataFrame, dict[str, Any], Any]
    ):
        """Explain a dimensionality reduction component using LASSO or Random Forest.

        Calculates which phenotypic properties (features) best explain the variability
        seen in a deep learning embedding dimension (the target, usually t-SNE 1 or 2).

        Workflow: dimensionality reduction is run on the full filtered embedding set
        first; regression then relates classical properties to the resulting axis
        coordinates. For transductive methods (t-SNE, UMAP), axis coordinates depend on
        all samples, so the returned ``r2`` (LASSO) describes descriptive on-axis alignment
        on the same dataset—not independent generalization to new plates or refit
        embeddings. Prefer ``drivers`` for hypothesis generation; interpret ``r2`` as an
        effect-size-style summary unless DR and evaluation are nested with proper holdouts.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap').
            component: Which component to explain (1, 2, ...).
            model_type: The regression model to use ('lasso' or 'random_forest').
                'lasso' uses L1 regularization for linear, sparse explanations.
                'random_forest' captures non-linear relationships.
            property_keys: Subset of properties to use as features.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions.
            normalize: Whether to normalize embeddings before dimensionality reduction
                (default: True). Property features are scaled fold-wise inside the LASSO
                pipeline (not globally) to avoid cross-validation leakage.
            cv: Number of cross-validation folds (only for 'lasso').
            stratify_by: Optional metadata key (e.g. ``"Treatment"``, ``"Batch"``) to
                stratify LASSO folds by class balance. Requires at least ``cv`` samples
                per class; otherwise falls back to unstratified KFold. Balances folds for
                ``lambda`` selection but does not correct global t-SNE/UMAP target leakage.
            rf_n_estimators: Number of trees (only for 'random_forest').
            seed: Random seed for reproducibility. If None, uses the pipeline's ``seed`` when set.
            plot: If True (default), show an interactive Plotly bar chart of top drivers.
                If False, log a plain-text summary via the package logger instead.
            return_fig: If True, include the Plotly figure in the return value (see Returns).
                When ``return_fig`` is True, ``fig.show()`` is not called; use
                ``plot=True`` with ``return_fig=False`` for interactive display.
            return_meta: If True, include a run-parameters dict in the return value.
            top_k: Number of top drivers to show in the plot or text summary.
            figsize: Figure size ``(width, height)`` passed through to Plotly layout; each
                value is multiplied by 100 to set width and height in layout pixels.

        Returns:
            pd.DataFrame or tuple, depending on *return_meta* and *return_fig*:

            - Default: ``drivers_df`` with columns ``feature``, ``weight``.
            - ``return_meta=True``: ``(drivers_df, meta)``; *meta* includes r2 (descriptive
              on-axis fit for globally fit DR targets), n_samples, n_features, method,
              model_type, target_component.
            - ``return_fig=True``: ``(drivers_df, fig)`` or ``(drivers_df, meta, fig)`` when
              *return_meta* is also True.
        """
        record_step(
            self,
            "compute_multivariate_interpretability",
            {
                "method": method,
                "component": component,
                "model_type": model_type,
                "property_keys": property_keys,
                "filters": filters,
                "exclude": exclude,
                "normalize": normalize,
                "cv": cv,
                "stratify_by": stratify_by,
                "rf_n_estimators": rf_n_estimators,
                "seed": seed,
                "plot": plot,
                "return_fig": return_fig,
                "return_meta": return_meta,
                "top_k": top_k,
            },
        )
        effective_seed = seed if seed is not None else getattr(self, "seed", None)

        # Step 1: Global DR on all filtered embeddings → fixed axis coordinates (y).
        # t-SNE/UMAP are transductive: y already encodes the full-sample layout, so any
        # later CV in compute_lasso_interpretability cannot claim out-of-sample DR validity.
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
            df = pd.DataFrame()
            return pack_df_meta_fig(df, {}, None, return_meta=return_meta, return_fig=return_fig)

        comp_col = f"Component {component}"
        if comp_col not in dr_results.columns:
            logger.warning("Target component '%s' not found in DR results.", comp_col)
            df = pd.DataFrame()
            return pack_df_meta_fig(df, {}, None, return_meta=return_meta, return_fig=return_fig)

        y = dr_results[comp_col].values
        valid_indices = dr_results["Index"].values.tolist()

        # Step 2: Property matrix for the same samples. Fold-wise StandardScaler in LASSO
        # handles normalization; skip global property scaling to avoid holdout leakage.
        # Drop NaN properties (columns) rather than samples so sparse missing features
        # (e.g. all-NaN blur metrics) do not wipe the whole cohort.
        matrix, prop_valid_indices, keys = self._get_property_matrix(  # type: ignore[attr-defined]
            indices=valid_indices,
            property_keys=property_keys,
            normalize=False,
            handle_nans="drop_columns",
        )

        if len(matrix) == 0 or matrix.shape[1] == 0 or not keys:
            logger.warning("No valid samples with properties for interpretability.")
            df = pd.DataFrame()
            return pack_df_meta_fig(df, {}, None, return_meta=return_meta, return_fig=return_fig)

        prepared = prepare_interpretability_inputs(y, valid_indices, matrix, prop_valid_indices)
        if prepared is None:
            logger.warning("No valid samples after filtering non-finite targets.")
            df = pd.DataFrame()
            return pack_df_meta_fig(df, {}, None, return_meta=return_meta, return_fig=return_fig)
        matrix, y, prop_valid_indices = prepared

        stratify_labels = None
        if stratify_by is not None and model_type == "lasso":
            stratify_labels = collect_metadata_labels(self.results, prop_valid_indices, stratify_by)
            if len(stratify_labels) != len(matrix):
                logger.warning(
                    "Stratify label length mismatch (%d vs %d); ignoring stratify_by.",
                    len(stratify_labels),
                    len(matrix),
                )
                stratify_labels = None

        # Step 3: Run regression model. LASSO/forest fits do not report fold progress.
        model_label = "LASSO" if model_type == "lasso" else "random forest"
        log_computing(
            f"Fitting {model_label} model on {len(y)} samples and {matrix.shape[1]} properties..."
        )
        if model_type == "lasso":
            results = compute_lasso_interpretability(
                x=matrix,
                y=y,
                feature_names=keys,
                cv=cv,
                seed=effective_seed,
                stratify=stratify_labels,
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
                "stratify_by": stratify_by,
                "dr_object": dr_obj,
            }
        )

        drivers = results.pop("drivers", [])
        df = pd.DataFrame(columns=["feature", "weight"]) if not drivers else pd.DataFrame(drivers)
        meta = {k: v for k, v in results.items() if k != "dr_object"}

        logger.info(
            "Multivariate Interpretability (%s, %s, %s): R^2=%.2f, %d drivers found.",
            format_dr_method(method),
            comp_col,
            model_type,
            meta.get("r2", 0.0),
            len(df),
        )

        results.pop("interpretability_fig", None)
        fig = _display_multivariate_interpretability(
            df,
            meta,
            plot=plot,
            return_fig=return_fig,
            top_k=top_k,
            figsize=figsize,
        )

        return pack_df_meta_fig(df, meta, fig, return_meta=return_meta, return_fig=return_fig)

    # ------------------------------------------------------------------
    # Outlier detection
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        method: Literal["z-score", "iqr"] = "z-score",
        threshold: float = 3.0,
        group_by: str | list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        normalize: bool = True,
        drop_outliers: bool = False,
        plot: bool = True,
    ) -> pd.DataFrame:
        """Detect outliers based on distance to centroid.

        When *source* is ``'embeddings'`` and *normalize* is True (default), embeddings
        are L2-normalized and group centroids use the **arithmetic mean** (not necessarily
        a unit vector). Euclidean distance to that centroid is not the same as angular
        distance to the mean direction; consider property-based or combined sources when
        a different geometry is needed.

        Args:
            method: ``'z-score'`` or ``'iqr'``.
            threshold: Threshold multiplier.
            group_by: Property/metadata key(s) for grouping. Can be a single string
                or a list of strings. If a list is provided, groups are formed by
                combining values from all specified columns (e.g., "drug1-10uM").
                If *None*, all images are treated as one group.
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
            pd.DataFrame: Tier-A DataFrame containing detected outliers.
                Columns include:
                - ``image_index``: Global image index.
                - ``image_path``: Path to the image file.
                - ``image_name``: Basename of the image file.
                - Additional columns for each grouping key (if *group_by* was set).
                - ``threshold``: The threshold value used for the image's group.
                - ``distance_to_centroid``: Distance to the group centroid.

                If no outliers are detected, returns an empty DataFrame.
                The index is a range index from 0 to N-1.
        """
        record_step(
            self,
            "detect_outliers",
            {
                "method": method,
                "threshold": threshold,
                "group_by": group_by,
                "filters": filters,
                "exclude": exclude,
                "source": source,
                "property_keys": property_keys,
                "normalize": normalize,
                "drop_outliers": drop_outliers,
                "plot": plot,
            },
        )
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
            raise ValueError(f"Outlier detection failed: {exc}") from exc

        if data is None or len(data) == 0:
            return pd.DataFrame()

        data = _helpers.numpy_for_torch(data)

        n_samples = len(data)
        outlier_mask = np.zeros(n_samples, dtype=bool)
        distances = np.full(n_samples, np.nan, dtype=np.float32)

        device = self.device

        # Convert data to GPU once (avoid repeated conversions in loop)
        data_t = torch.from_numpy(data).to(device) if device.type != "cpu" else None

        def _outliers_for(
            vectors: np.ndarray, vectors_t: torch.Tensor | None = None
        ) -> tuple[np.ndarray, np.ndarray, Any]:
            if vectors_t is not None:
                vec_t = vectors_t
            else:
                vec_t = torch.from_numpy(_helpers.numpy_for_torch(vectors)).to(device)

            with torch.no_grad():
                centroid = torch.mean(vec_t, dim=0)
                dists_t = torch.norm(vec_t - centroid.unsqueeze(0), dim=1)
                dists = dists_t.cpu().numpy() if device.type != "cpu" else dists_t.numpy()

            return _helpers.outlier_mask_from_distances(dists, method, threshold)

        # Build groups
        groups: dict[Any, np.ndarray] = {}
        actual_group_by: str | list[str] | None = None

        if group_by:
            requested_cols = [group_by] if isinstance(group_by, str) else group_by
            found_cols = resolve_result_keys(self.results, requested_cols)

            if not found_cols:
                logger.warning("No matching grouping keys found. Treating as single group.")
                groups["all"] = np.arange(n_samples)
            else:
                groups, actual_group_by = build_subset_groups(
                    self.results, valid_indices, found_cols
                )
        else:
            groups["all"] = np.arange(n_samples)

        thresholds: dict[Any, float] = {}
        for gname, didx in track_steps(
            groups.items(),
            desc="Detecting outliers",
            when_single=f"Detecting outliers with {method} on {n_samples} images...",
        ):
            if len(didx) < 2:
                continue
            didx = _helpers.index_array_for_torch(didx)
            # Pass pre-converted tensor slice if available
            vec_t_slice = data_t[didx] if data_t is not None else None
            d, is_out, tv = _outliers_for(data[didx], vec_t_slice)
            distances[didx] = d
            outlier_mask[didx] = is_out
            thresholds[gname] = tv

        # Build the result DataFrame for outliers only
        rows = []
        rel_out = np.where(outlier_mask)[0]
        outlier_indices = [valid_indices[i] for i in rel_out]

        for i, idx in enumerate(outlier_indices):
            # Fetch basic info without all metadata/properties
            path = self.results.img_path[idx]

            row = {
                IMAGE_INDEX: idx,
                IMAGE_PATH: path,
                "image_name": path_basename(path),
            }

            if actual_group_by:
                if isinstance(actual_group_by, list):
                    vals = [get_result_value(self.results, idx, k) for k in actual_group_by]
                    group_val = " | ".join(str(normalize_group_value(v)) for v in vals)
                    for k, val in zip(actual_group_by, vals, strict=True):
                        row[k] = val
                else:
                    val = get_result_value(self.results, idx, actual_group_by)
                    group_val = normalize_group_value(val)
                    row[actual_group_by] = val

                row["_internal_group"] = group_val
                row["threshold"] = thresholds.get(group_val, 0.0)
            else:
                row["_internal_group"] = "all"
                row["threshold"] = thresholds.get("all", 0.0)

            row["distance_to_centroid"] = float(distances[rel_out[i]])
            rows.append(row)

        result_df = pd.DataFrame(rows)

        logger.info(
            "Detected %d outliers out of %d images.", len(outlier_indices), len(valid_indices)
        )

        if plot and outlier_indices:
            import matplotlib.pyplot as plt

            # Group outliers by group name
            outliers_by_group = (
                result_df.groupby("_internal_group")[IMAGE_INDEX].apply(list).to_dict()
            )

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
                fig.subplots_adjust(top=0.85)
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
            validate_results(self.results)

            logger.info(
                "Removed %d outliers from results. Remaining: %d images.",
                len(outlier_indices),
                len(self.results.img_path),
            )

        if not result_df.empty:
            result_df.reset_index(drop=True, inplace=True)
            if "_internal_group" in result_df.columns:
                result_df.drop(columns=["_internal_group"], inplace=True)

        return result_df

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
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
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
                - ``'mutual_info'``: MI-derived correlation coefficient (detects any dependency,
                  mapped to [0, 1] via the Gaussian bivariate transform, **not** standard NMI)
            plot: If True (default), show an interactive Plotly faceted bar chart (one row per
                component). If False, log a plain-text summary via the package logger instead
                (unless *return_fig* requests a figure).
            return_fig: If True, return a tuple of ``(correlation_df, figure)``.
                When ``return_fig`` is True, ``fig.show()`` is not called.
            figsize: Figure size ``(width, height)`` in inches for the Plotly layout.

        Returns:
            pd.DataFrame | tuple[pd.DataFrame, Any]:
                - If ``return_fig`` is False (default): Correlation DataFrame with a
                  ``property`` column and ``component_1``, ``component_2``, … columns.
                - If ``return_fig`` is True: Returns a tuple of ``(correlation_df, figure)``.

        Raises:
            ValueError: If dimensionality reduction fails, no components are found,
                no numeric properties are available, or correlation computation fails.

        Examples:
            Get correlations directly::

                corr = pheno.compute_component_correlation(plot=False)

            Get correlations and save the figure::

                corr, fig = pheno.compute_component_correlation(return_fig=True)
                fig.write_html("component_corr.html")
        """
        record_step(
            self,
            "compute_component_correlation",
            {
                "method": method,
                "n_components": n_components,
                "source": source,
                "property_keys": property_keys,
                "filters": filters,
                "exclude": exclude,
                "top_k": top_k,
                "normalize": normalize,
                "correlation_method": correlation_method,
                "plot": plot,
                "return_fig": return_fig,
            },
        )
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
            raise ValueError("Dimensionality reduction returned empty result.")

        # Component columns use common naming: Component 1, Component 2, ... (all methods)
        comp_cols = [f"Component {i + 1}" for i in range(n_components)]
        comp_cols = [c for c in comp_cols if c in df.columns]
        if not comp_cols:
            raise ValueError(f"No components found for method {method}.")

        avail = set(self.get_available_property_keys())  # type: ignore[attr-defined]
        prop_cols = [c for c in df.columns if c in avail and pd.api.types.is_numeric_dtype(df[c])]
        if not prop_cols:
            raise ValueError("No numeric properties found to correlate.")

        # Normalize properties using the unified PhenoMeProperties helper if requested.
        # This ensures consistency and uses the internal normalization cache.
        if normalize and prop_cols:
            # We use the existing indices from the DataFrame to ensure alignment.
            indices = df["Index"].tolist()
            # Keep NaNs for pairwise correlation (one summary below). DR may already
            # have filtered the embedding matrix, but property columns can still miss values.
            norm_matrix, _, _ = self._get_property_matrix(  # type: ignore[attr-defined]
                indices=indices,
                property_keys=prop_cols,
                normalize=True,
                handle_nans="keep",
            )
            # Update DataFrame with normalized values
            for idx, col in enumerate(prop_cols):
                df[col] = norm_matrix[:, idx]

        nan_counts = {
            col: int(df[col].isna().sum()) for col in prop_cols if bool(df[col].isna().any())
        }
        if nan_counts:
            logger.info(
                "Properties with missing values (pairwise complete used for correlations): %s",
                nan_counts,
            )

        corr = compute_component_property_correlation_matrix(
            df,
            prop_cols,
            comp_cols,
            correlation_method,
            self._compute_correlation,
            show_progress=correlation_method != "pearson",
            progress_desc=(
                f"Computing {format_correlation_method(correlation_method)} correlations"
            ),
        )

        if corr.empty:
            raise ValueError("Correlation matrix is empty.")

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
        title = (
            f"Component-property correlations ({format_dr_method(method)}, "
            f"{format_correlation_method(correlation_method)})"
        )
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

        corr.index.name = "property"
        corr = corr.reset_index()
        component_rename = {comp: f"component_{i}" for i, comp in enumerate(comp_cols, start=1)}
        corr = corr.rename(columns=component_rename)

        if return_fig and fig is not None:
            return corr, fig
        return corr

    # ------------------------------------------------------------------
    # Group enrichment
    # ------------------------------------------------------------------

    def analyze_group_enrichment(
        self,
        group_by: str | list[str] = "cluster",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        plot: bool = True,
        return_fig: bool = False,
        top_k: int | None = None,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        correct_multiple_testing: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any]:
        """Compute z-score enrichment of properties per group (e.g. cluster labels).

        For each group, z-scores are computed against a **leave-group-out**
        population (all samples *except* the current group).  When
        *correct_multiple_testing* is True (default), Welch two-sample t-tests
        (group vs leave-one-out rest) are run per property and Benjamini-Hochberg
        FDR correction is applied across all (group, property) pairs; a
        ``significant`` column is added to the output DataFrame.

        Args:
            group_by: Property/metadata key(s) holding group labels (e.g. ``"cluster"``).
                Can be a single string or a list of strings for composite grouping.
            property_keys: Properties to analyse (all numeric if *None*).
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            plot: If True (default), show an interactive Plotly faceted bar chart (one row per
                group). If False, log a plain-text summary via the package logger instead
                (unless *return_fig* requests a figure).
            return_fig: If True, return a tuple ``(df, fig)`` where ``fig`` is the Plotly figure.
                When *return_fig* is True, ``fig.show()`` is not called; use
                ``plot=True`` with ``return_fig=False`` for the default interactive display.
            top_k: Max properties per group in the figure and in the text summary (``None`` = all).
            figsize: Figure size ``(width, height)`` in inches for the Plotly layout.
            title: Optional figure title.
            correct_multiple_testing (bool): If True (default), run Welch t-tests
                per (group, property), apply Benjamini-Hochberg FDR, and add
                ``p_value`` / ``significant`` columns (alpha = 0.05).

        Returns:
            pd.DataFrame or tuple[pd.DataFrame, Any]:
                - If *return_fig* is False (default): pd.DataFrame with columns:
                  [group_by columns], property, score, mean_group, mean_pop, p_value, abs_score,
                  and optionally significant when *correct_multiple_testing* is True.
                - If *return_fig* is True: A tuple (enrichment_df, fig).
        """
        record_step(
            self,
            "analyze_group_enrichment",
            {
                "group_by": group_by,
                "property_keys": property_keys,
                "filters": filters,
                "exclude": exclude,
                "plot": plot,
                "return_fig": return_fig,
                "top_k": top_k,
                "title": title,
                "correct_multiple_testing": correct_multiple_testing,
            },
        )
        working_df = self._build_properties_dataframe(include_metadata=True)  # type: ignore[attr-defined]

        # Resolve group columns (case-insensitive)
        resolved_cols = resolve_dataframe_columns(working_df, group_by)

        empty_cols = [
            *resolved_cols,
            "property",
            "score",
            "mean_group",
            "mean_pop",
            "p_value",
            "abs_score",
        ]
        if correct_multiple_testing:
            empty_cols.append("significant")
        if working_df.empty:
            edf = pd.DataFrame(columns=empty_cols)
            fig = self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=pd.DataFrame(columns=["Group", *empty_cols]),
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return (edf, fig) if return_fig else edf

        if filters or exclude:
            indices = filter_indices(self.results, filters, exclude)
            if IMAGE_INDEX not in working_df.columns:
                raise ValueError(
                    "Filtered group enrichment requires an 'image_index' column in the "
                    "properties DataFrame. Rebuild with include_metadata=True or avoid "
                    "filters/exclude on misaligned DataFrames."
                )
            working_df = working_df[working_df[IMAGE_INDEX].isin(indices)]

        # Drop unlabeled rows before N/A normalization so missing labels are not
        # turned into a synthetic enrichment group.
        missing_mask = working_df[resolved_cols].map(_is_missing_store_value).any(axis=1)
        n_dropped = int(missing_mask.sum())
        if n_dropped:
            logger.warning(
                "Excluding %d samples with missing group labels from enrichment.",
                n_dropped,
            )
        working_df = working_df.loc[~missing_mask].copy()

        # Create internal Group column for z-score calculation and plotting
        internal_group_col = "_internal_group_label_"
        if len(resolved_cols) == 1:
            working_df[internal_group_col] = working_df[resolved_cols[0]].map(normalize_group_value)
        else:
            working_df[internal_group_col] = build_composite_group_series(working_df, resolved_cols)

        cc = internal_group_col
        if property_keys is None:
            exclude_cols = set(TIER_A_EXCLUDE_FROM_PROPERTIES) | {
                "Index",
                "Image",
                internal_group_col,
            }
            exclude_cols.update(resolved_cols)
            exclude_cols.update(c for c in working_df.columns if c.startswith("Component "))
            property_keys = [
                c
                for c in working_df.columns
                if c not in exclude_cols and pd.api.types.is_numeric_dtype(working_df[c])
            ]
        if not property_keys:
            logger.warning("No numeric properties to analyse.")
            edf = pd.DataFrame(columns=empty_cols)
            fig = self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=pd.DataFrame(columns=["Group", *empty_cols]),
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return (edf, fig) if return_fig else edf

        working_df = working_df.dropna(subset=[cc])
        if working_df.empty:
            logger.warning("No samples with valid group labels.")
            edf = pd.DataFrame(columns=empty_cols)
            fig = self._plot_group_enrichment(  # type: ignore[attr-defined]
                enrichment_df=pd.DataFrame(columns=["Group", *empty_cols]),
                group_names=None,
                plot=plot,
                return_fig=return_fig,
                top_k=top_k,
                figsize=figsize,
                title=title,
            )
            return (edf, fig) if return_fig else edf

        edf = compute_leave_one_out_zscore_enrichment(
            working_df,
            cc,
            property_keys,
            resolved_cols,
            correct_fdr=correct_multiple_testing,
        )

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

        if not edf.empty:
            # Remove internal Group column and reorder
            edf = edf.drop(columns=["Group"])
            cols_order = resolved_cols + [c for c in edf.columns if c not in resolved_cols]
            edf = edf[cols_order].reset_index(drop=True)

        return (edf, fig) if return_fig else edf

    # ------------------------------------------------------------------
    # Prototypes
    # ------------------------------------------------------------------

    def find_prototypes(
        self,
        cluster_col: str | list[str] | None = "cluster",
        n_prototypes: int = 5,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
        metric: str = "euclidean",
        normalize: bool = True,
        plot: bool = True,
    ) -> pd.DataFrame:
        """Find images closest to each group centroid.

        Args:
            cluster_col: Property/metadata key(s) for grouping. Can be a single string
                or a list of strings. If a list is provided, groups are formed by
                combining values from all specified columns (e.g., "drug1-10uM").
                If *None*, all images are treated as one group.
            n_prototypes: How many prototypes per group.
            source: ``'embeddings'``, ``'properties'``, or ``'combined'``.
            property_keys: Property subset when *source='properties'*.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            metric: ``'euclidean'`` or ``'cosine'``. With ``metric='euclidean'``, the
                group centroid is the arithmetic mean (even when embeddings are
                L2-normalized). With ``metric='cosine'``, the centroid direction is
                L2-normalized before ranking by similarity.
            normalize: Whether to normalize data before finding prototypes (default: True).
                For embeddings, uses L2 normalization. For properties, uses StandardScaler.
            plot: If True (default), shows one matplotlib figure per group: subplots for
                that group's prototypes and a figure title naming the group (``Group: …``).
                Images are shown without pipeline ``image_transforms`` (same as
                ``apply_transforms=False`` for display).

        Returns:
            pd.DataFrame: DataFrame with one row per prototype. Columns include grouping
                keys (split into individual columns if multiple), ``image_index``,
                ``image_path``, ``image_name``, and ``rank_in_group``.
        """
        record_step(
            self,
            "find_prototypes",
            {
                "cluster_col": cluster_col,
                "n_prototypes": n_prototypes,
                "source": source,
                "property_keys": property_keys,
                "filters": filters,
                "exclude": exclude,
                "metric": metric,
                "normalize": normalize,
                "plot": plot,
            },
        )
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
            raise ValueError(f"find_prototypes failed: {exc}") from exc

        # Build group labels
        n_total = len(self.results.img_path)
        found_cols: list[str] = []
        group_labels: list[Any] = [None] * n_total

        if cluster_col:
            requested_cols = [cluster_col] if isinstance(cluster_col, str) else cluster_col
            group_labels, found_cols = build_index_group_labels(
                self.results, requested_cols, n_total, prefer="properties"
            )
            if not found_cols:
                logger.warning("No matching columns found. Treating as one group.")

        valid_gl = [group_labels[i] for i in valid_indices]
        unique_groups: list
        if found_cols:
            has_none = None in valid_gl
            others = sorted({v for v in valid_gl if v is not None})
            unique_groups = ([None] if has_none else []) + others
        else:
            unique_groups = [None]

        device = self.device

        matrix = _helpers.numpy_for_torch(matrix)

        # Convert entire matrix to GPU once (avoid repeated conversions per group)
        matrix_t = torch.from_numpy(matrix).to(device) if device.type != "cpu" else None

        results_list = []
        for grp in track_steps(
            unique_groups,
            desc="Finding prototypes",
            when_single="Finding prototypes...",
        ):
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
                    sims_t = torch.matmul(gm_norm, centroid_norm.T).reshape(-1)
                    sims = sims_t.cpu().numpy() if device.type != "cpu" else sims_t.numpy()
                    top = np.argsort(sims)[::-1][:n_prototypes]
                else:
                    raise ValueError(f"Unknown metric '{metric}'.")

            group_name = str(grp) if grp is not None else "All"

            for rank, i in enumerate(top, start=1):
                global_idx = gi[i]
                row = {
                    IMAGE_INDEX: global_idx,
                    IMAGE_PATH: self.results.img_path[global_idx],
                    "image_name": path_basename(self.results.img_path[global_idx]),
                    "rank_in_group": rank,
                }

                # Add grouping columns
                if not found_cols:
                    row["group"] = group_name
                elif len(found_cols) == 1:
                    row[found_cols[0]] = grp
                else:
                    # Multi-column: split composite label (built with " | " separator)
                    vals = group_name.split(" | ")
                    for col_name, val in zip(found_cols, vals, strict=False):
                        row[col_name] = val

                results_list.append(row)

        df = pd.DataFrame(results_list)

        if plot and not df.empty:
            import matplotlib.pyplot as plt

            # Determine grouping columns for plotting
            plot_group_cols = found_cols if found_cols else ["group"]

            for group_vals, group_df in df.groupby(plot_group_cols, sort=False):
                indices = group_df[IMAGE_INDEX].tolist()
                if isinstance(group_vals, tuple):
                    group_name = " | ".join(map(str, group_vals))
                else:
                    group_name = str(group_vals)

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
                fig.subplots_adjust(top=0.85)
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

        return df

    # ------------------------------------------------------------------
    # Embedding-property correlation
    # ------------------------------------------------------------------

    def compute_embedding_property_correlations(
        self,
        property_keys: list[str] | None = None,
        normalize: bool = True,
        method: Literal["pearson", "spearman", "distance_correlation", "mutual_info"] = "pearson",
        n_jobs: int = 1,
    ) -> dict[str, np.ndarray]:
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
                - ``'mutual_info'``: MI-derived correlation coefficient (detects any dependency,
                  mapped to [0, 1] via the Gaussian bivariate transform, **not** standard NMI)

        Returns:
            Dict mapping property names to correlation arrays (n_dims,)
        """
        record_step(
            self,
            "compute_embedding_property_correlations",
            {
                "property_keys": property_keys,
                "normalize": normalize,
                "method": method,
                "n_jobs": n_jobs,
            },
        )
        embeddings = self.get_embeddings()  # type: ignore[attr-defined]
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings not available. Run process_images() first.")
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D embeddings, got shape {embeddings.shape}")

        n_samples, n_dims = embeddings.shape
        log_computing(
            f"Computing {format_correlation_method(method)} correlations "
            f"across {n_dims} embedding dimensions..."
        )

        # Apply L2 normalization if requested
        if normalize:
            embeddings = self._normalize_embeddings_l2(embeddings)

        if property_keys is None:
            property_keys = self.get_available_property_keys()  # type: ignore[attr-defined]
        if not property_keys:
            raise ValueError("No properties found. Run compute_properties() first.")

        props_list = self.results.properties
        valid_props = filter_valid_property_arrays(
            props_list, property_keys, n_samples, min_finite=3
        )
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
                handle_nans="keep",
            )
            # Update valid_props with normalized values
            for idx, key in enumerate(valid_prop_keys):
                valid_props[key] = norm_matrix[:, idx]

        nan_counts = {
            key: int(np.isnan(arr).sum())
            for key, arr in valid_props.items()
            if bool(np.isnan(arr).any())
        }
        if nan_counts:
            logger.info(
                "Properties with missing values (pairwise complete used for correlations): %s",
                nan_counts,
            )

        corrs: dict[str, np.ndarray] = {}

        method_label = format_correlation_method(method)
        desc = f"Computing {method_label} correlations"
        # Dimension bars only in the serial path. Parallel workers would each open one.
        inner_progress = n_jobs == 1 and method in ("spearman", "distance_correlation")

        def _worker(pname: str, parr: np.ndarray) -> tuple[str, np.ndarray | None]:
            ok = np.isfinite(parr)
            if ok.sum() < 3:
                return pname, None

            r = self._compute_correlation(
                method,
                embeddings[ok],
                parr[ok],
                show_progress=inner_progress and int(embeddings.shape[1]) > 1,
            )

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
            for pname, parr in track(valid_props.items(), desc=desc):
                pname_res, r = _worker(pname, parr)
                if r is not None:
                    corrs[pname_res] = r
        else:
            items = list(valid_props.items())
            results = parallel_with_progress(
                (delayed(_worker)(pname, parr) for pname, parr in items),
                n_jobs=n_jobs,
                desc=desc,
                total=len(items),
            )
            for pname, r in results:
                if r is not None:
                    corrs[pname] = r

        logger.info(
            "Computed embedding-property correlations: %d properties, method=%s",
            len(valid_props),
            format_correlation_method(method),
        )

        return corrs

    def summarize_embedding_property_correlations(
        self,
        correlations: dict[str, np.ndarray],
        order_by: str = "mean_abs",
        top_k: int | None = 20,
        plot: bool = True,
        return_fig: bool = False,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Property Correlations with Embeddings",
        embedding_shape: tuple[int, int] | None = None,
        n_properties: int | None = None,
        correlation_method: str = "pearson",
    ) -> pd.DataFrame | tuple[pd.DataFrame, Any | None]:
        """Summarize embedding-property correlations, optionally plot, and/or return a Plotly figure.

        Computes per-property ``mean_abs``, ``std`` (across dimensions), ``max_abs``, ``min_abs``,
        and ``mean``, orders rows by ``order_by``, and either shows a
        horizontal violin plot of the distribution of |r| across dimensions per property or logs
        a plain-text table when ``plot`` is *False*.

        Args:
            correlations: Output from :meth:`compute_embedding_property_correlations`
                (dict mapping property names to correlation arrays).
            order_by: Metric used to sort properties (descending):
                ``mean_abs`` | ``max_abs`` | ``mean`` | ``std``.
            top_k: Number of top properties shown in the plot and listed in ``top_properties``;
                full sorted table is always in ``summary`` (*None* = all).
            plot: If *True* (default), show a Plotly violin plot of |r| per dimension for the top
                ``top_k`` properties by ``order_by`` (*None* = all). Highest metric at the **top**
                of the y-axis.
            return_fig: If *True*, return both the summary DataFrame and the
                Plotly figure.
            figsize: Figure size in inches, converted to pixels for Plotly layout.
            title: Chart title.
            embedding_shape: Shape of embeddings (n_samples, n_dims) for metadata display.
            n_properties: Number of properties processed for metadata display.
            correlation_method: Method used for correlation computation for metadata display.

        Returns:
            Sorted DataFrame with ``property``, ``mean_abs``, ``std``, ``max_abs``, ``min_abs``,
            and ``mean``. If ``return_fig`` is *True*, returns ``(summary, figure)``.
        """
        record_step(
            self,
            "summarize_embedding_property_correlations",
            {
                "order_by": order_by,
                "top_k": top_k,
                "plot": plot,
                "return_fig": return_fig,
                "title": title,
                "correlation_method": correlation_method,
            },
        )
        if order_by not in _helpers.ORDER_METRICS:
            raise ValueError(
                f"Unknown order_by: {order_by!r}; expected one of {_helpers.ORDER_METRICS}"
            )
        if not correlations:
            raise ValueError("correlations dict cannot be empty.")

        if embedding_shape is None:
            embedding_shape = getattr(self, "cls_shape", (0, 0))
        if n_properties is None:
            n_properties = len(correlations)

        per_prop: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {m: {} for m in _helpers.ORDER_METRICS}
        metrics["min_abs"] = {}

        for pname, r in correlations.items():
            vc = r[np.isfinite(r)]
            if len(vc) == 0:
                row = {
                    "property": pname,
                    "mean": np.nan,
                    "std": np.nan,
                    "mean_abs": np.nan,
                    "max_abs": np.nan,
                    "min_abs": np.nan,
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
                }
            per_prop.append(row)
            for name in _helpers.ORDER_METRICS:
                metrics[name][pname] = float(row[name])
            metrics["min_abs"][pname] = float(row["min_abs"])

        summary = pd.DataFrame(per_prop)
        summary = summary.sort_values(order_by, ascending=False, na_position="last")
        summary = summary.reset_index(drop=True)

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
                correlations=correlations,
                order_by=order_by,
                title=title,
                figsize=figsize,
                correlation_method=correlation_method,
            )
            if plot and out_fig is not None:
                out_fig.show()

        if not plot:
            _log_embedding_property_correlation_summary(summary, order_by, top_k)

        if return_fig:
            return summary, out_fig
        return summary

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

    def _compute_spearman_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute Spearman rank correlation between x and y (delegates to core)."""
        return compute_spearman_correlation(x, y, device=self.device, show_progress=show_progress)

    def _compute_distance_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute distance correlation between x and y (delegates to core)."""
        return compute_distance_correlation(x, y, device=self.device, show_progress=show_progress)

    def _compute_mutual_info(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Compute normalized mutual information (delegates to core)."""
        return compute_mutual_info(
            x,
            y,
            seed=getattr(self, "seed", None),
            device=self.device,
            show_progress=show_progress,
        )

    def _compute_correlation(
        self,
        method: str,
        x: np.ndarray,
        y: np.ndarray,
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Dispatch to the correlation function for the given method."""
        fn_name = _CORRELATION_METHOD_REGISTRY.get(method)
        if fn_name is None:
            raise ValueError(f"Unknown correlation method: {method}")
        fn = getattr(self, fn_name)
        if method == "pearson":
            return fn(x, y)
        return fn(x, y, show_progress=show_progress)

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
