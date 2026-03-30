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
    compute_mutual_info,
    compute_pearson_correlation,
    compute_spearman_correlation,
    filter_indices,
    get_metadata_value_from_dict,
    run_dimensionality_reduction,
    run_dimensionality_reduction_matrix,
)
from ..core.pipeline_results import PhenoMeResults
from ..core.protocols import PhenoMeProtocol

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
    """Pipeline providing analysis methods for PhenoMe.

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
        """Perform clustering and store labels as property ``'cluster'``.

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
        if reduce_dim is not None and reduce_dim > 0:
            n_comp = min(reduce_dim, matrix.shape[0] - 1, matrix.shape[1])
            if n_comp > 0:
                orig_dim = matrix.shape[1]
                n_samples = matrix.shape[0]
                seed = random_state if random_state is not None else getattr(self, "seed", None)
                dev = getattr(self, "device", None)
                dr_kwargs: dict[str, Any] = {}
                if reduce_method == "tsne":
                    # Perplexity must be < n_samples (sklearn constraint)
                    dr_kwargs["perplexity"] = min(30.0, n_samples - 1)
                elif reduce_method == "umap":
                    # n_neighbors must be <= n_samples
                    dr_kwargs["n_neighbors"] = min(15, n_samples - 1) if n_samples > 1 else 1

                matrix, _, _ = run_dimensionality_reduction_matrix(
                    matrix,
                    method=reduce_method,
                    n_components=n_comp,
                    seed=seed,
                    device=dev,
                    use_gpu=True,
                    **dr_kwargs,
                )
                logger.info(
                    "Clustering: %s extracted %d components from %d dimensions before %s.",
                    reduce_method.upper(),
                    n_comp,
                    orig_dim,
                    clustering_method.upper(),
                )

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
        full_labels = self._map_to_full(labels, valid_indices, n_total)

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
            vec_t = vectors_t if vectors_t is not None else torch.from_numpy(vectors).to(device)

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

        n_total = len(self.results.img_path)
        full_dists = self._map_to_full(distances, valid_indices, n_total)

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
    ) -> dict[str, Any]:
        """Correlate dim-reduction components with phenotypic properties.

        Args:
            method: Dimensionality reduction method.
            n_components: Number of components to compute.
            source: ``'embeddings'``, ``'properties'``, or ``'combined'``.
            property_keys: Property subset when *source* is ``'properties'`` or ``'combined'``.
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).
            top_k: Number of top correlations to return in summary.
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

        Returns:
            Dict with keys:
            - correlation_df: pd.DataFrame. Rows=properties, cols=components. Correlation values.
            - summary: pd.DataFrame. Columns: Component, Property, Correlation, AbsCorrelation.
            - component_names: List[str]. Column names (Component 1, Component 2, ...) for all methods.
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

        return {
            "correlation_df": corr,
            "summary": pd.DataFrame(rows),
            "component_names": comp_cols,
        }

    # ------------------------------------------------------------------
    # Cluster enrichment
    # ------------------------------------------------------------------

    def analyze_cluster_enrichment(
        self,
        cluster_col: str = "cluster",
        property_keys: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Compute z-score enrichment of properties per cluster.

        Uses sample mean and sample standard deviation (ddof=1) for the
        population statistics when computing z-scores.

        Args:
            cluster_col: Property/metadata key holding cluster labels.
            property_keys: Properties to analyse (all numeric if *None*).
            filters: Optional metadata filters.
            exclude: Optional metadata exclusions (same structure as filters).

        Returns:
            pd.DataFrame: Columns: Cluster, Property, Score, Mean_Cluster, Mean_Pop, AbsScore.
                Z-score enrichment per cluster-property pair.
        """
        working_df = self._build_properties_dataframe()  # type: ignore[attr-defined]
        if working_df.empty:
            return pd.DataFrame()

        if filters or exclude:
            indices = filter_indices(self.results, filters, exclude)
            working_df = working_df.iloc[indices]

        # Resolve cluster column (case-insensitive)
        cc = cluster_col
        if cc not in working_df.columns:
            for variant in (cc.capitalize(), cc.lower()):
                if variant in working_df.columns:
                    cc = variant
                    break
            else:
                raise ValueError(
                    f"Cluster column '{cluster_col}' not found. Columns: {list(working_df.columns)}"
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
            return pd.DataFrame()

        working_df = working_df.dropna(subset=[cc])
        if working_df.empty:
            logger.warning("No samples with valid cluster labels.")
            return pd.DataFrame()

        pop_mean = working_df[property_keys].mean()
        pop_std = working_df[property_keys].std()
        # Replace near-zero variance to avoid huge z-scores (exact 0 and fp noise)
        pop_std = pop_std.where(pop_std >= 1e-10, np.nan)

        all_clusters = sorted(working_df[cc].unique())
        skipped_small = sum(1 for c in all_clusters if len(working_df[working_df[cc] == c]) < 3)
        if skipped_small > 0:
            logger.info(
                "Skipped %d cluster(s) with < 3 samples; z-score enrichment requires at least 3.",
                skipped_small,
            )

        rows = []
        for cluster in all_clusters:
            cdf = working_df[working_df[cc] == cluster]
            if len(cdf) < 3:
                continue
            cmean = cdf[property_keys].mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                zs = (cmean - pop_mean) / pop_std
            for prop in property_keys:
                s = zs[prop]
                if pd.notna(s):
                    rows.append(
                        {
                            "Cluster": cluster,
                            "Property": prop,
                            "Score": float(s),
                            "Mean_Cluster": float(cmean[prop]),
                            "Mean_Pop": float(pop_mean[prop]),
                        }
                    )

        edf = pd.DataFrame(rows)
        if not edf.empty:
            edf["AbsScore"] = edf["Score"].abs()
            edf = edf.sort_values(["Cluster", "AbsScore"], ascending=[True, False])
        return edf

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
                gm_t = torch.from_numpy(gm).to(device)

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

    def aggregate_embedding_property_correlations(
        self,
        correlation_results: dict[str, Any],
        aggregation: str = "mean_abs",
        top_k: int | None = 20,
    ) -> dict[str, Any]:
        """Aggregate pre-computed embedding-property correlations using the specified method.

        This is a fast operation that takes the output of compute_embedding_property_correlations
        and applies aggregation to produce summary statistics.

        Args:
            correlation_results: Output dict from compute_embedding_property_correlations containing:
                - ``correlations``: Dict mapping property names to correlation arrays
                - ``embedding_shape``: Shape of embeddings
                - ``n_properties``: Number of properties
                - ``correlation_method``: Method used
            aggregation: Aggregation method to apply. Options:
                - ``'mean_abs'``: Mean of absolute correlations (default)
                - ``'max_abs'``: Maximum absolute correlation
                - ``'mean'``: Mean correlation
                - ``'std'``: Standard deviation of correlations
            top_k: Return only top-k properties in summary (*None* = all).

        Returns:
            Dict with:
                - ``correlations``: Original correlations dict
                - ``aggregated``: Dict mapping property names to aggregated values
                - ``summary``: DataFrame with properties sorted by aggregated correlation
                - ``top_properties``: List of top-k property names
                - ``embedding_shape``: Shape of embeddings
                - ``n_properties``: Number of properties
                - ``aggregation_method``: Aggregation method used
                - ``correlation_method``: Correlation method used
        """
        corrs = correlation_results["correlations"]
        embedding_shape = correlation_results.get(
            "embedding_shape", correlation_results.get("cls_shape")
        )
        n_properties = correlation_results.get(
            "n_properties", correlation_results.get("n_features")
        )
        correlation_method = correlation_results["correlation_method"]

        agg_map = {
            "mean_abs": lambda c: np.mean(np.abs(c)),
            "max_abs": lambda c: np.max(np.abs(c)),
            "mean": np.mean,
            "std": np.std,
        }
        if aggregation not in agg_map:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        agg_fn = agg_map[aggregation]

        aggs: dict[str, float] = {}
        for pname, r in corrs.items():
            vc = r[np.isfinite(r)]
            aggs[pname] = float(agg_fn(vc)) if len(vc) else np.nan

        summary = pd.DataFrame(
            {"property": list(aggs.keys()), "aggregated_correlation": list(aggs.values())}
        )
        summary = summary.sort_values(
            "aggregated_correlation", ascending=False, na_position="last"
        ).reset_index(drop=True)
        top_properties = (
            summary.head(top_k)["property"].tolist() if top_k else summary["property"].tolist()
        )

        logger.info(
            "Aggregated embedding-property correlations: aggregation=%s, top_k=%s",
            aggregation,
            top_k,
        )

        return {
            "correlations": corrs,
            "aggregated": aggs,
            "summary": summary,
            "top_properties": top_properties,
            "embedding_shape": embedding_shape,
            "n_properties": n_properties,
            "aggregation_method": aggregation,
            "correlation_method": correlation_method,
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
        data_t = torch.from_numpy(data).to(device)

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
        return compute_spearman_correlation(x, y)

    def _compute_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute distance correlation between x and y (delegates to core)."""
        return compute_distance_correlation(x, y)

    def _compute_mutual_info(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute normalized mutual information (delegates to core)."""
        return compute_mutual_info(x, y, seed=getattr(self, "seed", None))

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

    # ------------------------------------------------------------------
    # Map-back helper
    # ------------------------------------------------------------------

    @staticmethod
    def _map_to_full(values: np.ndarray, valid_indices: list[int], n_total: int) -> np.ndarray:
        """Map values for valid_indices into a full-length NaN array.

        Args:
            values: np.ndarray shape (len(valid_indices),).
            valid_indices: List[int]. Global indices where values apply.
            n_total: Total number of images.

        Returns:
            np.ndarray shape (n_total,), dtype float32. NaN where not in valid_indices.
        """
        full = np.full(n_total, np.nan, dtype=np.float32)
        full[np.asarray(valid_indices, dtype=np.int64)] = values.astype(np.float32)
        return full
