"""
Distance computation for the phenotyping pipeline.
Provides efficient GPU-accelerated distance calculations.
"""

from typing import Any, Literal, cast

import numpy as np
import torch

from .._logging import get_logger
from ..core import build_combined_features
from ..core import filter_indices as common_filter_indices
from ..core.pipeline_results import PhenoMeResults
from ..core.protocols import PhenoMeProtocol

logger = get_logger(__name__)

_DISTANCE_BATCH_SIZE = 512  # Process in batches to manage GPU memory


class PhenoMeDistances:
    """
    @section Analysis
    @order 6

    Provides distance computation methods for PhenoMe.

    Expects the following attributes from the pipeline:
        - self.results: PhenoMeResults containing processed results
        - self.device: torch.device for GPU computation
    """

    results: PhenoMeResults
    device: torch.device
    # _get_property_matrix is provided by PhenoMeProperties (pipeline MRO)

    def compute_reference_distances(
        self,
        reference_filters: dict[str, Any],
        filters: dict[str, Any | list[Any]] | None = None,
        exclude: dict[str, Any | list[Any]] | None = None,
        source: Literal["embeddings", "properties", "combined"] = "embeddings",
        mode: Literal["centroid", "all_to_all"] = "centroid",
        distance_type: Literal["euclidean", "cosine"] = "euclidean",
        property_keys: list[str] | None = None,
        group_by: str | list[str] | None = None,
        dist_range: tuple[float, float] = (0.0, 100.0),
        figsize: tuple[int, int] = (10, 6),
        plot: bool = True,
        return_fig: bool = False,
        points: Literal["all", "outliers", False] | None = None,
    ) -> dict:
        """Compute distances from all images to reference group.

        Args:
            reference_filters: Dict mapping metadata keys to values for reference group.
                Example: {'drug': 'Control', 'time': '60_min'}
                If a value is a list, matches any value in the list.
            filters: Optional generic filters applied before computing distances.
                Reference selection still uses reference_filters.
            exclude: Optional metadata exclusions (same structure as filters).
            source: 'embeddings' for model embeddings, 'properties' for scalar properties,
                or 'combined' for the normalized concatenation of both.
            mode: 'centroid' computes distance to mean embedding of reference,
                'all_to_all' computes min distance to any reference image.
            distance_type: 'euclidean' (default) or 'cosine'.
            property_keys: Optional subset of property names when *source* is ``'properties'``
                or ``'combined'``.
            group_by: If set, optionally visualize or print grouped statistics after
                computing distances. Pass ``None`` to skip post-processing.
            dist_range: When using *group_by*, keep distances in ``[min, max]`` for
                display and summary.
            figsize: Figure size in inches (width, height) when a plot is built.
            plot: If True and *group_by* is set, show or return a violin plot only (no
                per-group text summary to the logger). If False, log per-group summary
                statistics where applicable (e.g. text-only mode, or with *return_fig*).
            return_fig: If True and *group_by* is set, add key ``"figure"`` to the
                return dict and do not call ``fig.show()``.
            points: Violin plot point overlay: ``'all'``, ``'outliers'``, or ``False``;
                ``None`` auto-selects by data size.

        Returns:
            Dict containing:
                - distances: np.ndarray shape (N,), dtype float32. Distance per image; NaN for invalid/filtered.
                - reference_indices: list of reference image indices
                - reference_filters: dict of filters used
                - mode: str ('centroid' or 'all_to_all')
                - source: str ('embeddings', 'properties', or 'combined')
                - distance_type: str ('euclidean' or 'cosine')
                - figure: (optional) Plotly figure if ``return_fig=True`` and *group_by* is set

        Example:
            >>> dist_results = pipeline.compute_reference_distances(
            ...     reference_filters={'condition': 'Control'},
            ...     source='embeddings',
            ...     mode='centroid',
            ...     distance_type='euclidean',
            ...     group_by='condition',
            ... )
            >>> distances = dist_results['distances']
        """
        if not reference_filters:
            raise ValueError(
                "reference_filters must be provided. Example: {'plate': 'P1', 'condition': 'Control'}"
            )

        # Apply optional global filters
        filtered_indices = common_filter_indices(self.results, filters, exclude)
        if not filtered_indices:
            raise ValueError("No images left after applying filters.")

        ref_indices_all = common_filter_indices(self.results, reference_filters)
        # Reference must be subset of filtered set
        ref_indices = [i for i in ref_indices_all if i in filtered_indices]
        if not ref_indices:
            filter_str = ", ".join([f"{k}={v}" for k, v in reference_filters.items()])
            raise ValueError(f"No reference images found matching filters: {filter_str}")

        filter_str = ", ".join([f"{k}={v}" for k, v in reference_filters.items()])
        logger.info("Reference group: %s", filter_str)
        logger.info("  Found %d reference images", len(ref_indices))
        logger.info("  Mode: %s, Source: %s, Distance: %s", mode, source, distance_type)

        if source == "properties":
            data_matrix, valid_indices, _ = self._get_property_matrix(  # type: ignore[attr-defined]
                indices=filtered_indices,
                property_keys=property_keys,
                normalize=True,
                handle_nans="filter",
            )

            # Map reference indices to positions in valid_indices array
            ref_indices_mapped = [
                i for i, orig_idx in enumerate(valid_indices) if orig_idx in ref_indices
            ]

            if not ref_indices_mapped:
                filter_str = ", ".join([f"{k}={v}" for k, v in reference_filters.items()])
                raise ValueError(
                    f"No reference images found in valid (non-NaN) samples matching filters: {filter_str}. "
                    f"Consider checking for NaNs using _detect_nan_properties()."
                )
            embeddings = data_matrix
        elif source == "combined":
            embeddings, filtered_indices = build_combined_features(
                cast(PhenoMeProtocol, self),
                filtered_indices,
                property_keys=property_keys,
                normalize=True,  # Distances always benefit from normalized features
            )
            ref_indices_mapped = [
                i for i, orig_idx in enumerate(filtered_indices) if orig_idx in ref_indices
            ]
        else:
            # Load ONLY the filtered rows lazily
            embeddings = self.get_embeddings(filtered_indices)  # type: ignore[attr-defined]
            if embeddings is None or len(embeddings) == 0:
                raise ValueError(
                    "No embeddings available for distance computation. Run process_images() first."
                )
            ref_indices_mapped = [
                i for i, orig_idx in enumerate(filtered_indices) if orig_idx in ref_indices
            ]

        # centroid: distance to mean of reference group; all_to_all: min distance to any reference
        if mode == "centroid":
            distances_valid = self._compute_centroid_distances(
                embeddings, ref_indices_mapped, distance_type
            )
        else:  # all_to_all
            distances_valid = self._compute_all_to_all_distances(
                embeddings, ref_indices_mapped, distance_type
            )

        # Map distances back to original indices (NaN samples get NaN distance)
        n_total = len(self.results.img_path)
        distances = np.full(n_total, np.nan, dtype=np.float32)

        if source == "properties":
            valid_indices_arr = np.asarray(valid_indices, dtype=np.int64)
        else:
            valid_indices_arr = np.asarray(filtered_indices, dtype=np.int64)

        distances_valid = np.asarray(distances_valid).flatten()
        if len(distances_valid) != len(valid_indices_arr):
            raise ValueError(
                f"Length mismatch: distances_valid has {len(distances_valid)} elements, "
                f"but valid_indices has {len(valid_indices_arr)} elements. "
                f"This suggests an issue in distance computation."
            )

        distances[valid_indices_arr] = distances_valid
        n_valid_distances = np.sum(~np.isnan(distances))
        logger.info(
            "  Mapped %d valid distances out of %d total images", n_valid_distances, n_total
        )
        if n_valid_distances > 0:
            logger.info(
                "  Distance range: [%.4f, %.4f]", np.nanmin(distances), np.nanmax(distances)
            )

        result: dict[str, Any] = {
            "distances": distances,
            "reference_indices": ref_indices,
            "reference_filters": reference_filters,
            "filters": filters or {},
            "mode": mode,
            "source": source,
            "distance_type": distance_type,
        }

        if group_by is not None:
            # Visualization lives on ``PhenoMeVisualization`` (PhenoMe MRO)
            fig = self._plot_distance_distribution(  # type: ignore[attr-defined]
                result,
                group_by=group_by,
                dist_range=dist_range,
                figsize=figsize,
                plot=plot,
                return_fig=return_fig,
                points=points,
            )
            if return_fig and fig is not None:
                result["figure"] = fig

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_embedding_or_property_data(self, source: str) -> tuple[np.ndarray, bool]:
        """Get data for distance computation based on feature space.

        Args:
            source: 'embeddings' for model embeddings or 'properties' for scalar properties.

        Returns:
            Tuple of (data_array, is_patch_metric) where is_patch_metric is always False.

        Raises:
            ValueError: If embeddings are not available and source='embeddings'.
        """
        embeddings = self.get_embeddings()  # type: ignore[attr-defined]
        has_embeddings = embeddings is not None and len(embeddings) > 0

        if source == "embeddings" and has_embeddings:
            return embeddings, False
        elif source == "properties":
            props_matrix, _, _ = self._get_property_matrix(handle_nans="filter")  # type: ignore[attr-defined]
            return props_matrix, False
        elif has_embeddings:
            logger.warning("source='%s' not available, using 'embeddings'.", source)
            return embeddings, False
        else:
            raise ValueError(
                "No embeddings available for distance computation. Run process_images() first."
            )

    def _compute_centroid_distances(
        self,
        embeddings: np.ndarray,
        ref_indices: list,
        distance_type: Literal["euclidean", "cosine"] = "euclidean",
    ) -> np.ndarray:
        """Compute distances from all embeddings to centroid of reference group.

        Uses GPU-accelerated vectorized computation.

        Args:
            embeddings: Embedding matrix (N, n_features).
            ref_indices: List of reference image indices.
            distance_type: 'euclidean' or 'cosine'.

        Returns:
            Array of distances (N,).
        """
        n_samples = embeddings.shape[0]
        batch_size = _DISTANCE_BATCH_SIZE
        all_emb = torch.from_numpy(embeddings).to(self.device)
        ref_emb = all_emb[ref_indices]
        centroid = torch.mean(ref_emb, dim=0)

        distances_out = np.zeros(n_samples, dtype=np.float32)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_emb = all_emb[start:end]
            if distance_type == "cosine":
                batch_norm = torch.nn.functional.normalize(batch_emb, dim=-1)
                cent_norm = torch.nn.functional.normalize(centroid, dim=-1)
                sim = torch.matmul(batch_norm, cent_norm)
                distances_out[start:end] = (1.0 - sim).cpu().numpy()
            else:
                distances_out[start:end] = (
                    torch.norm(batch_emb - centroid.unsqueeze(0), dim=1).cpu().numpy()
                )

        return distances_out

    def _compute_all_to_all_distances(
        self,
        embeddings: np.ndarray,
        ref_indices: list,
        distance_type: Literal["euclidean", "cosine"] = "euclidean",
    ) -> np.ndarray:
        """Compute minimum distance from each image to any reference image.

        Uses batched GPU operations for memory efficiency.

        Args:
            embeddings: Embedding matrix (N, n_features).
            ref_indices: List of reference image indices.
            distance_type: 'euclidean' or 'cosine'.

        Returns:
            Array of minimum distances (N,).
        """
        n_samples = embeddings.shape[0]
        batch_size = _DISTANCE_BATCH_SIZE

        all_emb = torch.from_numpy(embeddings).to(self.device)
        ref_emb = all_emb[ref_indices]

        # Pre-normalize reference embeddings for cosine (done once)
        if distance_type == "cosine":
            ref_norm = torch.nn.functional.normalize(ref_emb, dim=-1)

        min_distances = np.zeros(n_samples, dtype=np.float32)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_emb = all_emb[start:end]

            if distance_type == "cosine":
                batch_norm = torch.nn.functional.normalize(batch_emb, dim=-1)
                sim = torch.matmul(batch_norm, ref_norm.T)  # (batch_size, n_refs)
                dists = 1.0 - sim
            else:
                dists = torch.cdist(batch_emb, ref_emb, p=2)

            min_dists = dists.min(dim=1).values
            min_distances[start:end] = min_dists.cpu().numpy()

        return min_distances
