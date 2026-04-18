"""
Property computation for the phenotyping pipeline.

Provides methods to compute per-image scalar properties from images and masks,
aggregate statistics by group, and format property reports.
"""

import os
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..._logging import get_logger
from ...core import metadata_to_stable_key, optimize_property_types
from ...core.pipeline_results import PhenoMeResults
from ...io.checkpoint_alignment import (
    compute_metadata_alignment as _compute_metadata_alignment_fn,
)
from ...io.checkpoint_alignment import (
    compute_path_alignment as _compute_path_alignment_fn,
)
from ...utils.path_utils import primary_path as _primary_path
from ...utils.property_factories import get_preset_property_functions
from ._paths import build_file_df_lookup, resolve_image_paths, resolve_mask_paths
from ._report import build_properties_dataframe as _build_properties_dataframe_fn
from ._workers import (
    _determine_channel_range,
    _load_image_and_mask_stacks,
    _should_call_property_function,
    collect_property_names_for_stacks,
    compute_properties_worker,
)
from .grouping import (
    filter_properties_by_group as _filter_properties_by_group_fn,
)
from .grouping import (
    print_property_stats_by_group as _print_property_stats_by_group_fn,
)
from .grouping import (
    print_top_properties_vs_reference as _print_top_properties_vs_reference_fn,
)
from .grouping import (
    top_properties_different_from_reference as _top_properties_different_from_reference_fn,
)
from .matrix import (
    detect_nan_properties as _detect_nan_properties_fn,
)
from .matrix import (
    get_property_matrix as _get_property_matrix_fn,
)
from .matrix import (
    warn_if_nan_properties as _warn_if_nan_properties_fn,
)

logger = get_logger(__name__)


class PhenoMeProperties:
    """Pipeline providing property computation and reporting for PhenoMe.

    Expected attributes from the parent class:
        - self.results: PhenoMeResults with img_path, metadata, properties, embeddings
        - self._processing_params: Optional[Dict[str, Any]]
    """

    # Type hints for pipeline attributes (provided by parent class)
    results: PhenoMeResults
    _processing_params: dict[str, Any] | None
    _property_norm_cache: dict[str, Any] | None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_properties(self, verbose: bool = False) -> None:
        """Reset only the computed properties.

        Args:
            verbose: If True, logs a confirmation message.
        """
        self.results.properties = []
        self._property_norm_cache = None
        if verbose:
            logger.info("Reset: Computed properties cleared.")

    def get_available_property_keys(self) -> list[str]:
        """Return sorted list of property keys stored in results."""
        properties_list = self.results.properties
        return sorted(properties_list[0].keys()) if properties_list else []

    def transfer_metadata_to_properties(self, columns: str | list[str]) -> None:
        """Transfer specified metadata columns to properties.

        Args:
            columns: A single metadata column name or a list of names to transfer.
        """
        if isinstance(columns, str):
            columns = [columns]

        metadata_list = self.results.metadata
        properties_list = self.results.properties

        n_images = len(self.results.img_path)

        # Ensure properties list is populated
        while len(properties_list) < n_images:
            properties_list.append({})

        for i in range(n_images):
            if i < len(metadata_list):
                meta = metadata_list[i]
                if not isinstance(properties_list[i], dict):
                    properties_list[i] = {}

                for col in columns:
                    if col in meta:
                        properties_list[i][col] = meta[col]

        self.results.properties = properties_list
        logger.info("Transferred metadata columns %s to properties.", columns)

    # ------------------------------------------------------------------
    # Grouping & statistics
    # ------------------------------------------------------------------

    def filter_properties_by_group(
        self,
        group_by: list[str] | None = None,
        properties: list[str] | None = None,
    ) -> pd.DataFrame:
        """Group property DataFrame and compute per-group statistics.

        Uses `_build_properties_dataframe` from computed
        ``results.properties`` and metadata (same source as
        [compute_properties](pipeline.md#api-phenome-compute_properties)).

        Args:
            group_by: Metadata columns to group by. If None, auto-selects first 2.
            properties: Property columns to include. If None, auto-detects numeric.

        Returns:
            Aggregated DataFrame with mean/std/min/max per property per group.

        Raises:
            TypeError: If group_by/properties are invalid types.
            ValueError: If group_by keys are not in DataFrame columns.
        """
        df = self._build_properties_dataframe()
        return _filter_properties_by_group_fn(df, group_by=group_by, properties=properties)

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def print_property_stats_by_group(
        self,
        df: pd.DataFrame,
        properties: list[str] | None = None,
        column_width: int = 20,
    ) -> None:
        """Print formatted table of grouped property statistics.

        Args:
            df: Aggregated DataFrame from filter_properties_by_group().
            properties: Property names to print. If None, prints all.
            column_width: Column width for each property column.
        """
        _print_property_stats_by_group_fn(
            df,
            properties=properties,
            column_width=column_width,
            available_property_keys=self.get_available_property_keys(),
        )

    def top_properties_different_from_reference(
        self,
        df: pd.DataFrame,
        reference_group: dict[str, Any],
        k: int = 5,
        properties: list[str] | None = None,
        metric: Literal["cohens_d", "mean_diff"] = "cohens_d",
        print_output: bool = True,
        column_width: int = 20,
    ) -> pd.DataFrame:
        """For each non-reference group, return the top k properties that most differentiate it from the reference.

        Uses Cohen's d (effect size) by default to rank properties by how different each group
        is from the reference. Both higher and lower values count as "different" (uses |d|).

        Cohen's d formula (pooled): d = (mean_group - mean_ref) / s_pooled, where
        s_pooled = sqrt([(n_ref-1)*std_ref² + (n_other-1)*std_other²] / (n_ref + n_other - 2)).
        Requires sample std (ddof=1) from filter_properties_by_group.

        Args:
            df: Aggregated DataFrame from filter_properties_by_group().
            reference_group: Dict mapping grouping column names to values (e.g. {"drug": "Control", "time": "60_min"}).
            k: Number of top properties per group.
            properties: Property names to consider. If None, uses all in DataFrame.
            metric: 'cohens_d' (effect size) or 'mean_diff' (absolute mean difference).
            print_output: If True, pretty-print the results.
            column_width: Column width for property, effect size, and mean diff columns.

        Returns:
            DataFrame with columns: grouping cols, property, effect_size, mean_diff, ref_mean,
            group_mean, rank. One row per (group, property) for top-k only.
        """
        result_df = _top_properties_different_from_reference_fn(
            df,
            reference_group=reference_group,
            k=k,
            properties=properties,
            metric=metric,
            available_property_keys=self.get_available_property_keys(),
        )
        if result_df.empty and print_output:
            logger.warning("DataFrame is empty or no valid comparisons.")
        elif not result_df.empty and print_output:
            from ._report import parse_grouped_stats_dataframe

            grouping_cols, _ = parse_grouped_stats_dataframe(df)
            _print_top_properties_vs_reference_fn(
                result_df, grouping_cols, reference_group, k, column_width
            )
        return result_df

    # ------------------------------------------------------------------
    # Core: compute_properties
    # ------------------------------------------------------------------

    def compute_properties(
        self,
        metadata_config: Any | None = None,
        property_preset: str | None = None,
        additional_property_functions: dict[str, Callable | list[Callable]] | None = None,
        checkpoint_path: str | None = None,
        save_every: int = 50,
        n_jobs: int = 1,
        lazy_checkpoint: bool = True,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """Compute per-image properties using presets and/or custom functions.

        Each function receives a 2D image slice and/or mask slice and returns
        a dict[str, float]. When multiple channels exist, properties are
        suffixed with ``_ch{idx}``.

        Use ``property_preset`` for built-in property sets and
        ``additional_property_functions`` to add custom functions on top
        (optional).

        This method now relies solely on ``file_df`` (typically produced by
        [PhenoMe.find_files](pipeline.md#api-phenome-find_files)) for resolving image and mask
        paths. Legacy ``image_dir`` / ``mask_dir`` parameters are no longer
        supported.

        Args:
            metadata_config: Optional MetadataBase instance (e.g. from
                `phenome.metadata`). When it has ``mask_dir`` and
                ``mask_filename_column``, uses explicit mask lookup.
            property_preset: Preset name. Use ``\"none\"`` or ``None`` for no
                preset. Valid: ``\"none\"``, ``\"basic\"``, ``\"regionprops\"``,
                ``\"intensity\"``, ``\"full\"``, ``\"full_extended\"``.
            additional_property_functions: Extra property functions to add on
                top of preset. Dict of {requirement: fn_or_list} where
                requirement is ``\"image\"``, ``\"mask\"``, ``\"both\"``, or
                ``\"any\"``. Merged with preset when both are used; used alone
                when ``property_preset`` is ``\"none\"`` or ``None``.
            checkpoint_path: HDF5 checkpoint for incremental persistence.
            save_every: Commit frequency (images) when checkpointing.
            n_jobs: Number of parallel jobs for property computation.
                ``1`` (default) = sequential. Use ``-1`` to use all available
                CPU cores.
            lazy_checkpoint: If True (default), keep the checkpoint file open.
                If False, load all data into RAM and close the file.
            force_update: If True and ``checkpoint_path`` points to an existing file,
                clears stored properties in that file, writes current
                ``_processing_params`` into the checkpoint config, and recomputes
                properties for every image (no resume). Use this after changing
                ``property_preset`` or custom functions, or after log warnings about
                missing checkpoint property keys. If ``checkpoint_path``
                is set but the file does not exist yet, only an info log is emitted;
                computation proceeds as for a new checkpoint.

        Returns:
            DataFrame with all properties and metadata columns. Also populates
            ``results.properties`` as ``List[dict]`` (one dict per image).

        Raises:
            ValueError: If no images processed or invalid preset/requirement
                keys are used.
            TypeError: If ``additional_property_functions`` values are not
                callable.
        """
        self.reset_properties()

        if not self.results.img_path:
            raise ValueError("No images processed yet. Run process_images() first.")

        file_df = self._require_file_df(method_name="compute_properties")  # type: ignore[attr-defined]

        property_functions = self._normalize_property_functions(
            property_preset=property_preset,
            additional_property_functions=additional_property_functions,
        )
        if not property_functions:
            return pd.DataFrame()

        image_paths, mask_paths = self._resolve_properties_paths(
            file_df=file_df,
            metadata_config=metadata_config,
        )

        expected_property_keys = self._infer_expected_property_keys(
            property_functions, image_paths, mask_paths
        )

        if force_update and not checkpoint_path:
            logger.info(
                "force_update=True has no effect without checkpoint_path; "
                "properties are computed in memory only.",
            )

        if force_update and checkpoint_path:
            if os.path.isfile(checkpoint_path):
                from ...io import CheckpointManager

                ckpt_force = CheckpointManager(checkpoint_path, lazy=lazy_checkpoint)
                ckpt_force.clear_properties()
                pparams = getattr(self, "_processing_params", None)
                if isinstance(pparams, dict) and pparams:
                    ckpt_force.set_processing_params(pparams)
                ckpt_force.close()
                logger.info(
                    "force_update=True: cleared properties in %s, refreshed checkpoint "
                    "config from current pipeline parameters, and will recompute all rows.",
                    checkpoint_path,
                )
            else:
                logger.info(
                    "force_update=True: checkpoint %s does not exist yet; "
                    "computing properties and writing a new file on save.",
                    checkpoint_path,
                )

        # Setup checkpoint resume (uses resolved image_paths for alignment)
        ckpt, n_already, path_alignment = self._setup_properties_checkpoint_resume(
            checkpoint_path,
            image_paths,
            lazy=lazy_checkpoint,
            expected_property_keys=expected_property_keys,
        )
        # When ckpt is None and n_already > 0, all properties were loaded from checkpoint
        # (ckpt was closed in _setup_properties_checkpoint_resume). Return early.
        if ckpt is None and n_already > 0:
            self._warn_if_nan_properties(
                property_keys=sorted(expected_property_keys) if expected_property_keys else None,
                count_missing_key_as_nan=not bool(expected_property_keys),
            )
            return _build_properties_dataframe_fn(self.results)

        # Process images (path_alignment is None when no checkpoint)
        all_property_names, feature_buffers, last_committed = self._process_all_images(
            image_paths,
            mask_paths,
            property_functions,
            ckpt,
            checkpoint_path,
            save_every,
            path_alignment,
            n_jobs,
        )

        # Finalize and return
        full_props = self._finalize_properties_computation(
            feature_buffers,
            all_property_names,
            path_alignment,
            last_committed,
            ckpt,
            checkpoint_path,
            image_paths,
            lazy=lazy_checkpoint,
        )

        self.results.properties = full_props
        nan_check_keys: list[str] | None = None
        count_missing_as_nan = True
        if expected_property_keys:
            nan_check_keys = sorted(set(expected_property_keys) | set(all_property_names))
            count_missing_as_nan = False
        elif all_property_names:
            nan_check_keys = sorted(all_property_names)
        self._warn_if_nan_properties(
            property_keys=nan_check_keys,
            count_missing_key_as_nan=count_missing_as_nan,
        )
        return _build_properties_dataframe_fn(
            self.results,
            list(all_property_names) if all_property_names else None,
        )

    def _build_properties_dataframe(
        self,
        property_keys: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build DataFrame from computed properties and metadata.

        Used by PhenoMeAnalysis (e.g. analyze_cluster_enrichment) and report sections.

        Args:
            property_keys: Optional list of property names to include.
                If None, uses all properties in results.

        Returns:
            DataFrame with metadata columns, img_name, img_path, and property columns.
        """
        return _build_properties_dataframe_fn(self.results, property_keys)

    # ------------------------------------------------------------------
    # Property matrix & NaN detection (used by distances, viz, analysis)
    # ------------------------------------------------------------------

    def _detect_nan_properties(
        self,
        property_keys: list[str] | None = None,
        indices: list[int] | None = None,
        count_missing_key_as_nan: bool = True,
    ) -> tuple[dict[str, int], dict[str, list[int]]]:
        """Detect which properties contain NaNs and count occurrences."""
        return _detect_nan_properties_fn(
            self.results,
            property_keys=property_keys,
            indices=indices,
            count_missing_key_as_nan=count_missing_key_as_nan,
        )

    def _warn_if_nan_properties(
        self,
        property_keys: list[str] | None = None,
        indices: list[int] | None = None,
        count_missing_key_as_nan: bool = True,
    ) -> None:
        """Log warning if any properties contain NaNs."""
        _warn_if_nan_properties_fn(
            self.results,
            property_keys=property_keys,
            indices=indices,
            count_missing_key_as_nan=count_missing_key_as_nan,
        )

    def _get_property_matrix(
        self,
        indices: list[int] | None = None,
        property_keys: list[str] | None = None,
        normalize: bool = True,
        handle_nans: Literal["filter", "impute", "warn"] = "filter",
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Build property matrix from numeric properties.

        Used by distances, visualization, and analysis modules.
        """
        matrix, valid_indices, keys, cache = _get_property_matrix_fn(
            self.results,
            cache=getattr(self, "_property_norm_cache", None),
            indices=indices,
            property_keys=property_keys,
            normalize=normalize,
            handle_nans=handle_nans,
        )
        if cache is not None:
            self._property_norm_cache = cache
        return matrix, valid_indices, keys

    # ------------------------------------------------------------------
    # compute_properties helpers
    # ------------------------------------------------------------------

    def _resolve_properties_paths(
        self,
        file_df: pd.DataFrame,
        metadata_config: Any | None,
    ) -> tuple[list[str | list[str]], list[str | None]]:
        """Resolve image and mask paths for property computation.

        Uses the provided ``file_df`` (typically from [find_files](pipeline.md#api-phenome-find_files)) as the
        single source of truth for both image and mask paths. The DataFrame is
        aligned to ``self.results.img_path`` via a lookup on the canonical path
        representation.

        Returns:
            Tuple of (image_paths, mask_paths), each aligned to
            ``self.results.img_path``.
        """
        file_df_lookup = build_file_df_lookup(file_df)
        image_paths = resolve_image_paths(
            self.results.img_path,
            file_df_lookup=file_df_lookup,
        )
        meta_cfg = metadata_config or getattr(self, "_metadata_config", None)
        metadata_list = self.results.metadata or []
        mask_paths = resolve_mask_paths(
            self.results.img_path,
            mask_dir=None,
            image_paths=image_paths,
            image_dir=None,
            metadata_list=metadata_list,
            metadata_config=meta_cfg,
            file_df_lookup=file_df_lookup,
        )
        return image_paths, mask_paths

    def _normalize_property_functions(
        self,
        property_preset: str | None = None,
        additional_property_functions: dict[str, Callable | list[Callable]] | None = None,
    ) -> dict[str, list[Callable]]:
        """Normalize property functions input to standardized dict format.

        Combines property_preset (optional) with additional_property_functions.
        """
        base: dict[str, list[Callable]] = {}
        if property_preset is not None and property_preset != "none":
            if property_preset not in (
                "basic",
                "regionprops",
                "intensity",
                "full",
                "full_extended",
            ):
                raise ValueError(
                    f"Unknown preset '{property_preset}'. Valid presets: "
                    "'none', 'basic', 'regionprops', 'intensity', 'full', 'full_extended'"
                )
            base = get_preset_property_functions(property_preset)

        extras_normalized = (
            self._normalize_property_functions_dict(additional_property_functions)
            if additional_property_functions
            else {}
        )

        if not base and not extras_normalized:
            logger.warning("No property functions provided, nothing to compute.")
            return {}

        return self._merge_property_dicts(base, extras_normalized)

    def _normalize_property_functions_dict(
        self,
        property_functions: str | dict[str, Callable | list[Callable]],
    ) -> dict[str, list[Callable]]:
        """Normalize a property_functions dict or preset string to standardized format."""
        if isinstance(property_functions, str):
            resolved: dict[str, list[Callable]] = get_preset_property_functions(property_functions)
        else:
            resolved = property_functions

        if not isinstance(resolved, dict):
            raise TypeError(
                f"property_functions must be a dict or preset string, got: {type(property_functions)}"
            )

        valid_requirements = {"image", "mask", "both", "any"}
        bad_keys = [k for k in resolved if k not in valid_requirements]
        if bad_keys:
            raise ValueError(
                f"Invalid requirement keys: {bad_keys}. Must be one of: {valid_requirements}"
            )

        normalized: dict[str, list[Callable]] = {}
        for req, val in resolved.items():
            if callable(val):
                normalized[req] = [val]
            elif isinstance(val, list):
                bad_idx = [i for i, fn in enumerate(val) if not callable(fn)]
                if bad_idx:
                    raise TypeError(
                        f"property_functions['{req}'] has non-callable at indices {bad_idx}"
                    )
                normalized[req] = val
            else:
                raise TypeError(
                    f"property_functions['{req}'] must be callable or list, got: {type(val)}"
                )

        return normalized

    @staticmethod
    def _merge_property_dicts(
        base: dict[str, list[Callable]],
        extras: dict[str, list[Callable]],
    ) -> dict[str, list[Callable]]:
        """Merge two property dicts by concatenating lists for each requirement key."""
        merged: dict[str, list[Callable]] = dict(base)
        for req, fns in extras.items():
            merged[req] = merged.get(req, []) + fns
        return merged

    def _compute_metadata_alignment(
        self,
        image_paths: list[str | list[str]],
        image_metadata: list[dict],
        ckpt_paths: list[str],
        ckpt_metadata: list[dict],
        n_committed_properties: int,
    ) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
        """Compute metadata-based alignment (portable across devices).

        Delegates to checkpoint_alignment module. Falls back to path-based when
        metadata is invalid.
        """
        return _compute_metadata_alignment_fn(
            image_paths,
            image_metadata,
            ckpt_paths,
            ckpt_metadata,
            n_committed_properties,
            _primary_path,
        )

    def _refine_path_alignment_for_unprocessed(
        self,
        path_alignment: tuple[
            dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]
        ],
        image_paths: list[str | list[str]],
        existing_props: list[dict],
    ) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
        """Move unprocessed (missing or all-NaN) entries from identifiers_with_props to paths_to_compute."""
        id_to_ckpt_idx, identifiers_with_props, paths_to_compute, identifier_list = path_alignment
        paths_to_compute_dict: dict[int, str | list[str]] = dict(paths_to_compute)
        for i, identifier in enumerate(identifier_list):
            if identifier in identifiers_with_props:
                idx = id_to_ckpt_idx.get(identifier, -1)
                if 0 <= idx < len(existing_props):
                    prop = existing_props[idx]
                    if self._is_property_dict_unprocessed(prop):
                        identifiers_with_props.discard(identifier)
                        paths_to_compute_dict[i] = image_paths[i]
        return (
            id_to_ckpt_idx,
            identifiers_with_props,
            sorted(paths_to_compute_dict.items(), key=lambda x: x[0]),
            identifier_list,
        )

    @staticmethod
    def _is_property_dict_unprocessed(prop: dict) -> bool:
        """Return True if property dict is empty or all values are NaN/None (unprocessed).

        Used to detect checkpoint entries that need recomputation.
        """
        if not prop:
            return True
        for v in prop.values():
            if v is None:
                continue
            if isinstance(v, (int, bool, np.integer)):
                return False
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    continue
                return False
            return False  # string, list, etc. - has a value
        return True

    @staticmethod
    def _compute_path_alignment(
        image_paths: list[str | list[str]],
        ckpt_paths: list[str],
        n_committed_properties: int,
    ) -> tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]]:
        """Compute path-based alignment (fallback when metadata invalid).

        Delegates to checkpoint_alignment module.
        """
        return _compute_path_alignment_fn(
            image_paths, ckpt_paths, n_committed_properties, _primary_path
        )

    def _infer_expected_property_keys(
        self,
        property_functions: dict[str, list[Callable]],
        image_paths: list[str | list[str]],
        mask_paths: list[str | None],
    ) -> set[str]:
        """Infer property column names using the first image/mask (channel layout and presets).

        Heterogeneous datasets with different channel counts may differ from later images;
        warnings are best-effort for checkpoint compatibility.
        """
        if not image_paths:
            return set()
        any_requires_image = any(r in ("image", "both", "any") for r in property_functions)
        any_requires_mask = any(r in ("mask", "both", "any") for r in property_functions)
        img0 = image_paths[0]
        mask0: str | None = None
        if mask_paths and len(mask_paths) > 0:
            mask0 = mask_paths[0]
        img_stack, mask_stack = _load_image_and_mask_stacks(
            img0, mask0, any_requires_image, any_requires_mask, property_functions
        )
        return collect_property_names_for_stacks(img_stack, mask_stack, property_functions)

    def _warn_checkpoint_missing_requested_keys(
        self,
        checkpoint_path: str,
        expected: set[str],
        props_list: list[dict],
    ) -> None:
        """Log a warning if the checkpoint does not define all requested property keys."""
        if not expected:
            return
        union_ckpt: set[str] = set()
        for p in props_list:
            if isinstance(p, dict):
                union_ckpt.update(p.keys())
        missing = expected - union_ckpt
        if not missing:
            return
        logger.warning(
            "Checkpoint %s does not store all requested properties; missing %d key(s): %s. "
            "This often happens after changing property_preset or custom functions. "
            "Call compute_properties(..., force_update=True) with the same checkpoint_path "
            "to recompute all properties and refresh the file, or use a new path / delete "
            "the file to start fresh.",
            checkpoint_path,
            len(missing),
            sorted(missing),
        )

    def _setup_properties_checkpoint_resume(
        self,
        checkpoint_path: str | None,
        image_paths: list[str | list[str]] | None = None,
        lazy: bool = True,
        expected_property_keys: set[str] | None = None,
    ) -> tuple[
        Any | None,
        int,
        tuple[dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]] | None,
    ]:
        """Setup checkpoint resume for properties computation.

        Args:
            checkpoint_path: Path to checkpoint file.
            image_paths: Resolved image paths (from _resolve_image_paths). If None,
                uses self.results.img_path
            lazy: If True, keep the checkpoint file open.
            expected_property_keys: If set, compared against checkpoint property keys;
                a warning is logged when the checkpoint is missing requested columns.

        Returns:
            Tuple of (ckpt, n_already, path_alignment):
            - When no checkpoint: (None, 0, None)
            - When all properties done: (None, n_already, None) - loads and filters
              properties to match self.results.img_path before returning
            - When partial: (ckpt, n_already, path_alignment) - path_alignment for
              path-based resume when image_paths may be a filtered subset
        """
        from ...io import CheckpointManager

        if checkpoint_path is None or not os.path.isfile(checkpoint_path):
            return None, 0, None

        try:
            ckpt = CheckpointManager(checkpoint_path, lazy=lazy)
        except Exception:
            raise

        n_already = ckpt.n_committed_props
        n_with_emb = ckpt.n_committed
        if image_paths is None:
            image_paths = list(self.results.img_path)
        image_metadata = list(self.results.metadata) or [{} for _ in range(len(image_paths))]
        ckpt_paths = ckpt.get_committed_paths_list()
        ckpt_metadata = ckpt.get_committed_metadata_list()

        try:
            if n_already >= n_with_emb > 0:
                loaded = ckpt.load_committed_results()
                all_props = loaded.properties
                if expected_property_keys:
                    self._warn_checkpoint_missing_requested_keys(
                        checkpoint_path, expected_property_keys, list(all_props)
                    )
                # Filter properties by metadata (portable) or path (fallback)
                filtered_props = []
                try:
                    id_to_idx = {}
                    for i, meta in enumerate(ckpt_metadata):
                        key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                        id_to_idx[key] = i
                    for _i, meta in enumerate(image_metadata):
                        key = metadata_to_stable_key(meta if isinstance(meta, dict) else {})
                        idx = id_to_idx.get(key, -1)
                        if 0 <= idx < len(all_props):
                            filtered_props.append(all_props[idx])
                        else:
                            filtered_props.append({})
                except ValueError:
                    path_to_idx = {p: i for i, p in enumerate(ckpt_paths)}
                    for p in image_paths:
                        idx = path_to_idx.get(_primary_path(p), -1)
                        if 0 <= idx < len(all_props):
                            filtered_props.append(all_props[idx])
                        else:
                            filtered_props.append({})

                # Check for missing images (no match in checkpoint) or unprocessed (NaN) entries
                unprocessed_indices = [
                    i
                    for i, prop in enumerate(filtered_props)
                    if self._is_property_dict_unprocessed(prop)
                ]
                if not unprocessed_indices:
                    logger.info(
                        "Checkpoint: loaded %d properties from %s", n_with_emb, checkpoint_path
                    )
                    self.results.properties = filtered_props
                    self._property_norm_cache = None
                    ckpt.close()
                    return None, n_already, None

                # Continue from checkpoint: compute missing and unprocessed
                path_alignment = self._compute_metadata_alignment(
                    image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already
                )
                existing_props = ckpt.load_committed_properties()
                path_alignment = self._refine_path_alignment_for_unprocessed(
                    path_alignment, image_paths, existing_props
                )
                n_to_compute = len(path_alignment[2])
                logger.info(
                    "Checkpoint: %d missing or unprocessed (NaN), computing %d remaining.",
                    len(unprocessed_indices),
                    n_to_compute,
                )
                return ckpt, n_already, path_alignment
            elif n_already > 0:
                path_alignment = self._compute_metadata_alignment(
                    image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already
                )
                existing_props = ckpt.load_committed_properties()
                if expected_property_keys:
                    self._warn_checkpoint_missing_requested_keys(
                        checkpoint_path, expected_property_keys, list(existing_props)
                    )
                path_alignment = self._refine_path_alignment_for_unprocessed(
                    path_alignment, image_paths, existing_props
                )
                paths_to_compute = path_alignment[2]
                logger.info(
                    "Checkpoint: loaded %d properties, computing remaining %d.",
                    n_already,
                    len(paths_to_compute),
                )
                return ckpt, n_already, path_alignment

            path_alignment = self._compute_metadata_alignment(
                image_paths, image_metadata, ckpt_paths, ckpt_metadata, n_already
            )
            return ckpt, n_already, path_alignment
        except Exception:
            raise

    def _process_all_images(
        self,
        image_paths: list[str | list[str]],
        mask_paths: list[str | None],
        property_functions: dict[str, list[Callable]],
        ckpt: Any | None,
        checkpoint_path: str | None,
        save_every: int,
        path_alignment: tuple[
            dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]
        ]
        | None,
        n_jobs: int = 1,
    ) -> tuple[set, dict[str, list[float]], int]:
        """Process all images and compute properties.

        Args:
            image_paths: List of image file paths.
            mask_paths: List of mask file paths (may contain None).
            property_functions: Normalized property functions dictionary.
            ckpt: Checkpoint manager instance or None.
            checkpoint_path: Path to checkpoint file or None.
            save_every: Frequency of checkpoint commits.
            path_alignment: Optional (path_to_ckpt_idx, paths_with_props, paths_to_compute).
                When None, processes all images. When set, processes only paths_to_compute.

        Returns:
            Tuple of (all_property_names, feature_buffers, last_committed).
        """
        any_requires_image = any(r in ("image", "both", "any") for r in property_functions)
        any_requires_mask = any(r in ("mask", "both", "any") for r in property_functions)

        # path_alignment None: process all images. With path_alignment: only paths_to_compute.
        if path_alignment is None:
            paths_to_compute = [(i, p) for i, p in enumerate(image_paths)]
        else:
            _, _, paths_to_compute, _ = path_alignment

        all_property_names: set = set()
        feature_buffers: dict[str, list[float]] = {}
        images_computed = 0
        last_committed = 0
        # Only commit incrementally when computing all images; partial resume
        # requires a full replace at the end (cannot append subset)
        use_ckpt = checkpoint_path is not None
        use_ckpt_incremental = use_ckpt and path_alignment is None
        n_to_process = len(paths_to_compute)
        paths_list = list(paths_to_compute)

        if n_jobs != 1:
            # Parallel path: process in batches for checkpoint commits and progress updates
            batch_size = (
                save_every
                if (use_ckpt_incremental and ckpt is not None)
                else max(1, min(50, n_to_process))
            )
            with tqdm(
                total=n_to_process,
                desc="Computing properties",
                unit="img",
            ) as pbar:
                for batch_start in range(0, n_to_process, batch_size):
                    batch_end = min(batch_start + batch_size, n_to_process)
                    batch_tasks = [
                        (
                            buf_idx,
                            paths_list[buf_idx][1],
                            mask_paths[paths_list[buf_idx][0]],
                        )
                        for buf_idx in range(batch_start, batch_end)
                    ]
                    results = list(
                        Parallel(n_jobs=n_jobs)(
                            delayed(compute_properties_worker)(
                                buf_idx,
                                img_path,
                                mask_path,
                                property_functions,
                                any_requires_image,
                                any_requires_mask,
                            )
                            for buf_idx, img_path, mask_path in batch_tasks
                        )
                    )
                    results.sort(key=lambda x: x[0])
                    batch_prop_names = set()
                    for _, props_dict in results:
                        batch_prop_names.update(props_dict.keys())
                    all_property_names.update(batch_prop_names)

                    for buf_idx, props_dict in results:
                        for pn in all_property_names:
                            val = props_dict.get(pn, np.nan)
                            if pn not in feature_buffers:
                                feature_buffers[pn] = [np.nan] * buf_idx
                            feature_buffers[pn].append(val)
                    images_computed = batch_end
                    if use_ckpt_incremental and ckpt is not None:
                        new_dicts = self._feature_buffers_to_dicts(
                            feature_buffers, all_property_names, last_committed, images_computed
                        )
                        if new_dicts:
                            ckpt.buffer_properties(new_dicts)
                            ckpt.commit_properties()
                        last_committed = images_computed
                    pbar.update(batch_end - batch_start)
        else:
            # Sequential path
            if n_to_process > 0:
                paths_list = tqdm(
                    paths_list,
                    desc="Computing properties",
                    total=n_to_process,
                    unit="img",
                )
            for buf_idx, (img_idx, img_path) in enumerate(paths_list):
                mask_path = mask_paths[img_idx]

                img_stack, mask_stack = _load_image_and_mask_stacks(
                    img_path,
                    mask_path,
                    any_requires_image,
                    any_requires_mask,
                    property_functions=property_functions,
                )
                n_img_ch = 0 if img_stack is None else img_stack.shape[-1]
                n_mask_ch = 0 if mask_stack is None else mask_stack.shape[-1]
                start_ch, end_ch = _determine_channel_range(property_functions, n_img_ch, n_mask_ch)
                n_channels = end_ch - start_ch

                all_computed: set = set()
                mask_props_done: set = set()
                for ch in range(start_ch, end_ch):
                    _, computed, _ = self._compute_properties_for_channel(
                        ch,
                        img_stack,
                        mask_stack,
                        n_img_ch,
                        n_mask_ch,
                        property_functions,
                        feature_buffers,
                        buf_idx,
                        n_channels,
                        start_ch=start_ch,
                        mask_properties_computed=mask_props_done,
                    )
                    all_computed.update(computed)

                for pn in list(feature_buffers):
                    if pn not in all_computed:
                        feature_buffers[pn].append(np.nan)

                all_property_names.update(feature_buffers.keys())
                images_computed += 1

                if use_ckpt_incremental and ckpt is not None and images_computed % save_every == 0:
                    new_dicts = self._feature_buffers_to_dicts(
                        feature_buffers, all_property_names, last_committed, images_computed
                    )
                    if new_dicts:
                        ckpt.buffer_properties(new_dicts)
                        ckpt.commit_properties()
                    last_committed = images_computed

        return all_property_names, feature_buffers, last_committed

    def _finalize_properties_computation(
        self,
        feature_buffers: dict[str, list[float]],
        all_property_names: set,
        path_alignment: tuple[
            dict[str, int], set[str], list[tuple[int, str | list[str]]], list[str]
        ]
        | None,
        last_committed: int,
        ckpt: Any | None,
        checkpoint_path: str | None,
        image_paths: list[str | list[str]],
        lazy: bool = True,
    ) -> list[dict[str, Any]]:
        """Finalize property computation and handle checkpoint commit.

        Builds full_props in image_paths order by merging checkpoint data (for
        paths_with_props) with newly computed data (for paths_to_compute).
        Only persists to checkpoint when image_paths matches checkpoint order
        (no filtered subset); otherwise skips persistence and logs a warning.

        Args:
            feature_buffers: Dictionary mapping property names to value lists.
            all_property_names: Set of all property names.
            path_alignment: Optional (path_to_ckpt_idx, paths_with_props, paths_to_compute).
            last_committed: Index of last committed in feature_buffers (for remaining commit).
            ckpt: Checkpoint manager instance or None.
            checkpoint_path: Path to checkpoint file or None.
            image_paths: Full list of image paths (order for full_props).
            lazy: If True, keep the checkpoint file open.

        Returns:
            List of property dictionaries for all images, in image_paths order.
        """
        from ...io import CheckpointManager

        images_computed = max((len(buf) for buf in feature_buffers.values()), default=0)
        new_props = self._feature_buffers_to_dicts(
            feature_buffers, all_property_names, 0, images_computed
        )

        if (
            path_alignment is not None
            and ckpt is not None
            and path_alignment[1]  # identifiers_with_props non-empty
        ):
            id_to_ckpt_idx, identifiers_with_props, paths_to_compute, identifier_list = (
                path_alignment
            )
            existing_props = ckpt.load_committed_properties()
            img_idx_to_new_idx = {img_idx: i for i, (img_idx, _) in enumerate(paths_to_compute)}

            # Merge: checkpoint data first (identifiers_with_props), then newly computed.
            all_keys = set(all_property_names)
            for prop_dict in existing_props:
                if isinstance(prop_dict, dict):
                    all_keys.update(prop_dict.keys())

            full_props = []
            for i in range(len(image_paths)):
                identifier = (
                    identifier_list[i]
                    if i < len(identifier_list)
                    else _primary_path(image_paths[i] if i < len(image_paths) else "")
                )
                if identifier in identifiers_with_props:
                    idx = id_to_ckpt_idx.get(identifier, -1)
                    if 0 <= idx < len(existing_props):
                        prop = dict(existing_props[idx]) if existing_props[idx] else {}
                    else:
                        prop = {}
                else:
                    new_idx = img_idx_to_new_idx.get(i, -1)
                    if 0 <= new_idx < len(new_props):
                        prop = dict(new_props[new_idx]) if new_props[new_idx] else {}
                    else:
                        prop = {}

                for key in all_keys:
                    if key not in prop:
                        prop[key] = np.nan
                full_props.append(optimize_property_types(prop))
        else:
            full_props = new_props

        # Checkpoint persistence: only when image_paths matches checkpoint (no filter)
        ckpt_created_this_run = False
        if checkpoint_path is not None:
            if ckpt is None:
                # No existing checkpoint open — create a new one and seed it with
                # the paths/embeddings/metadata currently in self.results.
                eager_emb = self.results.embeddings  # np.ndarray or None
                emb_dim = (
                    int(eager_emb.shape[1])
                    if isinstance(eager_emb, np.ndarray) and eager_emb.ndim == 2
                    else None
                )
                ckpt = CheckpointManager(
                    checkpoint_path,
                    embedding_dim=emb_dim,
                    processing_params=self._processing_params,
                    lazy=lazy,
                )

                img_paths_list = list(self.results.img_path)
                metadata_list = list(self.results.metadata) or [{} for _ in img_paths_list]

                if emb_dim is not None:
                    ckpt.buffer_embeddings(eager_emb, img_paths_list, metadata_list)
                    ckpt.commit_embeddings()
                else:
                    ckpt.buffer_paths_and_metadata(img_paths_list, metadata_list)
                    ckpt.commit_embeddings()
                ckpt_created_this_run = True

            if ckpt is not None:
                ckpt_paths = ckpt.get_committed_paths_list()

                def _norm_paths(paths: list[str]) -> list[str]:
                    return [os.path.normpath(p) for p in paths]

                primary_paths = [_primary_path(p) for p in image_paths]
                paths_match = (
                    len(image_paths) == len(ckpt_paths)
                    and (
                        _norm_paths(primary_paths) == _norm_paths(ckpt_paths)
                        or ckpt_created_this_run  # New ckpt: we wrote paths from self.results, same order
                    )
                )
                have_full_props = len(full_props) == len(image_paths)

                if paths_match and have_full_props:
                    if path_alignment is None:
                        # Full compute: append any remaining (incremental commits may have run)
                        remaining = self._feature_buffers_to_dicts(
                            feature_buffers, all_property_names, last_committed, images_computed
                        )
                        if remaining:
                            ckpt.buffer_properties(remaining)
                            n_total = ckpt.commit_properties()
                            logger.info(
                                "Checkpoint: saved %d properties to %s", n_total, checkpoint_path
                            )
                    else:
                        # Partial resume: replace entire properties (clear + write full)
                        ckpt.clear_properties()
                        ckpt.buffer_properties(full_props)
                        n_total = ckpt.commit_properties()
                        logger.info(
                            "Checkpoint: saved %d properties (replaced) to %s",
                            n_total,
                            checkpoint_path,
                        )
                else:
                    if not paths_match:
                        logger.warning(
                            "Skipping checkpoint persistence: image_paths do not match "
                            "checkpoint paths (different resolution or count)."
                        )
                    else:
                        logger.warning(
                            "Skipping checkpoint persistence: properties count (%d) "
                            "does not match image count (%d).",
                            len(full_props),
                            len(image_paths),
                        )
                ckpt.close()

        return full_props

    # ------------------------------------------------------------------
    # Path resolution helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_buffers_to_dicts(
        feature_buffers: dict[str, list[float]],
        all_property_names: set,
        start: int,
        end: int,
    ) -> list[dict[str, Any]]:
        """Convert slice of feature_buffers into list of property dictionaries.

        Args:
            feature_buffers: Dictionary mapping property names to lists of values.
            all_property_names: Set of all property names to include.
            start: Start index (inclusive) for the slice.
            end: End index (exclusive) for the slice.

        Returns:
            List of property dictionaries, one per image index in range [start, end).
        """
        result: list[dict[str, Any]] = []
        for buf_idx in range(start, end):
            props: dict[str, Any] = {}
            for pn in all_property_names:
                if pn in feature_buffers and buf_idx < len(feature_buffers[pn]):
                    v = feature_buffers[pn][buf_idx]
                    props[pn] = float(v) if not np.isnan(v) else np.nan
                else:
                    props[pn] = np.nan
            result.append(optimize_property_types(props))
        return result

    def _compute_properties_for_channel(
        self,
        ch: int,
        img_stack: np.ndarray | None,
        mask_stack: np.ndarray | None,
        n_img_ch: int,
        n_mask_ch: int,
        property_functions: dict[str, list[Callable]],
        feature_buffers: dict[str, list[float]],
        img_idx: int,
        n_channels: int,
        start_ch: int = 0,
        mask_properties_computed: set | None = None,
    ) -> tuple[set, set, set]:
        """Compute properties for one channel, populating feature_buffers.

        Args:
            ch: Channel index to process.
            img_stack: Image stack array (H, W, C) or None.
            mask_stack: Mask stack array (H, W, C) or None.
            n_img_ch: Number of image channels.
            n_mask_ch: Number of mask channels.
            property_functions: Dictionary mapping requirement types to function lists.
            feature_buffers: Dictionary to accumulate property values.
            img_idx: Current image index in buffers.
            n_channels: Total number of channels being processed.
            start_ch: Starting channel index (for mask-only properties).
            mask_properties_computed: Set tracking which mask properties were computed.

        Returns:
            Tuple of (base_names, computed, mask_base) sets.
        """
        has_img_fn = any(r in ("image", "both") for r in property_functions)
        has_mask_fn = any(r in ("mask", "both", "any") for r in property_functions)
        only_mask = (
            "mask" in property_functions and not has_img_fn and "any" not in property_functions
        )

        # Extract 2-D slices
        image2d: np.ndarray | None = None
        if (
            img_stack is not None
            and has_img_fn
            and not only_mask
            and (
                (n_channels == n_img_ch and ch < n_img_ch)
                or (n_channels == n_mask_ch and n_img_ch > 0 and ch < n_img_ch)
            )
        ):
            image2d = img_stack[..., ch]

        mask2d: np.ndarray | None = None
        if mask_stack is not None and has_mask_fn:
            if only_mask:
                mask2d = mask_stack[..., 0] if mask_stack.ndim == 3 else mask_stack
            elif n_mask_ch == 1:
                mask2d = mask_stack[..., 0]
            elif n_channels == n_mask_ch and ch < n_mask_ch:
                mask2d = mask_stack[..., ch]
            elif n_channels == n_img_ch:
                if ch < n_mask_ch:
                    mask2d = mask_stack[..., ch]
                elif n_mask_ch == 1:
                    mask2d = mask_stack[..., 0]

        base_names: set = set()
        computed: set = set()
        mask_base: set = set()
        if mask_properties_computed is None:
            mask_properties_computed = set()

        for req_type, fn_list in property_functions.items():
            if req_type == "mask" and ch != start_ch:
                continue
            if not _should_call_property_function(req_type, image2d, mask2d):
                continue

            for fn in fn_list:
                try:
                    feat_dict = fn(image2d, mask2d)
                    if not isinstance(feat_dict, dict):
                        logger.warning(
                            "Property function %s did not return a dict, skipping.",
                            getattr(fn, "__name__", "unknown"),
                        )
                        continue
                    base_names.update(feat_dict.keys())
                    if req_type == "mask":
                        mask_base.update(feat_dict.keys())
                except Exception as exc:
                    logger.warning(
                        "Property function %s failed: %s", getattr(fn, "__name__", "unknown"), exc
                    )
                    feat_dict = {}

                for bname, value in feat_dict.items():
                    if req_type == "mask":
                        name = bname
                        if name in mask_properties_computed:
                            continue
                        mask_properties_computed.add(name)
                    else:
                        name = f"{bname}_ch{ch}" if n_channels > 1 else bname

                    if name not in feature_buffers:
                        feature_buffers[name] = [np.nan] * img_idx
                    feature_buffers[name].append(float(value) if value is not None else np.nan)
                    computed.add(name)

        return base_names, computed, mask_base
