"""
Property computation for the phenotyping pipeline.

Provides methods to compute per-image scalar properties from images and masks,
aggregate statistics by group, and format property reports.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..._logging import get_logger
from ...core.pipeline_results import PhenoMeResults
from ...io import CheckpointManager
from . import _compute
from ._report import build_properties_dataframe as _build_properties_dataframe_fn
from ._report import parse_grouped_stats_dataframe as _parse_grouped_stats_dataframe_fn
from .grouping import (
    print_top_properties_vs_reference as _print_top_properties_vs_reference_fn,
)
from .grouping import (
    property_stats_by_group as _property_stats_by_group_fn,
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
    """
    @section Analysis
    @order 4

    Pipeline providing property computation and reporting for PhenoMe.

    Expected attributes from the parent class:
        - self.results: PhenoMeResults with img_path, metadata, properties, embeddings
        - self._processing_params: Optional[Dict[str, Any]]
        - self._db: Optional[CheckpointManager]
        - self._ram_internal: Optional[List[Dict[str, Any]]]
    """

    # Type hints for pipeline attributes (provided by parent class)
    results: PhenoMeResults
    _processing_params: dict[str, Any] | None
    _property_norm_cache: dict[str, Any] | None
    _db: CheckpointManager | None

    def save_results(
        self,
        path: str | None = None,
        compression: str = "gzip",
    ) -> None:
        """Persist results HDF5; real implementation lives on ``PhenoMe``."""
        ...

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

    def property_stats_by_group(
        self,
        group_by: list[str] | None = None,
        properties: list[str] | None = None,
        *,
        print_table: bool = True,
        print_properties: list[str] | None = None,
        group_column_width_max: int = 25,
        content_col_width_max: int = 25,
    ) -> pd.DataFrame:
        """Group properties, compute per-group statistics, and optionally print a table.

        Uses `_build_properties_dataframe` from computed
        ``results.properties`` and metadata (same source as
        [compute_properties](pipeline.md#api-phenomeproperties-compute_properties)).

        Args:
            group_by: Metadata columns to group by. If None, auto-selects first 2.
            properties: Property columns to include in aggregation. If None, auto-detects numeric.
            print_table: If True, log a formatted mean±std table.
            print_properties: Subset of properties to show in the table. If None, shows all.
            group_column_width_max: Maximum width for each grouping column when printing.
            content_col_width_max: Maximum width for each property statistic column when printing.

        Returns:
            Aggregated DataFrame with mean/std/min/max per property per group.

        Raises:
            TypeError: If group_by/properties are invalid types.
            ValueError: If group_by keys are not in DataFrame columns.
        """
        df = self._build_properties_dataframe()
        return _property_stats_by_group_fn(
            df,
            group_by=group_by,
            properties=properties,
            print_table=print_table,
            print_properties=print_properties,
            group_column_width_max=group_column_width_max,
            content_col_width_max=content_col_width_max,
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
        group_column_width_max: int = 25,
        top_property_col_width_max: int = 25,
        top_effect_col_width_min: int = 12,
    ) -> pd.DataFrame:
        """For each non-reference group, return the top k properties that most differentiate it from the reference.

        Uses Cohen's d (effect size) by default to rank properties by how different each group
        is from the reference. Both higher and lower values count as "different" (uses |d|).

        Cohen's d formula (pooled): d = (mean_group - mean_ref) / s_pooled, where
        s_pooled = sqrt([(n_ref-1)*std_ref² + (n_other-1)*std_other²] / (n_ref + n_other - 2)).
        Requires sample std (ddof=1) from property_stats_by_group.

        Args:
            df: Aggregated DataFrame from property_stats_by_group().
            reference_group: Dict mapping grouping column names to values (e.g. {"drug": "Control", "time": "60_min"}).
            k: Number of top properties per group.
            properties: Property names to consider. If None, uses all in DataFrame.
            metric: 'cohens_d' (effect size) or 'mean_diff' (absolute mean difference).
            print_output: If True, pretty-print the results.
            group_column_width_max: Maximum width for each grouping column in the printed table.
            top_property_col_width_max: Max width for the property name column in the printed table.
            top_effect_col_width_min: Min width for Cohen's d and mean diff columns when printing.

        Returns:
            DataFrame with columns: grouping cols, property, effect_size (Cohen's d when
            ``metric='cohens_d'``), mean_diff, ref_mean, group_mean, rank. One row per
            (group, property) for top-k only.
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
            grouping_cols, _ = _parse_grouped_stats_dataframe_fn(df)
            _print_top_properties_vs_reference_fn(
                result_df,
                grouping_cols,
                reference_group,
                k,
                group_column_width_max=group_column_width_max,
                metric=metric,
                top_property_col_width_max=top_property_col_width_max,
                top_effect_col_width_min=top_effect_col_width_min,
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

        property_functions = _compute.normalize_property_functions(
            property_preset=property_preset,
            additional_property_functions=additional_property_functions,
        )
        if not property_functions:
            raise ValueError(
                "No property functions provided. Pass a property_preset "
                "(e.g. 'basic', 'morphology') or additional_property_functions."
            )

        image_paths, mask_paths = _compute.resolve_properties_paths(
            results=self.results,
            file_df=file_df,
            metadata_config=metadata_config or getattr(self, "_metadata_config", None),
        )

        expected_property_keys = _compute.infer_expected_property_keys(
            property_functions, image_paths, mask_paths
        )

        if force_update and not checkpoint_path:
            logger.info(
                "force_update=True has no effect without checkpoint_path; "
                "properties are computed in memory only.",
            )

        import os

        if (
            checkpoint_path
            and not os.path.isfile(checkpoint_path)
            and getattr(self, "_db", None) is not None
        ):
            logger.info(
                "Checkpoint not found. Initializing %s from active database to preserve embeddings.",
                checkpoint_path,
            )
            self.save_results(path=checkpoint_path)

        if force_update and checkpoint_path:
            if os.path.isfile(checkpoint_path):
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

        # Setup checkpoint resume
        ckpt, n_already, path_alignment = _compute.setup_properties_checkpoint_resume(
            checkpoint_path,
            results=self.results,
            active_db=getattr(self, "_db", None),
            image_paths=image_paths,
            lazy=lazy_checkpoint,
            expected_property_keys=expected_property_keys,
        )
        if ckpt is None and n_already > 0:
            self._property_norm_cache = None
            self._warn_if_nan_properties(
                property_keys=sorted(expected_property_keys) if expected_property_keys else None,
                count_missing_key_as_nan=not bool(expected_property_keys),
            )
            return _build_properties_dataframe_fn(self.results)

        # Process images
        all_property_names, feature_buffers, last_committed, internal_buffer = (
            _compute.process_all_images(
                image_paths,
                mask_paths,
                property_functions,
                ckpt,
                checkpoint_path,
                save_every,
                path_alignment,
                n_jobs,
                expected_property_keys=expected_property_keys,
            )
        )

        # Finalize
        full_props, full_internal = _compute.finalize_properties_computation(
            results=self.results,
            feature_buffers=feature_buffers,
            all_property_names=all_property_names,
            path_alignment=path_alignment,
            last_committed=last_committed,
            ckpt=ckpt,
            checkpoint_path=checkpoint_path,
            image_paths=image_paths,
            processing_params=getattr(self, "_processing_params", None),
            lazy=lazy_checkpoint,
            internal_buffer=internal_buffer,
        )

        self.results.properties = full_props
        if hasattr(self, "_ram_internal"):
            self._ram_internal = full_internal

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
        """Build DataFrame from computed properties and metadata."""
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
        """Build property matrix from numeric properties."""
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
