"""
Results export and DataFrame builders.

Creates DataFrames from pipeline results for export and visualization.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from .._logging import get_logger
from ..utils.path_utils import path_basename
from .pipeline_results import PhenoMeResults
from .results_metadata import (
    build_metadata_columns,
    get_all_metadata_keys,
    get_metadata_value_from_dict,
)

logger = get_logger(__name__)


def prepare_embedding_dataframe(
    results: PhenoMeResults | dict,
    transformed: np.ndarray,
    indices: list[int],
    names: list[str],
) -> pd.DataFrame:
    """Create a DataFrame with embedding/reduction coordinates, metadata, and properties.

    Used by dimensionality reduction (PCA, t-SNE, UMAP) and embedding visualizations.

    Args:
        results: PhenoMeResults or dict with 'img_path', 'metadata', 'properties'.
        transformed: np.ndarray, shape (n_samples, n_components), coordinate matrix.
        indices: List[int], length n_samples. Image indices for each row.
        names: List[str], length n_components. Column names (e.g. ['Component 1', 'Component 2']).

    Returns:
        pd.DataFrame with coordinate columns, Image, Index, metadata, and property columns.
    """
    df = pd.DataFrame({n: transformed[:, i] for i, n in enumerate(names)})

    img_path = (
        results.img_path if isinstance(results, PhenoMeResults) else results.get("img_path", [])
    )
    df["Image"] = [path_basename(img_path[i]) for i in indices]
    df["Index"] = indices

    metadata_cols = build_metadata_columns(results, indices, capitalize=True)
    if metadata_cols:
        metadata_df = pd.DataFrame(metadata_cols)
        duplicate_cols = [c for c in metadata_df.columns if c in df.columns]
        if duplicate_cols:
            metadata_df = metadata_df.drop(columns=duplicate_cols)
        df = pd.concat([df, metadata_df], axis=1)

    props_list = (
        results.properties if isinstance(results, PhenoMeResults) else results.get("properties", [])
    )
    if props_list:
        prop_keys: set[str] = set()
        for i in indices:
            if i < len(props_list) and isinstance(props_list[i], dict):
                prop_keys.update(props_list[i].keys())
        for k in sorted(prop_keys):
            values = []
            for i in indices:
                if i < len(props_list) and isinstance(props_list[i], dict):
                    val = props_list[i].get(k)
                    values.append(np.nan if val is None else val)
                else:
                    values.append(np.nan)
            df[k.capitalize()] = values

    return df


def build_export_dataframe(
    results: PhenoMeResults | dict,
    dist_results: dict[str, Any] | None = None,
    include_embeddings: bool | Literal["separate"] = False,
) -> pd.DataFrame:
    """Build a DataFrame with the complete dataset for export.

    Includes metadata, distances (if provided), computed properties, and
    optionally embeddings as columns.

    Args:
        results: PhenoMeResults or dict with 'img_path', 'metadata', 'properties',
            and optionally 'embeddings'.
        dist_results: Optional dict from compute_reference_distances. Expected key:
            'distances': np.ndarray shape (N,). Adds a 'distance' column when provided.
        include_embeddings: If True, adds embedding columns (embedding_0, embedding_1, ...).
            If 'separate', embeddings are omitted here (caller saves them separately).

    Returns:
        pd.DataFrame with image_index, image_path, metadata keys, property keys,
        distance (if dist_results provided), and embedding columns (if requested).
    """
    # Normalise input
    if isinstance(results, PhenoMeResults):
        img_paths = results.img_path
        metadata_list = results.metadata
        properties_list = results.properties
        embeddings = results.embeddings
    else:
        img_paths = results.get("img_path", [])
        metadata_list = results.get("metadata", [])
        properties_list = results.get("properties", [])
        embeddings = results.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == 0:
            embeddings = None

    n_images = len(img_paths)
    if n_images == 0:
        return pd.DataFrame()

    export_data: dict[str, list[Any]] = {
        "image_index": list(range(n_images)),
        "image_path": list(img_paths),
    }

    # Metadata columns
    if metadata_list:
        meta_keys = get_all_metadata_keys(results)
        for key in meta_keys:
            column_values: list[Any] = []
            for m in metadata_list:
                if isinstance(m, dict):
                    value = get_metadata_value_from_dict(m, key)
                    column_values.append("" if value is None else value)
                else:
                    column_values.append("")
            export_data[key] = column_values

    # Distance column
    if dist_results and "distances" in dist_results:
        distances = dist_results["distances"]
        dist_list = distances.tolist() if isinstance(distances, np.ndarray) else list(distances)
        if len(dist_list) != n_images:
            logger.warning(
                "dist_results['distances'] length (%d) does not match n_images (%d); "
                "distance column omitted to avoid misalignment.",
                len(dist_list),
                n_images,
            )
        else:
            export_data["distance"] = dist_list

    # Property columns
    if properties_list:
        prop_keys: set = set()
        for p in properties_list:
            if isinstance(p, dict):
                prop_keys.update(p.keys())
        for key in sorted(prop_keys):
            export_data[key] = [
                p.get(key, np.nan) if isinstance(p, dict) else np.nan for p in properties_list
            ]

    # Embedding columns (optional)
    if include_embeddings is True and isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
        n_dims = embeddings.shape[1]
        for d in range(n_dims):
            export_data[f"embedding_{d}"] = embeddings[:, d].tolist()

    return pd.DataFrame(export_data)
