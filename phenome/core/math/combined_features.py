"""
Combined embedding-property feature construction.

Provides build_combined_features for DRY concatenation of embeddings and properties
with balanced normalization. Used by dimensionality reduction, analysis, and distances.
"""

import numpy as np

from ..protocols import PhenoMeProtocol


def build_combined_features(
    pipeline: PhenoMeProtocol,
    indices: list[int],
    property_keys: list[str] | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Build combined embedding + property matrix with balanced normalization.

    Aligns embeddings to property-valid indices (NaN-filtered), L2-normalizes
    embeddings, scales properties by 1/sqrt(n_props) for balance, and concatenates.

    Args:
        pipeline: PhenoMe (or protocol-compatible) with get_embeddings,
            _get_property_matrix, and _normalize_embeddings_l2.
        indices: Candidate image indices.
        property_keys: Property subset. None = all.
        normalize: If True, L2-normalize embeddings and scale properties for balance.

    Returns:
        Tuple of (matrix, valid_indices):
        - matrix: np.ndarray shape (n_valid, n_emb_dims + n_props).
        - valid_indices: List[int] of global image indices (intersection of indices
          with property-valid samples).

    Raises:
        ValueError: When embeddings unavailable, no valid property samples, or
            inconsistent indices.
    """
    emb_all = pipeline.get_embeddings(indices)
    if emb_all is None or len(emb_all) == 0:
        raise ValueError(
            "No embedding data available for combined features. Run process_images() first."
        )

    prop_matrix, prop_valid_indices, _ = pipeline._get_property_matrix(
        indices=indices,
        property_keys=property_keys,
        normalize=normalize,
        handle_nans="filter",
    )
    if len(prop_valid_indices) == 0:
        raise ValueError("No valid samples after filtering NaNs for properties (combined source).")

    prop_valid_arr = np.asarray(prop_valid_indices, dtype=np.int64)
    idx_map = {orig_idx: pos for pos, orig_idx in enumerate(indices)}
    try:
        pos_indices = [idx_map[orig_idx] for orig_idx in prop_valid_arr]
    except KeyError as e:
        raise ValueError(
            f"Inconsistent indices when constructing combined features: "
            f"index {e.args[0]} not found in candidate indices."
        ) from e

    if not pos_indices:
        raise ValueError(
            "No overlapping samples between embeddings and properties for combined source."
        )
    emb_aligned = emb_all[pos_indices]

    if normalize:
        emb_aligned = pipeline._normalize_embeddings_l2(emb_aligned)
        n_props = prop_matrix.shape[1]
        if n_props > 0:
            prop_matrix = prop_matrix / np.sqrt(n_props)

    if emb_aligned.shape[0] != prop_matrix.shape[0]:
        raise ValueError(
            f"Combined feature construction failed: embeddings rows={emb_aligned.shape[0]}, "
            f"properties rows={prop_matrix.shape[0]}."
        )

    matrix = np.concatenate([emb_aligned, prop_matrix], axis=1)
    return matrix, prop_valid_indices
