"""
Results validation utilities.

Checks consistency of pipeline results structure.
"""

import numpy as np

from .._logging import get_logger
from .pipeline_results import PhenoMeResults

logger = get_logger(__name__)


def validate_results(results: PhenoMeResults) -> bool:
    """Check that pipeline results have consistent lengths across all arrays.

    Logs a warning if any mismatch is found.  Useful for debugging after
    load_results or load_committed_results.

    Args:
        results: PhenoMeResults instance.

    Returns:
        True if all lengths are consistent (or results are empty), False otherwise.
    """
    n_paths = results.n_images
    n_meta = len(results.metadata)
    n_props = len(results.properties)
    emb = results.embeddings

    n_emb = (emb.shape[0] if emb.ndim > 0 else 0) if isinstance(emb, np.ndarray) else 0

    ref = n_paths
    consistent = True

    # n_emb == 0 is valid when embeddings are lazy (HDF5-backed) or not computed.
    # Only warn when both sides are non-zero and disagree.
    if n_emb != ref and n_emb > 0 and ref > 0:
        logger.warning("Results length mismatch: embeddings=%d, img_path=%d", n_emb, ref)
        consistent = False
    if n_meta != ref and (n_meta > 0 or ref > 0):
        logger.warning("Results length mismatch: metadata=%d, img_path=%d", n_meta, ref)
        consistent = False
    if n_props != ref and (n_props > 0 or ref > 0):
        logger.warning("Results length mismatch: properties=%d, img_path=%d", n_props, ref)
        consistent = False

    return consistent
