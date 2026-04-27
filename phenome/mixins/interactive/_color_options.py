"""Color-by dropdown option discovery for the interactive explorer.

Pure helpers kept out of :mod:`~phenome.mixins.interactive.explorer` so the host
class module stays focused on UI wiring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import get_all_metadata_keys, get_metadata_value_from_dict
from ._constants import DR_RANDOM_STATE, INTERACTIVE_COLOR_SCAN_MAX_INDICES


def _indices_for_color_scan(n_total: int, *, seed: int = DR_RANDOM_STATE) -> np.ndarray:
    """Return image indices spanning the dataset for Color-by dropdown informativeness.

    Uses all indices when ``n_total`` is small; otherwise a fixed-size random sample
    so large runs stay bounded while still covering the index range (unlike a prefix).
    """
    if n_total <= 0:
        return np.array([], dtype=np.int64)
    cap = INTERACTIVE_COLOR_SCAN_MAX_INDICES
    if n_total <= cap:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=cap, replace=False)


def _build_informative_color_columns(results: Any, property_keys: list[str]) -> list[str]:
    """Compute metadata/property columns eligible for the explorer Color-by dropdown."""
    meta_keys_list = get_all_metadata_keys(results)
    meta_keys_set = frozenset(meta_keys_list)
    cols = sorted(set(meta_keys_list).union(property_keys))
    informative_cols: list[str] = []
    n_total = results.n_images
    scan_ix = _indices_for_color_scan(n_total)

    for col in cols:
        vals: list[Any] = []
        is_meta = col in meta_keys_set
        if is_meta:
            for i in scan_ix:
                ii = int(i)
                meta = results.metadata[ii]
                if isinstance(meta, dict):
                    v = get_metadata_value_from_dict(meta, col)
                    if v is not None:
                        vals.append(v)
        else:
            for i in scan_ix:
                ii = int(i)
                props = results.properties[ii]
                if isinstance(props, dict):
                    v = props.get(col)
                    if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)):
                        vals.append(v)

        if not vals:
            continue

        unique_vals = set(vals)
        if len(unique_vals) <= 1:
            continue

        if (
            len(unique_vals) > n_total * 0.9
            and n_total > 10
            and col.lower() in ("id", "filename", "file_path", "path", "index")
        ):
            continue

        informative_cols.append(col)

    return sorted(informative_cols)
