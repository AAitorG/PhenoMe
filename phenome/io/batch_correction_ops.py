"""HDF5 persistence for batch-correction statistics (optional /audit trail)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, cast

import h5py
import numpy as np

from ..core.batch_correction import BatchCorrectionStats, MethodName

_GZIP_LEVEL = 4


def _batch_subgroup_name(batch_id: str) -> str:
    """Stable HDF5 subgroup name (avoids illegal / special characters in batch_id)."""
    h = hashlib.sha256(batch_id.encode("utf-8")).hexdigest()[:32]
    return f"batch_{h}"


def write_batch_correction_group(
    f: h5py.File,
    stats_by_batch: dict[str, BatchCorrectionStats],
    *,
    method: MethodName,
    batch_metadata_key: str,
    source: str,
) -> None:
    """Create or replace ``/batch_correction`` with serialized per-batch stats."""
    if "batch_correction" in f:
        del f["batch_correction"]
    grp = f.create_group("batch_correction")
    grp.attrs["method"] = method
    grp.attrs["batch_metadata_key"] = batch_metadata_key
    grp.attrs["source"] = source
    grp.attrs["written_at"] = datetime.now(UTC).isoformat()

    for batch_id, st in stats_by_batch.items():
        sub = grp.create_group(_batch_subgroup_name(batch_id))
        sub.attrs["batch_id"] = batch_id
        sub.attrs["n_controls"] = int(st.n_controls)
        sub.create_dataset(
            "mean",
            data=np.asarray(st.mean, dtype=np.float32),
            compression="gzip",
            compression_opts=_GZIP_LEVEL,
        )
        if st.whiten_mat is not None:
            sub.create_dataset(
                "whiten_mat",
                data=np.asarray(st.whiten_mat, dtype=np.float32),
                compression="gzip",
                compression_opts=_GZIP_LEVEL,
            )
        if st.std is not None:
            sub.create_dataset(
                "std",
                data=np.asarray(st.std, dtype=np.float32),
                compression="gzip",
                compression_opts=_GZIP_LEVEL,
            )


def read_batch_correction_group(
    f: h5py.File,
) -> tuple[dict[str, BatchCorrectionStats], dict[str, Any]] | None:
    """Load ``/batch_correction`` if present; otherwise return None."""
    if "batch_correction" not in f:
        return None
    grp = cast(h5py.Group, f["batch_correction"])
    meta = {
        "method": str(grp.attrs.get("method", "")),
        "batch_metadata_key": str(grp.attrs.get("batch_metadata_key", "")),
        "source": str(grp.attrs.get("source", "")),
        "written_at": str(grp.attrs.get("written_at", "")),
    }
    stats: dict[str, BatchCorrectionStats] = {}
    for key in grp:
        sub = grp[key]
        if not isinstance(sub, h5py.Group):
            continue
        batch_id = str(sub.attrs.get("batch_id", key))
        n_controls = int(sub.attrs.get("n_controls", 0))
        mean = np.asarray(cast(h5py.Dataset, sub["mean"])[:], dtype=np.float32)
        whiten = None
        std = None
        if "whiten_mat" in sub:
            whiten = np.asarray(cast(h5py.Dataset, sub["whiten_mat"])[:], dtype=np.float32)
        if "std" in sub:
            std = np.asarray(cast(h5py.Dataset, sub["std"])[:], dtype=np.float32)
        stats[batch_id] = BatchCorrectionStats(
            batch_id=batch_id,
            mean=mean,
            n_controls=n_controls,
            whiten_mat=whiten,
            std=std,
        )
    return stats, meta
