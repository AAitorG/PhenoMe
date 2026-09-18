"""Critical CPU smokes: construct, I/O, core pipeline, checkpoint, presets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenome import PhenoMe, __version__, get_preset_property_functions
from phenome.core import validate_results
from phenome.io import read_image
from tests.helpers import DummyModelWrapper, write_rgb_png

_PROCESS_KW = {
    "batch_size": 2,
    "num_workers": 0,
    "channel_mode": "combined",
    "resize_size": 32,
}


def test_construct_phenome_on_cpu() -> None:
    """The public constructor and package version must work without a GPU."""
    assert __version__
    pm = PhenoMe(device="cpu", seed=0)
    assert str(pm.device) == "cpu"
    assert pm.results.n_images == 0
    assert pm.get_embeddings() is None


def test_read_image_png_and_npy(tmp_path: Path) -> None:
    """PNG and .npy loaders must return float32 arrays with the stored shape."""
    png_path = tmp_path / "solid.png"
    write_rgb_png(png_path, 42)
    png = read_image(str(png_path))
    assert png.dtype == np.float32
    assert png.shape == (16, 16, 3)
    np.testing.assert_allclose(png, 42.0)

    npy_path = tmp_path / "tiny.npy"
    stored = np.arange(24, dtype=np.uint16).reshape(2, 4, 3)
    np.save(npy_path, stored)
    loaded = read_image(str(npy_path))
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, stored.astype(np.float32))


def test_find_files_discovers_pngs(rgb_image_dir: Path) -> None:
    """find_files must return one row per image."""
    pm = PhenoMe(device="cpu", seed=0)
    df = pm.find_files(str(rgb_image_dir), extensions=[".png"])
    assert len(df) == 3
    assert "file_path" in df.columns
    assert all(Path(p).is_file() for p in df["file_path"])


def test_process_images_properties_and_clustering(rgb_image_dir: Path) -> None:
    """find_files → process_images → compute_properties → clustering on CPU."""
    pm = PhenoMe(device="cpu", seed=0)
    pm.find_files(str(rgb_image_dir), extensions=[".png"])
    pm.process_images(DummyModelWrapper(device=pm.device), **_PROCESS_KW)

    embeddings = pm.get_embeddings()
    assert embeddings is not None
    assert embeddings.shape == (3, 8)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    # Distinct images must not collapse to a single embedding (dummy uses intensity).
    assert np.unique(embeddings, axis=0).shape[0] == 3

    props = pm.compute_properties(property_preset="intensity")
    assert len(props) == 3
    assert len(pm.results.properties) == 3
    intensity_cols = [c for c in props.columns if "intensity_mean" in c]
    assert intensity_cols, f"missing intensity_mean columns: {list(props.columns)}"
    assert np.isfinite(props[intensity_cols].to_numpy(dtype=float)).all()
    assert validate_results(pm.results)

    clusters = pm.compute_clustering(n_clusters=2, clustering_method="kmeans")
    assert len(clusters) == 3
    assert "cluster" in clusters.columns
    labels = clusters["cluster"]
    assert labels.notna().all()
    assert set(labels.astype(int)) <= {0, 1}


def test_checkpoint_roundtrip_embeddings(rgb_image_dir: Path, tmp_path: Path) -> None:
    """Embeddings written to HDF5 must reload with the same values."""
    ckpt = tmp_path / "run.h5"
    pm = PhenoMe(device="cpu", seed=0)
    pm.find_files(str(rgb_image_dir), extensions=[".png"])
    pm.process_images(
        DummyModelWrapper(device=pm.device),
        checkpoint_path=str(ckpt),
        lazy_checkpoint=False,
        save_every=1,
        **_PROCESS_KW,
    )
    original = pm.get_embeddings()
    assert original is not None
    original = np.array(original, copy=True)
    assert original.shape == (3, 8)
    assert ckpt.is_file()
    assert validate_results(pm.results)

    # Close the writer handle before reopening the same HDF5 file.
    pm.reset()
    pm.load_results(str(ckpt), lazy_checkpoint=False)
    restored = pm.get_embeddings()
    assert restored is not None
    np.testing.assert_allclose(restored, original, rtol=1e-5, atol=1e-5)
    assert validate_results(pm.results)


def test_property_presets_resolve() -> None:
    """Every advertised preset must return at least one callable."""
    for name in ("basic", "shape", "intensity", "standard", "complete"):
        preset = get_preset_property_functions(name)
        callables = [fn for fns in preset.values() for fn in fns]
        assert callables, f"preset {name!r} has no functions"
        assert all(callable(fn) for fn in callables)


@pytest.mark.parametrize("bad_preset", ["", "unknown", "BASIC"])
def test_unknown_property_preset_raises(bad_preset: str) -> None:
    """Invalid preset names must fail closed."""
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset_property_functions(bad_preset)
