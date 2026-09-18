"""Shared fixtures for PhenoMe smoke tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from phenome.utils.device import set_default_device
from tests.helpers import write_rgb_png


@pytest.fixture(autouse=True)
def _reset_default_device() -> Iterator[None]:
    """Keep tests from leaking a global torch device into later cases."""
    set_default_device(None)
    yield
    set_default_device(None)


@pytest.fixture
def rgb_image_dir(tmp_path: Path) -> Path:
    """Three distinct RGB PNGs so clustering and file discovery have samples."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i, value in enumerate((10, 80, 200), start=1):
        write_rgb_png(image_dir / f"sample_{i}.png", value)
    return image_dir
