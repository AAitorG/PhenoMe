"""Smoke tests for installed library compatibility with PhenoMe."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import subprocess
import sys

import pytest

import phenome


def test_installed_packages_are_mutually_compatible() -> None:
    """Fail if the installed environment has unsatisfied package requirements."""
    if importlib.util.find_spec("pip") is None:
        pytest.skip("pip is not installed in this interpreter")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_phenome_modules_import() -> None:
    """Import the public API and every phenome.* module; collect all failures."""
    failures: list[str] = []

    missing = [name for name in phenome.__all__ if not hasattr(phenome, name)]
    if missing:
        failures.append(f"phenome.__all__ missing attributes: {', '.join(missing)}")

    def _onerror(name: str) -> None:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    for module in pkgutil.walk_packages(
        phenome.__path__,
        phenome.__name__ + ".",
        onerror=_onerror,
    ):
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            failures.append(f"{module.name}: {type(exc).__name__}: {exc}")

    assert not failures, "\n".join(failures)
