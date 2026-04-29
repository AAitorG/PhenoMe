"""
Path resolution utilities for HDF5 results files.
"""

import os


def _resolve_results_hdf5_path(path: str | None) -> str:
    """Resolve a unified HDF5 results path for save/load.

    - ``None`` or blank -> ``<cwd>/phenome_results.h5``
    - Otherwise the argument is always a **filename** (``.h5`` / ``.hdf5`` optional)
      with an **optional directory prefix**:
      - No directory in the string (no path separator) -> file in the **current
        working directory** (e.g. ``out`` -> ``<cwd>/out.h5``).
      - Path separators present -> relative or absolute path to that file
        (e.g. ``results/run`` -> ``<cwd>/results/run.h5``).
    - Trailing separator only (directory, no basename) -> ``phenome_results.h5``
      inside that directory (``results/`` -> ``results/phenome_results.h5``).
    """
    default_base = "phenome_results.h5"
    if path is None:
        return os.path.normpath(os.path.join(os.getcwd(), default_base))

    s = path.strip()
    if not s:
        return os.path.normpath(os.path.join(os.getcwd(), default_base))

    expanded = os.path.expanduser(s)

    if expanded.endswith(os.sep) or (os.altsep is not None and expanded.endswith(os.altsep)):
        parent = os.path.abspath(expanded)
        return os.path.normpath(os.path.join(parent, default_base))

    has_sep = os.sep in expanded or (os.altsep is not None and os.altsep in expanded)
    candidate = os.path.join(os.getcwd(), expanded) if not has_sep else os.path.abspath(expanded)

    _, ext = os.path.splitext(candidate)
    if ext.lower() not in (".h5", ".hdf5"):
        candidate = candidate + ".h5"

    return os.path.normpath(candidate)
