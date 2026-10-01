#!/usr/bin/env python3
"""Thin entry point for the modular :mod:`api_docs` package.

Run from the repo root or ``docs/``; this script is invoked by
``npm run build``/``npm run dev`` in ``docs/package.json``. All logic now
lives in :mod:`docs.scripts.api_docs`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ``npm run dev`` calls ``python3``. When that is older than 3.12, re-run
# under a 3.12 interpreter before importing the package (PEP 695 syntax).
if sys.version_info < (3, 12):  # noqa: UP036
    _repo_root = Path(__file__).resolve().parents[2]
    _candidates = [_repo_root / ".venv" / "bin" / "python"]
    _python312 = shutil.which("python3.12")
    if _python312:
        _candidates.append(Path(_python312))
    _current = Path(sys.executable).resolve()
    for _candidate in _candidates:
        if not _candidate.is_file() or _candidate.resolve() == _current:
            continue
        try:
            _probe = subprocess.run(
                [
                    _candidate,
                    "-c",
                    "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)",
                ],
                check=False,
            )
        except OSError:
            continue
        if _probe.returncode == 0:
            os.execv(_candidate, [str(_candidate), *sys.argv])
    sys.stderr.write(
        "generate_api_docs.py requires Python 3.12 or newer "
        f"(this interpreter is {sys.version.split()[0]}).\n"
    )
    raise SystemExit(1)

# Ensure ``docs/scripts/`` is on ``sys.path`` so ``api_docs`` imports regardless
# of the current working directory when the script is invoked.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from api_docs.site import build  # noqa: E402  (deliberate late import)

if __name__ == "__main__":
    build()
