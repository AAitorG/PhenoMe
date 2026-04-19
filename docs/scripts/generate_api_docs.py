#!/usr/bin/env python3
"""Thin entry point for the modular :mod:`api_docs` package.

Run from the repo root or ``docs/``; this script is invoked by
``npm run prebuild``/``predev`` in ``docs/package.json``. All logic now
lives in :mod:`docs.scripts.api_docs`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``docs/scripts/`` is on ``sys.path`` so ``api_docs`` imports regardless
# of the current working directory when the script is invoked.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from api_docs.site import build  # noqa: E402  (deliberate late import)

if __name__ == "__main__":
    build()
