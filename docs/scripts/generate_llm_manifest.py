#!/usr/bin/env python3
"""Write the public PhenoMe API manifest to the Astro public directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phenome.mcp.manifest import build_manifest  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "phenome-api.json"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote LLM API manifest to {OUTPUT}")


if __name__ == "__main__":
    main()
