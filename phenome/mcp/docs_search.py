"""Search local docs or the published llms-small.txt corpus."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_LLMS_SMALL_URL = "https://AAitorG.github.io/PhenoMe/llms-small.txt"
_MAX_HITS = 8
_SNIPPET = 280


def docs_content_dir() -> Path | None:
    """Return the Starlight content tree when this checkout includes docs."""
    candidate = Path(__file__).resolve().parents[2] / "docs" / "src" / "content" / "docs"
    return candidate if candidate.is_dir() else None


def search_corpus(query: str) -> str:
    """Keyword-search local markdown docs, then recipes, then published llms-small.txt."""
    q = query.strip()
    if not q:
        return "Provide a non-empty query."
    hits = _search_local_docs(q)
    if hits:
        return hits
    from .recipes import RECIPES

    recipe_hits = []
    ql = q.lower()
    for name, body in RECIPES.items():
        if ql in name.lower() or ql in body.lower():
            recipe_hits.append(f"### recipe:{name}\n{body.strip()[:_SNIPPET]}")
    if recipe_hits:
        return "\n\n".join(recipe_hits[:_MAX_HITS])
    remote = _search_llms_small(q)
    if remote:
        return remote
    return (
        "No matches in local docs, bundled recipes, or "
        f"{_LLMS_SMALL_URL}. Try get_recipe or list_public_api."
    )


def _search_local_docs(query: str) -> str:
    """Return scored snippets from local markdown/MDX docs, or empty string."""
    root = docs_content_dir()
    if root is None:
        return ""
    tokens = [t.lower() for t in query.split() if t]
    scored: list[tuple[int, Path, str]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lower = text.lower()
        score = sum(lower.count(t) for t in tokens)
        if score <= 0:
            continue
        snippet = _first_snippet(text, tokens[0] if tokens else query)
        rel = path.relative_to(root)
        scored.append((score, rel, snippet))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ""
    parts = []
    for score, rel, snippet in scored[:_MAX_HITS]:
        parts.append(f"### {rel} (score={score})\n{snippet}")
    return "\n\n".join(parts)


def _first_snippet(text: str, token: str) -> str:
    """Return a short window of *text* around the first match of *token*."""
    lower = text.lower()
    idx = lower.find(token.lower())
    if idx < 0:
        return text[:_SNIPPET].strip()
    start = max(0, idx - 80)
    return text[start : start + _SNIPPET].replace("\n", " ").strip()


def _search_llms_small(query: str) -> str:
    """Fetch and search the published llms-small.txt; empty on network failure."""
    try:
        with urllib.request.urlopen(_LLMS_SMALL_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return ""
    tokens = [t.lower() for t in query.split() if t]
    chunks = text.split("\n#")
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        lower = chunk.lower()
        score = sum(lower.count(t) for t in tokens)
        if score:
            scored.append((score, chunk[:_SNIPPET].strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ""
    return "\n\n".join(f"(llms-small, score={s})\n{c}" for s, c in scored[:_MAX_HITS])
