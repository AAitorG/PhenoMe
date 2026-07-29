"""Optional progress reporting for long-running PhenoMe operations.

GUIs and notebooks can pass a ``progress_callback`` to ``process_images`` /
``compute_properties``. When omitted, those methods keep the default ``tqdm``
bars. The library itself has no Qt/napari/ipywidgets dependency.
"""

from collections.abc import Callable

# current, total, description  (total may be 0 if unknown)
ProgressCallback = Callable[[int, int, str], None]


def report_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    desc: str = "",
) -> None:
    """Invoke *callback* if provided; no-op when ``None``.

    ``ProgressCallback`` is ``Callable[[int, int, str], None]`` —
    ``(current, total, desc)``. Pass the same callable to
    ``PhenoMe.process_images`` / ``compute_properties`` via
    ``progress_callback=...``. When a callback is set, those methods disable
    ``tqdm`` so GUIs are not fighting a second progress UI.

    Args:
        callback: Optional progress hook, or ``None``.
        current: Units completed so far (batches or images).
        total: Expected total units; may be ``0`` if unknown.
        desc: Short stage label (e.g. ``\"Processing batches\"``).
    """
    if callback is not None:
        callback(current, total, desc)
