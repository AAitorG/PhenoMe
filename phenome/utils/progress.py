"""Optional progress reporting for long-running PhenoMe operations.

GUIs and notebooks can pass a ``progress_callback`` to ``process_images`` /
``compute_properties``. When omitted, those methods keep the default ``tqdm``
bars. The library itself has no Qt/napari/ipywidgets dependency.

Analysis steps that walk a known sequence (properties, batches, groups) use
:func:`track`. :func:`track_steps` is the same bar, except a single step
stays a one-line status via :func:`log_computing`. Parallel property loops
use :func:`parallel_with_progress`. Steps that are one library call (PCA, a
vectorized correlation) call :func:`log_computing` directly.
"""

from collections.abc import Callable, Iterable, Sized
from typing import Any

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


def track[T](
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap *iterable* in a ``tqdm`` progress bar.

    Args:
        iterable: Sequence of work units whose length is known, or can be
            passed as *total*.
        desc: Short label shown beside the bar.
        total: Number of units. Inferred for sized sequences when omitted.
        leave: If False (default), remove the bar when iteration finishes.

    Returns:
        A ``tqdm`` iterator over *iterable*.
    """
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc, total=total, leave=leave)


def track_steps[T](
    items: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    bar: bool = True,
    when_single: str | None = None,
) -> Iterable[T]:
    """Progress bar for more than one step; one status line otherwise.

    A one-step bar adds nothing. *when_single* is logged in that case, and
    also when *bar* is False. Omit it when the caller already logged the
    surrounding step.

    Args:
        items: Work units.
        desc: Short label shown beside the bar.
        total: Number of units. Inferred when *items* has a length.
        bar: If False, never open a bar.
        when_single: Status line used when a bar would not add information.

    Returns:
        *items*, or a ``tqdm`` wrapper when a bar is shown.
    """
    counted = total
    if counted is None and isinstance(items, Sized):
        counted = len(items)
    if not bar or (counted is not None and counted <= 1):
        if when_single:
            log_computing(when_single)
        return items
    return track(items, desc=desc, total=counted)


def parallel_with_progress(
    tasks: Iterable[Any],
    *,
    n_jobs: int,
    desc: str,
    total: int,
) -> list[Any]:
    """Run joblib tasks, with a bar when tqdm's joblib bridge is installed.

    Args:
        tasks: ``joblib.delayed`` objects, one per work unit.
        n_jobs: Worker count forwarded to ``joblib.Parallel``.
        desc: Short label shown beside the bar.
        total: Number of tasks.

    Returns:
        Results in task order.
    """
    from joblib import Parallel

    try:
        from tqdm.auto import tqdm
        from tqdm.contrib.joblib import tqdm_joblib
    except ImportError:
        return Parallel(n_jobs=n_jobs)(tasks)
    with tqdm_joblib(tqdm(desc=desc, total=total, leave=False)):
        return Parallel(n_jobs=n_jobs)(tasks)


def log_computing(message: str) -> None:
    """Log a one-line status when a step cannot report a progress bar.

    Notebook output uses the package logger (INFO, message only), same as
    other PhenoMe progress lines.

    Args:
        message: Status sentence, for example
            ``"Computing Pearson correlations..."``.
    """
    from .._logging import get_logger

    get_logger("phenome.progress").info(message)
