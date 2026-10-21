"""Show-function templates for volumes, waveforms, and text.

Each function returns an array of shape ``(H, W)`` or ``(H, W, C)`` in ``[0, 1]``.
Pass one as ``show_fn`` to ``PhenoMe.set_data_io``. Nothing here is applied
unless the caller passes it.
"""

from __future__ import annotations

import numpy as np


def orthogonal_projections(volume: np.ndarray, axis_order: str = "zyx") -> np.ndarray:
    """Tile the three maximum-intensity projections of a volume.

    ``axis_order`` names the axes of ``volume`` from left to right. The default
    ``"zyx"`` means ``volume.shape`` is ``(Z, Y, X)``. This function does not
    call ``ensure_hwc``, so a thin axis stays spatial.

    The returned image is three panels, left to right: projection along Z,
    along Y, then along X. Each panel is scaled to ``[0, 1]`` on its own.

    Args:
        volume: Array with ``ndim == 3``.
        axis_order: Permutation of ``"zyx"``. Default ``"zyx"``.

    Returns:
        np.ndarray: Float32 image of shape ``(H, W)`` in ``[0, 1]``.

    Raises:
        ValueError: If ``volume`` is not 3D or ``axis_order`` is not a
            permutation of ``"zyx"``.
    """
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"orthogonal_projections expects a 3D volume, got shape {array.shape}.")
    order = axis_order.lower()
    if len(order) != 3 or sorted(order) != ["x", "y", "z"]:
        raise ValueError('axis_order must be a permutation of "zyx".')
    axis_of = {name: index for index, name in enumerate(order)}
    panels: list[np.ndarray] = []
    for name in ("z", "y", "x"):
        panel = array.max(axis=axis_of[name])
        low = float(panel.min())
        high = float(panel.max())
        panel = (panel - low) / (high - low) if high > low else np.zeros_like(panel)
        panels.append(panel.astype(np.float32, copy=False))
    height = max(panel.shape[0] for panel in panels)
    gap = np.zeros((height, 4), dtype=np.float32)
    tiles: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < height:
            pad = np.zeros((height - panel.shape[0], panel.shape[1]), dtype=np.float32)
            panel = np.concatenate([panel, pad], axis=0)
        tiles.append(panel)
        tiles.append(gap)
    return np.concatenate(tiles[:-1], axis=1)


def spectrogram_image(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 256,
    hop: int = 128,
) -> np.ndarray:
    """Draw a log-magnitude spectrogram of a 1D waveform.

    ``sample_rate`` is recorded for the caller. The image uses a Hann window
    and ``numpy.fft.rfft``. Magnitude is ``log1p`` and scaled to ``[0, 1]``.

    Args:
        waveform: 1D samples. A short waveform is zero-padded to ``n_fft``.
        sample_rate: Samples per second. Must be positive. Not drawn as an axis.
        n_fft: FFT length. Default 256.
        hop: Hop between frames. Default 128.

    Returns:
        np.ndarray: Float32 image of shape ``(n_freq, n_frames)`` in ``[0, 1]``.

    Raises:
        ValueError: If the waveform is empty, ``sample_rate`` is not positive,
            or ``hop`` / ``n_fft`` are not positive.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if n_fft <= 0 or hop <= 0:
        raise ValueError(f"n_fft and hop must be positive, got n_fft={n_fft}, hop={hop}.")
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("spectrogram_image received an empty waveform.")
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    window = np.hanning(n_fft).astype(np.float32)
    frames = [
        signal[start : start + n_fft] * window for start in range(0, signal.size - n_fft + 1, hop)
    ]
    spectrum = np.fft.rfft(np.stack(frames, axis=0), axis=1)
    magnitude = np.log1p(np.abs(spectrum)).T.astype(np.float32)
    low = float(magnitude.min())
    high = float(magnitude.max())
    scaled = (magnitude - low) / (high - low) if high > low else np.zeros_like(magnitude)
    return np.asarray(scaled, dtype=np.float32)


def text_card(text: str, width: int = 320, height: int = 96) -> np.ndarray:
    """Render text as an RGB image the existing viewers can draw.

    Args:
        text: Characters to draw. Empty text becomes ``"(empty)"``.
        width: Image width in pixels. Default 320.
        height: Image height in pixels. Default 96.

    Returns:
        np.ndarray: Float32 RGB image of shape ``(height, width, 3)`` in ``[0, 1]``.

    Raises:
        ValueError: If ``width`` or ``height`` is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}.")
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    body = str(text).strip() or "(empty)"
    draw.multiline_text((8, 8), body[:500], fill=(20, 20, 20), spacing=4)
    return np.asarray(image, dtype=np.float32) / 255.0
