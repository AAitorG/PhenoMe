"""
Embedding extraction from images using vision models.
"""

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .._logging import get_logger
from ..io import CheckpointManager
from ..utils.model_wrapper import ModelWrapper
from ..utils.progress import ProgressCallback, report_progress

logger = get_logger(__name__)

_ZERO_NORM_EPS = 1e-12


def l2_normalize_skip_zeros(
    toks: torch.Tensor,
    *,
    channel_index: int | None = None,
    eps: float = _ZERO_NORM_EPS,
) -> torch.Tensor:
    """L2-normalize rows, leaving near-zero vectors as zeros instead of NaN."""
    norms = torch.linalg.vector_norm(toks, ord=2, dim=-1, keepdim=True)
    zero = norms <= eps
    if bool(zero.any()):
        n_zero = int(zero.sum().item())
        logger.warning(
            "Zero-norm embedding at channel %s for %d vector(s); left as zeros.",
            channel_index if channel_index is not None else "?",
            n_zero,
        )
    safe = norms.clamp_min(eps)
    out = toks / safe
    return torch.where(zero, torch.zeros_like(out), out)


class EmbeddingExtractor:
    """
    @section I/O & Discovery
    @order 15

    Extracts embeddings from images using a ModelWrapper and DataLoader.

    Handles both combined mode (single embedding per image) and split mode
    (concatenate embeddings from each channel).

    In split mode, each channel embedding is L2-normalized before concatenation
    by default so channels contribute equally. Set ``l2_normalize_channels=False``
    to preserve raw channel magnitudes (one channel may then dominate). Use
    combined mode when channels should be fused into a single model input instead.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        device: torch.device,
        *,
        l2_normalize_channels: bool = True,
    ):
        """Initialize the extractor.

        Args:
            model_wrapper: ModelWrapper instance for vision model feature extraction.
            device: Device to run inference on.
            l2_normalize_channels: If True (default), L2-normalize each split-mode
                channel embedding before concatenation.
        """
        self.model_wrapper = model_wrapper
        self.device = device
        self.l2_normalize_channels = l2_normalize_channels
        self.model_wrapper.sync_device(device)

    def extract_batch(
        self,
        batch_tensor: torch.Tensor,
        items: tuple,
        *,
        l2_normalize_channels: bool | None = None,
    ) -> tuple[np.ndarray, list[str], list[dict]]:
        """Process one batch and return embeddings, paths, and metadata.

        Args:
            batch_tensor: torch.Tensor, shape (B, C, H, W) combined mode or
                (B, N_ch, C, H, W) split mode (concatenate channel embeddings).
            items: Tuple of item dicts from dataset. Each dict has 'file_path', 'metadata'.
            l2_normalize_channels: Override for split-mode per-channel L2. If None,
                uses the extractor's ``l2_normalize_channels`` setting.

        Returns:
            Tuple of (embeddings, paths, metadata):
            - embeddings: np.ndarray shape (B, D) or (B, D_combined) for split mode.
            - paths: List[str], length B.
            - metadata: List[dict], length B.
        """
        is_split = batch_tensor.ndim == 5

        if not is_split:
            flat = batch_tensor.to(self.device, non_blocking=True)
            embeddings = self.model_wrapper.extract_embeddings(flat).float().cpu().numpy()
            return (
                embeddings,
                [it.get("file_path", "") for it in items],
                [it.get("metadata", {}) for it in items],
            )

        do_l2 = (
            self.l2_normalize_channels
            if l2_normalize_channels is None
            else bool(l2_normalize_channels)
        )
        # Split mode: optionally L2-normalize each channel embedding, then concatenate
        batch_sz, n_ch, _c, _h, _w = batch_tensor.shape
        per_img: list[list[np.ndarray]] = [[] for _ in range(batch_sz)]

        for ch in range(n_ch):
            flat = batch_tensor[:, ch].to(self.device, non_blocking=True)
            toks = self.model_wrapper.extract_embeddings(flat).float()
            if do_l2:
                toks = l2_normalize_skip_zeros(toks, channel_index=ch)
            toks = toks.cpu().numpy()
            for i in range(batch_sz):
                per_img[i].append(toks[i])

        embeddings_list, paths, metas = [], [], []
        for i, item in enumerate(items):
            paths.append(item.get("file_path", ""))
            metas.append(item.get("metadata", {}))
            if per_img[i]:
                embeddings_list.append(np.concatenate(per_img[i], axis=-1))
            else:
                dim = sum(t.shape[-1] for t in per_img[0]) if per_img[0] else 0
                embeddings_list.append(np.zeros(dim, dtype=np.float32))

        return np.stack(embeddings_list), paths, metas

    def extract_from_dataloader(
        self,
        dataloader: Any,
        results: Any,
        emb_buffer: list[np.ndarray],  # accumulates embeddings in-place
        checkpoint_path: str | None = None,
        ckpt: CheckpointManager | None = None,
        cur_params: dict | None = None,
        save_every: int = 5,
        lazy_checkpoint: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> CheckpointManager | None:
        """Process all batches and populate results or checkpoint.

        When *checkpoint_path* is set, batches are buffered and committed
        incrementally.  When not using a checkpoint, embeddings are appended
        to *emb_buffer* and paths/metadata are appended to *results* in-place.

        Args:
            dataloader: PyTorch DataLoader yielding (batch_tensor, items).
            results: PhenoMeResults instance. Paths and metadata are appended
                in-place when not using a checkpoint.
            emb_buffer: List that accumulates embedding arrays in-place.
                Filled only in the non-checkpoint path.
            checkpoint_path: Optional path for HDF5 checkpoint.
            ckpt: Optional CheckpointManager (created if checkpoint_path set and ckpt None).
            cur_params: Processing params for a new checkpoint.
            save_every: Checkpoint commit frequency (batches).
            lazy_checkpoint: If True (default), keep the checkpoint file open.
            progress_callback: Optional ``(current, total, desc) -> None`` hook.
                When set, ``tqdm`` is disabled and the callback is invoked after
                each batch.

        Returns:
            The CheckpointManager used, or None.  Caller must commit and close.
        """
        use_ckpt = checkpoint_path is not None
        batch_count = 0
        skipped_batches = 0
        # Do not default to True here — that would override l2_normalize_channels=False.
        l2_override: bool | None = None
        if cur_params is not None and cur_params.get("l2_normalize_channels") is not None:
            l2_override = bool(cur_params["l2_normalize_channels"])

        total_batches = len(dataloader)
        desc = "Processing batches"
        disable_tqdm = progress_callback is not None

        for batch_i, (batch_tensor, items) in enumerate(
            tqdm(dataloader, desc=desc, total=total_batches, disable=disable_tqdm),
            start=1,
        ):
            if batch_tensor is None:
                skipped_batches += 1
                for it in items or ():
                    file_path = it.get("file_path", "?") if isinstance(it, dict) else "?"
                    logger.debug("Skipped image (failed load): %s", file_path)
                report_progress(progress_callback, batch_i, total_batches, desc)
                continue

            batch_embeddings, batch_paths, batch_meta = self.extract_batch(
                batch_tensor,
                items,
                l2_normalize_channels=l2_override,
            )

            if use_ckpt and checkpoint_path is not None:
                if ckpt is None:
                    if batch_embeddings.ndim != 2:
                        raise ValueError(
                            "batch_embeddings must be 2D (n_batch, n_features); "
                            f"got shape {batch_embeddings.shape}."
                        )
                    dim = int(batch_embeddings.shape[-1])
                    ckpt = CheckpointManager(
                        checkpoint_path,
                        embedding_dim=dim,
                        processing_params=cur_params,
                        lazy=lazy_checkpoint,
                    )
                ckpt.buffer_embeddings(batch_embeddings, batch_paths, batch_meta)
                batch_count += 1
                if batch_count % save_every == 0:
                    ckpt.commit_embeddings()
            else:
                # Accumulate into buffer and results (no checkpoint)
                for i in range(batch_embeddings.shape[0]):
                    emb_buffer.append(batch_embeddings[i])
                    results.img_path.append(batch_paths[i])
                    results.metadata.append(batch_meta[i])

            report_progress(progress_callback, batch_i, total_batches, desc)

        if skipped_batches > 0:
            logger.info("Skipped %d batches (failed image loads).", skipped_batches)

        return ckpt if use_ckpt else None
