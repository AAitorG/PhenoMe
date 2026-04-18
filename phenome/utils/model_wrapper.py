"""
Model wrappers for vision model embedding extraction.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from .device import get_default_device


class ModelWrapper(ABC):
    """
    @section Model wrappers
    @order 15

    Base wrapper for vision model embedding extraction.

    Provides only basics: model.eval(), model.to(device), torch.no_grad(), and
    a sanity check that output is (B, D). Dict extraction, key lookup, and
    tensor flattening are the heritage class's responsibility.

    Args:
        model: PyTorch vision model.
        device: Optional torch.device. If None, uses get_default_device().
    """

    def __init__(self, model: Any, device: torch.device | None = None):
        self.model = model
        self.device = device if device is not None else get_default_device()

        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """@section Model wrappers
        @order 20

        Extract per-image embeddings.

        Delegates to _get_embeddings (heritage class) and validates output shape (B, D).

        Args:
            tensor: Input tensor (B, C, H, W) on device.

        Returns:
            Embedding tensor (B, D).
        """
        with torch.no_grad():
            out = self._get_embeddings(tensor)
            return _validate_embedding_out(out)

    @abstractmethod
    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return embeddings as (B, D) tensor.

        Heritage class must implement. Handles dict extraction, key lookup,
        squeeze (B,1,D), flatten, etc. as needed for the specific model architecture.

        Args:
            tensor: Input tensor (B, C, H, W) on device.

        Returns:
            Embedding tensor (B, D).
        """
        pass


class DinoV2ModelWrapper(ModelWrapper):
    """
    @section Model wrappers
    @order 25

    Heritage wrapper for DINOv2 models from Meta AI.

    Handles dict output with 'x_norm_clstoken' (default key) and squeeze (B,1,D) -> (B,D).

    Args:
        model: DINOv2 model (from torch.hub.load('facebookresearch/dinov2', ...)).
        device: Optional torch.device. If None, uses get_default_device().
        embedding_output_key: Key for global embedding in forward_features output.
                              Default 'x_norm_clstoken' (required for DINOv2).
    """

    def __init__(
        self,
        model: Any,
        device: torch.device | None = None,
        embedding_output_key: str = "x_norm_clstoken",
    ):
        super().__init__(model, device=device)
        self.embedding_output_key = embedding_output_key

    def _get_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract CLS-token embeddings from DINOv2 ``forward_features`` output."""
        out = self.model.forward_features(tensor)
        if not isinstance(out, dict):
            raise ValueError(f"DINOv2 forward_features expected dict, got {type(out)}")
        emb = out.get(self.embedding_output_key)
        if emb is None:
            raise ValueError(
                f"DINOv2 output missing '{self.embedding_output_key}', keys: {list(out.keys())}"
            )
        # Squeeze (B, 1, D) -> (B, D)
        if emb.ndim == 3:
            emb = emb.squeeze(1)
        return emb


def load_dinov2_model(
    model_name: str = "dinov2_vitb14_reg",
    device: torch.device | None = None,
) -> DinoV2ModelWrapper:
    """
    @section Model wrappers
    @order 5

    Load a DINOv2 model and return a DinoV2ModelWrapper ready for processing.

    Convenience helper for DINOv2 users. Uses get_default_device() when
    device is not specified. Returns the heritage DinoV2ModelWrapper, which
    subclasses ModelWrapper and is fully compatible with the pipeline.

    Args:
        model_name: DINOv2 model identifier for torch.hub.
            Common: 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'
        device: Optional torch.device. If None, uses get_default_device().

    Returns:
        DinoV2ModelWrapper. Use with process_images(). The underlying
        ``torch.nn.Module`` is ``wrapper.model``.

    Example:
        >>> wrapper = load_dinov2_model()
        >>> pheno.find_files("path/to/images")
        >>> pheno.process_images(wrapper)
    """
    dev = device if device is not None else get_default_device()
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    return DinoV2ModelWrapper(model, dev)


def _validate_embedding_out(tensor: torch.Tensor) -> torch.Tensor:
    """Validate embedding output shape (B, D).

    Args:
        tensor: Output tensor to validate.

    Returns:
        Validated tensor.

    Raises:
        ValueError: If tensor is not a torch.Tensor or doesn't have shape (B, D).
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Embedding output must be torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 2:
        raise ValueError(
            f"Embedding output must have shape (B, D), got {tensor.shape} "
            f"(ndim={tensor.ndim}). Heritage class must return (B, D)."
        )
    return tensor
