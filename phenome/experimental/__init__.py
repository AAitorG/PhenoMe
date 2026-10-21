"""Experimental helpers for data that is not a 2D image.

The session methods are ``PhenoMe.set_data_io`` and ``PhenoMe.process_data``.
Templates here are optional show functions. They are not used unless the
caller passes them to ``set_data_io``.
"""

from .templates import orthogonal_projections, spectrogram_image, text_card

__all__ = [
    "orthogonal_projections",
    "spectrogram_image",
    "text_card",
]
