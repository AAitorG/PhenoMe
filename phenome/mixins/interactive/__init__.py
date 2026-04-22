"""Interactive Jupyter explorer for PhenoMe results.

Public exports (preserve historical import path
``phenome.mixins.interactive.PhenoMeInteractive``):

    - :class:`PhenoMeInteractive`
    - :func:`create_interactive_explorer`
    - :class:`_InteractiveExplorerProtocol` (internal; used by the visualization mixin)
"""

from ._protocol import _InteractiveExplorerProtocol
from .explorer import PhenoMeInteractive, create_interactive_explorer

__all__ = [
    "PhenoMeInteractive",
    "_InteractiveExplorerProtocol",
    "create_interactive_explorer",
]
