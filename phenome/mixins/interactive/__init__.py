"""Interactive Jupyter explorer for PhenoMe results.

Public exports (preserve historical import path
``phenome.mixins.interactive.PhenoMeInteractive``):

    - :class:`PhenoMeInteractive`
    - :func:`create_interactive_explorer`
    - :func:`create_colab_interactive_explorer`
    - :class:`_InteractiveExplorerProtocol` (internal; used by the visualization mixin)
"""

from ._protocol import _InteractiveExplorerProtocol
from .colab_explorer import ColabInteractiveExplorer, create_colab_interactive_explorer
from .explorer import PhenoMeInteractive, create_interactive_explorer

__all__ = [
    "ColabInteractiveExplorer",
    "PhenoMeInteractive",
    "_InteractiveExplorerProtocol",
    "create_colab_interactive_explorer",
    "create_interactive_explorer",
]
