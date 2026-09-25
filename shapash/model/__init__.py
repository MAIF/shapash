"""Model capability layer — modality-agnostic adapters for the shapash engine.

This package introduces a first-class ``Model`` abstraction (a peer to the planned
``shapash/data/`` layer). Explanation backends and counterfactual generators depend on
*declared capabilities* rather than a concrete model type, so a component can call
``is_compatible(model)`` and only run when the required capabilities are present.

The design deliberately adopts the *intent* of Google PAIR LIT's ``Model`` API — components
stay model-agnostic by introspecting what a model can do — but expresses it Pythonically with
abstract base classes and capability mixins (``SupportsTokenization`` / ``SupportsEmbeddings`` /
``SupportsGradients``) instead of LIT's runtime ``JsonDict``/``LitType`` specs. This matches the
typed-dataclass style already used elsewhere in shapash (e.g. ``NlpContributions``).

All full-capability adapters share one spine, :class:`~shapash.model.encoder.EncoderClassifierModel`
(encoder + pooling + classification head), and differ only in how the backbone is assembled:
``HFClassifierModel`` (the classifier is the backbone), ``SentenceTransformerModel`` and
``TorchClassifierModel`` (a body + pooling + head fused into a backbone).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapash.model.base import (
    SupportsCaptumIG,
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
    has_capabilities,
)
from shapash.model.encoder import EncoderClassifierModel
from shapash.model.hf import HFClassifierModel, HFPipelineModel

if TYPE_CHECKING:  # import for type checkers only — the runtime path is the lazy __getattr__ below
    from shapash.model.torch_models import SentenceTransformerModel, TorchClassifierModel

# Adapters imported on demand. ``shapash.model`` is loaded by a plain ``import shapash``, but
# ``torch_models`` needs ``torch`` at *module* level (it subclasses ``nn.Module`` to fuse body+head into
# a backbone), and torch is an optional dependency. Deferring the import keeps core installs working and
# means users who never touch these adapters don't pay to load them. ``__getattr__`` (PEP 562) makes
# ``from shapash.model import TorchClassifierModel`` work exactly as before.
_LAZY_ADAPTERS = {"SentenceTransformerModel", "TorchClassifierModel"}


def __getattr__(name: str):
    """Resolve the torch-only adapters on first access (see :data:`_LAZY_ADAPTERS`)."""
    if name in _LAZY_ADAPTERS:
        from shapash.model import torch_models  # noqa: PLC0415

        return getattr(torch_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-resolved adapters in ``dir()`` and tab-completion."""
    return sorted(__all__)


__all__ = [
    "TextModel",
    "SupportsTokenization",
    "SupportsEmbeddings",
    "SupportsGradients",
    "SupportsCaptumIG",
    "has_capabilities",
    "HFPipelineModel",
    "HFClassifierModel",
    "EncoderClassifierModel",
    "SentenceTransformerModel",
    "TorchClassifierModel",
]
