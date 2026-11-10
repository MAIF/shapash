"""Modality-agnostic backend contract.

``Backend`` is the abstract base shared by every explainability backend, tabular
or text.  It declares ``run_explainer`` without committing to a concrete return type, so a
single interface spans both modalities.

Return-type contract (migration target)
----------------------------------------
Text backends already return a typed structure (``NlpContributions``). Tabular
backends still return a ``dict`` pending migration to the same typed contract.
``Backend`` therefore types the method loosely (``Any``); tighten the annotation
once the tabular path has been migrated to the typed structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Abstract base class shared by all explainability backends.

    Concrete subclasses implement ``run_explainer``: run the underlying explainer
    and return its backend-specific output. The concrete input and return types
    depend on the modality (tabular vs text); see the module docstring for the
    intended convergence.
    """

    # ``name`` identifies the backend for name-based construction via
    # ``get_backend_cls_from_name``.  The ``"base"`` sentinel marks abstract
    # bases (this class and per-modality ABCs), which are skipped by the
    # registry; concrete backends override it with a unique name.
    name = "base"

    @abstractmethod
    def run_explainer(self, x: Any) -> Any:
        """Run the explainer and return its raw, backend-specific output.

        Parameters
        ----------
        x : Any
            The observations to explain (a ``pd.DataFrame`` for tabular
            backends, a sequence of strings for text backends).

        Returns
        -------
        Any
            The backend-specific contributions.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of Backend and must implement `run_explainer`"
        )
