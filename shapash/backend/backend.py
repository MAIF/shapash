"""Modality-agnostic backend contract.

``Backend`` is the abstract base shared by every explainability backend, tabular
or text.  It declares the two-method seam — ``run_explainer`` then
``get_local_contributions`` — without committing to concrete return types, so a
single interface spans both modalities.

Return-type contract (migration target)
----------------------------------------
Text backends already return a typed structure (``NlpContributions``, from both
``run_explainer`` and ``get_local_contributions``). Tabular backends still return
``dict`` / ``pd.DataFrame`` pending migration to the same typed contract.
``Backend`` therefore types both methods loosely (``Any``); tighten these
annotations once the tabular path has been migrated to the typed structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Abstract base class shared by all explainability backends.

    Concrete subclasses implement two methods:

    * ``run_explainer`` — run the underlying explainer and return its raw,
      backend-specific output.
    * ``get_local_contributions`` — convert that raw output into the
      caller-facing local-contribution structure.

    The concrete input and return types depend on the modality (tabular vs
    text); see the module docstring for the intended convergence.
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
            Raw explainer output consumed by ``get_local_contributions``.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of Backend and must implement `run_explainer`"
        )

    @abstractmethod
    def get_local_contributions(self, x: Any, explain_data: Any, subset: list[int] | None = None) -> Any:
        """Convert raw explainer output into caller-facing local contributions.

        Parameters
        ----------
        x : Any
            The same observations passed to ``run_explainer``.
        explain_data : Any
            The raw output returned by ``run_explainer``.
        subset : list[int], optional
            Positional indices selecting a subset of samples.

        Returns
        -------
        Any
            The caller-facing local-contribution structure.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of Backend and must implement `get_local_contributions`"
        )
