"""``WebappComponent`` contract + capability resolution for what-if panels.

A component declares the capabilities it needs via ``requires``; :func:`available_capabilities`
computes what the app's :class:`AppContext` actually provides, and :meth:`WebappComponent.is_available`
gates mounting on ``requires <= available``. This is the mechanism that makes the What-if Lab appear
only when the explainer holds a live (and, for counterfactuals, gradient-capable) model.

Components read the immutable :class:`~shapash.explainer.nlp_explanation.NlpExplanation` directly and
never write to it: every display choice lives in a Dash ``dcc.Store`` or a callback argument, so the
artifact a component renders is the same one that was saved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from shapash.compute.diagnostics.label_noise import has_usable_probabilities
from shapash.explainer.interactive import InteractiveEngine
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.model.base import SupportsGradients, has_capabilities

# Capability tokens components may require.
CAP_PREDICT = "engine:predict"
CAP_COUNTERFACTUAL = "engine:counterfactual"
CAP_GRADIENTS = "model:gradients"
CAP_SIMILAR = "engine:similar"
CAP_LABELS = "data:labels"
CAP_GROUND_TRUTH = "data:ground_truth"
CAP_PROJECTION = "data:projection"


@dataclass(frozen=True)
class AppContext:
    """What every panel reads: the explanation, the live engine, and the scatter coordinates.

    Built and validated once by :class:`~shapash.webapp.nlp_app.NlpWebApp`; not a user-facing type.

    Parameters
    ----------
    explanation : NlpExplanation
        The immutable artifact every panel reads.
    engine : InteractiveEngine or None
        Live engine for what-if actions, or ``None`` for a snapshot.
    coords : np.ndarray or None
        ``(n_samples, 2)`` scatter coordinates, already checked against *explanation*.
    """

    explanation: NlpExplanation
    engine: InteractiveEngine | None = None
    coords: np.ndarray | None = None


def available_capabilities(ctx: AppContext) -> frozenset[str]:
    """Return the capability tokens *ctx* satisfies.

    Parameters
    ----------
    ctx : AppContext
        The explanation, engine and coordinates this app was built from.

    Returns
    -------
    frozenset[str]
        Satisfied capability tokens (e.g. ``{"engine:predict", "engine:counterfactual",
        "model:gradients"}``).
    """
    explanation, engine = ctx.explanation, ctx.engine
    caps: set[str] = set()
    # Data capabilities are read from the compiled batch and the coordinates beside it, so they
    # survive a snapshot — they sit outside the engine guard below on purpose.
    if ctx.coords is not None:
        caps.add(CAP_PROJECTION)
    if explanation.y_true is not None:
        caps.add(CAP_GROUND_TRUTH)
        if has_usable_probabilities(explanation.y_prob):
            caps.add(CAP_LABELS)
    if engine is not None:
        if engine.can_edit():
            caps.add(CAP_PREDICT)
        if engine.can_find_similar():
            caps.add(CAP_SIMILAR)
        if engine.can_counterfactual():
            caps.add(CAP_COUNTERFACTUAL)
            # Advertise gradients only when the *bound* generator actually operates on a
            # gradient-capable model — a forward-pass-only generator (AblationFlip) must not.
            generator = getattr(engine, "cf_generator", None)
            if has_capabilities(getattr(generator, "model", None), SupportsGradients):
                caps.add(CAP_GRADIENTS)
    return frozenset(caps)


def error_mask(explanation: NlpExplanation) -> np.ndarray | None:
    """Boolean array, ``True`` where the prediction disagrees with the ground truth.

    ``None`` when either is unavailable. Compared as strings, the same way the dataset table's
    "Model Errors" filter does, so every panel that scopes itself to errors scopes to exactly the
    same rows. :func:`error_positions` is this same comparison, shaped as a set of positions instead
    of a boolean array — use whichever shape the caller needs, they must never drift apart.
    """
    y_true, y_pred = explanation.y_true, explanation.y_pred
    if y_true is None or y_pred is None:
        return None
    return np.asarray(y_true).astype(str) != np.asarray(y_pred).astype(str)


def error_positions(explanation: NlpExplanation) -> set[int] | None:
    """Positional indices of the samples the model got wrong, or ``None`` without ground truth.

    See :func:`error_mask` for the boolean-array shape of the same comparison.
    """
    mask = error_mask(explanation)
    if mask is None:
        return None
    return set(np.where(mask)[0].tolist())


def compose_selection(
    selected_indices: list[int] | None,
    cell_indices: list[int] | None,
    errors: set[int] | None,
) -> list[int] | None:
    """Intersect the app's active sample filters into one index list.

    Each argument is an independent filter that may be inactive (``None``): the scatter box/lasso
    selection, the confusion-matrix cell, and — when the errors-only switch is on — the set of
    misclassified positions. Active filters intersect; returns ``None`` when none are active.

    Lives here rather than in the app shell because every panel that honours the global selection
    (the word-importance chart in the shell, the single-word profile component) has to compose it
    identically — a panel with its own precedence rules would silently show a different subset than
    the table beside it.
    """
    combined = selected_indices
    if cell_indices is not None:
        cell_set = set(cell_indices)
        combined = list(cell_indices) if combined is None else [i for i in combined if i in cell_set]
    if errors is not None:
        combined = sorted(errors) if combined is None else [i for i in combined if i in errors]
    return combined


class WebappComponent(ABC):
    """Base class for a self-contained, registrable webapp panel.

    Subclasses set ``id``/``name``/``scope``/``requires`` and implement :meth:`layout` and
    :meth:`register_callbacks`. All Dash ids a component creates must be namespaced with its ``id``
    to avoid collisions.
    """

    id: str = "component"
    name: str = "Component"
    scope: str = "local"  # "global" | "local"
    requires: frozenset[str] = frozenset()

    @classmethod
    def is_available(cls, ctx: AppContext) -> bool:
        """Whether the component's ``requires`` are satisfied by *ctx*."""
        return cls.requires <= available_capabilities(ctx)

    @abstractmethod
    def layout(self, ctx: AppContext):
        """Return the Dash layout for this component.

        The whole context is passed because a component's initial UI may depend on more than the
        explanation — the counterfactual panel renders its config controls from the live generator's
        spec, and the scatter panel needs the coordinates.
        """

    @abstractmethod
    def register_callbacks(self, app, ctx: AppContext, stores: dict) -> None:
        """Register this component's Dash callbacks.

        Parameters
        ----------
        app : dash.Dash
            The Dash application.
        ctx : AppContext
            The explanation, engine and coordinates to read (never written to). Display state lives
            in the ``dcc.Store``s.
        stores : dict
            Shared ``dcc.Store`` ids the What-if Lab wires between components
            (e.g. ``{"apply": "whatif-apply-store"}``).
        """
