"""NLP backend infrastructure — shared base for text explainability backends.

``NlpContributions`` is the single typed payload every concrete ``run_explainer``
returns: per-sample token strings, contribution values, and baseline predictions —
no batch metadata (predictions, ground truth, class names) attached. It carries no
behavior either; word-importance aggregation and persistence live on
:class:`~shapash.explainer.nlp_explanation.NlpExplanation`, the batch artifact that
wraps this data together with everything else ``NlpExplainer.explain()`` knows.
The same slim type is also what :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.explain_text`
returns for a single live (uncommitted) text — a one-sample batch.

``NlpBackend`` is the abstract base class that owns the common ``__init__``
skeleton and the ``get_local_contributions`` implementation.  Concrete
subclasses (``NlpShapBackend``, ``NlpLimeBackend``) only need to implement
``run_explainer``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from shapash._optional import import_optional_module
from shapash.backend.backend import Backend
from shapash.model.base import has_capabilities

# A unit made entirely of punctuation/symbol characters — no letters, digits or underscore. Word
# segmentation deliberately emits these as their own units (the model reads them, and "!!!" or "?"
# genuinely moves a sentiment prediction), so they stay visible in a local highlight. They are noise
# in a *corpus-level* ranking, though, which is why ``NlpExplanation.word_importance`` filters them
# by default.
_PUNCTUATION_RE = re.compile(r"^[^\w\s]+$")


def is_punctuation(word: str) -> bool:
    """True when ``word`` consists solely of punctuation or symbol characters."""
    stripped = word.strip()
    return bool(stripped) and bool(_PUNCTUATION_RE.match(stripped))


@dataclass
class NlpContributions:
    """Raw per-sample token contributions — a backend's output, no batch metadata attached.

    Attributes
    ----------
    token_strings : list[list[str]]
        Word (or LIME-vocabulary) strings per sample, variable length.
    values : list[np.ndarray]
        Per-token contribution values, one array per sample.
        Each array has shape ``(n_tokens_i, n_classes)`` for multi-class models
        or ``(n_tokens_i,)`` for binary/regression.
    base_values : np.ndarray or None
        Baseline prediction for each sample, shape ``(n_samples, n_classes)``
        or ``(n_samples,)``. ``None`` when the backend has no reference at all
        (see :attr:`NlpBackend.reference_kind`).
    token_spans : list[list[tuple[tuple[int, int], ...]]] or None
        Per sample, per unit: the ``(start, end)`` character spans of the source text the unit
        stands for — one for a word, one per occurrence for a bag-of-words unit, none when the
        backend could not place it. The coordinate backends share (see
        :mod:`shapash.compute.spans`). ``None`` when the backend records no spans at all.
    """

    token_strings: list[list[str]]
    values: list[np.ndarray]
    base_values: np.ndarray | None
    token_spans: list[list[tuple[tuple[int, int], ...]]] | None = None


class NlpBackend(Backend):
    """Abstract base class for NLP explainability backends.

    Owns the ``__init__`` skeleton shared by all text backends (including the
    ``requires_model_capabilities`` check) and the ``get_local_contributions``
    implementation.  Concrete subclasses must implement ``run_explainer`` and
    return an ``NlpContributions``, and must declare ``reference_kind``,
    ``is_additive`` and ``output_space`` below.

    Class Attributes
    -----------------
    reference_kind : {"distribution", "statistics", "point", "none"}
        What kind of reference, if any, this method's numbers are measured against —
        not every method has a baseline, so this is a four-valued question, not a
        boolean. ``"distribution"``: rows the model is compared against (a genuine
        baseline, e.g. SHAP kernel/permutation on tabular data). ``"statistics"``:
        summaries used to perturb/discretize, never subtracted (e.g. LIME tabular's
        discretizer). ``"point"``: one constructed input built by the model/tokenizer,
        not learned from data (e.g. Captum LIG's pad/mask reference ids). ``"none"``:
        no reference exists, or none is learned from data (e.g. a masker that infers
        itself, or a pointwise gradient method).
    is_additive : bool
        Whether this method's contributions satisfy an efficiency/completeness axiom —
        i.e. sum to a well-defined total (the prediction gap for Shapley-family methods,
        ``logits(x) - logits(baseline)`` for Integrated Gradients). Licenses feature
        grouping and waterfall/force-style charts; ``False`` for LIME, whose local
        surrogate coefficients carry no such guarantee.
    output_space : {"probability", "logit"}
        Which model output the contributions explain — read off the backend *instance* at
        ``explain()`` time and recorded on :class:`~shapash.explainer.nlp_explanation.NlpExplanation`
        so a saved artifact says which. ``"probability"``: the explained quantity is a
        (softmax) probability, which forces per-token cross-class cancellation. ``"logit"``: the
        explained quantity is the model's raw pre-softmax output, which does not cancel. Not an
        affine rescaling of one another; a caller comparing two backends is comparing numbers on
        different scales unless both report the same space. The class attribute is the default; a
        backend whose space is a constructor choice (``NlpShapBackend``) overrides it per instance,
        which is why it also enters ``NlpExplainer``'s cache key.
    is_signed : bool
        Whether a contribution's sign carries meaning (pushes toward / away from the class). ``True``
        for every attribution method so far; ``False`` for a magnitude-only method such as unsigned
        saliency, whose values the diverging colour scales and "pushes toward" wording would misread.
        Defaults to ``True``.
    baseline_token : str or None
        Instance attribute: the token string a perturbation/path method substitutes for "word absent"
        (``"[MASK]"``, ``"[PAD]"``, ``"..."``), or ``None`` when the method has no single one (LIME
        removes words) or it is unknown. The baseline input — every content token replaced by it — is
        what ``base_values`` is the model's output on, for SHAP (the fully masked text) and LIG (the
        integration start) alike. Two backends measured against different baselines answer different
        questions even in the same ``output_space``, so it is recorded and keyed too.
    requires_model_capabilities : tuple[type, ...]
        Capability ABCs (from :mod:`shapash.model.base`) the bound model must satisfy,
        checked once here so a new backend declares its needs instead of failing at
        first call. Empty when the backend only needs a plain scoring callable.

    Parameters
    ----------
    model : callable
        Text model or pipeline accepted by the concrete backend.
    preprocessing : None
        Unused; accepted only for signature compatibility with the concrete
        subclasses' ``super().__init__`` calls.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    explainer_args : dict, optional
        Keyword arguments forwarded to the underlying explainer constructor.
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to the explainer call / ``explain_instance``.

    Raises
    ------
    TypeError
        If ``model`` does not satisfy every capability in ``requires_model_capabilities``.
    """

    reference_kind: ClassVar[Literal["distribution", "statistics", "point", "none"]]
    is_additive: ClassVar[bool]
    # Not a ClassVar: the class value is the default, an instance may override it (see above).
    output_space: Literal["probability", "logit"]
    is_signed: ClassVar[bool] = True
    baseline_token: str | None = None
    requires_model_capabilities: ClassVar[tuple[type, ...]] = ()

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        required = type(self).requires_model_capabilities
        if required and not has_capabilities(model, *required):
            names = ", ".join(c.__name__ for c in required)
            raise TypeError(
                f"{type(self).__name__} requires a model implementing {names}; "
                "got a model without the required attribution surface."
            )
        self.model = model
        self._classes = label_names or []
        self.explainer_args = explainer_args or {}
        self.explainer_compute_args = explainer_compute_args or {}

    def get_local_contributions(
        self, x, explain_data: NlpContributions, subset: list[int] | None = None
    ) -> NlpContributions:
        """Optionally select a subset of ``run_explainer``'s output.
        TODO: dead-code, not used by any current NLP backend.  Keep for now to avoid breaking the interface.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples (not used here; kept for interface compatibility).
        explain_data : NlpContributions
            The output of ``run_explainer``.
        subset : list[int], optional
            Positional indices to select a subset of samples. Returns ``explain_data`` unchanged
            when ``None`` (the default).

        Returns
        -------
        NlpContributions
            Token strings, contribution values, and baseline predictions for
            each sample.
        """
        if subset is None:
            return explain_data
        base_values = explain_data.base_values
        return NlpContributions(
            token_strings=[explain_data.token_strings[i] for i in subset],
            values=[explain_data.values[i] for i in subset],
            base_values=None if base_values is None else base_values[subset],
            token_spans=None if explain_data.token_spans is None else [explain_data.token_spans[i] for i in subset],
        )

    def _progress_iter(self, items: list[str]) -> Iterable[str]:
        """Wrap ``items`` in a ``tqdm`` bar when the backend's ``show_progress`` is set, else return it unchanged.

        Shared by the backends that loop over texts one at a time (LIG, LIME). Best-effort and
        dependency-free: ``tqdm`` is imported with ``errors="ignore"`` so a missing install simply
        yields the plain list rather than raising. Read with ``getattr`` so a backend without the
        option (or one built via ``object.__new__``) runs silently.
        """
        if not getattr(self, "show_progress", False):
            return items
        tqdm_mod = import_optional_module("tqdm", errors="ignore")
        if tqdm_mod is None:
            return items
        return tqdm_mod.tqdm(items, desc=f"{self.name} attribution", unit="text")
