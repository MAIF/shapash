"""Counterfactual generator interface + declarative config spec.

Design mirrors both LIT's ``Generator`` (``generate`` + ``config_spec`` + compatibility check) and
the shapash ``NlpBackend`` ABC style. The tiny :class:`Field` spec types let a generator declare its
tunable parameters once; the webapp renders controls from that declaration automatically, so adding
a new knob or a new generator needs no UI changes.

The ABC also owns the **minimal-perturbation search** (:meth:`CounterfactualGenerator.search_minimal`)
that every generator ends in: try perturbation sets of increasing size, rebuild the text, re-predict,
keep the flips, never return a superset of a smaller success. Concrete generators differ only in
*which* positions they nominate and *what* they put there, so they supply a ranked candidate list and
a per-position replacement, and the search is written once — batched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from shapash.compute.generators.cf_utils import format_edit, is_prediction_flip
from shapash.model.base import SupportsTokenization, TextModel

#: Candidate texts scored per ``predict`` call when the model declares no batch size of its own.
DEFAULT_PREDICT_BATCH_SIZE = 32


@dataclass(frozen=True)
class Counterfactual:
    """One generated counterfactual for a single input text.

    Attributes
    ----------
    original_text : str
        The input text the counterfactual was derived from.
    new_text : str
        The perturbed text (reconstructed from the modified tokens).
    tokens : list[str]
        The original content tokens the generator operated on.
    flipped_positions : list[int]
        Positions (into ``tokens``) that were modified.
    substitutions : list[tuple[int, str, str]]
        ``(position, old_token, new_token)`` for each modification.
    orig_label : str
        Predicted label of ``original_text``.
    new_label : str
        Predicted label of ``new_text`` (differs from ``orig_label`` on a successful flip).
    orig_prob : float
        Probability of ``orig_label`` on the original text.
    new_prob : float
        Probability of ``new_label`` on the counterfactual text.
    prob_delta : float
        Drop in the original label's probability (``orig_prob`` minus its probability on the CF).

    Examples
    --------
    >>> import dataclasses, pandas as pd
    >>> res = xpl.generate_counterfactuals(text, generator="ablation_flip")
    >>> pd.DataFrame([dataclasses.asdict(cf) for cf in res])
    """

    original_text: str
    new_text: str
    tokens: list[str]
    flipped_positions: list[int]
    substitutions: list[tuple[int, str, str]]
    orig_label: str
    new_label: str
    orig_prob: float
    new_prob: float
    prob_delta: float

    def __repr__(self) -> str:
        """Concise, edit-first summary — the shape that matters when scanning results in a notebook.

        The default dataclass repr dumps every field (``tokens`` included), burying the one thing
        worth reading at a glance under noise, and leaves a removal's edit (``old_token, ""``) to be
        inferred from an empty string rather than stated. :func:`~.cf_utils.format_edit` spells out a
        substitution (``old→new``) and a removal (``−old``) the same way here and in the webapp's
        results table, so the two read consistently wherever a counterfactual is shown.
        """
        edits = ", ".join(format_edit(old, new) for _, old, new in self.substitutions) or "no edits"
        return (
            f'Counterfactual("{self.original_text}" → "{self.new_text}", {self.orig_label}→{self.new_label}, {edits})'
        )


@dataclass(frozen=True)
class Field:
    """Base declarative spec for a configurable generator parameter."""

    label: str
    default: Any


@dataclass(frozen=True)
class IntField(Field):
    """Integer parameter rendered as a bounded numeric control."""

    minimum: int = 1
    maximum: int = 10


@dataclass(frozen=True)
class TokenListField(Field):
    """List-of-strings parameter (e.g. tokens to ignore) rendered as a tag/multi-select input."""

    default: list[str] = field(default_factory=list)


class CounterfactualGenerator(ABC):
    """Abstract base for counterfactual generators.

    Concrete generators implement :meth:`config_spec`, :meth:`is_compatible` and :meth:`generate`.
    They depend on a :class:`~shapash.model.base.TextModel` through its declared capabilities, never
    on a concrete model class.
    """

    name: str = "counterfactual"
    #: Human-readable label for the webapp method selector (falls back to a title-cased ``name``).
    display_name: str = "Counterfactual"
    #: How many ranked positions a generator may offer :meth:`search_minimal`.
    #:
    #: Deliberately **not** a :meth:`config_spec` field. ``max_flips`` already is one, and the two
    #: multiply: the search scores ``sum(C(n, s) for s in 1..max_flips)`` combinations, so 10 positions
    #: at ``max_flips=5`` is 637 candidate texts while 30 positions is 174,436 — an unbounded wait in a
    #: blocking webapp callback, chosen by the person least placed to predict it. Overridable per
    #: subclass or per instance (``generator.max_candidate_positions = 20``) for deliberate experiments.
    max_candidate_positions: int = 10

    def __init__(self, model: TextModel) -> None:
        self.model = model

    @abstractmethod
    def config_spec(self) -> dict[str, Field]:
        """Return the tunable parameters as ``name -> Field`` (drives the webapp controls)."""

    @classmethod
    @abstractmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Return ``True`` when ``model`` exposes the capabilities this generator needs."""

    @abstractmethod
    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate counterfactuals for ``text``.

        Parameters
        ----------
        text : str
            Input text to perturb.
        target_label : str or None
            Desired label to flip *towards*. When ``None``, any label change counts as success.
        config : dict or None
            Overrides for :meth:`config_spec` defaults (keys match the spec).

        Returns
        -------
        list[Counterfactual]
            Successful counterfactuals, minimal and ordered by increasing number of flips.
        """

    def resolve_config(self, config: dict | None) -> dict:
        """Merge user ``config`` over the spec defaults into a plain value dict."""
        resolved = {name: fld.default for name, fld in self.config_spec().items()}
        if config:
            resolved.update({k: v for k, v in config.items() if k in resolved})
        return resolved

    @property
    def predict_batch_size(self) -> int:
        """Number of candidate texts to score per :meth:`~shapash.model.base.TextModel.predict` call.

        Read from the model, which is the only layer that knows its device and how much it can hold in
        one forward pass: a CPU adapter configured with a small ``batch_size`` keeps its small batches,
        a GPU one keeps its large ones, and neither is second-guessed here. Models that batch
        internally and expose nothing (e.g. a pipeline adapter) get
        :data:`DEFAULT_PREDICT_BATCH_SIZE`.
        """
        return max(1, int(getattr(self.model, "batch_size", DEFAULT_PREDICT_BATCH_SIZE)))

    def search_minimal(
        self,
        text: str,
        tokens: Sequence[str],
        candidates: Sequence[int],
        replacement: Callable[[int], str],
        *,
        max_size: int,
        num_examples: int,
        orig_probs: np.ndarray,
        target_class: int | None = None,
    ) -> list[Counterfactual]:
        """Search for minimal perturbation sets over ``candidates`` that flip the prediction.

        Tries every combination of 1..``max_size`` candidate positions, rebuilds the text by applying
        ``replacement`` at each chosen position, re-predicts, and keeps the combinations that flip.
        A combination that is a superset of an already-successful one is never evaluated, so results
        are minimal and ordered by increasing perturbation size.

        Candidates within one size are scored in **batches**, not one text per ``predict`` call. This
        is exact, not an approximation: two distinct combinations of the same size can never be
        supersets of one another, so nothing inside a size level can prune anything else inside it —
        only smaller sizes can, and those are already resolved when the level starts. Batches are
        capped at :attr:`predict_batch_size` rather than run per level, so ``num_examples`` still
        stops the search promptly instead of after a whole level has been scored.

        Parameters
        ----------
        text : str
            The original input text (recorded on each returned counterfactual).
        tokens : Sequence[str]
            Tokenization of ``text``; positions in ``candidates`` index into it.
        candidates : Sequence[int]
            Perturbable positions, best first. The caller is responsible for ranking and for
            capping how many are offered — the combination count grows steeply with this length.
        replacement : Callable[[int], str]
            The token to substitute at a given position. Returning ``""`` removes the token, which
            is how removal-based generators express an ablation.
        max_size : int
            Largest number of positions to perturb simultaneously.
        num_examples : int
            Stop once this many counterfactuals have been found.
        orig_probs : np.ndarray, shape (n_classes,)
            Class probabilities of ``text``, used as the flip reference.
        target_class : int or None
            When given, only a flip *to* that class counts as success.

        Returns
        -------
        list[Counterfactual]
            Minimal successful counterfactuals, ordered by increasing perturbation size.
        """
        model = self.model
        if not isinstance(model, SupportsTokenization):  # rebuilding perturbed text needs detokenize
            raise TypeError(f"{type(self).__name__} requires a model implementing SupportsTokenization.")
        label_names = model.label_names or []
        orig_class = int(np.argmax(orig_probs))
        orig_label = _label_at(label_names, orig_class)
        token_list = list(tokens)
        batch_size = self.predict_batch_size

        results: list[Counterfactual] = []
        successful: list[frozenset[int]] = []
        for size in range(1, max_size + 1):
            # Prune supersets of smaller successes up front — same-size combinations cannot prune
            # each other, so this is the complete pruning available at this level.
            combos = [c for c in combinations(candidates, size) if not any(prev <= set(c) for prev in successful)]
            for start in range(0, len(combos), batch_size):
                chunk = combos[start : start + batch_size]
                built = [_apply_replacements(token_list, combo, replacement) for combo in chunk]
                texts = [model.detokenize(new_tokens) for new_tokens, _ in built]
                cf_probs = model.predict(texts)
                for combo, new_text, (_, subs), probs in zip(chunk, texts, built, cf_probs, strict=True):
                    if not is_prediction_flip(orig_probs, probs, target_class):
                        continue
                    cf_class = int(np.argmax(probs))
                    results.append(
                        Counterfactual(
                            original_text=text,
                            new_text=new_text,
                            tokens=token_list,
                            flipped_positions=list(combo),
                            substitutions=subs,
                            orig_label=orig_label,
                            new_label=_label_at(label_names, cf_class),
                            orig_prob=float(orig_probs[orig_class]),
                            new_prob=float(probs[cf_class]),
                            prob_delta=float(orig_probs[orig_class] - probs[orig_class]),
                        )
                    )
                    successful.append(frozenset(combo))
                    if len(results) >= num_examples:
                        return results
        return results


def _label_at(label_names: list[str], index: int) -> str:
    """Return ``label_names[index]``, falling back to the bare index for an unnamed class."""
    return label_names[index] if index < len(label_names) else str(index)


def _apply_replacements(
    tokens: list[str], combo: Sequence[int], replacement: Callable[[int], str]
) -> tuple[list[str], list[tuple[int, str, str]]]:
    """Apply ``replacement`` at every position of ``combo``, returning ``(new_tokens, substitutions)``.

    An empty replacement drops the token instead of substituting it, which is what makes one search
    serve both substitution-based and removal-based generators.
    """
    chosen = set(combo)
    new_tokens: list[str] = []
    subs: list[tuple[int, str, str]] = []
    for i, token in enumerate(tokens):
        if i not in chosen:
            new_tokens.append(token)
            continue
        new_token = replacement(i)
        subs.append((i, token, new_token))
        if new_token:
            new_tokens.append(new_token)
    return new_tokens, subs
