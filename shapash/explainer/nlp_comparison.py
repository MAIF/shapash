"""Put several :class:`~shapash.explainer.nlp_explanation.NlpExplanation` objects of the same texts side by side.

The array-level work — aligning word units, rescaling, scoring agreement — lives in
:mod:`shapash.compute.backend_comparison`, which knows nothing about explanations. This module is the
layer above it: it names the explanations, checks they explain the same texts and classes, warns
when they are in different output spaces, and reads one text's values out of each.
:meth:`NlpPlotter.compare <shapash.explainer.nlp_plotter.NlpPlotter.compare>` uses it for one
text, :func:`corpus_agreement` for many.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from shapash.compute.backend_comparison import AGREEMENT_SCORES, align_units, backend_agreement
from shapash.compute.spans import UnitSpans, locate_all, locate_spans
from shapash.explainer.nlp_explanation import select_label_column

if TYPE_CHECKING:
    from shapash.explainer.nlp_explanation import NlpExplanation

AGREEMENT_COLUMNS = ["row", "text", "label", *AGREEMENT_SCORES]


def name_explanations(
    reference: NlpExplanation,
    others: NlpExplanation | Sequence[NlpExplanation] | Mapping[str, NlpExplanation],
) -> dict[str, NlpExplanation]:
    """Reference first, then ``others``, under unique display labels.

    A default label is the backend name without its ``nlp_`` prefix; two explanations from the
    same backend are told apart by ``output_space`` first (SHAP in probability vs logit space),
    then by a counter. A label given in a mapping is kept.
    """
    if isinstance(others, Mapping):
        pairs: list[tuple[str | None, NlpExplanation]] = [(str(k), v) for k, v in others.items()]
    elif isinstance(others, Sequence):
        pairs = [(None, o) for o in others]
    else:
        pairs = [(None, others)]
    pairs.insert(0, (None, reference))

    defaults = [o.backend_name.removeprefix("nlp_") for _, o in pairs]
    # Only entries left to their default label can clash by backend: a caller-chosen label is kept.
    unnamed = [d for (given, _), d in zip(pairs, defaults, strict=True) if given is None]
    named: dict[str, NlpExplanation] = {}
    for (given, other), default in zip(pairs, defaults, strict=True):
        if given is not None:
            label = given
        elif unnamed.count(default) > 1:
            label = f"{default} ({other.output_space})"
        else:
            label = default
        base, n = label, 2
        while label in named:
            label, n = f"{base} #{n}", n + 1
        named[label] = other
    return named


def check_comparable(named: Mapping[str, NlpExplanation], stacklevel: int = 2) -> None:
    """Raise if the explanations explain different classes; warn on different output spaces or missing spans.

    Parameters
    ----------
    named : mapping of str to NlpExplanation
        Reference first, as returned by :func:`name_explanations`.
    stacklevel : int
        Passed to :func:`warnings.warn`, so the warning points at the caller's line.
    """
    ref_name, reference = next(iter(named.items()))
    for name, other in named.items():
        if other.n_classes != reference.n_classes or (
            other.label_names and reference.label_names and list(other.label_names) != list(reference.label_names)
        ):
            raise ValueError(f"{name!r} explains different classes than the reference explanation {ref_name!r}.")

    spaces = {name: other.output_space for name, other in named.items()}
    if len(set(spaces.values())) > 1:
        listing = ", ".join(f"{name}={space}" for name, space in spaces.items())
        warnings.warn(
            f"Comparing backends across output spaces ({listing}). Normalisation removes the scale gap "
            "but not the softmax non-linearity, so rankings and agreement scores mix method and space "
            "differences. For a like-for-like comparison with a logit-space backend, use "
            "NlpShapBackend(..., output_space='logit').",
            UserWarning,
            stacklevel=stacklevel + 1,
        )

    unplaced = [name for name, other in named.items() if other.token_spans is None]
    if unplaced:
        warnings.warn(
            f"{', '.join(map(repr, unplaced))} carry no character spans (saved by an older shapash), so "
            "their units are located by searching the text for their strings. That misplaces words a "
            "tokenizer rewrote (stripped accents, [UNK], normalised characters). Re-run explain() for "
            "an exact alignment.",
            UserWarning,
            stacklevel=stacklevel + 1,
        )


def predicted_label_idx(explanation: NlpExplanation, row: int) -> int:
    """Column index of the class ``explanation`` predicted for ``row`` (0 when it cannot be resolved)."""
    return explanation.label_to_idx.get(str(explanation.y_pred.iloc[row]), 0)


def unit_spans(explanation: NlpExplanation, row: int) -> list[UnitSpans]:
    """Each unit's character spans in ``row``'s text.

    Read off :attr:`~shapash.explainer.nlp_explanation.NlpExplanation.token_spans`. An artifact saved
    before units carried spans gets them by search instead: every occurrence of each word for LIME's
    bag of words, successive occurrences otherwise — approximate, which :func:`check_comparable` warns
    about.
    """
    if explanation.token_spans is not None:
        return list(explanation.token_spans[row])
    text, tokens = str(explanation.texts.iloc[row]), explanation.token_strings[row]
    return locate_all(text, tokens) if explanation.backend_name == "nlp_lime" else locate_spans(text, tokens)


def align_row(named: Mapping[str, NlpExplanation], row: int, label_idx: int) -> tuple[list[str], dict[str, np.ndarray]]:
    """One text's contributions for one class, every backend read on the same units.

    Parameters
    ----------
    named : mapping of str to NlpExplanation
        Reference first, as returned by :func:`name_explanations`.
    row : int
        Positional index of the text; it must be the same text in every explanation.
    label_idx : int
        Output column to read.

    Returns
    -------
    tuple of (list of str, dict of str to np.ndarray)
        The units as written in the text, and per label the aligned 1-D values (``NaN`` = not
        attributed). Units are every backend's units merged where they overlap
        (:func:`~shapash.compute.backend_comparison.align_units`): the reference's words wherever no
        other backend groups the text more coarsely.
    """
    reference = next(iter(named.values()))
    if not -len(reference) <= row < len(reference):
        raise IndexError(f"row={row} is out of range for {len(reference)} sample(s).")
    text = str(reference.texts.iloc[row])

    units = {}
    for name, other in named.items():
        if other is not reference and (len(other) != len(reference) or str(other.texts.iloc[row]) != text):
            raise ValueError(f"Row {row} holds a different text in {name!r} than in the reference explanation.")
        units[name] = (unit_spans(other, row), select_label_column(other.values[row], label_idx))
    groups, aligned = align_units(units)
    return [text[start:end] for start, end in groups], aligned


def corpus_agreement(
    reference: NlpExplanation,
    others: NlpExplanation | Sequence[NlpExplanation] | Mapping[str, NlpExplanation],
    rows: Sequence[int] | None = None,
    label_idx: int | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """Agreement between backends on every text of a corpus (or a sample of it), one line per text and pair.

    The per-text scores are those :meth:`NlpPlotter.compare
    <shapash.explainer.nlp_plotter.NlpPlotter.compare>` shows in its subtitle; collecting them over
    many texts shows how far two backends agree *in general* on this model and data, and which texts
    they disagree on most — the ones worth opening with ``compare(row=...)``. Nothing is recomputed:
    every explanation must already hold the texts.

    Parameters
    ----------
    reference : NlpExplanation
        The explanation the others are named and checked against; units are shared by all (see
        :func:`align_row`).
    others : NlpExplanation, sequence of NlpExplanation, or mapping of str to NlpExplanation
        Explanations of the same texts by other backends. Labels as in ``compare``.
    rows : sequence of int, optional
        Positional indices of the texts to score. ``None`` (default) scores every text.
    label_idx : int, optional
        Class to compare on. ``None`` (default) uses each text's predicted class (by the reference).
    top_k : int
        Size of the top set in ``top_k_overlap``.

    Returns
    -------
    pd.DataFrame
        Columns ``row``, ``text``, ``label`` (the class compared), ``backend_a``, ``backend_b``,
        ``spearman``, ``pearson``, ``cosine``, ``sign_agreement``, ``top_k_overlap``, ``top_share_a``,
        ``top_share_b`` — see :func:`~shapash.compute.backend_comparison.backend_agreement`. A low
        ``spearman`` with a high ``cosine`` and a high top share means the backends agree on the
        word that matters and only order the small rest differently. Every pair of backends is scored,
        not only pairs with the reference. Aggregate with ``groupby(["backend_a", "backend_b"])``.

    Warns
    -----
    UserWarning
        When the explanations are in different output spaces (see ``compare``).

    Examples
    --------
    >>> scores = corpus_agreement(shap_exp, {"LIG": lig_exp}, rows=range(50))
    >>> scores.groupby(["backend_a", "backend_b"])[["spearman", "top_k_overlap"]].median()
    >>> scores.nsmallest(5, "spearman")[["row", "text", "spearman"]]  # where they disagree most
    """
    named = name_explanations(reference, others)
    check_comparable(named, stacklevel=2)
    return score_rows(named, rows=rows, label_idx=label_idx, top_k=top_k)


def score_rows(
    named: Mapping[str, NlpExplanation],
    rows: Sequence[int] | None = None,
    label_idx: int | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """The body of :func:`corpus_agreement`, on explanations already named and checked."""
    reference = next(iter(named.values()))
    rows = range(len(reference)) if rows is None else rows
    names = reference.label_names

    frames = []
    for row in rows:
        idx = predicted_label_idx(reference, row) if label_idx is None else label_idx % reference.n_classes
        _, aligned = align_row(named, row, idx)
        scores = backend_agreement(aligned, top_k=top_k)
        scores.insert(0, "label", names[idx] if names is not None and idx < len(names) else idx)
        scores.insert(0, "text", str(reference.texts.iloc[row]))
        scores.insert(0, "row", int(row))
        frames.append(scores)
    if not frames:
        return pd.DataFrame(columns=AGREEMENT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[AGREEMENT_COLUMNS]
