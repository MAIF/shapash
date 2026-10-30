"""Compare NLP backends' contributions on one text, at the array level: align, rescale, score agreement.

This module knows nothing about explanations; the layer that reads them is
:mod:`shapash.explainer.nlp_comparison`. Its first job is alignment, since backends do not agree on
what a unit is, even after each one merges subwords back into words:

- ``nlp_shap`` splits wherever a letter meets a non-letter: ``"didn" "'" "t"``, ``"great" "."``.
- ``nlp_captum_lig`` groups by the tokenizer's ``word_ids()``, i.e. its pre-tokenizer's pieces:
  ``"didn" "'" "t"`` on WordPiece, ``"didn" "'t"`` on byte-level BPE, ``"didn't"`` and ``"great."``
  on SentencePiece, which splits on whitespace only.
- ``nlp_lime`` (``bow=True``, its default) emits each *distinct* word once, punctuation dropped; one
  weight stands for every occurrence.

So neither position nor string identifies a unit across backends. Character spans do (see
:mod:`shapash.compute.spans`): every unit records the source characters it covers, and
:func:`align_units` reads all backends on the finest groups they can all be expressed in — the
spans of every unit, merged where they overlap. A group's value is the sum of the backend's units
inside it (each occurrence of a bag-of-words unit counts), and ``NaN`` when the backend has no unit
there: it did not attribute those characters (LIME and punctuation), which is different from
attributing them zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from shapash.compute.spans import Span, aggregate, association_matrix, merge_overlapping


def align_units(
    units: Mapping[str, tuple[Sequence[Sequence[Span]], np.ndarray]],
) -> tuple[list[Span], dict[str, np.ndarray]]:
    """Read several backends' unit values on one set of groups of the same text.

    Parameters
    ----------
    units : mapping of str to (sequence of spans per unit, np.ndarray)
        Backend name → its units' spans and their values, ``(n_units,)`` or
        ``(n_units, n_classes)``. A unit with no span cannot be placed and is left out.

    Returns
    -------
    tuple of (list of (int, int), dict of str to np.ndarray)
        The groups' spans, in text order, and per backend its values on them (``NaN`` where it has
        no unit). Groups are the units' spans merged where they overlap
        (:func:`~shapash.compute.spans.merge_overlapping`), so where every backend splits the text
        alike, a group is exactly one unit of each.
    """
    groups = merge_overlapping(span for spans, _ in units.values() for unit in spans for span in unit)
    aligned = {}
    for name, (spans, raw) in units.items():
        values = np.asarray(raw, dtype=float)
        if len(spans) != len(values):
            raise ValueError(f"{name!r}: {len(values)} value(s) for {len(spans)} unit(s).")
        aligned[name] = aggregate(association_matrix(groups, spans), values)
    return groups, aligned


def normalize_contributions(values: np.ndarray, method: str | None = "max_abs") -> np.ndarray:
    """Rescale one backend's aligned values so different backends share a scale.

    Backends explain different quantities — a probability, a logit, a local linear surrogate's
    weights — so their raw magnitudes are not comparable; only the *shape* of each attribution is.

    Parameters
    ----------
    values : np.ndarray
        1-D aligned contributions, ``NaN`` for unattributed units (kept as ``NaN``).
    method : {"max_abs", "sum_abs", None}
        ``"max_abs"`` divides by the largest magnitude (values land in ``[-1, 1]``, the strongest
        unit reads ±1); ``"sum_abs"`` divides by the total magnitude (each value is a signed share
        of the attribution mass); ``None`` leaves the raw values.

    Returns
    -------
    np.ndarray
    """
    values = np.asarray(values, dtype=float)
    if method is None:
        return values
    if method == "max_abs":
        scale = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 0.0
    elif method == "sum_abs":
        scale = np.nansum(np.abs(values))
    else:
        raise ValueError(f"normalize must be 'max_abs', 'sum_abs' or None, got {method!r}.")
    return values / scale if scale > 0 else values


# Columns of :func:`backend_agreement`, in order.
AGREEMENT_SCORES = [
    "backend_a",
    "backend_b",
    "spearman",
    "pearson",
    "cosine",
    "sign_agreement",
    "top_k_overlap",
    "top_share_a",
    "top_share_b",
    "coverage",
]


def backend_agreement(aligned: Mapping[str, np.ndarray], top_k: int = 5) -> pd.DataFrame:
    """Pairwise agreement between backends' aligned contributions on one text.

    Parameters
    ----------
    aligned : mapping of str to np.ndarray
        Backend name → 1-D contributions on the same reference units (``NaN`` = unattributed).
    top_k : int
        Size of the top set for the ``top_k_overlap`` column.

    Returns
    -------
    pd.DataFrame
        One row per unordered backend pair. Scores over the units both backends attributed:

        - ``spearman`` — rank correlation of the signed values. Every unit counts the same, so
          when one word carries most of the attribution, the order of the near-zero rest decides it.
        - ``pearson`` — linear correlation of the signed values: each unit counts by its size, but
          measured around each backend's mean, so a shift both backends share (every word slightly
          positive) does not count as agreement.
        - ``cosine`` — cosine similarity of the signed values, i.e. Pearson around zero rather than
          around the mean: zero is "no effect" for an attribution, so a shared shift does count.
          Each unit counts by its size, so it asks whether the two tell the same story on the
          words that matter.
        - ``sign_agreement`` — share of units where both signs match.
        - ``top_k_overlap`` — share of the ``top_k`` highest-magnitude units the two have in common.

        And, per backend, over the units it attributed: ``top_share_a`` / ``top_share_b``, the
        largest unit's share of the total magnitude — how concentrated the attribution is, which
        says how much to trust ``spearman`` against ``cosine``.

        And ``coverage``: the share of units both backends attributed, i.e. what the scores above
        stand on. Low coverage is expected against LIME on punctuation-heavy text; on two sequence
        backends it means they attribute different parts of the text (one truncated, say).

        All are scale-free, so they are safe across backends with different output spaces.
    """
    names = list(aligned)
    rows = []
    for a_pos, a in enumerate(names):
        for b in names[a_pos + 1 :]:
            va, vb = np.asarray(aligned[a], dtype=float), np.asarray(aligned[b], dtype=float)
            both = np.isfinite(va) & np.isfinite(vb)
            xa, xb = va[both], vb[both]
            # A correlation is undefined when either side is constant (e.g. LIME all zeros on a
            # class it did not select a word for); say so with NaN rather than a numpy warning.
            ranked = both.sum() > 1 and np.ptp(xa) > 0 and np.ptp(xb) > 0
            spearman = float(pd.Series(xa).corr(pd.Series(xb), method="spearman")) if ranked else np.nan
            pearson = float(np.corrcoef(xa, xb)[0, 1]) if ranked else np.nan
            norm = float(np.linalg.norm(xa) * np.linalg.norm(xb))
            cosine = float(xa @ xb) / norm if norm > 0 else np.nan
            signs = float(np.mean(np.sign(xa) == np.sign(xb))) if both.any() else np.nan
            k = min(top_k, int(both.sum()))
            if k > 0:
                top_a = set(np.argsort(-np.abs(xa), kind="stable")[:k])
                top_b = set(np.argsort(-np.abs(xb), kind="stable")[:k])
                overlap = len(top_a & top_b) / k
            else:
                overlap = np.nan
            rows.append(
                {
                    "backend_a": a,
                    "backend_b": b,
                    "spearman": spearman,
                    "pearson": pearson,
                    "cosine": cosine,
                    "sign_agreement": signs,
                    "top_k_overlap": overlap,
                    "top_share_a": _top_share(va),
                    "top_share_b": _top_share(vb),
                    "coverage": float(both.mean()) if len(both) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=AGREEMENT_SCORES)


def _top_share(values: np.ndarray) -> float:
    """The largest attributed unit's share of the total magnitude; ``NaN`` when there is none."""
    magnitude = np.abs(values[np.isfinite(values)])
    total = magnitude.sum()
    return float(magnitude.max() / total) if total > 0 else np.nan
