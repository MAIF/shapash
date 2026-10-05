"""Compare NLP backends' contributions on one text, at the array level: align, rescale, score agreement.

This module knows nothing about explanations; the layer that reads them is
:mod:`shapash.explainer.nlp_comparison`. Its first job is alignment, since backends do not
agree on what a "token" is, even after each one merges subwords back into words:

- ``nlp_shap`` and ``nlp_captum_lig`` emit words in sentence order, punctuation as its own unit —
  but LIG reads them back off the tokenizer, so an uncased model hands back ``"the"`` where SHAP
  kept ``"The"``.
- ``nlp_lime`` (``bow=True``, its default) emits each *distinct* word once, in first-appearance
  order, case-sensitive, with punctuation dropped by its ``split_expression``. A word that occurs
  twice gets one weight that stands for every occurrence.

A positional ``zip`` across backends is therefore wrong as soon as LIME is involved. This module
locates every backend's tokens in the source text to get character spans, then reads each
reference unit's value off the other backend in three steps:

1. **span** — the other backend's token that starts inside the reference unit's span;
2. **exact word** — otherwise, the other backend's token with the same string (this is how a
   repeated word reaches LIME's single bag-of-words weight);
3. **case-folded word** — otherwise, the same match ignoring case.

A unit none of these resolve is ``NaN``: the backend did not attribute it (LIME and punctuation),
which is different from attributing it zero.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def locate_tokens(text: str, tokens: Sequence[str]) -> list[tuple[int, int] | None]:
    """Character span of each token in ``text``, searched left to right, ignoring case.

    Each search starts where the previous match ended, so a repeated word maps to its successive
    occurrences. A token that cannot be found (special tokens, normalisation the tokenizer applied)
    gets ``None`` and leaves the cursor where it was, so one miss does not shift every later token.

    Parameters
    ----------
    text : str
        The source text the tokens were produced from.
    tokens : sequence of str
        Token strings in the order the backend emitted them.

    Returns
    -------
    list of (int, int) or None
        ``(start, end)`` per token, ``None`` where the token was not found.
    """
    haystack = text.casefold()
    cursor = 0
    spans: list[tuple[int, int] | None] = []
    for token in tokens:
        needle = str(token).strip().casefold()
        start = _find_word(haystack, needle, cursor) if needle else -1
        if start < 0:
            spans.append(None)
            continue
        end = start + len(needle)
        spans.append((start, end))
        cursor = end
    return spans


def _find_word(haystack: str, needle: str, cursor: int) -> int:
    """First occurrence of ``needle`` at or after ``cursor`` that is not part of a longer word.

    A plain substring search would place LIME's ``"i"`` inside ``"didn"``. Word-boundary guards
    apply only on the sides where ``needle`` itself starts/ends with a word character, so
    punctuation tokens and contraction pieces (``"'"``, ``"t"`` after ``"didn'"``) still match.
    Falls back to the plain substring search for scripts written without spaces (CJK), where no
    boundary exists to guard.
    """
    left = r"(?<!\w)" if re.match(r"\w", needle) else ""
    right = r"(?!\w)" if re.search(r"\w$", needle) else ""
    found = re.compile(left + re.escape(needle) + right).search(haystack, cursor)
    return found.start() if found else haystack.find(needle, cursor)


def align_token_values(
    text: str,
    reference_tokens: Sequence[str],
    tokens: Sequence[str],
    values: np.ndarray,
) -> np.ndarray:
    """Project one backend's per-token ``values`` onto ``reference_tokens``.

    See the module docstring for the three matching steps.

    Parameters
    ----------
    text : str
        The source text both tokenizations come from.
    reference_tokens : sequence of str
        The units to align onto, in sentence order.
    tokens : sequence of str
        The other backend's tokens.
    values : np.ndarray
        The other backend's 1-D contributions, one per entry of ``tokens``.

    Returns
    -------
    np.ndarray
        Shape ``(len(reference_tokens),)``. ``NaN`` where the other backend attributed nothing.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != len(tokens):
        raise ValueError(f"values must be 1-D with one entry per token, got shape {values.shape} for {len(tokens)}.")

    ref_spans = locate_tokens(text, reference_tokens)
    spans = locate_tokens(text, tokens)

    by_start = {span[0]: i for i, span in enumerate(spans) if span is not None}
    exact: dict[str, int] = {}
    folded: dict[str, int] = {}
    for i, token in enumerate(tokens):
        exact.setdefault(str(token), i)
        folded.setdefault(str(token).casefold(), i)

    out = np.full(len(reference_tokens), np.nan)
    for j, (ref_token, ref_span) in enumerate(zip(reference_tokens, ref_spans, strict=True)):
        match = None
        if ref_span is not None:
            match = next((by_start[p] for p in range(*ref_span) if p in by_start), None)
        if match is None:
            match = exact.get(str(ref_token), folded.get(str(ref_token).casefold()))
        if match is not None:
            out[j] = values[match]
    return out


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
                }
            )
    return pd.DataFrame(rows, columns=AGREEMENT_SCORES)


def _top_share(values: np.ndarray) -> float:
    """The largest attributed unit's share of the total magnitude; ``NaN`` when there is none."""
    magnitude = np.abs(values[np.isfinite(values)])
    total = magnitude.sum()
    return float(magnitude.max() / total) if total > 0 else np.nan
