"""Character spans — the coordinate every NLP backend's units share.

Backends do not agree on what a unit is: SHAP merges its masker's segments into words, LIG groups
subwords by the tokenizer's ``word_ids()``, LIME splits on a regex and keeps one weight per distinct
word. They do not share a token axis either — SHAP segments the source string, LIME never sees a
tokenizer. What they all can say is *which characters of the source text a unit stands for*, so
that is the coordinate: each unit carries the ``(start, end)`` spans it covers, as Python string
indices into the text the backend explained.

A unit has **one or more** spans: one for a word in a sequence backend, one per occurrence for a
bag-of-words unit (LIME's single weight for every occurrence of a word). A unit with no span is
one the backend could not place (only possible on a legacy, string-located artifact).

This module is the array-level vocabulary on top of that coordinate: build spans
(:func:`span_from_offsets`, and the legacy :func:`locate_spans` / :func:`locate_all`), partition a
text into groups every unit fits in (:func:`merge_overlapping`), and aggregate unit values onto
groups (:func:`association_matrix`, :func:`aggregate`). Cross-backend comparison is one use;
reading explanations at word or sentence granularity is another.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

import numpy as np

Span = tuple[int, int]
UnitSpans = tuple[Span, ...]

# Scripts written without inter-word spaces: a word-boundary test sees a whole sentence as one
# word, so neither a boundary-guarded search nor a word/non-word split applies to them. Hangul is
# deliberately absent: Korean *is* space-segmented.
_UNSEGMENTED_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR")


def is_unsegmented_script(char: str) -> bool:
    """True when ``char`` belongs to a script written without spaces between words."""
    return unicodedata.name(char, "").startswith(_UNSEGMENTED_SCRIPTS)


def span_from_offsets(offsets: Sequence[Sequence[int] | None], positions: Iterable[int]) -> Span | None:
    """The character span covered by the tokens at ``positions``, from a tokenizer's offset mapping.

    Empty offsets (``start == end``: special tokens, and the zero-width offsets some tokenizers give
    a bare word-start marker) cover no character and are skipped.

    Parameters
    ----------
    offsets : sequence of (int, int) or None
        ``offset_mapping`` of one encoded text, one entry per token.
    positions : iterable of int
        The token indices composing the unit.

    Returns
    -------
    (int, int) or None
        ``(min start, max end)`` over the covering tokens; ``None`` when none covers a character.
    """
    covered: list[Span] = []
    for i in positions:
        offset = offsets[i]
        if offset is not None and offset[1] > offset[0]:
            covered.append((int(offset[0]), int(offset[1])))
    if not covered:
        return None
    return min(start for start, _ in covered), max(end for _, end in covered)


def trim_span(text: str, span: Span | None) -> Span | None:
    """``span`` shrunk to the visible characters of ``text`` it covers; ``None`` when there are none.

    Tokenizers disagree on whether an offset includes the space before a word: most trim it, some
    byte-level BPE tokenizers (ModernBERT) keep it, so ``" didn"`` would cover ``(1, 6)`` where
    another backend's ``"didn"`` covers ``(2, 6)``. Trimming every span makes units comparable and
    their text clean, and a unit covering only whitespace (a lone ``Ġ``/``Ċ``) has no span at all.
    """
    if span is None:
        return None
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def merge_overlapping(spans: Iterable[Span]) -> list[Span]:
    """Partition the covered text into the smallest groups that every input span fits inside.

    Spans that overlap — directly or through a chain of others — end up in one group; spans that
    merely touch (``a.end == b.start``) do not. Applied to several backends' units at once, this is
    the finest common unit they can all be read on: where they split a text alike, the groups are
    their units; where one backend's unit spans several of another's (``"didn't"`` against
    ``"didn" "'" "t"``), the group is the larger one.

    Parameters
    ----------
    spans : iterable of (int, int)
        Character spans, in any order. Empty spans are ignored.

    Returns
    -------
    list of (int, int)
        Disjoint groups, in text order.
    """
    groups: list[Span] = []
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        if groups and start < groups[-1][1]:
            groups[-1] = (groups[-1][0], max(groups[-1][1], end))
        else:
            groups.append((start, end))
    return groups


def association_matrix(groups: Sequence[Span], units: Sequence[Sequence[Span]]) -> np.ndarray:
    """How many of each unit's spans fall in each group: shape ``(n_groups, n_units)``.

    A span belongs to the group that contains its start. With groups from :func:`merge_overlapping`
    over the same spans it is always wholly inside that group; a coarser partition built otherwise
    (sentences) assigns a span straddling two groups to the first. A bag-of-words unit counts once
    per occurrence, so its single weight reaches every place the word appears.

    Parameters
    ----------
    groups : sequence of (int, int)
        Disjoint spans, in text order.
    units : sequence of sequence of (int, int)
        Each unit's spans.

    Returns
    -------
    np.ndarray
        Integer counts.
    """
    starts = np.array([g[0] for g in groups], dtype=int)
    ends = np.array([g[1] for g in groups], dtype=int)
    matrix = np.zeros((len(groups), len(units)), dtype=int)
    for j, unit in enumerate(units):
        for start, end in unit:
            if end <= start:
                continue
            i = int(np.searchsorted(starts, start, side="right")) - 1
            if i >= 0 and start < ends[i]:
                matrix[i, j] += 1
    return matrix


def aggregate(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each group's value: the sum of its units' values, ``NaN`` where no unit falls in it.

    A sum, because a group of an additive backend's units then keeps the additive guarantee (its
    value is the attribution of all of them together). ``NaN`` keeps "no unit here" apart from
    "units here that sum to zero".

    Parameters
    ----------
    matrix : np.ndarray
        ``(n_groups, n_units)``, from :func:`association_matrix`.
    values : np.ndarray
        ``(n_units,)`` or ``(n_units, n_classes)``.

    Returns
    -------
    np.ndarray
        ``(n_groups,)`` or ``(n_groups, n_classes)``.
    """
    values = np.asarray(values, dtype=float)
    out = matrix.astype(float) @ values
    empty = matrix.sum(axis=1) == 0
    out[empty] = np.nan
    return out


# --- Legacy: spans recovered from strings ------------------------------------------------------
#
# An artifact saved before units carried spans holds strings only. These recover spans by search,
# which is what alignment used to rely on everywhere. It is approximate by nature — a tokenizer's
# string is not always a substring of the text (stripped accents, ``[UNK]``, NFKC-normalised
# characters), and a miss cannot be told from a word that is not there — so it is only ever a
# fallback for such files.


def locate_spans(text: str, tokens: Sequence[str]) -> list[UnitSpans]:
    """Each token's span in ``text``, searched left to right, ignoring case.

    Each search starts where the previous match ended, so a repeated word maps to its successive
    occurrences. A token that cannot be found gets no span and leaves the cursor where it was.

    Parameters
    ----------
    text : str
        The source text the tokens were produced from.
    tokens : sequence of str
        Token strings in the order the backend emitted them.

    Returns
    -------
    list of tuple of (int, int)
        One span per located token, ``()`` for one not found.
    """
    # Offsets found in the folded text must be offsets in ``text``, so the fold must keep its length:
    # ``casefold`` does not always ("ß" -> "ss"), and then ``lower``, then no folding, stands in.
    fold = next(
        (f for f in (str.casefold, str.lower) if len(f(text)) == len(text)),
        lambda s: s,
    )
    haystack = fold(text)
    cursor = 0
    spans: list[UnitSpans] = []
    for token in tokens:
        needle = fold(str(token).strip())
        start = _find_word(haystack, needle, cursor) if needle else -1
        if start < 0:
            spans.append(())
            continue
        end = start + len(needle)
        spans.append(((start, end),))
        cursor = end
    return spans


def locate_all(text: str, tokens: Sequence[str]) -> list[UnitSpans]:
    """Every whole-word occurrence of each token in ``text``, case-sensitive — for a bag of words.

    Parameters
    ----------
    text : str
        The source text.
    tokens : sequence of str
        Distinct words, each standing for all its occurrences.

    Returns
    -------
    list of tuple of (int, int)
        All spans of each token, ``()`` where it does not occur.
    """
    out: list[UnitSpans] = []
    for token in tokens:
        needle = str(token).strip()
        if not needle:
            out.append(())
            continue
        found = tuple((m.start(), m.end()) for m in _word_pattern(needle).finditer(text))
        out.append(found)
    return out


def _word_pattern(needle: str) -> re.Pattern:
    """``needle`` not glued to a longer word on a side where it begins/ends with a word character."""
    left = r"(?<!\w)" if re.match(r"\w", needle) else ""
    right = r"(?!\w)" if re.search(r"\w$", needle) else ""
    return re.compile(left + re.escape(needle) + right)


def _find_word(haystack: str, needle: str, cursor: int) -> int:
    """First occurrence of ``needle`` at or after ``cursor`` that is not part of a longer word.

    A plain substring search would place ``"i"`` inside ``"didn"``, and once one token lands too far
    right every token after it is searched past its true place. The plain search is kept only for
    scripts written without spaces, where there is no boundary to guard.
    """
    found = _word_pattern(needle).search(haystack, cursor)
    if found:
        return found.start()
    if any(is_unsegmented_script(c) for c in needle):
        return haystack.find(needle, cursor)
    return -1
