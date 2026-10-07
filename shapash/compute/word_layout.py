"""Words of an encoded text: which tokens make each word, and where each word sits in the text.

A sequence backend explains a text on the model's own encoding and reports words. The tokenizer
says which tokens belong to which word (its ``word_ids()``, through
:meth:`~shapash.model.base.SupportsCaptumIG.word_alignment`) and which characters each token covers
(its offset mapping). :func:`word_layout` turns the two into a :class:`WordLayout`, and
:meth:`WordLayout.aggregate` sums per-token or per-piece values into words. The LIG and SHAP
backends both build their words this way, so they report the same units.

Three rules keep a word's value additive and its string as written:

- a character belongs to one word and one piece. Tokens sharing a character (a byte-level BPE emoji
  or ``é`` split across bytes, a SentencePiece ``▁`` given the next character's offsets) are one
  piece, and a word whose first token shares a character with the previous word joins that word;
- a word's span is its tokens' offsets, whitespace trimmed, and its string is the text at that span,
  never the tokenizer's rebuilt string (which an uncased tokenizer lowercases and strips of accents,
  and which reads ``[UNK]`` for an emoji);
- values belonging to no visible word fold into the baseline, so ``base + Σ words`` is unchanged.
  That covers special tokens, and a word covering only whitespace: byte-level BPE gives a lone
  ``Ġ``/``Ċ`` its own word id on a double space or a newline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from shapash.compute.spans import Span, span_from_offsets, trim_span


@dataclass(frozen=True)
class WordLayout:
    """One encoded text, cut into words and each word into pieces.

    Attributes
    ----------
    text : str
        The encoded text.
    words : list of list of list of int
        Per word, its pieces; per piece, its token positions. In text order.
    spans : list of (int, int) or None
        Each word's character span in ``text``, whitespace trimmed; ``None`` for a word covering no
        visible character.
    """

    text: str
    words: list[list[list[int]]]
    spans: list[Span | None]

    @property
    def pieces(self) -> list[list[int]]:
        """Every piece's token positions, in text order."""
        return [piece for word in self.words for piece in word]

    @property
    def n_pieces(self) -> int:
        """How many pieces the words hold."""
        return sum(len(word) for word in self.words)

    def aggregate(
        self, values: np.ndarray, base_values: np.ndarray, *, per: Literal["piece", "token"] = "piece"
    ) -> tuple[list[str], np.ndarray, np.ndarray, list[tuple[Span, ...]]]:
        """Sum values into words; fold whatever is not a visible word into the baseline.

        Parameters
        ----------
        values : np.ndarray
            ``(n_pieces, n_classes)`` with ``per="piece"``; ``(n_tokens, n_classes)``, one row per
            token of the encoding, with ``per="token"``. A token in no word (a special token) then
            folds into the baseline.
        base_values : np.ndarray
            ``(n_classes,)``.
        per : {"piece", "token"}, default "piece"
            What a row of ``values`` stands for.

        Returns
        -------
        tuple[list[str], np.ndarray, np.ndarray, list[tuple[tuple[int, int], ...]]]
            Word strings (read from ``text``), per-word values ``(n_words, n_classes)``, the baseline
            with every folded value, and each word's span.
        """
        base = np.asarray(base_values, dtype=float).reshape(-1).copy()
        n_classes = base.shape[0]
        values = np.asarray(values, dtype=float)
        if per == "token":
            values = values.reshape(-1, n_classes)
            in_word = np.zeros(len(values), dtype=bool)
            in_word[[p for piece in self.pieces for p in piece]] = True
            base = base + values[~in_word].sum(axis=0)
            values = np.array([values[piece].sum(axis=0) for piece in self.pieces])
        values = values.reshape(self.n_pieces, n_classes)

        words: list[str] = []
        rows: list[np.ndarray] = []
        spans: list[tuple[Span, ...]] = []
        first = 0
        for word, span in zip(self.words, self.spans, strict=True):
            row = values[first : first + len(word)].sum(axis=0)
            first += len(word)
            if span is None:
                base = base + row
                continue
            words.append(self.text[span[0] : span[1]])
            rows.append(row)
            spans.append((span,))
        stacked = np.stack(rows, axis=0) if rows else np.zeros((0, n_classes))
        return words, stacked, base, spans


def word_layout(text: str, word_positions: Sequence[Sequence[int]], offsets: Sequence[Sequence[int]]) -> WordLayout:
    """Cut an encoded text into words and pieces, from the tokenizer's word grouping and offsets.

    Parameters
    ----------
    text : str
        The encoded text.
    word_positions : sequence of sequence of int
        Per word, its token positions, as ``word_alignment`` reports them (special tokens are in no
        word).
    offsets : sequence of (int, int)
        The tokenizer's offset mapping for ``text``, one entry per token of the encoding.

    Returns
    -------
    WordLayout

    Raises
    ------
    ValueError
        When a position indexes past ``offsets``: the two come from separate tokenizer calls, so a
        tokenizer disagreeing with itself would misplace every value.
    """
    if any(p >= len(offsets) for positions in word_positions for p in positions):
        raise ValueError(f"Word positions index past the {len(offsets)} token offsets of {text!r}.")
    words = _group_pieces(word_positions, offsets)
    spans = [trim_span(text, span_from_offsets(offsets, [p for piece in word for p in piece])) for word in words]
    return WordLayout(text=text, words=words, spans=spans)


def _group_pieces(word_positions: Sequence[Sequence[int]], offsets: Sequence[Sequence[int]]) -> list[list[list[int]]]:
    """Group each word's tokens into pieces: tokens sharing a character are one piece.

    A token with no extent (a bare word-start marker some tokenizers give empty offsets) joins the
    piece it sits in. A word whose first token shares a character with the previous word's last
    piece is the same word: a character is never split between two words.
    """
    words: list[list[list[int]]] = []
    end = -1  # end of the current piece's characters; -1 while it covers none
    for positions in word_positions:
        for k, p in enumerate(positions):
            start, stop = offsets[p]
            empty = stop <= start
            overlaps = not empty and 0 <= start < end
            if k == 0 and not (words and overlaps):
                words.append([[p]])
                end = -1
            elif k == 0 or empty or end < 0 or overlaps:
                words[-1][-1].append(p)
            else:
                words[-1].append([p])
                end = -1
            if not empty:
                end = max(end, stop)
    return words
