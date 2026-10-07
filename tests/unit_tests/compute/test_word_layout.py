"""Unit tests for ``shapash.compute.word_layout`` — the words of an encoded text, shared by LIG and SHAP."""

import numpy as np
import pytest

from shapash.compute.word_layout import _group_pieces, word_layout

TEXT = "un café 😍\n\nok"
# Specials at 0 and 9; "é" split over two byte tokens; a zero-width "▁" before the emoji; a newline
# with a word id of its own (byte-level "Ċ").
OFFSETS = [(0, 0), (0, 2), (3, 6), (6, 7), (6, 7), (8, 8), (8, 9), (9, 11), (11, 13), (0, 0)]
WORDS = [[1], [2, 3, 4], [5, 6], [7], [8]]


class TestWordLayout:
    def test_pieces_never_split_a_character(self):
        layout = word_layout(TEXT, WORDS, OFFSETS)
        # "é"'s two tokens are one piece; the zero-width marker joins the emoji.
        assert layout.words == [[[1]], [[2], [3, 4]], [[5, 6]], [[7]], [[8]]]
        assert layout.pieces == [[1], [2], [3, 4], [5, 6], [7], [8]]
        assert layout.n_pieces == 6

    def test_spans_are_trimmed_and_whitespace_words_have_none(self):
        assert word_layout(TEXT, WORDS, OFFSETS).spans == [(0, 2), (3, 7), (8, 9), None, (11, 13)]

    def test_a_word_sharing_a_character_with_the_previous_one_joins_it(self):
        assert _group_pieces([[0], [1, 2]], [(0, 3), (2, 4), (4, 6)]) == [[[0, 1], [2]]]

    def test_positions_past_the_offsets_are_refused(self):
        with pytest.raises(ValueError, match="index past"):
            word_layout(TEXT, [[99]], OFFSETS)


class TestAggregate:
    def test_per_piece_sums_words_and_folds_whitespace_words(self):
        layout = word_layout(TEXT, WORDS, OFFSETS)
        values = np.arange(12, dtype=float).reshape(6, 2)
        words, word_values, base, spans = layout.aggregate(values, np.array([100.0, 200.0]))
        assert words == ["un", "café", "😍", "ok"]
        np.testing.assert_array_equal(word_values, [[0, 1], [6, 8], [6, 7], [10, 11]])
        np.testing.assert_array_equal(base, [108.0, 209.0])  # + the newline's (8, 9)
        assert spans == [((0, 2),), ((3, 7),), ((8, 9),), ((11, 13),)]

    def test_per_token_folds_specials_and_keeps_the_total(self):
        layout = word_layout(TEXT, WORDS, OFFSETS)
        values = np.random.default_rng(0).normal(size=(len(OFFSETS), 3))
        base = np.array([0.5, -0.5, 1.0])
        words, word_values, new_base, _ = layout.aggregate(values, base, per="token")
        assert words == ["un", "café", "😍", "ok"]
        np.testing.assert_allclose(word_values[1], values[2:5].sum(axis=0))
        np.testing.assert_allclose(new_base, base + values[[0, 7, 9]].sum(axis=0))  # specials + newline
        np.testing.assert_allclose(new_base + word_values.sum(axis=0), base + values.sum(axis=0))

    def test_words_are_shown_as_written(self):
        # distilbert-base-uncased rebuilds "cafe" and "[UNK]"; the text says otherwise.
        layout = word_layout("Le café 😍", [[1], [2, 3], [4]], [(0, 0), (0, 2), (3, 6), (6, 7), (8, 9), (0, 0)])
        words, values, base, spans = layout.aggregate(np.arange(6.0)[:, None], np.zeros(1), per="token")
        assert words == ["Le", "café", "😍"]
        assert spans == [((0, 2),), ((3, 7),), ((8, 9),)]
        np.testing.assert_allclose(values[:, 0], [1.0, 5.0, 4.0])
        assert base[0] == 0 + 5

    def test_lone_whitespace_tokens_fold_into_the_baseline(self):
        # roberta-base on "Hello  world\n\nok": a lone "Ġ" (zero-width) and two "Ċ" get word ids.
        text = "Hello  world\n\nok"
        offsets = [(0, 0), (0, 5), (6, 6), (7, 12), (12, 13), (13, 14), (14, 16), (0, 0)]
        layout = word_layout(text, [[1], [2], [3], [4], [5], [6]], offsets)
        words, values, base, _ = layout.aggregate(np.arange(1.0, 7.0)[:, None], np.zeros(1))
        assert words == ["Hello", "world", "ok"]
        np.testing.assert_allclose(values[:, 0], [1.0, 3.0, 6.0])
        assert base[0] == 2 + 4 + 5

    def test_no_words(self):
        layout = word_layout("   ", [], [(0, 0), (0, 0)])
        words, values, base, spans = layout.aggregate(np.ones((2, 2)), np.zeros(2), per="token")
        assert (words, spans, values.shape) == ([], [], (0, 2))
        np.testing.assert_array_equal(base, [2.0, 2.0])
