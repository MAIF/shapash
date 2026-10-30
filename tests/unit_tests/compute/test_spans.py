"""Unit tests for ``shapash.compute.spans`` — character spans, the coordinate NLP backends share."""

import numpy as np
import pytest

from shapash.compute.spans import (
    aggregate,
    association_matrix,
    locate_all,
    locate_spans,
    merge_overlapping,
    span_from_offsets,
    trim_span,
)


class TestSpanFromOffsets:
    def test_covers_the_tokens_of_a_word(self):
        offsets = [(0, 0), (0, 3), (3, 7), (8, 10), (0, 0)]
        assert span_from_offsets(offsets, [1, 2]) == (0, 7)

    def test_zero_width_and_missing_offsets_cover_nothing(self):
        # A special token (0, 0), a bare word-start marker (3, 3), a None from a tokenizer.
        offsets = [(0, 0), (3, 3), None, (3, 4)]
        assert span_from_offsets(offsets, [0, 1, 2]) is None
        assert span_from_offsets(offsets, [1, 3]) == (3, 4)

    def test_tokens_sharing_a_character(self):
        # Byte-level BPE splits "é" into two tokens that both map to it.
        assert span_from_offsets([(6, 7), (6, 7)], [0, 1]) == (6, 7)


class TestTrimSpan:
    @pytest.mark.parametrize(
        ("span", "trimmed"),
        [((1, 6), (2, 6)), ((0, 3), (0, 3)), ((6, 8), None), ((2, 7), (2, 6)), (None, None)],
    )
    def test_shrinks_to_visible_characters(self, span, trimmed):
        # ModernBERT keeps the space before a word in its offsets: " didn" at (1, 6).
        assert trim_span("I didn  t", span) == trimmed


class TestMergeOverlapping:
    def test_identical_partitions_stay_as_they_are(self):
        units = [(0, 4), (4, 5), (5, 6)]
        assert merge_overlapping(units + units) == units

    def test_a_coarser_unit_absorbs_the_finer_ones_it_covers(self):
        # SHAP "didn" "'" "t" against SentencePiece LIG "didn't".
        assert merge_overlapping([(2, 6), (6, 7), (7, 8), (2, 8)]) == [(2, 8)]

    def test_chains_of_overlap_merge_and_touching_spans_do_not(self):
        assert merge_overlapping([(0, 3), (2, 5), (4, 6), (6, 8)]) == [(0, 6), (6, 8)]

    def test_order_and_empty_spans_are_ignored(self):
        assert merge_overlapping([(5, 6), (3, 3), (0, 2)]) == [(0, 2), (5, 6)]


class TestAssociationAndAggregate:
    def test_each_occurrence_of_a_bag_of_words_unit_counts(self):
        groups = [(0, 4), (5, 9), (10, 14)]
        units = [((0, 4), (10, 14)), ((5, 9),), ()]
        matrix = association_matrix(groups, units)
        np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        np.testing.assert_array_equal(aggregate(matrix, np.array([2.0, 3.0, 99.0])), [2.0, 3.0, 2.0])

    def test_a_group_sums_its_units_and_a_group_without_any_is_nan(self):
        groups = [(0, 8), (8, 9)]
        matrix = association_matrix(groups, [((2, 6),), ((6, 7),), ((7, 8),)])
        out = aggregate(matrix, np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
        np.testing.assert_array_equal(out[0], [6.0, 60.0])
        assert np.isnan(out[1]).all()

    def test_a_span_straddling_two_groups_goes_to_the_one_holding_its_start(self):
        np.testing.assert_array_equal(association_matrix([(0, 5), (5, 10)], [((3, 7),)]), [[1], [0]])


class TestLocateSpans:
    """The legacy path, for artifacts saved before units carried spans."""

    def test_repeated_words_map_to_successive_occurrences(self):
        assert locate_spans("a cat and a cat", ["a", "cat", "and", "a", "cat"]) == [
            ((0, 1),),
            ((2, 5),),
            ((6, 9),),
            ((10, 11),),
            ((12, 15),),
        ]

    def test_case_is_ignored(self):
        assert locate_spans("The cat", ["the", "CAT"]) == [((0, 3),), ((4, 7),)]

    def test_a_short_word_is_not_found_inside_a_longer_one(self):
        assert locate_spans("I didn't, did i", ["didn", "i"]) == [((2, 6),), ((14, 15),)]

    def test_contraction_pieces_still_match(self):
        assert locate_spans("didn't", ["didn", "'", "t"]) == [((0, 4),), ((4, 5),), ((5, 6),)]

    def test_missing_token_gets_no_span_and_keeps_the_cursor(self):
        assert locate_spans("a b", ["a", "[SEP]", "b"]) == [((0, 1),), (), ((2, 3),)]

    def test_a_rewritten_word_is_not_placed_inside_another(self):
        # An uncased tokenizer's "tre" (from "très") used to land inside a later "entre", sending
        # every following token past its place.
        assert locate_spans("très bien entre nous", ["tre", "bien", "entre"]) == [(), ((5, 9),), ((10, 15),)]

    def test_unspaced_script_falls_back_to_substring_search(self):
        assert locate_spans("我喜欢你", ["我", "喜欢", "你"]) == [((0, 1),), ((1, 3),), ((3, 4),)]

    def test_length_changing_case_fold_keeps_offsets_on_the_text(self):
        text = "Straße ok"
        assert [text[a:b] for ((a, b),) in locate_spans(text, ["Straße", "ok"])] == ["Straße", "ok"]


class TestLocateAll:
    def test_every_whole_word_occurrence_case_sensitive(self):
        assert locate_all("Good good, Good! goods", ["Good", "good", "bad", ""]) == [
            ((0, 4), (11, 15)),
            ((5, 9),),
            (),
            (),
        ]
