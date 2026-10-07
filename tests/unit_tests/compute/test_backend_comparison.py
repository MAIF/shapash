"""Unit tests for ``shapash.compute.backend_comparison`` — putting backends' units side by side.

Token lists are copied from real ``nlp_shap`` / ``nlp_captum_lig`` / ``nlp_lime`` runs, each on the
tokenizer named in the test. Spans are what those runs record: the characters of the text a unit
covers, whatever string the backend printed for it.
"""

import numpy as np
import pytest

from shapash.compute.backend_comparison import align_units, backend_agreement, normalize_contributions

TEXT = "The waiting room was cold and I felt nervous, really nervous, about the results."
SHAP_TOKENS = ["The", "waiting", "room", "was", "cold", "and", "I", "felt", "nervous", ","]
SHAP_TOKENS += ["really", "nervous", ",", "about", "the", "results", "."]
LIME_TOKENS = ["The", "waiting", "room", "was", "cold", "and", "I", "felt", "nervous", "really", "about", "the"]
LIME_TOKENS += ["results"]


def _spans(text, words):
    """Each word's span, taking successive occurrences — what a sequence backend records."""
    out, cursor = [], 0
    for word in words:
        start = text.index(word, cursor)
        out.append(((start, start + len(word)),))
        cursor = start + len(word)
    return out


def _bag_spans(text, words):
    """Every occurrence of each distinct word — what LIME's bag of words records."""
    import re

    return [tuple((m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(w)}\b", text)) for w in words]


def _labels(text, groups):
    return [text[a:b] for a, b in groups]


class TestAlignUnits:
    def test_identical_units_are_the_identity(self):
        values = np.arange(len(SHAP_TOKENS), dtype=float)
        groups, aligned = align_units(
            {"a": (_spans(TEXT, SHAP_TOKENS), values), "b": (_spans(TEXT, SHAP_TOKENS), values)}
        )
        assert _labels(TEXT, groups) == SHAP_TOKENS
        np.testing.assert_array_equal(aligned["b"], values)

    def test_strings_play_no_part(self):
        # distilbert-base-uncased LIG on "Le café était très bon.": the tokenizer lowercases and strips
        # accents, so none of its strings but "bon" and "." occur in the text. Its spans do.
        text = "Le café était très bon."
        shap = _spans(text, ["Le", "café", "était", "très", "bon", "."])
        values = np.arange(6, dtype=float)
        groups, aligned = align_units({"shap": (shap, values), "lig": (shap, values * 10)})
        assert _labels(text, groups) == ["Le", "café", "était", "très", "bon", "."]
        np.testing.assert_array_equal(aligned["lig"], values * 10)

    def test_bag_of_words_reaches_every_occurrence_and_skips_punctuation(self):
        lime_values = np.arange(1.0, len(LIME_TOKENS) + 1)
        groups, aligned = align_units(
            {
                "shap": (_spans(TEXT, SHAP_TOKENS), np.zeros(len(SHAP_TOKENS))),
                "lime": (_bag_spans(TEXT, LIME_TOKENS), lime_values),
            }
        )
        lime = dict(zip(LIME_TOKENS, lime_values, strict=True))
        for label, value in zip(_labels(TEXT, groups), aligned["lime"], strict=True):
            if label in {",", "."}:
                assert np.isnan(value)
            else:
                assert value == lime[label]
        # Both "nervous": the one bag-of-words weight stands for each.
        assert aligned["lime"][8] == aligned["lime"][11] == lime["nervous"]

    def test_bag_of_words_keeps_case_distinct_words_apart(self):
        lime_values = np.arange(1.0, len(LIME_TOKENS) + 1)
        _, aligned = align_units(
            {
                "shap": (_spans(TEXT, SHAP_TOKENS), np.zeros(len(SHAP_TOKENS))),
                "lime": (_bag_spans(TEXT, LIME_TOKENS), lime_values),
            }
        )
        assert aligned["lime"][0] == lime_values[LIME_TOKENS.index("The")]
        assert aligned["lime"][14] == lime_values[LIME_TOKENS.index("the")]

    def test_a_coarser_backend_merges_the_finer_units_it_spans(self):
        # xlm-roberta LIG groups by whitespace ("didn't", "great."), SHAP splits at punctuation.
        text = "I didn't love it, great."
        shap_words = ["I", "didn", "'", "t", "love", "it", ",", "great", "."]
        lig_words = ["I", "didn't", "love", "it,", "great."]
        shap_values = np.arange(1.0, 10.0)
        groups, aligned = align_units(
            {"shap": (_spans(text, shap_words), shap_values), "lig": (_spans(text, lig_words), np.arange(5.0))}
        )
        assert _labels(text, groups) == lig_words
        # Each group carries the sum of the finer backend's units inside it: nothing is dropped.
        np.testing.assert_array_equal(aligned["shap"], [1.0, 2 + 3 + 4, 5.0, 6 + 7, 8 + 9])
        np.testing.assert_array_equal(aligned["lig"], np.arange(5.0))
        assert aligned["shap"].sum() == shap_values.sum()

    def test_a_unit_starting_inside_another_is_not_lost(self):
        # roberta LIG keeps "'t" whole; LIME drops the apostrophe and keeps "t" — which starts after
        # the group starts, and used to come back NaN.
        text = "I didn't"
        _, aligned = align_units(
            {
                "lig": (_spans(text, ["I", "didn", "'t"]), np.array([1.0, 2.0, 3.0])),
                "lime": (_bag_spans(text, ["I", "didn", "t"]), np.array([10.0, 20.0, 30.0])),
            }
        )
        np.testing.assert_array_equal(aligned["lime"], [10.0, 20.0, 30.0])

    def test_a_backend_that_stops_early_is_nan_on_the_rest(self):
        # LIG truncates at the model's max length; SHAP does not.
        text = "one two three"
        _, aligned = align_units(
            {
                "shap": (_spans(text, ["one", "two", "three"]), np.ones(3)),
                "lig": (_spans(text, ["one", "two"]), np.ones(2)),
            }
        )
        assert np.isnan(aligned["lig"][2]) and not np.isnan(aligned["shap"]).any()

    def test_a_unit_without_spans_is_left_out(self):
        _, aligned = align_units({"a": ([((0, 1),), ()], np.array([1.0, 5.0]))})
        np.testing.assert_array_equal(aligned["a"], [1.0])

    def test_two_dimensional_values(self):
        _, aligned = align_units({"a": ([((0, 1),), ((0, 1),)], np.array([[1.0, 2.0], [3.0, 4.0]]))})
        np.testing.assert_array_equal(aligned["a"], [[4.0, 6.0]])

    def test_rejects_mismatched_values(self):
        with pytest.raises(ValueError, match="1 value"):
            align_units({"a": ([((0, 1),), ((2, 3),)], np.array([1.0]))})


class TestNormalize:
    def test_max_abs(self):
        np.testing.assert_allclose(normalize_contributions(np.array([2.0, -4.0, np.nan])), [0.5, -1.0, np.nan])

    def test_sum_abs(self):
        np.testing.assert_allclose(normalize_contributions(np.array([1.0, -3.0]), "sum_abs"), [0.25, -0.75])

    def test_none_is_raw(self):
        np.testing.assert_array_equal(normalize_contributions(np.array([2.0, -4.0]), None), [2.0, -4.0])

    @pytest.mark.parametrize("values", [np.zeros(3), np.full(2, np.nan)])
    def test_degenerate_input_is_left_alone(self, values):
        np.testing.assert_array_equal(normalize_contributions(values), values)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="normalize"):
            normalize_contributions(np.ones(2), "zscore")


class TestBackendAgreement:
    def test_one_row_per_pair_and_identical_backends_agree_fully(self):
        v = np.array([0.1, -0.5, 0.9, 0.0, -0.2])
        df = backend_agreement({"a": v, "b": v * 10, "c": -v}, top_k=2)
        assert list(zip(df.backend_a, df.backend_b, strict=True)) == [("a", "b"), ("a", "c"), ("b", "c")]
        ab = df.iloc[0]
        assert ab.spearman == pytest.approx(1.0)
        assert ab.pearson == pytest.approx(1.0)
        assert ab.cosine == pytest.approx(1.0)
        assert ab.sign_agreement == 1.0
        assert ab.top_k_overlap == 1.0
        assert df.iloc[1].spearman == pytest.approx(-1.0)
        assert df.iloc[1].cosine == pytest.approx(-1.0)
        # Each backend's own concentration: 0.9 of a total magnitude of 1.7.
        assert ab.top_share_a == ab.top_share_b == pytest.approx(0.9 / 1.7)

    def test_one_dominant_unit_cosine_agrees_where_spearman_does_not(self):
        # The demo's "I'm furious." case: same dominant word, the small rest ordered in reverse.
        a = np.array([5.0, 0.1, 0.2, 0.3, 0.4])
        b = np.array([5.0, 0.4, 0.3, 0.2, 0.1])
        row = backend_agreement({"a": a, "b": b}).iloc[0]
        assert row.spearman < 0.5
        assert row.cosine > 0.99
        assert row.pearson > 0.99
        assert row.top_share_a == pytest.approx(5.0 / 6.0)

    def test_pearson_ignores_a_shift_both_share_and_cosine_does_not(self):
        a = np.array([1.0, 1.1, 0.9])
        b = np.array([1.0, 0.9, 1.1])
        row = backend_agreement({"a": a, "b": b}).iloc[0]
        assert row.pearson == pytest.approx(-1.0)
        assert row.cosine > 0.98

    def test_top_share_ignores_unattributed_units(self):
        row = backend_agreement({"a": np.array([1.0, 3.0]), "b": np.array([np.nan, 2.0])}).iloc[0]
        assert row.top_share_a == pytest.approx(0.75)
        assert row.top_share_b == 1.0

    def test_nan_units_are_excluded(self):
        df = backend_agreement({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, np.nan, 3.0])}, top_k=5)
        row = df.iloc[0]
        assert row.spearman == pytest.approx(1.0)
        assert row.top_k_overlap == 1.0  # k shrinks to the 2 shared units

    def test_constant_backend_has_no_rank_correlation(self):
        df = backend_agreement({"a": np.array([1.0, 2.0, 3.0]), "b": np.zeros(3)})
        assert np.isnan(df.iloc[0].spearman) and np.isnan(df.iloc[0].pearson)

    def test_coverage_is_the_share_of_units_both_attributed(self):
        df = backend_agreement({"a": np.array([1.0, 2.0, 3.0, np.nan]), "b": np.array([1.0, np.nan, 3.0, 4.0])})
        assert df.iloc[0].coverage == 0.5

    def test_no_shared_unit_gives_nan(self):
        df = backend_agreement({"a": np.array([1.0, np.nan]), "b": np.array([np.nan, 1.0])})
        assert df.iloc[0][["spearman", "pearson", "cosine", "sign_agreement", "top_k_overlap"]].isna().all()

    def test_all_zero_backend_has_no_cosine_or_top_share(self):
        row = backend_agreement({"a": np.array([1.0, 2.0]), "b": np.zeros(2)}).iloc[0]
        assert np.isnan(row.cosine) and np.isnan(row.top_share_b)
