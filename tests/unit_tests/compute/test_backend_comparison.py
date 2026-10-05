"""Unit tests for ``shapash.compute.backend_comparison`` — putting backends' word units side by side.

The token lists below are copied from real ``nlp_shap`` / ``nlp_lime`` / ``nlp_captum_lig`` runs on
``distilbert-base-uncased-emotion``: SHAP keeps case and punctuation, LIG lowercases, LIME emits a
case-sensitive bag of distinct words with punctuation dropped.
"""

import numpy as np
import pytest

from shapash.compute.backend_comparison import (
    align_token_values,
    backend_agreement,
    locate_tokens,
    normalize_contributions,
)

TEXT = "The waiting room was cold and I felt nervous, really nervous, about the results."
SHAP_TOKENS = ["The", "waiting", "room", "was", "cold", "and", "I", "felt", "nervous", ","]
SHAP_TOKENS += ["really", "nervous", ",", "about", "the", "results", "."]
LIG_TOKENS = [t.lower() for t in SHAP_TOKENS]
LIME_TOKENS = ["The", "waiting", "room", "was", "cold", "and", "I", "felt", "nervous", "really", "about", "the"]
LIME_TOKENS += ["results"]


class TestLocateTokens:
    def test_repeated_words_map_to_successive_occurrences(self):
        spans = locate_tokens(TEXT, SHAP_TOKENS)
        assert [TEXT[a:b] for a, b in spans] == SHAP_TOKENS
        assert spans[8] != spans[11]  # the two "nervous"

    def test_case_is_ignored(self):
        assert locate_tokens(TEXT, LIG_TOKENS) == locate_tokens(TEXT, SHAP_TOKENS)

    def test_a_short_word_is_not_found_inside_a_longer_one(self):
        # A plain substring search would place "i" inside "didn".
        assert locate_tokens("I didn't, did i", ["didn", "i"]) == [(2, 6), (14, 15)]

    def test_contraction_pieces_still_match(self):
        assert locate_tokens("didn't", ["didn", "'", "t"]) == [(0, 4), (4, 5), (5, 6)]

    def test_missing_token_gets_none_and_keeps_the_cursor(self):
        assert locate_tokens("a b", ["a", "[SEP]", "b"]) == [(0, 1), None, (2, 3)]

    def test_unspaced_script_falls_back_to_substring_search(self):
        assert locate_tokens("我喜欢你", ["我", "喜欢", "你"]) == [(0, 1), (1, 3), (3, 4)]


class TestAlignTokenValues:
    def test_identical_tokenization_is_the_identity(self):
        values = np.arange(len(SHAP_TOKENS), dtype=float)
        np.testing.assert_array_equal(align_token_values(TEXT, SHAP_TOKENS, SHAP_TOKENS, values), values)

    def test_lowercased_tokenization_aligns_by_span(self):
        values = np.arange(len(LIG_TOKENS), dtype=float)
        np.testing.assert_array_equal(align_token_values(TEXT, SHAP_TOKENS, LIG_TOKENS, values), values)

    def test_bag_of_words_reaches_every_occurrence_and_skips_punctuation(self):
        values = np.arange(1.0, len(LIME_TOKENS) + 1)
        out = align_token_values(TEXT, SHAP_TOKENS, LIME_TOKENS, values)
        lime = dict(zip(LIME_TOKENS, values, strict=True))
        for token, value in zip(SHAP_TOKENS, out, strict=True):
            if token in {",", "."}:
                assert np.isnan(value)
            else:
                assert value == lime[token]
        # The second "nervous" has no LIME span of its own: it gets the one bag-of-words weight.
        assert out[8] == out[11] == lime["nervous"]

    def test_bag_of_words_keeps_case_distinct_words_apart(self):
        values = np.arange(1.0, len(LIME_TOKENS) + 1)
        out = align_token_values(TEXT, SHAP_TOKENS, LIME_TOKENS, values)
        assert out[0] == values[LIME_TOKENS.index("The")]
        assert out[14] == values[LIME_TOKENS.index("the")]

    def test_rejects_mismatched_values(self):
        with pytest.raises(ValueError, match="one entry per token"):
            align_token_values("a b", ["a", "b"], ["a", "b"], np.array([1.0]))


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

    def test_no_shared_unit_gives_nan(self):
        df = backend_agreement({"a": np.array([1.0, np.nan]), "b": np.array([np.nan, 1.0])})
        assert df.iloc[0][["spearman", "pearson", "cosine", "sign_agreement", "top_k_overlap"]].isna().all()

    def test_all_zero_backend_has_no_cosine_or_top_share(self):
        row = backend_agreement({"a": np.array([1.0, 2.0]), "b": np.zeros(2)}).iloc[0]
        assert np.isnan(row.cosine) and np.isnan(row.top_share_b)
