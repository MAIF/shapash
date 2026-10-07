"""Unit tests for ``shapash.explainer.nlp_comparison`` — several explanations of the same texts, side by side."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from shapash.explainer.nlp_comparison import AGREEMENT_COLUMNS, align_row, corpus_agreement, name_explanations
from tests.unit_tests.explainer.test_nlp_plotter import _make_explanation


@pytest.fixture
def ref():
    return _make_explanation()


@pytest.fixture
def lime(ref):
    # Bag of words in its own order; on row 1 LIME and SHAP rank "happy" above "i" for class 1.
    return replace(
        ref,
        token_strings=[["world", "hello"], ["happy", "i"], ["ok"]],
        token_spans=[[((6, 11),), ((0, 5),)], [((5, 10),), ((0, 1),)], [((0, 2),)]],
        values=[
            np.array([[5.0, -5.0], [3.0, -3.0]]),
            np.array([[0.0, 8.0], [0.0, 2.0]]),
            np.array([[1.0, -1.0]]),
        ],
        backend_name="nlp_lime",
        is_additive=False,
    )


class TestCorpusAgreement:
    def test_one_line_per_text_and_pair_on_the_predicted_class(self, ref, lime):
        df = corpus_agreement(ref, lime)
        assert list(df.columns) == AGREEMENT_COLUMNS
        assert list(df.row) == [0, 1, 2]
        assert list(df.label) == ["pos", "neg", "pos"]  # y_pred
        assert set(zip(df.backend_a, df.backend_b, strict=True)) == {("shap", "lime")}
        assert df.text.iloc[1] == "i am happy today"
        # Row 0, class "pos": shap [-1, -2], lime aligned onto "hello world" [-3, -5] — same ranks.
        assert df.spearman.iloc[0] == pytest.approx(1.0)
        # Row 1, class "neg": LIME is constant zero; row 2 has one word. No rank correlation either way.
        assert df.spearman.iloc[1:].isna().all()

    def test_fixed_class_and_a_subset_of_rows(self, ref, lime):
        df = corpus_agreement(ref, {"LIME": lime}, rows=[1], label_idx=-1)
        assert list(df.row) == [1]
        assert list(df.label) == ["pos"]
        assert df.backend_b.iloc[0] == "LIME"
        assert df.spearman.iloc[0] == pytest.approx(1.0)

    def test_every_pair_is_scored(self, ref, lime):
        df = corpus_agreement(ref, [lime, replace(ref, backend_name="nlp_captum_lig")], rows=[0])
        assert list(zip(df.backend_a, df.backend_b, strict=True)) == [
            ("shap", "lime"),
            ("shap", "captum_lig"),
            ("lime", "captum_lig"),
        ]

    def test_label_is_the_column_index_without_class_names(self):
        one_d = _make_explanation(ndim=1)
        df = corpus_agreement(one_d, replace(one_d, backend_name="nlp_lime"), rows=[0])
        assert df.label.iloc[0] == 0

    def test_no_rows_gives_an_empty_table(self, ref, lime):
        df = corpus_agreement(ref, lime, rows=[])
        assert df.empty and list(df.columns) == AGREEMENT_COLUMNS

    def test_warns_across_output_spaces(self, ref, lime):
        with pytest.warns(UserWarning, match="output spaces") as record:
            corpus_agreement(ref, replace(lime, output_space="logit"), rows=[0])
        assert record[0].filename == __file__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            corpus_agreement(ref, lime, rows=[0])

    def test_rejects_a_different_corpus(self, ref, lime):
        other = replace(lime, texts=lime.texts.replace("ok", "fine"))
        with pytest.raises(ValueError, match="Row 2 holds a different text"):
            corpus_agreement(ref, other)


class TestAlignRow:
    def test_units_are_merged_where_a_backend_groups_more_coarsely(self, ref):
        # A SentencePiece-like LIG keeps "am happy" as one unit on row 1 (made up, for the shape).
        coarse = replace(
            ref,
            token_strings=[["hello", "world"], ["i", "am happy"], ["ok"]],
            token_spans=[[((0, 5),), ((6, 11),)], [((0, 1),), ((2, 10),)], [((0, 2),)]],
            values=[ref.values[0], np.array([[1.0, 1.0], [9.0, 9.0]]), ref.values[2]],
            backend_name="nlp_captum_lig",
        )
        units, aligned = align_row(name_explanations(ref, coarse), 1, 1)
        assert units == ["i", "am happy"]
        np.testing.assert_allclose(aligned["shap"], [0.2, 0.4 + 0.6])
        np.testing.assert_allclose(aligned["captum_lig"], [1.0, 9.0])

    def test_legacy_artifacts_without_spans_are_located_by_string_and_warn(self, ref, lime):
        legacy_ref, legacy_lime = replace(ref, token_spans=None), replace(lime, token_spans=None)
        with pytest.warns(UserWarning, match="no character spans"):
            df = corpus_agreement(legacy_ref, legacy_lime, rows=[0])
        assert df.spearman.iloc[0] == pytest.approx(1.0)
        # Same alignment as the exact path on text the tokenizer did not rewrite.
        exact = align_row(name_explanations(ref, lime), 1, 1)
        located = align_row(name_explanations(legacy_ref, legacy_lime), 1, 1)
        assert exact[0] == located[0]
        np.testing.assert_array_equal(exact[1]["lime"], located[1]["lime"])

    def test_out_of_range_row(self, ref, lime):
        with pytest.raises(IndexError, match="row=5"):
            align_row(name_explanations(ref, lime), 5, 0)
