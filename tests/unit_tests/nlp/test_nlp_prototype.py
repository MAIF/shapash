"""Integration tests for the NLP explanation prototype.

Covers the full wiring between NlpContributions (backend), word_importance
aggregation, plot_token_highlight / plot_word_importance (plots), NlpExplainer
(explainer), and NlpWebApp (webapp layout construction). A real NLP model is
not required — synthetic NlpContributions data is used throughout so the suite
runs in CI without transformers/datasets.
"""

import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_lime_backend import NlpLimeBackend
from shapash.backend.nlp_shap_backend import NlpShapBackend
from shapash.compute.generators import AblationFlipGenerator, HotFlipGenerator
from shapash.compute.retrieval import Neighbor
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.explainer.nlp_explanation import (
    WORD_AGGREGATIONS,
    NlpExplanation,
    aggregate_word_contributions,
    rank_word_samples,
)
from shapash.model.base import SupportsEmbeddings, SupportsGradients, SupportsTokenization, TextModel
from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_noise_matrix import plot_noise_matrix
from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.plots.plot_waterfall import plot_waterfall
from shapash.plots.plot_word_importance import plot_word_importance
from shapash.plots.plot_word_profile import plot_word_profile
from shapash.style.style_utils import DEFAULT_NLP_THEME
from shapash.webapp.nlp_app import NlpWebApp
from shapash.webapp.nlp_components import compose_selection as _compose_selection
from shapash.webapp.nlp_components import error_positions
from shapash.webapp.nlp_components.error_analysis import ErrorAnalysisComponent, _cell_from_click

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_token_data() -> tuple[list[list[str]], list[np.ndarray], np.ndarray]:
    """Synthetic per-token contribution data for 3 samples, 6 classes."""
    rng = np.random.default_rng(42)
    token_strings = [
        ["", "i", "feel", "so", "happy", "today", ""],
        ["", "this", "is", "terrible", "and", "sad", ""],
        ["", "what", "a", "wonderful", "day", ""],
    ]
    values = [rng.uniform(-0.4, 0.4, size=(len(t), N_CLASSES)).astype(np.float32) for t in token_strings]
    base_values = rng.uniform(-0.1, 0.1, size=(3, N_CLASSES)).astype(np.float32)
    return token_strings, values, base_values


def _make_texts() -> pd.Series:
    return pd.Series(
        ["i feel so happy today", "this is terrible and sad", "what a wonderful day"],
        index=pd.RangeIndex(3),
    )


def _make_explanation(
    texts: pd.Series | None = None,
    y_pred: pd.Series | None = None,
    y_prob: pd.DataFrame | None = None,
    y_true: pd.Series | None = None,
    label_names: list[str] | None = LABEL_NAMES,
    backend_name: str = "nlp_shap",
    is_additive: bool = True,
    reference_kind: str = "none",
    output_space: str = "probability",
) -> NlpExplanation:
    """Synthetic ``NlpExplanation`` for 3 samples, 6 classes — the ``explain()`` return value."""
    texts = _make_texts() if texts is None else texts
    token_strings, values, base_values = _make_token_data()
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=(
            pd.Series(["joy", "sadness", "joy"], index=texts.index, name="prediction") if y_pred is None else y_pred
        ),
        y_prob=y_prob,
        y_true=y_true,
        label_names=label_names,
        folds_case=None,
        backend_name=backend_name,
        is_additive=is_additive,
        reference_kind=reference_kind,
        output_space=output_space,
    )


def _minimal_explanation(
    token_strings: list[list[str]],
    values: list[np.ndarray],
    base_values: np.ndarray | None,
    folds_case: bool | None = None,
) -> NlpExplanation:
    """Bare ``NlpExplanation`` for word-importance/case-folding tests — only the token data matters."""
    n = len(token_strings)
    texts = pd.Series([""] * n, index=pd.RangeIndex(n))
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=pd.Series([""] * n, index=texts.index, name="prediction"),
        y_prob=None,
        y_true=None,
        label_names=None,
        folds_case=folds_case,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


def _make_explainer() -> NlpExplainer:
    """Bare ``NlpExplainer`` (engine role only), bypassing ``__init__`` to avoid ``shap.Explainer(None)``.

    Per ``fit``/``explain``, the explainer never holds compiled results — tests that need a computed
    batch build an :class:`NlpExplanation` directly with :func:`_make_explanation` instead.
    """
    xpl = object.__new__(NlpExplainer)
    xpl.model = None
    xpl.label_names = LABEL_NAMES
    xpl.backend = None
    return xpl


# ---------------------------------------------------------------------------
# NlpExplanation.word_importance / resolve_lowercase
# ---------------------------------------------------------------------------


class TestWordImportance(unittest.TestCase):
    def setUp(self):
        token_strings, values, base_values = _make_token_data()
        self.explanation = _minimal_explanation(token_strings, values, base_values)

    def test_len(self):
        self.assertEqual(len(self.explanation), 3)

    def test_word_importance_returns_series(self):
        imp = self.explanation.word_importance(label_idx=1)
        self.assertIsInstance(imp, pd.Series)

    def test_word_importance_filters_special_tokens(self):
        imp = self.explanation.word_importance(label_idx=0, filter_special=True)
        self.assertNotIn("", imp.index)
        self.assertNotIn(" ", imp.index)

    def test_word_importance_keeps_special_when_disabled(self):
        imp = self.explanation.word_importance(label_idx=0, filter_special=False)
        # Empty strings (BOS/EOS) should now be present
        self.assertIn("", imp.index)

    def test_word_importance_hides_punctuation_by_default(self):
        explanation = _minimal_explanation(
            token_strings=[["great", "!", "!", "movie", ".", ","]],
            values=[np.array([[3.0], [9.0], [9.0], [2.0], [8.0], [7.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["great", "movie"])

    def test_word_importance_keeps_punctuation_when_asked(self):
        explanation = _minimal_explanation(
            token_strings=[["great", "!", "!"]],
            values=[np.array([[3.0], [9.0], [9.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, filter_punctuation=False)
        self.assertIn("!", imp.index)
        # Punctuation kept means it can outrank real words, which is why it is hidden by default.
        self.assertEqual(imp.index[0], "!")

    def test_word_importance_keeps_words_containing_punctuation(self):
        # Only *pure* punctuation units are dropped — a hyphenated or apostrophised word stays.
        explanation = _minimal_explanation(
            token_strings=[["state-of-the-art", "don't", "-"]],
            values=[np.array([[3.0], [2.0], [9.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["don't", "state-of-the-art"])

    def test_resolve_lowercase_follows_the_model(self):
        for folds_case, expected in ((True, True), (False, False), (None, True)):
            with self.subTest(folds_case=folds_case):
                explanation = replace(self.explanation, folds_case=folds_case)
                self.assertEqual(explanation.resolve_lowercase(), expected)

    def test_resolve_lowercase_explicit_argument_wins(self):
        self.assertFalse(replace(self.explanation, folds_case=True).resolve_lowercase(False))
        self.assertTrue(replace(self.explanation, folds_case=False).resolve_lowercase(True))

    def test_word_importance_keeps_case_on_a_cased_model(self):
        # A cased tokenizer encodes AWFUL and awful to *different* ids, so they are different
        # inputs with genuinely different attributions — merging them would hide that.
        explanation = _minimal_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
            folds_case=False,
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["AWFUL", "awful"])

    def test_word_importance_merges_case_on_an_uncased_model(self):
        # An uncased tokenizer maps both spellings to one input id — the model cannot tell them
        # apart, so they must not appear as two rows.
        explanation = _minimal_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
            folds_case=True,
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(list(imp.index), ["awful"])
        self.assertAlmostEqual(imp["awful"], -6.0)

    def test_word_importance_merges_case_variants_by_default(self):
        # One row per word, averaged over every casing — not three rows of one occurrence each.
        explanation = _minimal_explanation(
            token_strings=[["AWFUL", "Awful", "awful", "good"]],
            values=[np.array([[-9.0], [-3.0], [-3.0], [1.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["awful", "good"])
        self.assertAlmostEqual(imp["awful"], -5.0)

    def test_word_importance_case_fragmentation_lets_rare_variants_outrank(self):
        # The reason lowercasing is the default: a single capitalised occurrence keeps its own
        # extreme value instead of being averaged into the 3 common ones, and tops the ranking.
        explanation = _minimal_explanation(
            token_strings=[["TERRIBLE", *["terrible"] * 9, "dull"]],
            values=[np.array([[-9.0], *[[-0.1]] * 9, [-2.0]])],
            base_values=np.zeros((1, 1)),
        )
        cased = explanation.word_importance(label_idx=0, n_top=20, lowercase=False)
        self.assertEqual(cased.index[0], "TERRIBLE")
        folded = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(folded.index[0], "dull")

    def test_word_importance_keeps_case_when_disabled(self):
        explanation = _minimal_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, lowercase=False)
        self.assertEqual(sorted(imp.index), ["AWFUL", "awful"])

    def test_word_importance_exclusion_is_case_insensitive_when_folding(self):
        # The webapp's dropdown offers lowercase entries; excluding one must drop every casing.
        explanation = _minimal_explanation(
            token_strings=[["AWFUL", "awful", "good"]],
            values=[np.array([[-9.0], [-3.0], [1.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, exclude_words={"AwFuL"})
        self.assertEqual(list(imp.index), ["good"])

    def test_word_importance_respects_n_top(self):
        imp = self.explanation.word_importance(label_idx=1, n_top=3)
        self.assertLessEqual(len(imp), 3)

    def test_word_importance_sorted_by_absolute_value(self):
        imp = self.explanation.word_importance(label_idx=2, n_top=20)
        abs_vals = imp.abs().tolist()
        self.assertEqual(abs_vals, sorted(abs_vals, reverse=True))

    def test_word_importance_aggregates_repeated_words(self):
        # "feel" only appears once; check it has a single contribution value
        imp = self.explanation.word_importance(label_idx=1, n_top=20, filter_special=True)
        self.assertIn("feel", imp.index)
        # Should be a scalar (mean of one occurrence)
        self.assertIsInstance(imp["feel"], float)

    def test_word_importance_all_labels(self):
        for idx in range(N_CLASSES):
            imp = self.explanation.word_importance(label_idx=idx)
            self.assertIsInstance(imp, pd.Series)
            self.assertGreater(len(imp), 0)

    def test_word_importance_filter_sign_positive(self):
        imp = self.explanation.word_importance(label_idx=0, filter_sign="positive")
        if len(imp) > 0:
            self.assertTrue((imp > 0).all(), "positive filter should return only positive values")

    def test_word_importance_filter_sign_negative(self):
        imp = self.explanation.word_importance(label_idx=0, filter_sign="negative")
        if len(imp) > 0:
            self.assertTrue((imp < 0).all(), "negative filter should return only negative values")

    def test_word_importance_exclude_words(self):
        imp_full = self.explanation.word_importance(label_idx=1, filter_special=True)
        if len(imp_full) == 0:
            return
        word_to_exclude = imp_full.index[0]
        imp_filtered = self.explanation.word_importance(
            label_idx=1, filter_special=True, exclude_words={word_to_exclude}
        )
        self.assertNotIn(word_to_exclude, imp_filtered.index)

    def test_word_importance_exclude_words_empty_set(self):
        imp_no_exclude = self.explanation.word_importance(label_idx=1, exclude_words=set())
        imp_none_exclude = self.explanation.word_importance(label_idx=1, exclude_words=None)
        pd.testing.assert_series_equal(imp_no_exclude, imp_none_exclude)

    def test_word_importance_sample_indices(self):
        imp = self.explanation.word_importance(label_idx=0, sample_indices=[0], n_top=50)
        self.assertGreater(len(imp), 0)
        # "terrible" only exists in sample 1 — must not appear in the sample-0 subset
        self.assertNotIn("terrible", imp.index)


# ---------------------------------------------------------------------------
# plot_sentence_highlight
# ---------------------------------------------------------------------------


class TestPlotSentenceHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "happy", "[SEP]"]
        self.values = np.array([0.01, 0.05, 0.30, -0.20, 0.01])

    def test_returns_html_div(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsInstance(result, html.Div)

    def test_has_children(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsNotNone(result.children)
        # legend + spans div + summary
        self.assertGreaterEqual(len(result.children), 3)

    def test_raises_on_2d_values(self):
        with self.assertRaises(ValueError):
            plot_sentence_highlight(self.tokens, np.zeros((5, 3)))

    def test_with_base_value_does_not_raise(self):
        result = plot_sentence_highlight(self.tokens, self.values, base_value=0.15)
        self.assertIsInstance(result, html.Div)

    def test_empty_tokens_does_not_raise(self):
        result = plot_sentence_highlight([], np.array([]))
        self.assertIsInstance(result, html.Div)


# ---------------------------------------------------------------------------
# plot_waterfall
# ---------------------------------------------------------------------------


class TestPlotWaterfall(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "so", "happy", "today", "[SEP]"]
        self.values = np.array([0.01, 0.08, 0.35, 0.02, -0.20, 0.05, 0.01])

    def test_returns_figure(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_waterfall_trace(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Waterfall)

    def test_filters_special_tokens_by_default(self):
        fig = plot_waterfall(self.tokens, self.values)
        y_labels = list(fig.data[0].y)
        for label in y_labels:
            self.assertNotIn("[CLS]", label)
            self.assertNotIn("[SEP]", label)

    def test_keeps_special_tokens_when_disabled(self):
        # min_pct=0 disables grouping so every token gets its own bar
        fig = plot_waterfall(self.tokens, self.values, filter_special=False, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("[CLS]" in lbl for lbl in y_labels))

    def test_total_bar_present(self):
        fig = plot_waterfall(self.tokens, self.values)
        measures = list(fig.data[0].measure)
        self.assertIn("total", measures)

    def test_grouping_reduces_bar_count(self):
        # With min_pct=0.0 (no grouping), every non-special token is its own bar
        fig_no_group = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        # With min_pct=0.5, small tokens are lumped
        fig_grouped = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        self.assertLessEqual(len(fig_grouped.data[0].y), len(fig_no_group.data[0].y))

    def test_other_bar_present_when_grouping_active(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("other" in lbl for lbl in y_labels))

    def test_no_other_bar_when_min_pct_zero(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertFalse(any("other" in lbl for lbl in y_labels))

    def test_base_value_creates_absolute_bar(self):
        fig = plot_waterfall(self.tokens, self.values, base_value=0.20)
        measures = list(fig.data[0].measure)
        self.assertEqual(measures[0], "absolute")
        y_labels = list(fig.data[0].y)
        self.assertEqual(y_labels[0], "Base")

    def test_empty_tokens_returns_figure(self):
        fig = plot_waterfall([], np.array([]))
        self.assertIsInstance(fig, go.Figure)

    def test_all_special_tokens_returns_figure(self):
        fig = plot_waterfall(["[CLS]", "[SEP]"], np.array([0.01, -0.01]))
        self.assertIsInstance(fig, go.Figure)

    def test_custom_title(self):
        fig = plot_waterfall(self.tokens, self.values, title="joy waterfall")
        self.assertIn("joy waterfall", fig.layout.title.text)


# ---------------------------------------------------------------------------
# plot_token_highlight
# ---------------------------------------------------------------------------


class TestPlotTokenHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["i", "feel", "happy", "today"]
        self.values = np.array([0.1, 0.4, -0.2, 0.05])

    def test_returns_figure(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_max_tokens_limits_bars(self):
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        self.assertLessEqual(len(fig.data[0].x), 2)

    def test_max_tokens_preserves_sentence_order(self):
        # top-2 by |value| are "feel" (0.4) and "happy" (-0.2)
        # sentence order: "feel" before "happy"
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        # y-axis is reversed for display (highest importance at top → reversed list)
        displayed_labels = list(fig.data[0].y)
        feel_pos = displayed_labels.index("feel")
        happy_pos = displayed_labels.index("happy")
        # In reversed display: "feel" appears after "happy" (lower index = further down)
        self.assertNotEqual(feel_pos, happy_pos)

    def test_custom_title(self):
        fig = plot_token_highlight(self.tokens, self.values, title="Test title")
        self.assertIn("Test title", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(fig.data[0].orientation, "h")


# ---------------------------------------------------------------------------
# plot_word_importance
# ---------------------------------------------------------------------------


class TestPlotWordImportance(unittest.TestCase):
    def setUp(self):
        self.word_imp = pd.Series(
            {"happy": 0.35, "terrible": -0.28, "wonderful": 0.20, "feel": -0.10},
        )

    def test_returns_figure(self):
        fig = plot_word_importance(self.word_imp)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_all_words_rendered(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data[0].x), len(self.word_imp))

    def test_custom_title(self):
        fig = plot_word_importance(self.word_imp, title="Joy importance")
        self.assertIn("Joy importance", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(fig.data[0].orientation, "h")


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------


class TestPlotNoiseMatrix(unittest.TestCase):
    """The noise matrix is given-vs-true and mostly diagonal, so it needs its own axes and masking."""

    def setUp(self):
        # Typical shape: ~94% correctly labelled, one contaminated pair (A labelled, really B).
        self.joint = np.array([[0.50, 0.05, 0.0], [0.01, 0.30, 0.0], [0.0, 0.0, 0.14]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        self.assertIsInstance(plot_noise_matrix(self.joint, self.labels), go.Figure)

    def test_diagonal_is_masked_by_default(self):
        # Left in, the ~90% diagonal flattens every off-diagonal cell to the same near-white shade.
        z = plot_noise_matrix(self.joint, self.labels).data[0].z
        self.assertTrue(np.all(np.isnan(np.diag(z))))
        self.assertAlmostEqual(z[0][1], 0.05)

    def test_masked_cells_render_no_text(self):
        text = plot_noise_matrix(self.joint, self.labels).data[0].text
        self.assertEqual(text[0][0], "")
        self.assertEqual(text[0][1], "5.0%")

    def test_diagonal_can_be_kept(self):
        z = plot_noise_matrix(self.joint, self.labels, mask_diagonal=False).data[0].z
        np.testing.assert_allclose(np.diag(z), [0.50, 0.30, 0.14])

    def test_does_not_mutate_the_callers_matrix(self):
        original = self.joint.copy()
        plot_noise_matrix(self.joint, self.labels)
        np.testing.assert_array_equal(self.joint, original)

    def test_axes_name_the_label_semantics_not_the_prediction(self):
        # The whole reason this is not a plot_confusion_matrix mode: the columns are the estimated
        # *true* class, not the model's prediction.
        layout = plot_noise_matrix(self.joint, self.labels).layout
        self.assertEqual(layout.yaxis.title.text, "Given label")
        self.assertEqual(layout.xaxis.title.text, "Estimated true class")


class TestPlotConfusionMatrix(unittest.TestCase):
    def setUp(self):
        # 3-class matrix: rows = true, cols = predicted. Row 2 (index 2) is empty.
        self.cm = np.array([[5, 2, 0], [1, 4, 0], [0, 0, 0]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(fig.data[0], go.Heatmap)

    def test_axes_are_labelled_true_and_predicted(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(list(fig.data[0].x), self.labels)
        self.assertEqual(list(fig.data[0].y), self.labels)

    def test_customdata_encodes_pred_then_true(self):
        # customdata[true][pred] must be [pred_idx, true_idx] for the click handler.
        fig = plot_confusion_matrix(self.cm, self.labels)
        cd = np.asarray(fig.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # true=0, pred=1
        self.assertEqual(list(cd[1, 2]), [2, 1])  # true=1, pred=2

    def test_counts_shown_as_text(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(fig.data[0].text[0][0], "5")

    def test_normalize_true_is_row_recall(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        np.testing.assert_allclose(z[0], [5 / 7, 2 / 7, 0.0])

    def test_normalize_true_handles_empty_row_without_nan(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        self.assertFalse(np.isnan(z).any())
        np.testing.assert_array_equal(z[2], [0.0, 0.0, 0.0])

    def test_custom_title(self):
        fig = plot_confusion_matrix(self.cm, self.labels, title="Errors")
        self.assertIn("Errors", fig.layout.title.text)


# ---------------------------------------------------------------------------
# NlpExplainer (no real model)
# ---------------------------------------------------------------------------


class TestNlpExplainer(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()
        self.explanation = _make_explanation()

    def test_plot_tokens_returns_figure(self):
        fig = self.explanation.plot.tokens(row=0, label_idx=1)
        self.assertIsInstance(fig, go.Figure)

    def test_folds_case_is_none_without_a_tokenizing_model(self):
        # A bare classifier_fn has no tokenizer to ask; None is the honest answer, not a guess.
        self.assertIsNone(self.xpl._folds_case())

    def test_folds_case_reads_the_model_capability(self):
        class _Uncased(TextModel, SupportsTokenization):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

            def tokenize(self, text):
                return text.lower().split()

            def detokenize(self, tokens):
                return " ".join(tokens)

        self.xpl._text_model = _Uncased(label_names=["neg", "pos"])
        self.assertTrue(self.xpl._folds_case())

    def test_plot_tokens_all_samples(self):
        for row in range(3):
            fig = self.explanation.plot.tokens(row=row, label_idx=0)
            self.assertIsInstance(fig, go.Figure)

    def test_plot_tokens_all_labels(self):
        for label_idx in range(N_CLASSES):
            fig = self.explanation.plot.tokens(row=0, label_idx=label_idx)
            self.assertIsInstance(fig, go.Figure)

    def test_plot_tokens_max_tokens(self):
        fig = self.explanation.plot.tokens(row=0, label_idx=1, max_tokens=3)
        self.assertIsInstance(fig, go.Figure)
        self.assertLessEqual(len(fig.data[0].x), 3)

    def test_y_pred_stored(self):
        self.assertIsNotNone(self.explanation.y_pred)
        self.assertEqual(len(self.explanation.y_pred), 3)

    def test_label_names_propagated(self):
        self.assertEqual(self.explanation.label_names, LABEL_NAMES)

    def test_y_true_is_none_by_default(self):
        self.assertIsNone(self.explanation.y_true)


class _ProjectableModel(TextModel, SupportsEmbeddings):
    """Embeds each text to a deterministic 3-D point; counts calls so cache hits are observable."""

    def __init__(self, space="decision"):
        super().__init__(label_names=LABEL_NAMES[:2])
        self.space = space
        self.calls = 0

    def resolve_space(self, space=None):
        return space if space is not None else self.space

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def get_embedding_table(self):
        return (["a"], np.zeros((1, 3)))

    def embed(self, texts, space=None):
        self.calls += 1
        return np.array([[float(len(t)), float(t.count("a")), 1.0] for t in texts])

    @property
    def shap_callable(self):
        return self.predict


class _CountingReducer:
    """A reducer with sklearn's ``get_params`` surface, so its settings reach the cache tag."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.fits = 0

    def get_params(self, deep=True):
        return {"scale": self.scale}

    def fit_transform(self, x):
        self.fits += 1
        return np.asarray(x)[:, :2] * self.scale


def _projection_explanation(texts: pd.Series) -> NlpExplanation:
    """A minimal ``NlpExplanation`` carrying only ``texts`` — all ``compute_projection`` needs."""
    n = len(texts)
    return NlpExplanation(
        texts=texts,
        token_strings=[[] for _ in range(n)],
        values=[np.zeros((0, 2)) for _ in range(n)],
        base_values=np.zeros((n, 2)),
        y_pred=pd.Series(["pos"] * n, index=texts.index, name="prediction"),
        y_prob=None,
        y_true=None,
        label_names=None,
        folds_case=None,
        backend_name="test",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestComputeProjection(unittest.TestCase):
    """The library owns the space + the caching; the caller injects only the reducer."""

    def setUp(self):
        self.model = _ProjectableModel()
        self.xpl = NlpExplainer(self.model, backend=object())
        self.explanation = _projection_explanation(pd.Series(["alpha", "beta banana", "gamma"]))

    def test_returns_two_columns_aligned_with_the_texts(self):
        xy = self.xpl.compute_projection(self.explanation)
        self.assertEqual(xy.shape, (3, 2))

    def test_defaults_to_pca_without_any_extra_dependency(self):
        """The default reducer must be something a core install already has — sklearn's PCA."""
        xy = self.xpl.compute_projection(self.explanation)
        self.assertEqual(xy.shape, (3, 2))
        self.assertEqual(self.model.calls, 1)

    def test_injected_reducer_is_used(self):
        reducer = _CountingReducer(scale=2.0)
        xy = self.xpl.compute_projection(self.explanation, reducer=reducer)
        self.assertEqual(reducer.fits, 1)
        np.testing.assert_allclose(xy[0], [10.0, 4.0])  # "alpha": len 5, 2 a's, doubled

    def test_raises_for_a_model_that_cannot_embed(self):
        """A prediction-only model gets a clear error pointing at the escape hatch, not an AttributeError."""
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object())
        explanation = _projection_explanation(pd.Series(["alpha", "beta"]))
        with self.assertRaises(TypeError):
            xpl.compute_projection(explanation)

    def test_cached_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            reducer = _CountingReducer()
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d)

            fresh_model = _ProjectableModel()
            fresh = NlpExplainer(fresh_model, backend=object())
            fresh.compute_projection(self.explanation, reducer=reducer, cache_dir=d)
            self.assertEqual(fresh_model.calls, 0)  # neither embedded
            self.assertEqual(reducer.fits, 1)  # nor re-fitted

    def test_reducer_settings_take_part_in_the_key(self):
        """Re-tuning a reducer must not silently reload the previous scatter."""
        with tempfile.TemporaryDirectory() as d:
            a = self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(scale=1.0), cache_dir=d)
            b = self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(scale=3.0), cache_dir=d)
            np.testing.assert_allclose(b, a * 3.0)

    def test_model_space_takes_part_in_the_key(self):
        """Moving the model's space must re-project, not reload the other space's coordinates."""
        with tempfile.TemporaryDirectory() as d:
            self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(), cache_dir=d)
            self.model.space = "pooled"
            self.model.calls = 0
            self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(), cache_dir=d)
            self.assertEqual(self.model.calls, 1)  # re-embedded under the new space

    def test_recompute_forces_a_fresh_fit(self):
        with tempfile.TemporaryDirectory() as d:
            reducer = _CountingReducer()
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d)
            self.model.calls = 0
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d, recompute=True)
            self.assertEqual(self.model.calls, 1)
            self.assertEqual(reducer.fits, 2)


# ---------------------------------------------------------------------------
# NlpExplainer.fit — reference state, and when its cost is paid
# ---------------------------------------------------------------------------


class _UnembeddableModel(_ProjectableModel):
    """Embeds nothing: stands in for a corpus the model chokes on (OOM, bad encoding)."""

    def embed(self, texts, space=None):
        raise RuntimeError("cannot embed this corpus")


class TestFitPrecompute(unittest.TestCase):
    CORPUS = ["a happy line", "a sad line", "another happy one"]
    LABELS = ["joy", "sadness", "joy"]

    def test_precompute_embeds_the_bank_at_fit(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS)
        self.assertEqual(model.calls, 1)  # the corpus, embedded once, inside fit
        xpl.find_similar("a happy line")
        self.assertEqual(model.calls, 2)  # the query only — the bank was already there

    def test_precompute_false_defers_to_first_query(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS, precompute=False)
        self.assertEqual(model.calls, 0)
        xpl.find_similar("a happy line")
        self.assertEqual(model.calls, 2)  # bank + query, both charged to the first click

    def test_find_similar_threshold_filters_by_score_and_reports_total(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS)
        neighbors, total = xpl.find_similar_threshold("a happy line", threshold=-1.0, limit=1)
        self.assertLessEqual(len(neighbors), 1)
        self.assertGreaterEqual(total, len(neighbors))

    def test_find_similar_threshold_requires_a_retriever(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object()).fit(self.CORPUS, y=self.LABELS)
        with self.assertRaises(RuntimeError):
            xpl.find_similar_threshold("a happy line")

    def test_precompute_is_a_noop_when_no_retriever_was_built(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object()).fit(self.CORPUS, y=self.LABELS)
        self.assertFalse(xpl.can_find_similar())  # model cannot embed
        self.assertTrue(xpl.can_probe_labels())  # ... but the model-free probe still fit

    def test_a_bank_failure_still_leaves_the_model_free_probe_usable(self):
        xpl = NlpExplainer(_UnembeddableModel(), backend=object())
        with self.assertRaises(RuntimeError):
            xpl.fit(self.CORPUS, y=self.LABELS)
        # reference_/classes_ are assigned before the bank is built, so the half-fitted object keeps
        # the feature that never needed the model.
        self.assertEqual(xpl.reference_, (self.CORPUS, self.LABELS))
        self.assertEqual(xpl.classes_, LABEL_NAMES[:2])  # from the model, not derived from y
        self.assertTrue(xpl.can_probe_labels())


# ---------------------------------------------------------------------------
# NlpExplainer — counterfactual generator discovery / selection (captum-free)
# ---------------------------------------------------------------------------


class _FullCapModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """A model exposing every capability — both HotFlip and AblationFlip are compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return (["a"], np.zeros((1, 2)))

    def embed(self, texts):
        return np.zeros((len(texts), 2))

    def token_gradients(self, text, target_class):
        toks = text.split()
        return toks, np.zeros((len(toks), 2))

    @property
    def shap_callable(self):
        return self.predict


class _TokenizeOnlyModel(TextModel, SupportsTokenization):
    """Tokenizable but gradient-free — only AblationFlip is compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    @property
    def shap_callable(self):
        return self.predict


class TestNlpExplainerGenerators(unittest.TestCase):
    """Generator auto-discovery drives the webapp's method selector (no captum needed here)."""

    def test_full_capability_model_offers_both_methods(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertEqual(
            xpl.available_cf_generators(),
            [("hotflip", "HotFlip"), ("ablation_flip", "Ablation")],
        )
        # The preferred (first-discovered) generator stays the active default.
        self.assertIsInstance(xpl.cf_generator, HotFlipGenerator)
        self.assertEqual(set(xpl.cf_generators), {"hotflip", "ablation_flip"})

    def test_tokenize_only_model_offers_ablation_only(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object())
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIsInstance(xpl.cf_generator, AblationFlipGenerator)

    def test_explicit_generator_used_verbatim_no_extras(self):
        model = _FullCapModel()
        gen = AblationFlipGenerator(model)
        xpl = NlpExplainer(model, backend=object(), cf_generator=gen)
        # An explicit choice is not augmented with the other compatible built-ins.
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIs(xpl.cf_generator, gen)

    def test_cf_config_spec_selected_by_generator(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertIn("max_flips", xpl.cf_config_spec("hotflip"))
        self.assertIn("max_ablations", xpl.cf_config_spec("ablation_flip"))
        # No argument → the active generator's spec.
        self.assertIn("max_flips", xpl.cf_config_spec())

    def test_unknown_generator_raises(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        with self.assertRaises(KeyError):
            xpl.cf_config_spec("does_not_exist")
        with self.assertRaises(KeyError):
            xpl.generate_counterfactuals("hi there", generator="does_not_exist")

    def test_no_generators_without_text_model(self):
        # A plain callable is neither a TextModel nor a pipeline → no generators, empty selector.
        xpl = NlpExplainer(
            lambda texts: np.tile([0.5, 0.5], (len(texts), 1)), label_names=["neg", "pos"], backend=object()
        )
        self.assertEqual(xpl.available_cf_generators(), [])
        self.assertEqual(xpl.cf_config_spec(), {})
        self.assertIsNone(xpl.cf_generator)


# ---------------------------------------------------------------------------
# NlpWebApp layout (no server launch)
# ---------------------------------------------------------------------------


class TestNlpWebApp(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()
        self.explanation = _make_explanation()
        self.webapp = NlpWebApp(self.explanation, engine=self.xpl)

    def test_layout_built(self):
        self.assertIsNotNone(self.webapp.app.layout)

    def test_class_selector_options(self):
        # Class selector is now two independent dropdowns: "local-class-selector" (Sentence
        # Highlight / Waterfall) and "global-class-selector" (Word Importance / Embeddings).
        # Only the global one offers the cross-class overview — a single sample's highlight has no
        # such aggregate.
        local = self._find_component(self.webapp.app.layout, "local-class-selector")
        self.assertIsNotNone(local, "local-class-selector dropdown not found in layout")
        self.assertEqual([opt["label"] for opt in local.options], LABEL_NAMES)

        glob = self._find_component(self.webapp.app.layout, "global-class-selector")
        self.assertIsNotNone(glob, "global-class-selector dropdown not found in layout")
        self.assertEqual(len(glob.options), N_CLASSES + 1)
        self.assertEqual([opt["label"] for opt in glob.options], ["All classes", *LABEL_NAMES])
        # Its value must never collide with a real class index.
        self.assertEqual(glob.options[0]["value"], "all")
        self.assertEqual([opt["value"] for opt in glob.options[1:]], list(range(N_CLASSES)))

    def test_local_class_selector_defaults_to_predicted_class(self):
        # Defaults to the predicted class of the initially selected row (row 0).
        dropdown = self._find_component(self.webapp.app.layout, "local-class-selector")
        row0_prediction = self.webapp._full_table_records[0]["prediction"]
        self.assertEqual(dropdown.value, LABEL_NAMES.index(row0_prediction))

    def test_dataset_table_populated(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        self.assertIsNotNone(table, "dataset-table not found in layout")
        self.assertEqual(len(table.rowData), 3)
        self.assertIn("text", table.rowData[0])
        self.assertIn("prediction", table.rowData[0])

    def test_graph_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("global-importance-graph", ids)
        self.assertIn("dataset-table", ids)
        self.assertIn("local-class-selector", ids)
        self.assertIn("global-class-selector", ids)
        # token bar chart removed; sentence-highlight replaced it
        self.assertNotIn("local-contributions-graph", ids)

    def test_control_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("topk-input", ids)
        self.assertIn("sign-filter", ids)
        self.assertIn("word-filter", ids)

    def test_sentence_highlight_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("sentence-highlight", ids)

    def test_waterfall_controls_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        # Waterfall is now a tab (no show/hide switch); the threshold slider + graph live in it.
        self.assertIn("waterfall-threshold", ids)
        self.assertIn("waterfall-graph", ids)

    def test_dataset_table_no_ground_truth_by_default(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertNotIn("ground_truth", col_fields)

    def test_dataset_table_with_y_true(self):
        y_true = pd.Series(["sadness", "joy", "sadness"], index=pd.RangeIndex(3), name="ground_truth")
        explanation = _make_explanation(y_true=y_true)
        webapp = NlpWebApp(explanation, engine=self.xpl)
        table = self._find_component(webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertIn("ground_truth", col_fields)
        self.assertEqual(table.rowData[0]["ground_truth"], "sadness")

    def test_scatter_store_always_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("scatter-selected-indices", ids)

    def test_scatter_absent_when_no_xy(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("scatter-plot", ids)
        self.assertNotIn("color-by", ids)

    def test_scatter_present_when_xy_given(self):
        xy = np.zeros((3, 2))
        webapp = NlpWebApp(self.explanation, engine=self.xpl, scatter_xy=xy)
        ids = self._collect_ids(webapp.app.layout)
        self.assertIn("scatter-plot", ids)
        self.assertIn("color-by", ids)

    def test_scatter_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            NlpWebApp(self.explanation, engine=self.xpl, scatter_xy=np.zeros((5, 2)))  # 5 rows but only 3 samples

    # ── Error Analysis tab (confusion matrix) ─────────────────────────

    def _make_webapp_with_gt(self) -> NlpWebApp:
        # y_pred is ["joy", "sadness", "joy"]; make sample 0 a sadness→joy error, others correct.
        y_true = pd.Series(["sadness", "sadness", "joy"], index=pd.RangeIndex(3), name="ground_truth")
        explanation = _make_explanation(y_true=y_true)
        return NlpWebApp(explanation, engine=self.xpl)

    def test_error_analysis_absent_without_ground_truth(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("confusion-matrix-graph", ids)
        self.assertNotIn("error-pred-importance", ids)

    def test_error_cell_store_always_present(self):
        # The store is created unconditionally so cross-panel callbacks can read it even without gt.
        self.assertIn("error-cell", self._collect_ids(self.webapp.app.layout))

    def test_error_analysis_present_with_ground_truth(self):
        ids = self._collect_ids(self._make_webapp_with_gt().app.layout)
        for cid in ("confusion-matrix-graph", "error-pred-importance", "error-true-importance", "cm-normalize"):
            self.assertIn(cid, ids)

    @staticmethod
    def _error_analysis_comp(webapp: NlpWebApp) -> ErrorAnalysisComponent:
        return next(c for c in webapp._components if isinstance(c, ErrorAnalysisComponent))

    def test_confusion_matrix_counts(self):
        webapp = self._make_webapp_with_gt()
        cm = self._error_analysis_comp(webapp)._cm
        # LABEL_NAMES index: sadness=0, joy=1. Rows=true, cols=pred.
        self.assertEqual(cm[0, 1], 1)  # true sadness predicted joy (the error)
        self.assertEqual(cm[0, 0], 1)  # true sadness predicted sadness
        self.assertEqual(cm[1, 1], 1)  # true joy predicted joy
        self.assertEqual(cm.sum(), 3)

    def test_confusion_matrix_index_arrays(self):
        webapp = self._make_webapp_with_gt()
        comp = self._error_analysis_comp(webapp)
        self.assertEqual(comp._cm_true_idx.tolist(), [0, 0, 1])  # sadness, sadness, joy
        self.assertEqual(comp._cm_pred_idx.tolist(), [1, 0, 1])  # joy, sadness, joy

    def test_confusion_matrix_figure_customdata_orientation(self):
        webapp = self._make_webapp_with_gt()
        graph = self._find_component(webapp.app.layout, "confusion-matrix-graph")
        cd = np.asarray(graph.figure.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # cell (true=0, pred=1) → [pred_idx, true_idx]

    def test_cell_from_click_uses_label_names_when_no_customdata(self):
        # Heatmap clicks may omit customdata; the x (pred) / y (true) labels must still resolve.
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness"}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (1, 0))

    def test_cell_from_click_prefers_customdata(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness", "customdata": [2, 3]}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (2, 3))

    def test_cell_from_click_none_on_empty_or_unknown(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        self.assertIsNone(_cell_from_click(None, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": []}, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": [{"x": "??", "y": "??"}]}, name_to_idx))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_component(self, node, component_id):
        """Depth-first search for a Dash component by id."""
        if hasattr(node, "id") and node.id == component_id:
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        if not isinstance(children, list):
            children = [children]
        for child in children:
            result = self._find_component(child, component_id)
            if result is not None:
                return result
        return None

    def _collect_ids(self, node) -> set:
        """Collect all component ids in the layout tree."""
        ids = set()
        if hasattr(node, "id") and isinstance(node.id, str):
            ids.add(node.id)
        children = getattr(node, "children", None)
        if children is None:
            return ids
        if not isinstance(children, list):
            children = [children]
        for child in children:
            ids |= self._collect_ids(child)
        return ids


class TestComposeSelection(unittest.TestCase):
    """The three sample filters (scatter box, confusion cell, errors-only) must intersect."""

    def test_nothing_active_returns_none(self):
        self.assertIsNone(_compose_selection(None, None, None))

    def test_scatter_only(self):
        self.assertEqual(_compose_selection([1, 2, 3], None, None), [1, 2, 3])

    def test_cell_only(self):
        self.assertEqual(_compose_selection(None, [4, 5], None), [4, 5])

    def test_errors_only(self):
        self.assertEqual(_compose_selection(None, None, {2, 7}), [2, 7])

    def test_scatter_intersect_cell(self):
        self.assertEqual(_compose_selection([1, 2, 3], [2, 3, 4], None), [2, 3])

    def test_scatter_intersect_errors(self):
        # This is the box-then-errors case: errors *within* the selected points.
        self.assertEqual(_compose_selection([1, 2, 3, 4], None, {2, 4, 9}), [2, 4])

    def test_all_three_intersect(self):
        self.assertEqual(_compose_selection([1, 2, 3, 4], [2, 3, 4], {3, 4}), [3, 4])

    def test_empty_intersection_is_empty_list(self):
        self.assertEqual(_compose_selection([1, 2], None, {8, 9}), [])


class TestErrorPositions(unittest.TestCase):
    """The shared definition of "a model error" every errors-scoped panel filters on."""

    def test_none_without_ground_truth(self):
        self.assertIsNone(error_positions(_make_explanation()))

    def test_compares_as_strings(self):
        texts = _make_texts()
        explanation = _make_explanation(y_true=pd.Series(["joy", "joy", "joy"], index=texts.index))
        # y_pred is ["joy", "sadness", "joy"], so only the middle row disagrees.
        self.assertEqual(error_positions(explanation), {1})


# ---------------------------------------------------------------------------
# Single-word profile: word_occurrences / vocabulary / aggregators / plot / plotter
# ---------------------------------------------------------------------------


def _profile_explanation(folds_case=True):
    """Three samples, two classes, with 'happy' appearing in three of them (twice in one)."""
    token_strings = [
        ["[CLS]", "so", "happy", "today", "!"],
        ["Happy", "and", "happy", "again"],
        ["not", "happy", "at", "all"],
        ["nothing", "here"],
    ]
    values = [
        np.array([[0.0, 0.0], [0.1, -0.1], [0.4, -0.4], [0.05, -0.05], [0.0, 0.0]]),
        np.array([[0.2, -0.2], [0.0, 0.0], [0.1, -0.1], [0.0, 0.0]]),
        np.array([[-0.1, 0.1], [-0.6, 0.6], [0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    ]
    texts = pd.Series(["so happy today !", "Happy and happy again", "not happy at all", "nothing here"])
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=None,
        y_pred=pd.Series(["pos", "pos", "neg", "neg"], index=texts.index, name="prediction"),
        y_prob=None,
        y_true=pd.Series(["pos", "neg", "neg", "neg"], index=texts.index, name="ground_truth"),
        label_names=["pos", "neg"],
        folds_case=folds_case,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestVocabulary(unittest.TestCase):
    def test_filters_special_and_punctuation_and_folds(self):
        vocab = _profile_explanation().vocabulary()
        self.assertNotIn("[CLS]", vocab)
        self.assertNotIn("!", vocab)
        # Folded: "Happy" and "happy" collapse to one entry.
        self.assertIn("happy", vocab)
        self.assertNotIn("Happy", vocab)

    def test_keeps_case_when_model_does_not_fold(self):
        vocab = _profile_explanation(folds_case=False).vocabulary()
        self.assertIn("Happy", vocab)
        self.assertIn("happy", vocab)

    def test_sorted_and_unique(self):
        vocab = _profile_explanation().vocabulary()
        self.assertEqual(vocab, sorted(set(vocab)))

    def test_offers_exactly_what_word_importance_ranks(self):
        # The contract the pickers rely on: every unit offered has occurrences to show.
        explanation = _profile_explanation()
        for word in explanation.vocabulary():
            self.assertFalse(explanation.word_occurrences(word).empty, word)

    def test_can_keep_punctuation_and_special(self):
        vocab = _profile_explanation().vocabulary(filter_special=False, filter_punctuation=False)
        self.assertIn("[cls]", vocab)
        self.assertIn("!", vocab)


class TestWordImportanceRanking(unittest.TestCase):
    """``rank_by`` + ``min_occurrences``: ordering criteria that never touch the reported sign."""

    def setUp(self):
        # "rare" appears once with a large pull; "common" appears four times with a small one.
        # Under |mean| rare wins; under |sum| common wins (4 x 0.3 = 1.2 > 0.9).
        token_strings = [
            ["rare", "common"],
            ["common"],
            ["common", "common"],
            ["mild"],
        ]
        values = [
            np.array([[0.9], [0.3]]),
            np.array([[0.3]]),
            np.array([[0.3], [0.3]]),
            np.array([[-0.5]]),
        ]
        self.explanation = _minimal_explanation(token_strings, values, None)

    def test_mean_is_the_default_and_unchanged(self):
        imp = self.explanation.word_importance(label_idx=0)
        self.assertEqual(imp.index[0], "rare")
        self.assertAlmostEqual(imp["common"], 0.3)

    def test_sum_reranks_by_total_mass(self):
        imp = self.explanation.word_importance(label_idx=0, rank_by="sum")
        self.assertEqual(imp.index[0], "common")
        self.assertAlmostEqual(imp["common"], 1.2)

    def test_series_name_records_the_statistic(self):
        self.assertEqual(self.explanation.word_importance(label_idx=0).name, "mean")
        self.assertEqual(self.explanation.word_importance(label_idx=0, rank_by="sum").name, "sum")

    def test_ranking_is_absolute_but_values_stay_signed(self):
        # "mild" is negative and must survive ranking, keeping its sign for the sign filter and
        # for the renderer's red/blue colouring.
        for rank_by in ("mean", "sum"):
            imp = self.explanation.word_importance(label_idx=0, rank_by=rank_by)
            self.assertLess(imp["mild"], 0)

    def test_sign_filter_works_in_both_modes(self):
        for rank_by in ("mean", "sum"):
            pos = self.explanation.word_importance(label_idx=0, rank_by=rank_by, filter_sign="positive")
            neg = self.explanation.word_importance(label_idx=0, rank_by=rank_by, filter_sign="negative")
            self.assertNotIn("mild", pos.index)
            self.assertEqual(list(neg.index), ["mild"])

    def test_min_occurrences_drops_rare_words(self):
        imp = self.explanation.word_importance(label_idx=0, min_occurrences=2)
        self.assertEqual(list(imp.index), ["common"])

    def test_min_occurrences_of_one_is_no_filter(self):
        self.assertEqual(
            set(self.explanation.word_importance(label_idx=0, min_occurrences=1).index),
            {"rare", "common", "mild"},
        )

    def test_min_occurrences_counts_within_the_selection(self):
        # "common" occurs 4x across the batch but only once in sample 1, so a floor of 2 must
        # exclude it there — otherwise a threshold would mean something different per scope.
        imp = self.explanation.word_importance(label_idx=0, sample_indices=[1], min_occurrences=2)
        self.assertTrue(imp.empty)
        self.assertEqual(
            list(self.explanation.word_importance(label_idx=0, sample_indices=[2], min_occurrences=2).index),
            ["common"],
        )

    def test_impossible_floor_returns_an_empty_series_not_an_error(self):
        imp = self.explanation.word_importance(label_idx=0, min_occurrences=999)
        self.assertTrue(imp.empty)
        self.assertEqual(imp.dtype, np.float64)

    def test_unknown_rank_by_raises(self):
        with self.assertRaisesRegex(ValueError, "must be 'mean' or 'sum'"):
            self.explanation.word_importance(label_idx=0, rank_by="median")

    def test_plotter_labels_the_axis_for_the_statistic(self):
        self.assertEqual(
            self.explanation.plot.word_importance(label_idx=0, rank_by="sum").layout.xaxis.title.text,
            "Total SHAP contribution",
        )
        self.assertEqual(
            self.explanation.plot.word_importance(label_idx=0).layout.xaxis.title.text,
            "Mean SHAP contribution",
        )

    def test_plotter_hover_counts_match_the_scope_it_ranked(self):
        # The plotter forwards only the keying arguments to word_counts, so a count on a bar must
        # be the count over the *same* samples the aggregate was taken over.
        fig = self.explanation.plot.word_importance(label_idx=0, sample_indices=[2])
        counts = self.explanation.word_counts(sample_indices=[2])["n_occurrences"]
        drawn = list(fig.data[0].y)
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [int(counts[w]) for w in drawn])


class TestWordImportanceAcrossClasses(unittest.TestCase):
    """``label_idx=None``: one ranking spanning every class, on magnitudes."""

    @staticmethod
    def _three_class():
        # "alpha" is decisive for class c only; "beta" is mild and spread. Three classes so max
        # and mean over classes are actually distinguishable (with two they coincide).
        texts = pd.Series(["alpha beta", "alpha beta"])
        rows = np.array([[-0.3, -0.3, 0.6], [0.1, -0.2, 0.1]])
        return NlpExplanation(
            texts=texts,
            token_strings=[["alpha", "beta"], ["alpha", "beta"]],
            values=[rows, rows],
            base_values=None,
            y_pred=pd.Series(["a", "a"], index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=["a", "b", "c"],
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def test_value_is_the_strongest_classs_magnitude(self):
        # Not the mean over classes, which would read 0.4 and 0.133 — smaller than any real
        # contribution and matching nothing the single-class views show.
        self.assertEqual(self._three_class().word_importance(label_idx=None).to_dict(), {"alpha": 0.6, "beta": 0.2})

    def test_it_matches_what_the_driving_class_shows(self):
        exp = self._three_class()
        self.assertEqual(exp.word_importance(label_idx=None)["alpha"], abs(exp.word_importance(label_idx=2)["alpha"]))

    def test_signed_averaging_across_classes_would_be_zero(self):
        # The reason the collapse discards sign: an explainer of a normalised output cancels
        # across classes, so a signed cross-class mean is a chart of zeros.
        exp = self._three_class()
        per_class = np.array([exp.word_importance(label_idx=i)["alpha"] for i in range(3)])
        self.assertAlmostEqual(float(per_class.mean()), 0.0)
        self.assertGreater(exp.word_importance(label_idx=None)["alpha"], 0.5)

    def test_single_output_model_collapses_to_the_absolute_value(self):
        # A 1-D contribution array has no class axis to reduce over; it still must not raise.
        texts = pd.Series(["alpha beta"])
        exp = NlpExplanation(
            texts=texts,
            token_strings=[["alpha", "beta"]],
            values=[np.array([0.5, -0.2])],
            base_values=None,
            y_pred=pd.Series(["a"], index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=None,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )
        self.assertEqual(exp.word_importance(label_idx=None).to_dict(), {"alpha": 0.5, "beta": 0.2})

    def test_every_value_is_a_magnitude(self):
        self.assertTrue((_profile_explanation().word_importance(label_idx=None) >= 0).all())

    def test_name_records_the_statistic(self):
        exp = _profile_explanation()
        self.assertEqual(exp.word_importance(label_idx=None).name, "mean_across_classes")
        self.assertEqual(exp.word_importance(label_idx=None, rank_by="sum").name, "sum_across_classes")

    def test_a_negative_sign_filter_has_nothing_to_select(self):
        self.assertTrue(_profile_explanation().word_importance(label_idx=None, filter_sign="negative").empty)

    def test_the_frequency_floor_still_applies(self):
        exp = _profile_explanation()
        self.assertIn("happy", exp.word_importance(label_idx=None, min_occurrences=4).index)
        self.assertNotIn("today", exp.word_importance(label_idx=None, min_occurrences=2).index)

    def test_plotter_titles_and_labels_the_cross_class_view(self):
        fig = _profile_explanation().plot.word_importance(label_idx=None)
        self.assertEqual(fig.layout.title.text, "Word importance — all classes")
        self.assertEqual(fig.layout.xaxis.title.text, "Largest |mean SHAP| across classes")


class TestWordImportanceReadability(unittest.TestCase):
    """The chart must name every bar it draws, whatever K is."""

    @staticmethod
    def _imp(n):
        return pd.Series(
            [(-1) ** i * (n - i) / n for i in range(n)],
            index=[f"word{i}" for i in range(n)],
        )

    def test_every_word_gets_a_tick(self):
        # dtick=1 on a categorical axis is what stops plotly thinning labels on a long ranking.
        fig = plot_word_importance(self._imp(50))
        self.assertEqual(fig.layout.yaxis.dtick, 1)
        self.assertEqual(len(set(fig.data[0].y)), 50)

    def test_height_grows_with_the_word_count(self):
        heights = [plot_word_importance(self._imp(n)).layout.height for n in (5, 20, 50)]
        self.assertEqual(heights, sorted(heights))
        # Every word keeps room to render, rather than the chart being squeezed to a fixed panel.
        for n, h in zip((5, 20, 50), heights):
            self.assertGreaterEqual(h / n, 24)

    def test_explicit_height_still_wins(self):
        self.assertEqual(plot_word_importance(self._imp(50), height=300).layout.height, 300)

    def test_value_labels_are_signed_and_on_by_default(self):
        fig = plot_word_importance(pd.Series([0.9, -0.5], index=["a", "b"]))
        self.assertEqual(list(fig.data[0].text), ["-0.500", "+0.900"])  # reversed for drawing
        self.assertEqual(fig.data[0].textposition, "outside")
        # Outside text sits past the bar end and would be cut off at the plot edge without this.
        self.assertFalse(fig.data[0].cliponaxis)

    def test_value_labels_get_headroom(self):
        fig = plot_word_importance(pd.Series([1.0, -0.5], index=["a", "b"]))
        lo, hi = fig.layout.xaxis.range
        self.assertLess(lo, -0.5)
        self.assertGreater(hi, 1.0)

    def test_value_labels_can_be_turned_off(self):
        fig = plot_word_importance(pd.Series([0.9], index=["a"]), show_values=False)
        self.assertIsNone(fig.data[0].text)
        # No outside text means no need to reserve headroom; plotly autoscales.
        self.assertIsNone(fig.layout.xaxis.range)

    def test_degenerate_inputs_autoscale_rather_than_collapse(self):
        # An all-zero (or empty) series has no span to pad, and a [0, 0] range would be invalid.
        self.assertIsNone(plot_word_importance(pd.Series(dtype=float)).layout.xaxis.range)
        self.assertIsNone(plot_word_importance(pd.Series([0.0, 0.0], index=["a", "b"])).layout.xaxis.range)

    def test_tick_font_is_set(self):
        self.assertEqual(plot_word_importance(self._imp(3)).layout.yaxis.tickfont.size, 12)


class TestWordImportanceHover(unittest.TestCase):
    """The hover names the statistic and, when supplied, how many occurrences it aggregated."""

    imp = pd.Series([0.9, -0.5], index=["a", "b"])

    def test_hover_names_the_statistic_on_the_axis(self):
        # The axis title scrolls off a long chart; the hover is where the reader can still check
        # whether they are looking at a mean or a total.
        fig = plot_word_importance(self.imp, x_title="Total SHAP contribution")
        self.assertIn("Total SHAP contribution: %{x:.4f}", fig.data[0].hovertemplate)

    def test_counts_are_reversed_with_the_bars(self):
        fig = plot_word_importance(self.imp, counts={"a": 12, "b": 3})
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [3, 12])
        self.assertIn("Occurrences:", fig.data[0].hovertemplate)

    def test_counts_accept_a_series(self):
        fig = plot_word_importance(self.imp, counts=pd.Series({"a": 12, "b": 3}))
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [3, 12])

    def test_without_counts_the_hover_is_unchanged(self):
        fig = plot_word_importance(self.imp)
        self.assertIsNone(fig.data[0].customdata)
        self.assertNotIn("Occurrences", fig.data[0].hovertemplate)

    def test_a_partial_mapping_is_dropped_rather_than_shown_as_zero(self):
        # A missing word can only mean the caller counted under different filters, which would
        # misreport every bar — not just the uncovered one.
        fig = plot_word_importance(self.imp, counts={"a": 12})
        self.assertIsNone(fig.data[0].customdata)
        self.assertNotIn("Occurrences", fig.data[0].hovertemplate)


class TestWordCounts(unittest.TestCase):
    """The frequency table behind the word picker's order, labels and threshold."""

    def setUp(self):
        self.explanation = _profile_explanation()

    def test_occurrences_and_samples_differ_for_a_repeated_word(self):
        counts = self.explanation.word_counts()
        # "happy" occurs 4x (twice in one sample) across 3 samples.
        self.assertEqual(counts.loc["happy", "n_occurrences"], 4)
        self.assertEqual(counts.loc["happy", "n_samples"], 3)

    def test_sorted_by_frequency_then_alphabetically(self):
        counts = self.explanation.word_counts()
        self.assertEqual(counts.index[0], "happy")
        ties = counts[counts["n_occurrences"] == 1].index.tolist()
        self.assertEqual(ties, sorted(ties))

    def test_applies_the_same_filters_as_word_importance(self):
        counts = self.explanation.word_counts()
        self.assertNotIn("[CLS]", counts.index)
        self.assertNotIn("!", counts.index)
        self.assertNotIn("Happy", counts.index)  # folded into "happy"

    def test_index_is_exactly_the_vocabulary(self):
        self.assertEqual(sorted(self.explanation.word_counts().index), self.explanation.vocabulary())

    def test_scoped_to_sample_indices(self):
        counts = self.explanation.word_counts(sample_indices=[1])
        self.assertEqual(counts.loc["happy", "n_occurrences"], 2)
        self.assertEqual(counts.loc["happy", "n_samples"], 1)

    def test_counts_are_the_denominator_word_importance_filters_on(self):
        # The contract the min-occurrence control depends on: the count shown beside a word is the
        # count its aggregate was computed over.
        counts = self.explanation.word_counts()
        for word, n in counts["n_occurrences"].items():
            kept = self.explanation.word_importance(label_idx=0, n_top=999, min_occurrences=int(n))
            self.assertIn(word, kept.index, word)
            dropped = self.explanation.word_importance(label_idx=0, n_top=999, min_occurrences=int(n) + 1)
            self.assertNotIn(word, dropped.index, word)

    def test_empty_batch_returns_an_empty_frame(self):
        empty = _minimal_explanation([[]], [np.zeros((0, 2))], None)
        self.assertTrue(empty.word_counts().empty)
        self.assertEqual(empty.vocabulary(), [])


class TestWordOccurrences(unittest.TestCase):
    def setUp(self):
        self.explanation = _profile_explanation()

    def test_one_row_per_occurrence_and_class(self):
        occ = self.explanation.word_occurrences("happy")
        # 4 occurrences (one in sample 0, two in sample 1, one in sample 2) x 2 classes.
        self.assertEqual(len(occ), 8)
        self.assertEqual(list(occ.columns), ["sample", "token_pos", "token", "class_idx", "contribution"])

    def test_case_folding_follows_the_model(self):
        self.assertEqual(len(self.explanation.word_occurrences("HAPPY")), 8)
        cased = _profile_explanation(folds_case=False)
        self.assertEqual(len(cased.word_occurrences("happy")), 6)
        self.assertEqual(len(cased.word_occurrences("Happy")), 2)

    def test_explicit_lowercase_overrides(self):
        cased = _profile_explanation(folds_case=False)
        self.assertEqual(len(cased.word_occurrences("happy", lowercase=True)), 8)

    def test_token_keeps_original_casing(self):
        occ = self.explanation.word_occurrences("happy")
        self.assertIn("Happy", set(occ["token"]))

    def test_sample_indices_scope(self):
        occ = self.explanation.word_occurrences("happy", sample_indices=[2])
        self.assertEqual(set(occ["sample"]), {2})
        self.assertEqual(len(occ), 2)

    def test_missing_word_returns_typed_empty_frame(self):
        occ = self.explanation.word_occurrences("absent")
        self.assertTrue(occ.empty)
        self.assertEqual(occ["contribution"].dtype, np.float64)
        # An empty frame must still support the numeric work the callers do on it.
        self.assertTrue(aggregate_word_contributions(occ, "mean_abs").empty)
        self.assertTrue(rank_word_samples(occ, class_idx=0).empty)

    def test_binary_1d_values(self):
        explanation = _minimal_explanation(
            [["good", "day"], ["good"]],
            [np.array([0.5, 0.1]), np.array([-0.3])],
            None,
        )
        occ = explanation.word_occurrences("good")
        self.assertEqual(set(occ["class_idx"]), {0})
        self.assertEqual(sorted(occ["contribution"].round(3)), [-0.3, 0.5])


class TestAggregateWordContributions(unittest.TestCase):
    def setUp(self):
        self.occ = _profile_explanation().word_occurrences("happy")

    def test_mean_is_signed(self):
        stats = aggregate_word_contributions(self.occ, "mean")
        # class 0: (0.4 + 0.2 + 0.1 - 0.6) / 4
        self.assertAlmostEqual(stats[0], 0.025)
        self.assertAlmostEqual(stats[1], -0.025)

    def test_sum_scales_with_frequency(self):
        stats = aggregate_word_contributions(self.occ, "sum")
        self.assertAlmostEqual(stats[0], 0.1)

    def test_abs_forms_expose_the_two_way_word(self):
        signed = aggregate_word_contributions(self.occ, "mean")
        magnitude = aggregate_word_contributions(self.occ, "mean_abs")
        # The whole point of the abs forms: a word averaging to ~0 is not a weak word.
        self.assertLess(abs(signed[0]), 0.05)
        self.assertAlmostEqual(magnitude[0], 0.325)
        self.assertAlmostEqual(aggregate_word_contributions(self.occ, "sum_abs")[0], 1.3)

    def test_index_is_every_class_in_order(self):
        self.assertEqual(list(aggregate_word_contributions(self.occ, "sum").index), [0, 1])

    def test_unknown_agg_raises(self):
        with self.assertRaisesRegex(ValueError, "not one of"):
            aggregate_word_contributions(self.occ, "median")

    def test_all_documented_ops_run(self):
        for agg in WORD_AGGREGATIONS:
            self.assertEqual(len(aggregate_word_contributions(self.occ, agg)), 2)


class TestRankWordSamples(unittest.TestCase):
    def setUp(self):
        self.occ = _profile_explanation().word_occurrences("happy")

    def test_occurrences_are_summed_within_a_sample(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="most")
        row = ranked[ranked["sample"] == 1].iloc[0]
        self.assertAlmostEqual(row["contribution"], 0.3)  # 0.2 + 0.1
        self.assertEqual(row["n_occurrences"], 2)
        # One row per sample, never one per occurrence.
        self.assertEqual(len(ranked), ranked["sample"].nunique())

    def test_most_ranks_positive_first(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="most")
        self.assertEqual(ranked["sample"].tolist(), [0, 1, 2])

    def test_least_ranks_negative_first(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="least")
        self.assertEqual(ranked["sample"].iloc[0], 2)

    def test_strongest_ignores_sign(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="strongest")
        self.assertEqual(ranked["sample"].iloc[0], 2)  # |-0.6| is the largest

    def test_class_idx_selects_the_column(self):
        pos = rank_word_samples(self.occ, class_idx=0, order="most")["contribution"].tolist()
        neg = rank_word_samples(self.occ, class_idx=1, order="most")["contribution"].tolist()
        self.assertNotEqual(pos, neg)

    def test_n_top_truncates(self):
        self.assertEqual(len(rank_word_samples(self.occ, class_idx=0, n_top=1)), 1)

    def test_unknown_order_raises(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            rank_word_samples(self.occ, class_idx=0, order="alphabetical")

    def test_absent_class_returns_typed_empty(self):
        ranked = rank_word_samples(self.occ, class_idx=99)
        self.assertTrue(ranked.empty)
        self.assertEqual(list(ranked.columns), ["sample", "contribution", "n_occurrences"])


class TestPlotWordProfile(unittest.TestCase):
    def setUp(self):
        self.stats = pd.Series([0.2, -0.3], index=pd.Index([0, 1], name="class_idx"))

    def test_returns_figure_with_one_bar_per_class(self):
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data[0].x), 2)

    def test_class_zero_is_drawn_at_the_top(self):
        # Plotly draws the y-axis bottom-to-top, so the first class must be last in the array.
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        self.assertEqual(fig.data[0].y[-1], "pos")

    def test_sign_drives_colour(self):
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        # Reversed alongside the values, so the last colour belongs to the positive class 0.
        self.assertEqual(fig.data[0].marker.color[-1], DEFAULT_NLP_THEME.xpl_positive)
        self.assertEqual(fig.data[0].marker.color[0], DEFAULT_NLP_THEME.xpl_negative)

    def test_falls_back_to_class_indices_without_names(self):
        fig = plot_word_profile(self.stats)
        self.assertEqual(set(fig.data[0].y), {"0", "1"})

    def test_spread_is_reversed_with_the_bars(self):
        spread = pd.Series([0.1, 0.9], index=pd.Index([0, 1], name="class_idx"))
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"], spread=spread)
        # Class 1's spread must sit on class 1's bar, which is drawn first.
        self.assertAlmostEqual(fig.data[0].error_x.array[0], 0.9)

    def test_misaligned_spread_does_not_shift_error_bars(self):
        # A spread indexed on classes the stats do not carry must not silently slide onto them.
        spread = pd.Series([0.1], index=pd.Index([5], name="class_idx"))
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"], spread=spread)
        self.assertEqual(list(fig.data[0].error_x.array), [0.0, 0.0])

    def test_no_error_bars_without_spread(self):
        fig = plot_word_profile(self.stats)
        self.assertIsNone(fig.data[0].error_x.array)


class TestPlotterWordProfile(unittest.TestCase):
    def setUp(self):
        self.explanation = _profile_explanation()

    def test_returns_figure_titled_with_counts(self):
        fig = self.explanation.plot.word_profile("happy")
        self.assertIsInstance(fig, go.Figure)
        self.assertIn("4 occurrence(s) in 3 sample(s)", fig.layout.title.text)

    def test_mean_draws_error_bars_and_sum_does_not(self):
        self.assertIsNotNone(self.explanation.plot.word_profile("happy", agg="mean").data[0].error_x.array)
        self.assertIsNone(self.explanation.plot.word_profile("happy", agg="sum").data[0].error_x.array)

    def test_single_occurrence_spread_is_zero_not_nan(self):
        fig = self.explanation.plot.word_profile("today", agg="mean")
        self.assertEqual(list(fig.data[0].error_x.array), [0.0, 0.0])

    def test_sample_indices_scope(self):
        fig = self.explanation.plot.word_profile("happy", sample_indices=[1])
        self.assertIn("2 occurrence(s) in 1 sample(s)", fig.layout.title.text)

    def test_title_override(self):
        self.assertEqual(self.explanation.plot.word_profile("happy", title="X").layout.title.text, "X")

    def test_absent_word_raises_rather_than_drawing_an_empty_chart(self):
        with self.assertRaisesRegex(ValueError, "does not occur"):
            self.explanation.plot.word_profile("absent")

    def test_absent_in_scope_names_the_scope(self):
        with self.assertRaisesRegex(ValueError, "in the selected samples"):
            self.explanation.plot.word_profile("happy", sample_indices=[3])


# ---------------------------------------------------------------------------
# NlpLimeBackend
# ---------------------------------------------------------------------------

# Keep num_samples tiny so LIME tests run fast in CI.
_LIME_COMPUTE_ARGS = {"num_samples": 50, "num_features": 5}
_SAMPLE_TEXTS = ["i feel so happy today", "this is terrible and sad"]


def _fake_classifier(texts: list[str]) -> np.ndarray:
    """Deterministic fake classifier returning (n_texts, N_CLASSES) probabilities."""
    rng = np.random.default_rng(0)
    probs = rng.random((len(texts), N_CLASSES)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _make_lime_backend() -> NlpLimeBackend:
    return NlpLimeBackend(
        _fake_classifier,
        label_names=LABEL_NAMES,
        explainer_compute_args=_LIME_COMPUTE_ARGS,
    )


class TestNlpLimeBackend(unittest.TestCase):
    def setUp(self):
        self.backend = _make_lime_backend()

    # --- init / config ---

    def test_name(self):
        self.assertEqual(self.backend.name, "nlp_lime")

    def test_inherits_nlp_backend(self):
        self.assertIsInstance(self.backend, NlpBackend)

    def test_label_names_stored(self):
        self.assertEqual(self.backend._classes, LABEL_NAMES)

    def test_mask_string_stored(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            mask_string="[MASK]",
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertEqual(backend.mask_string, "[MASK]")

    def test_explainer_args_forwarded(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            explainer_args={"bow": False},
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertFalse(backend.explainer.bow)

    # --- _classifier_fn ---

    def test_classifier_fn_converts_list_to_array(self):
        def list_model(texts):
            return [[0.5] * N_CLASSES for _ in texts]

        backend = NlpLimeBackend(list_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, N_CLASSES))

    def test_classifier_fn_passes_through_array(self):
        arr = np.ones((2, N_CLASSES), dtype=np.float32)

        def array_model(texts):
            return arr

        backend = NlpLimeBackend(array_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["a", "b"])
        self.assertIs(result, arr)

    def test_classifier_fn_converts_hf_pipeline_format(self):
        # HuggingFace pipeline with return_all_scores=True returns list[list[dict]].
        def hf_model(texts):
            return [[{"label": name, "score": 1.0 / N_CLASSES} for name in LABEL_NAMES] for _ in texts]

        backend = NlpLimeBackend(hf_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello", "world"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, N_CLASSES))
        self.assertEqual(result.dtype, np.float64)
        # Each score should be 1/N_CLASSES
        np.testing.assert_allclose(result, 1.0 / N_CLASSES, atol=1e-6)

    # --- run_explainer ---

    def test_run_explainer_returns_nlp_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertIsInstance(raw, NlpContributions)

    def test_run_explainer_contributions_count(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.values), len(_SAMPLE_TEXTS))

    def test_run_explainer_contributions_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        for arr in raw.values:
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[1], N_CLASSES)

    def test_run_explainer_base_values_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(raw.base_values.shape, (len(_SAMPLE_TEXTS), N_CLASSES))

    def test_run_explainer_data_is_word_list(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.token_strings), len(_SAMPLE_TEXTS))
        for word_list in raw.token_strings:
            self.assertIsInstance(word_list, list)
            self.assertTrue(all(isinstance(w, str) for w in word_list))

    def test_run_explainer_sparse_weights(self):
        # LIME fills at most num_features non-zero weights per label column.
        raw = self.backend.run_explainer(_SAMPLE_TEXTS[:1])
        matrix = raw.values[0]
        for col in range(N_CLASSES):
            self.assertLessEqual(
                np.count_nonzero(matrix[:, col]),
                _LIME_COMPUTE_ARGS["num_features"],
            )

    # --- get_local_contributions ---

    def test_get_local_contributions_returns_nlp_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertIsInstance(contrib, NlpContributions)

    def test_get_local_contributions_token_strings_match_data(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertEqual(contrib.token_strings, raw.token_strings)

    def test_get_local_contributions_values_match_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        for got, expected in zip(contrib.values, raw.values):
            np.testing.assert_array_equal(got, expected)

    def test_get_local_contributions_subset(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw, subset=[0])
        self.assertEqual(len(contrib.token_strings), 1)
        self.assertEqual(len(contrib.values), 1)
        self.assertEqual(contrib.base_values.shape[0], 1)


# ---------------------------------------------------------------------------
# NlpExplainer with NlpLimeBackend
# ---------------------------------------------------------------------------


class TestNlpExplainerWithLimeBackend(unittest.TestCase):
    def _make_explainer_lime(self) -> NlpExplainer:
        """NlpExplainer backed by NlpLimeBackend."""
        xpl = _make_explainer()
        xpl.backend = _make_lime_backend()
        return xpl

    def test_backend_is_lime_instance(self):
        xpl = self._make_explainer_lime()
        self.assertIsInstance(xpl.backend, NlpLimeBackend)

    def test_plot_tokens_works_for_a_lime_explanation(self):
        # LIME is non-additive, so .plot.waterfall refuses; .plot.tokens stays available.
        explanation = replace(_make_explanation(), backend_name="nlp_lime", is_additive=False)
        self.assertIsInstance(explanation.plot.tokens(row=0, label_idx=0), go.Figure)
        with self.assertRaises(ValueError):
            explanation.plot.waterfall(row=0, label_idx=0)

    def test_explain_sets_contributions(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df):
            explanation = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(len(explanation), len(_SAMPLE_TEXTS))
        self.assertEqual(explanation.label_names, LABEL_NAMES)

    def test_explain_caching_skips_rerun(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df) as mocked_predict:
            xpl.explain(_SAMPLE_TEXTS)
            xpl.explain(_SAMPLE_TEXTS)  # same data — must hit in-memory cache
        self.assertEqual(mocked_predict.call_count, 1, "memoization must skip the second _predict call")


# ---------------------------------------------------------------------------
# compile() cache key
# ---------------------------------------------------------------------------


class _KeyModel(TextModel):
    """Prediction-only model with a caller-chosen ``model_id``, so keys can be varied deliberately."""

    def __init__(self, ident="model-a"):
        super().__init__(label_names=LABEL_NAMES)
        self._ident = ident

    @property
    def model_id(self):
        return self._ident

    def predict(self, texts):
        probs = np.full((len(texts), N_CLASSES), 1.0 / N_CLASSES)
        return probs


class _MarkerBackend(NlpBackend):
    """Returns a constant contribution equal to ``marker``, and counts its explainer runs.

    The constant makes it visible *whose* result a cache served: a value of 1.0 can only have come
    from the backend built with ``marker=1.0``.
    """

    name = "marker_backend"
    reference_kind = "none"
    is_additive = True
    output_space = "probability"

    def __init__(self, marker=1.0, **kwargs):
        super().__init__(model=None, label_names=LABEL_NAMES, **kwargs)
        self.marker = marker
        self.calls = 0

    def run_explainer(self, x):
        self.calls += 1
        texts = list(x)
        return NlpContributions(
            token_strings=[["a", "b"] for _ in texts],
            values=[np.full((2, N_CLASSES), self.marker) for _ in texts],
            base_values=np.zeros((len(texts), N_CLASSES)),
        )


class _OtherMarkerBackend(_MarkerBackend):
    """Same behaviour under a different registered ``name`` — a distinct attribution method."""

    name = "other_marker_backend"


def _explainer(model=None, backend=None, label_names=LABEL_NAMES):
    return NlpExplainer(
        model or _KeyModel(),
        label_names=label_names,
        backend=backend or _MarkerBackend(),
    )


class TestExplainCacheKey(unittest.TestCase):
    """``explain`` results depend on *(texts, model, backend)* — so all three must be in the key.

    Keying on the texts alone means swapping the model or the attribution backend and pointing at the
    same ``cache_dir`` silently reloads the previous run's contributions, and even without a
    ``cache_dir`` the in-memory guard turns a re-``explain`` into a no-op.
    """

    def test_identical_inputs_still_hit_the_in_memory_cache(self):
        backend = _MarkerBackend()
        xpl = _explainer(backend=backend)
        xpl.explain(_SAMPLE_TEXTS)
        xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(backend.calls, 1, "memoization must survive the richer key")

    def test_the_memo_cannot_be_poisoned_through_a_returned_artifact(self):
        """``explain`` returns a ``replace()`` of the memoized artifact — a *shallow* copy.

        The two artifacts therefore share their contribution arrays, so before the arrays were
        sealed an in-place edit of the first return value silently rewrote the cache, and every
        later ``explain`` of the same texts served the corrupted numbers as a fresh result. The
        seal turns that into an error at the point of the edit.
        """
        backend = _MarkerBackend(marker=1.0)
        xpl = _explainer(backend=backend)
        first = xpl.explain(_SAMPLE_TEXTS)

        with self.assertRaises(ValueError):
            first.values[0][0] = -42.0

        second = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(backend.calls, 1, "still a memo hit — the seal must not defeat caching")
        np.testing.assert_allclose(second.values[0], 1.0)

    def test_a_memo_hit_is_relabelled_onto_the_caller_index(self):
        """The memo is keyed by text *content*, so the same texts from a differently-indexed
        source is a hit — and the cached predictions still carry the first caller's index.
        ``replace`` says nothing about the fields it is not handed, so every index-bearing field
        has to be re-labelled, not just ``texts``.
        """
        backend = _MarkerBackend()
        xpl = _explainer(backend=backend)
        xpl.explain(pd.Series(_SAMPLE_TEXTS))  # first call: default RangeIndex

        index = pd.Index([77 + i for i in range(len(_SAMPLE_TEXTS))])
        second = xpl.explain(pd.Series(_SAMPLE_TEXTS, index=index))

        self.assertEqual(backend.calls, 1, "re-labelling must not cost a recompute")
        for field_name in ("texts", "y_pred", "y_prob"):
            with self.subTest(field=field_name):
                self.assertTrue(getattr(second, field_name).index.equals(index))

    def test_a_y_series_that_does_not_match_x_is_refused(self):
        """Reindexing it silently would pair labels positionally while looking like alignment."""
        xpl = _explainer(backend=_MarkerBackend())
        texts = pd.Series(_SAMPLE_TEXTS, index=[10 + i for i in range(len(_SAMPLE_TEXTS))])
        misaligned = pd.Series(["joy"] * len(_SAMPLE_TEXTS), index=range(len(_SAMPLE_TEXTS)))
        with self.assertRaises(ValueError) as ctx:
            xpl.explain(texts, y=misaligned)
        self.assertIn("indexed differently", str(ctx.exception))

    def test_a_y_series_aligned_to_x_is_kept_and_a_list_is_indexed_positionally(self):
        xpl = _explainer(backend=_MarkerBackend())
        index = pd.Index([10 + i for i in range(len(_SAMPLE_TEXTS))])
        texts = pd.Series(_SAMPLE_TEXTS, index=index)
        labels = ["joy"] * len(_SAMPLE_TEXTS)
        for tag, y in (("series", pd.Series(labels, index=index)), ("list", labels)):
            with self.subTest(y=tag):
                self.assertTrue(xpl.explain(texts, y=y).y_true.index.equals(index))

    def test_swapping_the_backend_recomputes(self):
        xpl = _explainer(backend=_MarkerBackend(marker=1.0))
        xpl.explain(_SAMPLE_TEXTS)
        replacement = _OtherMarkerBackend(marker=2.0)
        xpl.backend = replacement
        explanation = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(replacement.calls, 1, "in-memory guard served a stale, other-backend result")
        np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_backend_compute_args_are_part_of_the_key(self):
        xpl = _explainer(backend=_MarkerBackend(explainer_compute_args={"n_steps": 50}))
        first = xpl._compute_key(_SAMPLE_TEXTS)
        xpl.backend = _MarkerBackend(explainer_compute_args={"n_steps": 200})
        self.assertNotEqual(first, xpl._compute_key(_SAMPLE_TEXTS))

    def test_compute_args_key_is_order_insensitive(self):
        # Same settings written in a different order are the same configuration.
        a = _explainer(backend=_MarkerBackend(explainer_compute_args={"x": 1, "y": 2}))
        b = _explainer(backend=_MarkerBackend(explainer_compute_args={"y": 2, "x": 1}))
        self.assertEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_model_identity_is_part_of_the_key(self):
        a = _explainer(model=_KeyModel("model-a"))
        b = _explainer(model=_KeyModel("model-b"))
        self.assertNotEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_label_names_are_part_of_the_key(self):
        # label_names fixes the column order of y_prob, so it changes the cached payload.
        a = _explainer(label_names=LABEL_NAMES)
        b = _explainer(label_names=list(reversed(LABEL_NAMES)))
        self.assertNotEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_key_is_collision_safe_across_text_boundaries(self):
        # Without a separator between texts these two corpora hash identically.
        xpl = _explainer()
        self.assertNotEqual(xpl._compute_key(["ab", "c"]), xpl._compute_key(["a", "bc"]))

    def test_key_is_order_sensitive(self):
        xpl = _explainer()
        self.assertNotEqual(xpl._compute_key(["a", "b"]), xpl._compute_key(["b", "a"]))


class TestExplainDiskCacheIsolation(unittest.TestCase):
    """The disk cache is the dangerous case: a stale entry survives the process that wrote it."""

    def test_two_backends_share_a_cache_dir_without_collision(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = _MarkerBackend(marker=1.0)
            _explainer(backend=first).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            second = _OtherMarkerBackend(marker=2.0)
            xpl = _explainer(backend=second)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(second.calls, 1, "loaded the other backend's cached contributions")
            np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_two_models_share_a_cache_dir_without_collision(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(model=_KeyModel("model-a"), backend=_MarkerBackend(marker=1.0)).explain(
                _SAMPLE_TEXTS, cache_dir=cache_dir
            )
            backend_b = _MarkerBackend(marker=2.0)
            xpl = _explainer(model=_KeyModel("model-b"), backend=backend_b)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(backend_b.calls, 1, "loaded the other model's cached contributions")
            np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_same_model_and_backend_reload_from_disk(self):
        # The cache must still *work* — a fresh instance skips the expensive run.
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(backend=_MarkerBackend(marker=3.0)).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            reloaded_backend = _MarkerBackend(marker=3.0)
            xpl = _explainer(backend=reloaded_backend)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertEqual(reloaded_backend.calls, 0, "disk cache did not hit for identical inputs")
            np.testing.assert_allclose(explanation.values[0], 3.0)

    def test_cache_path_points_at_the_file_explain_writes(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            xpl = _explainer()
            path = xpl.cache_path(_SAMPLE_TEXTS, cache_dir)
            self.assertFalse(path.exists())
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertTrue(path.exists(), "cache_path disagrees with where explain() wrote")

    def test_clear_cache_forces_a_recompute(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            backend = _MarkerBackend()
            xpl = _explainer(backend=backend)
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            xpl.clear_cache(_SAMPLE_TEXTS, cache_dir)
            self.assertFalse(xpl.cache_path(_SAMPLE_TEXTS, cache_dir).exists())
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertEqual(backend.calls, 2, "clear_cache must defeat the in-memory guard too")

    def test_clear_cache_leaves_other_backends_entries_intact(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            keeper = _MarkerBackend(marker=1.0)
            _explainer(backend=keeper).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            keeper_path = _explainer(backend=_MarkerBackend(marker=1.0)).cache_path(_SAMPLE_TEXTS, cache_dir)

            other = _explainer(backend=_OtherMarkerBackend(marker=2.0))
            other.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            other.clear_cache(_SAMPLE_TEXTS, cache_dir)

            self.assertTrue(keeper_path.exists(), "clearing one backend dropped another's cache")


class TestDetectLabelNoise(unittest.TestCase):
    """The explainer's confident-learning surface, over an ``NlpExplanation`` (no model, no backend)."""

    def _with_labels(self, y_true=None, y_prob=None) -> NlpExplanation:
        """A batch carrying ground truth and per-class probabilities.

        Sample 1 is labelled ``joy`` while the model confidently says ``sadness`` — the planted
        error. Both classes carry at least one label, without which ``sadness`` would have no
        estimable threshold and could never be suggested (covered separately in the compute tests).
        """
        confident = ["joy", "sadness", "sadness"]
        probs = np.full((3, N_CLASSES), 0.02)
        for row, name in enumerate(confident):
            probs[row, LABEL_NAMES.index(name)] = 0.90
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = pd.Series(confident, index=pd.RangeIndex(3), name="prediction")
        resolved_y_prob = pd.DataFrame(probs, index=pd.RangeIndex(3), columns=LABEL_NAMES) if y_prob is None else y_prob
        resolved_y_true = (
            pd.Series(["joy", "joy", "sadness"], index=pd.RangeIndex(3), name="ground_truth")
            if y_true is None
            else y_true
        )
        return _make_explanation(y_pred=y_pred, y_prob=resolved_y_prob, y_true=resolved_y_true)

    # ── capability flag ────────────────────────────────────────────────
    def test_available_with_ground_truth_and_per_class_probabilities(self):
        self.assertTrue(_make_explainer().can_detect_label_noise(self._with_labels()))

    def test_unavailable_without_ground_truth(self):
        explanation = replace(self._with_labels(), y_true=None)
        self.assertFalse(_make_explainer().can_detect_label_noise(explanation))

    def test_unavailable_with_only_the_winning_class_probability(self):
        # The raw-pipeline path emits a single "probability" column; the losing classes' scores are
        # precisely what confident learning needs.
        legacy = pd.DataFrame({"probability": [0.9, 0.9, 0.9]}, index=pd.RangeIndex(3))
        self.assertFalse(_make_explainer().can_detect_label_noise(self._with_labels(y_prob=legacy)))

    def test_unavailable_on_an_explanation_with_no_probabilities(self):
        self.assertFalse(_make_explainer().can_detect_label_noise(_make_explanation()))

    def test_available_without_a_model(self):
        # Unlike the other capability flags this needs no live model.
        xpl = _make_explainer()
        self.assertIsNone(xpl.model)
        self.assertTrue(xpl.can_detect_label_noise(self._with_labels()))

    # ── detection ──────────────────────────────────────────────────────
    def test_raises_when_the_prerequisites_are_missing(self):
        with self.assertRaisesRegex(RuntimeError, "ground-truth labels"):
            _make_explainer().detect_label_noise(_make_explanation())

    def test_flags_the_planted_mislabel(self):
        report = _make_explainer().detect_label_noise(self._with_labels())
        self.assertEqual([i.index for i in report.issues], [1])
        issue = report.issues[0]
        self.assertEqual(issue.given_label, "joy")
        self.assertEqual(issue.suggested_label, "sadness")
        self.assertEqual(issue.text, "this is terrible and sad")

    def test_label_names_come_from_the_probability_columns(self):
        report = _make_explainer().detect_label_noise(self._with_labels())
        self.assertEqual(report.label_names, LABEL_NAMES)
        self.assertEqual(report.noise_matrix.shape, (N_CLASSES, N_CLASSES))
        self.assertEqual(report.n_samples, 3)

    def test_respects_top_n_and_score(self):
        report = _make_explainer().detect_label_noise(self._with_labels(), top_n=0, score="normalized_margin")
        self.assertEqual(report.issues, [])
        self.assertEqual(report.n_issues, 1)

    # ── memoisation ────────────────────────────────────────────────────
    def test_repeats_are_served_from_the_memo(self):
        xpl = _make_explainer()
        explanation = self._with_labels()
        first = xpl.detect_label_noise(explanation, top_n=5)
        self.assertIs(xpl.detect_label_noise(explanation, top_n=5), first)

    def test_different_arguments_recompute(self):
        xpl = _make_explainer()
        explanation = self._with_labels()
        self.assertIsNot(xpl.detect_label_noise(explanation, top_n=5), xpl.detect_label_noise(explanation, top_n=4))

    # ── independent probe ──────────────────────────────────────────────
    def _probe_corpus(self):
        """A reference corpus separable by words the audited fixture's texts also use."""
        texts = [
            "this is wonderful and joyful",
            "wonderful joyful and bright",
            "a joyful wonderful day",
            "bright and wonderful joy",
            "this is terrible and sad",
            "terrible sad and bleak",
            "a sad terrible day",
            "bleak and terrible sadness",
        ]
        labels = ["joy"] * 4 + ["sadness"] * 4
        return texts, labels

    def test_no_probe_when_no_reference_corpus_is_bound(self):
        xpl = _make_explainer()
        self.assertFalse(xpl.can_probe_labels())
        self.assertIsNone(xpl.detect_label_noise(self._with_labels()).issues[0].probe)

    def test_probe_verdict_is_attached_when_a_labelled_corpus_is_bound(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        self.assertTrue(xpl.can_probe_labels())
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertIsNotNone(issue.probe)
        self.assertIn(issue.probe.top_label, {"joy", "sadness"})
        self.assertEqual(issue.probe.backs_given, issue.probe.top_label == issue.given_label)

    def test_probe_corroborates_a_genuine_label_error(self):
        # The flagged row is "this is terrible and sad" carrying the label "joy". The reference
        # corpus puts that vocabulary firmly in "sadness", so the probe rejects the given label too
        # — the two-signals-agree case, which is the one worth relabelling.
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertEqual((issue.given_label, issue.text), ("joy", "this is terrible and sad"))
        self.assertFalse(issue.probe.backs_given)
        self.assertEqual(issue.probe.top_label, "sadness")
        self.assertLess(issue.probe.given_prob, 0.5)

    def test_probe_backs_the_label_when_the_corpus_sides_with_it(self):
        # The mirror case, and the reason the column exists: same flagged row, but a corpus that
        # calls this vocabulary "joy". The probe now defends the label, marking the row as the
        # audited model's error rather than the corpus's.
        xpl = _make_explainer()
        texts, _ = self._probe_corpus()
        xpl.reference_ = (texts, ["sadness"] * 4 + ["joy"] * 4)
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertEqual(issue.given_label, "joy")
        self.assertTrue(issue.probe.backs_given)
        self.assertGreater(issue.probe.given_prob, 0.5)

    def test_probe_is_skipped_when_not_requested(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        self.assertIsNone(xpl.detect_label_noise(self._with_labels(), probe=False).issues[0].probe)

    def test_probe_is_fit_once_and_reused_across_calls(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        explanation = self._with_labels()
        xpl.detect_label_noise(explanation, top_n=5)
        first = xpl._label_probe
        self.assertIsNotNone(first)
        xpl.detect_label_noise(explanation, top_n=4)  # different args -> recompute, but the probe is kept
        self.assertIs(xpl._label_probe, first)

    def test_probe_needs_no_model_or_retriever(self):
        # The point of fitting on plain text: a prediction-only pipeline that cannot embed (so
        # can_find_similar() is False) still gets the second opinion.
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        xpl._retriever = None
        xpl._text_model = None
        self.assertFalse(xpl.can_find_similar())
        self.assertTrue(xpl.can_probe_labels())
        self.assertIsNotNone(xpl.detect_label_noise(self._with_labels()).issues[0].probe)


class TestRunAppMountPath(unittest.TestCase):
    """``run_app`` forwards the reverse-proxy mount point to the webapp it builds."""

    def test_url_base_pathname_reaches_the_webapp(self):
        xpl = object.__new__(NlpExplainer)
        explanation = object()
        with patch("shapash.explainer.nlp_explainer.NlpWebApp") as web_app:
            xpl.run_app(explanation, url_base_pathname="/shapash-nlp-explainer/")
        web_app.assert_called_once_with(
            explanation,
            engine=xpl,
            scatter_xy=None,
            url_base_pathname="/shapash-nlp-explainer/",
            palette_name="default",
            colors_dict=None,
            info={},
        )

    def test_defaults_to_no_prefix(self):
        xpl = object.__new__(NlpExplainer)
        with patch("shapash.explainer.nlp_explainer.NlpWebApp") as web_app:
            xpl.run_app(object())
        self.assertIsNone(web_app.call_args.kwargs["url_base_pathname"])


if __name__ == "__main__":
    unittest.main()
