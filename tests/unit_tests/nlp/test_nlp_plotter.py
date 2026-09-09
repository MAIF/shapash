"""Unit tests for ``NlpPlotter`` — the ``explanation.plot`` accessor.

The accessor exists so a *reloaded* explanation can be plotted: ``NlpExplanation.load()``
deliberately needs no model and no backend, but plotting used to require an ``NlpExplainer``.
These tests therefore check three things: the figures come out, the artifact is never written
to, and the two guards (non-additive backend, missing ground truth) refuse rather than draw
something meaningless.
"""

import copy
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from dash import html
from plotly import graph_objs as go

from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.explainer.nlp_plotter import NlpPlotter
from shapash.webapp.utils.dash_to_html import DashHtmlPreview


def _make_explanation(
    ndim: int = 2, is_additive: bool = True, with_true: bool = True, with_base: bool = True
) -> NlpExplanation:
    texts = pd.Series(["hello world", "i am happy today", "ok"], index=[10, 11, 12])
    if ndim == 2:
        values = [
            np.array([[1.0, -1.0], [2.0, -2.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.array([[0.7, -0.7]]),
        ]
        base_values = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]) if with_base else None
        label_names = ["neg", "pos"]
    else:
        values = [np.array([1.0, 2.0]), np.array([0.1, 0.3, 0.5]), np.array([0.7])]
        base_values = np.array([0.1, 0.2, 0.3]) if with_base else None
        label_names = None

    return NlpExplanation(
        texts=texts,
        token_strings=[["hello", "world"], ["i", "am", "happy"], ["ok"]],
        values=values,
        base_values=base_values,
        y_pred=pd.Series(["pos", "neg", "pos"], index=texts.index),
        y_prob=None,
        y_true=pd.Series(["pos", "pos", "pos"], index=texts.index) if with_true else None,
        label_names=label_names,
        folds_case=True,
        backend_name="nlp_shap" if is_additive else "nlp_lime",
        is_additive=is_additive,
        reference_kind="distribution" if with_base else "none",
        output_space="probability",
    )


class TestPlotAccessor(unittest.TestCase):
    def test_plot_returns_a_plotter_bound_to_this_explanation(self):
        explanation = _make_explanation()
        self.assertIsInstance(explanation.plot, NlpPlotter)
        self.assertIs(explanation.plot._exp, explanation)

    def test_plot_is_not_a_dataclass_field(self):
        """It must stay out of ``fields()`` or ``save()``/``load()`` would try to persist it."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(_make_explanation())}
        self.assertNotIn("plot", names)

    def test_plot_is_built_fresh_and_holds_no_state(self):
        explanation = _make_explanation()
        self.assertIsNot(explanation.plot, explanation.plot)

    def test_repr_names_the_batch_and_backend(self):
        self.assertIn("nlp_shap", repr(_make_explanation().plot))


class TestPerInstancePlots(unittest.TestCase):
    def setUp(self):
        self.explanation = _make_explanation()

    def test_tokens_returns_a_figure_for_every_row_and_class(self):
        for row in range(len(self.explanation)):
            for label_idx in range(self.explanation.n_classes):
                self.assertIsInstance(self.explanation.plot.tokens(row=row, label_idx=label_idx), go.Figure)

    def test_tokens_titles_the_class_when_names_are_known(self):
        self.assertIn("pos", self.explanation.plot.tokens(row=0, label_idx=1).layout.title.text)

    def test_tokens_falls_back_to_a_generic_title_without_class_names(self):
        explanation = _make_explanation(ndim=1)
        self.assertEqual(explanation.plot.tokens(row=0).layout.title.text, "Token contributions")

    def test_tokens_honours_max_tokens(self):
        fig = self.explanation.plot.tokens(row=1, label_idx=0, max_tokens=2)
        self.assertLessEqual(len(fig.data[0].x), 2)

    def test_waterfall_returns_a_figure(self):
        self.assertIsInstance(self.explanation.plot.waterfall(row=0, label_idx=1), go.Figure)

    def test_sentence_returns_a_dash_component(self):
        self.assertIsInstance(self.explanation.plot.sentence(row=0, label_idx=1), html.Div)

    def test_sentence_notebook_true_returns_a_previewable_wrapper(self):
        preview = self.explanation.plot.sentence(row=0, label_idx=1, notebook=True)
        self.assertIsInstance(preview, DashHtmlPreview)
        self.assertIsInstance(preview.component, html.Div)
        self.assertIn("<div", preview._repr_html_())

    def test_negative_row_counts_from_the_end(self):
        last = self.explanation.plot.tokens(row=-1, label_idx=0)
        self.assertEqual(list(last.data[0].y), ["ok"])

    def test_row_is_positional_not_a_texts_index_label(self):
        """``texts.index`` starts at 10; ``row=0`` must still mean the first sample."""
        self.assertEqual(list(self.explanation.texts.index)[0], 10)
        fig = self.explanation.plot.tokens(row=0, label_idx=0)
        self.assertEqual(set(fig.data[0].y), {"hello", "world"})

    def test_out_of_range_row_raises_with_the_batch_size(self):
        with self.assertRaises(IndexError) as ctx:
            self.explanation.plot.tokens(row=99)
        self.assertIn("3 sample(s)", str(ctx.exception))

    def test_out_of_range_label_idx_raises_listing_the_classes(self):
        with self.assertRaises(IndexError) as ctx:
            self.explanation.plot.tokens(row=0, label_idx=7)
        self.assertIn("neg", str(ctx.exception))

    def test_binary_1d_values_need_no_class_slicing(self):
        explanation = _make_explanation(ndim=1)
        self.assertIsInstance(explanation.plot.waterfall(row=1), go.Figure)

    def test_plots_work_without_base_values(self):
        """``reference_kind == "none"`` means no baseline bar, not a crash."""
        explanation = _make_explanation(with_base=False)
        self.assertIsInstance(explanation.plot.waterfall(row=0, label_idx=0), go.Figure)
        self.assertIsInstance(explanation.plot.sentence(row=0, label_idx=0), html.Div)


class TestBatchPlots(unittest.TestCase):
    def test_word_importance_returns_a_figure_titled_with_the_class(self):
        fig = _make_explanation().plot.word_importance(label_idx=1, n_top=5)
        self.assertIsInstance(fig, go.Figure)
        self.assertIn("pos", fig.layout.title.text)

    def test_word_importance_forwards_kwargs(self):
        explanation = _make_explanation()
        fig = explanation.plot.word_importance(label_idx=0, n_top=10, exclude_words={"hello"})
        self.assertNotIn("hello", set(fig.data[0].y))

    def test_confusion_returns_a_figure(self):
        self.assertIsInstance(_make_explanation().plot.confusion(), go.Figure)

    def test_confusion_refuses_without_ground_truth(self):
        explanation = _make_explanation(with_true=False)
        with self.assertRaises(ValueError) as ctx:
            explanation.plot.confusion()
        self.assertIn("y_true", str(ctx.exception))


class TestGuards(unittest.TestCase):
    def test_waterfall_refuses_on_a_non_additive_backend(self):
        """LIME contributions do not sum to the prediction, so the running total is nonsense."""
        explanation = _make_explanation(is_additive=False)
        with self.assertRaises(ValueError) as ctx:
            explanation.plot.waterfall(row=0, label_idx=0)
        message = str(ctx.exception)
        self.assertIn("nlp_lime", message)
        self.assertIn("tokens", message)  # points at the plot that is still valid

    def test_the_other_plots_stay_available_on_a_non_additive_backend(self):
        explanation = _make_explanation(is_additive=False)
        self.assertIsInstance(explanation.plot.tokens(row=0, label_idx=0), go.Figure)
        self.assertIsInstance(explanation.plot.word_importance(label_idx=0), go.Figure)


class TestArtifactIsNeverWritten(unittest.TestCase):
    def test_rendering_leaves_the_explanation_untouched(self):
        explanation = _make_explanation()
        before = copy.deepcopy(explanation)

        explanation.plot.tokens(row=0, label_idx=1, max_tokens=1)
        explanation.plot.waterfall(row=1, label_idx=0)
        explanation.plot.sentence(row=0, label_idx=0)
        explanation.plot.word_importance(label_idx=1, n_top=3)
        explanation.plot.confusion(normalize="true")

        self.assertEqual(explanation.backend_name, before.backend_name)
        self.assertEqual(explanation.label_names, before.label_names)
        self.assertEqual(explanation.token_strings, before.token_strings)
        for original, current in zip(before.values, explanation.values, strict=True):
            np.testing.assert_array_equal(original, current)
        np.testing.assert_array_equal(before.base_values, explanation.base_values)


class TestConfusionMatrixData(unittest.TestCase):
    def test_counts_are_true_by_predicted(self):
        explanation = _make_explanation()
        # y_true = [pos, pos, pos], y_pred = [pos, neg, pos]; idx: neg=0, pos=1
        cm = explanation.confusion_matrix()
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(cm[1, 1], 2)  # true pos, predicted pos
        self.assertEqual(cm[1, 0], 1)  # true pos, predicted neg
        self.assertEqual(cm[0].sum(), 0)  # no true-neg samples

    def test_unknown_labels_are_skipped_not_raised(self):
        base = _make_explanation()
        explanation = replace(base, y_pred=pd.Series(["pos", "surprise", "pos"], index=base.texts.index))
        cm = explanation.confusion_matrix()
        self.assertEqual(cm.sum(), 2)  # the unmatched row is dropped, the rest still counted

    def test_raises_without_ground_truth(self):
        with self.assertRaises(ValueError):
            _make_explanation(with_true=False).confusion_matrix()


if __name__ == "__main__":
    unittest.main()
