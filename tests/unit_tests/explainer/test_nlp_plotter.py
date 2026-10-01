"""Unit tests for ``NlpPlotter`` — the ``explanation.plot`` accessor.

The accessor exists so a *reloaded* explanation can be plotted: ``NlpExplanation.load()``
deliberately needs no model and no backend, but plotting used to require an ``NlpExplainer``.
These tests therefore check three things: the figures come out, the artifact is never written
to, and the two guards (non-additive backend, missing ground truth) refuse rather than draw
something meaningless.
"""

import copy
import unittest
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from dash import html
from plotly import graph_objs as go

from shapash.compute.embeddings import Embedding
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


class TestScatterPlot(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    def test_default_colors_by_prediction(self):
        explanation = _make_explanation()  # y_pred = [pos, neg, pos]
        fig = explanation.plot.scatter(self.xy)
        self.assertIsInstance(fig, go.Figure)
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos", "neg"})

    def test_ground_truth_falls_back_to_prediction_without_y_true(self):
        explanation = _make_explanation(with_true=False)  # y_pred = [pos, neg, pos]
        fig = explanation.plot.scatter(self.xy, color_by="ground_truth")
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos", "neg"})

    def test_word_contribution_colors_by_the_matching_samples(self):
        explanation = _make_explanation()  # sample 1 tokens: ["i", "am", "happy"]
        fig = explanation.plot.scatter(self.xy, color_by="word_contribution", words=["happy"], label_idx=1)
        self.assertEqual(len(fig.data), 2)  # absent layer + present layer
        absent_trace, present_trace = fig.data
        self.assertEqual([row[0] for row in present_trace.customdata], [1])
        self.assertEqual([row[0] for row in absent_trace.customdata], [0, 2])

    def test_word_contribution_without_words_falls_back_to_prediction(self):
        explanation = _make_explanation()
        fig = explanation.plot.scatter(self.xy, color_by="word_contribution", words=[])
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos", "neg"})

    def test_out_of_range_label_idx_raises_with_words(self):
        explanation = _make_explanation()
        with self.assertRaises(IndexError):
            explanation.plot.scatter(self.xy, color_by="word_contribution", words=["happy"], label_idx=5)

    def test_errors_only_runs_without_ground_truth(self):
        explanation = _make_explanation(with_true=False)
        fig = explanation.plot.scatter(self.xy, errors_only=True)
        self.assertIsInstance(fig, go.Figure)

    def test_ground_truth_and_errors_only_with_real_ground_truth(self):
        explanation = _make_explanation()  # y_true = [pos, pos, pos], y_pred = [pos, neg, pos]
        fig = explanation.plot.scatter(self.xy, color_by="ground_truth", errors_only=True)
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos"})


class TestScatterProjectionArgument(unittest.TestCase):
    """Wiring only — the check itself is tested in ``test_embeddings.TestProjectionCoords``."""

    @staticmethod
    def _projection(explanation, **overrides):
        fields = {
            "vectors": np.zeros((explanation.n_samples, 2)),
            "model_id": "fake:v1",
            "space": "decision",
            "corpus_id": explanation.corpus_id,
            "reducer_tag": "pca-1234",
        }
        fields.update(overrides)
        return Embedding(**fields)

    def test_a_typed_projection_is_accepted(self):
        explanation = _make_explanation()
        fig = explanation.plot.scatter(self._projection(explanation))
        self.assertIsInstance(fig, go.Figure)

    def test_a_projection_from_another_corpus_is_refused(self):
        explanation = _make_explanation()
        with self.assertRaises(ValueError) as ctx:
            explanation.plot.scatter(self._projection(explanation, corpus_id="a-different-corpus"))
        self.assertIn("different texts", str(ctx.exception))

    def test_a_bare_array_is_still_accepted_on_its_shape(self):
        explanation = _make_explanation()
        self.assertIsInstance(explanation.plot.scatter(np.zeros((3, 2))), go.Figure)


class TestGuards(unittest.TestCase):
    def test_waterfall_refuses_on_a_non_additive_backend(self):
        """LIME contributions do not sum to the prediction, so the running total is nonsense."""
        explanation = _make_explanation(is_additive=False)
        with self.assertRaises(ValueError) as ctx:
            explanation.plot.waterfall(row=0, label_idx=0)
        message = str(ctx.exception)
        self.assertIn("nlp_lime", message)
        self.assertIn("tokens", message)  # points at the plot that is still valid

    def test_sentence_omits_the_base_plus_sum_summary_on_a_non_additive_backend(self):
        # "Base + Σ = Total" asserts additivity; LIME's intercept + surrogate weights is no prediction.
        self.assertNotIn("Base:", str(_make_explanation(is_additive=False).plot.sentence(row=0, label_idx=0)))
        self.assertIn("Base:", str(_make_explanation(is_additive=True).plot.sentence(row=0, label_idx=0)))

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


def _word_profile_explanation(folds_case=True):
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


class TestPlotterWordProfile(unittest.TestCase):
    def setUp(self):
        self.explanation = _word_profile_explanation()

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


class TestPlotterCompare(unittest.TestCase):
    """``compare`` aligns other backends onto this one's units and hands the result to a renderer."""

    def setUp(self):
        self.ref = _make_explanation()
        # A LIME-shaped artifact of the same texts: bag of words, its own order, a different scale.
        self.lime = replace(
            self.ref,
            token_strings=[["world", "hello"], ["happy", "i"], ["ok"]],
            values=[
                np.array([[5.0, -5.0], [3.0, -3.0]]),
                np.array([[0.0, 8.0], [0.0, 2.0]]),
                np.array([[1.0, -1.0]]),
            ],
            backend_name="nlp_lime",
            is_additive=False,
        )

    def test_heatmap_aligns_the_other_backend_onto_reference_units(self):
        fig = self.ref.plot.compare(self.lime, row=1, label_idx=1, normalize=None)
        heat = fig.data[0]
        self.assertEqual(list(heat.y), ["shap", "lime"])
        self.assertEqual(list(fig.layout.xaxis.ticktext), ["i", "am", "happy"])
        z = np.asarray(heat.z, dtype=float)
        np.testing.assert_allclose(z[0], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(z[1], [2.0, np.nan, 8.0])
        self.assertIn("pos", fig.layout.title.text)
        self.assertIn("lime vs shap", fig.layout.title.text)

    def test_normalizes_by_default(self):
        z = np.asarray(self.ref.plot.compare(self.lime, row=1, label_idx=1).data[0].z, dtype=float)
        self.assertEqual(np.nanmax(np.abs(z[1])), 1.0)

    def test_default_class_is_the_predicted_one(self):
        # Row 1 predicts "neg" (index 0); LIME gave class 0 all zeros.
        z = np.asarray(self.ref.plot.compare(self.lime, row=1, normalize=None).data[0].z, dtype=float)
        np.testing.assert_allclose(z[0], [0.1, 0.3, 0.5])
        np.testing.assert_allclose(z[1], [0.0, np.nan, 0.0])

    def test_bars_and_highlight(self):
        bars = self.ref.plot.compare([self.lime], row=0, kind="bars")
        self.assertEqual([t.name for t in bars.data], ["shap", "lime"])
        self.assertIsInstance(self.ref.plot.compare(self.lime, kind="highlight"), html.Div)
        self.assertIsInstance(self.ref.plot.compare(self.lime, kind="highlight", notebook=True), DashHtmlPreview)

    def test_max_tokens_keeps_the_strongest_units_in_sentence_order(self):
        # Normalised: shap [0.33, 0.67, 1.0], lime [0.25, nan, 1.0] — "i" is weakest in both.
        fig = self.ref.plot.compare(self.lime, row=1, label_idx=1, max_tokens=2)
        self.assertEqual(list(fig.layout.xaxis.ticktext), ["am", "happy"])

    def test_mapping_sets_labels_and_same_backend_is_told_apart_by_space(self):
        logit = replace(self.ref, output_space="logit")
        fig = self.ref.plot.compare({"LIME": self.lime}, row=0)
        self.assertEqual(list(fig.data[0].y), ["shap", "LIME"])
        # pytest.warns, not assertWarns: the latter walks sys.modules and trips transformers' lazy imports
        with pytest.warns(UserWarning):
            fig = self.ref.plot.compare(logit, row=0)
        self.assertEqual(list(fig.data[0].y), ["shap (probability)", "shap (logit)"])
        fig = self.ref.plot.compare({"shap": replace(self.ref)}, row=0)
        self.assertEqual(list(fig.data[0].y), ["shap", "shap #2"])

    def test_warns_across_output_spaces_and_not_within_one(self):
        lig = replace(self.lime, backend_name="nlp_captum_lig", output_space="logit")
        with pytest.warns(UserWarning, match="shap=probability, captum_lig=logit"):
            self.ref.plot.compare(lig, row=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.ref.plot.compare(self.lime, row=0)

    def test_rejects_a_different_text(self):
        other = replace(self.lime, texts=self.ref.texts.replace("hello world", "goodbye world"))
        with self.assertRaisesRegex(ValueError, "different text"):
            self.ref.plot.compare(other, row=0)

    def test_rejects_different_classes(self):
        other = replace(self.lime, label_names=["a", "b"])
        with self.assertRaisesRegex(ValueError, "different classes"):
            self.ref.plot.compare(other, row=0)

    def test_rejects_unknown_kind(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            self.ref.plot.compare(self.lime, kind="violin")

    def test_does_not_touch_either_artifact(self):
        before = [copy.deepcopy(e.values) for e in (self.ref, self.lime)]
        self.ref.plot.compare(self.lime, row=1, kind="bars")
        for exp, values in zip((self.ref, self.lime), before, strict=True):
            for a, b in zip(exp.values, values, strict=True):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
