"""Unit tests for shapash.webapp.nlp_components.scatter: ScatterComponent."""

import unittest

import numpy as np
import pandas as pd

from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import LABEL_NAMES, _callback, _ctx


class TestScatterComponentWordContributionOption(unittest.TestCase):
    """``offer_word_contribution`` gates the "Word contribution" color mode, not the whole panel."""

    @staticmethod
    def _explanation():
        # Same fixture as TestGlobalWordImportancePanel (tests/unit_tests/webapp/nlp_components/
        # test_word_importance.py): "rare" pulls hard once, "common" mildly four times. Duplicated
        # here rather than imported across test modules, matching this repo's no-cross-file-test-
        # helper-sharing convention (sharing goes through _shared.py only).
        token_strings = [["rare", "common"], ["common"], ["common", "common"], ["mild"]]
        values = [
            np.array([[0.9, -0.9], [0.3, -0.3]]),
            np.array([[0.3, -0.3]]),
            np.array([[0.3, -0.3], [0.3, -0.3]]),
            np.array([[-0.5, 0.5]]),
        ]
        texts = pd.Series(["rare common", "common", "common common", "mild"])
        return NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["neg"] * 4, index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )

    def _color_options(self, offer_word_contribution: bool):
        from shapash.webapp.nlp_components import ScatterComponent

        comp = ScatterComponent(offer_word_contribution=offer_word_contribution)
        explanation = self._explanation()
        layout = comp.layout(_ctx(explanation, None, coords=np.zeros((explanation.n_samples, 2))))
        found = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            children = getattr(node, "children", None)
            for ch in children if isinstance(children, (list, tuple)) else [children]:
                if ch is not None and not isinstance(ch, str):
                    walk(ch)

        walk(layout)
        return [opt["value"] for opt in found["color-by"].options]

    def test_offered_when_word_importance_is_mounted(self):
        self.assertIn("word_contribution", self._color_options(offer_word_contribution=True))

    def test_omitted_without_word_importance(self):
        # Scatter still works standalone (Prediction/Ground-Truth) — it just cannot source a class
        # index for word-contribution coloring without Word Importance's control.
        options = self._color_options(offer_word_contribution=False)
        self.assertNotIn("word_contribution", options)
        self.assertIn("prediction", options)


class TestBuildScatterFig(unittest.TestCase):
    """``_build_scatter_fig``'s own dispatch — the part left after extracting the drawing itself
    into :func:`~shapash.plots.plot_scatter.plot_scatter`."""

    @staticmethod
    def _explanation(with_true: bool, with_pred: bool):
        texts = pd.Series(["rare common", "common"])
        return NlpExplanation(
            texts=texts,
            token_strings=[["rare", "common"], ["common"]],
            values=[np.array([[0.9, -0.9], [0.3, -0.3]]), np.array([[0.3, -0.3]])],
            base_values=None,
            y_pred=pd.Series(["neg", "pos"], index=texts.index) if with_pred else None,
            y_prob=None,
            y_true=pd.Series(["pos", "pos"], index=texts.index) if with_true else None,
            label_names=None,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def _component(self):
        from shapash.webapp.nlp_components import ScatterComponent

        return ScatterComponent(offer_word_contribution=True)

    def _ctx_for(self, explanation):
        # The figure takes its two coordinate columns from the projection and every visual encoding
        # — colours, legend, hover, error mask — from the explanation, so both have to be bound.
        return _ctx(explanation, None, coords=np.zeros((explanation.n_samples, 2)))

    def test_ground_truth_colors_by_y_true(self):
        fig = self._component()._build_scatter_fig(self._ctx_for(self._explanation(True, True)), "ground_truth")
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos"})  # both samples are "pos" in y_true

    def test_without_prediction_or_ground_truth_draws_a_single_unlabeled_group(self):
        fig = self._component()._build_scatter_fig(self._ctx_for(self._explanation(False, False)), "prediction")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(len(fig.data[0].x), 2)


class TestScatterComponentCallbacks(unittest.TestCase):
    """The scatter panel's own callbacks: color-mode/word-list sync and box/lasso/click selection."""

    @staticmethod
    def _explanation():
        texts = pd.Series(["rare common", "common"])
        return NlpExplanation(
            texts=texts,
            token_strings=[["rare", "common"], ["common"]],
            values=[np.array([[0.9, -0.9], [0.3, -0.3]]), np.array([[0.3, -0.3]])],
            base_values=None,
            y_pred=pd.Series(["neg", "pos"], index=texts.index, name="prediction"),
            y_prob=None,
            y_true=pd.Series(["pos", "pos"], index=texts.index, name="ground_truth"),
            label_names=None,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def _app(self):
        return NlpWebApp(self._explanation(), engine=None, projection=np.zeros((2, 2)))

    def test_ground_truth_option_offered_when_available(self):
        app = self._app()
        # Walk the layout for the color-by dropdown directly, mirroring TestScatterComponentWordContributionOption.
        found = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            children = getattr(node, "children", None)
            for ch in children if isinstance(children, (list, tuple)) else [children]:
                if ch is not None and not isinstance(ch, str):
                    walk(ch)

        walk(app.app.layout)
        dropdown = found["color-by"]
        self.assertIn("ground_truth", [opt["value"] for opt in dropdown.options])

    def test_sync_color_by_switches_to_word_contribution_when_words_selected(self):
        app = self._app()
        sync = _callback(app, "color-by.value")
        self.assertEqual(sync(["common"]), "word_contribution")

    def test_sync_color_by_prevents_update_when_words_cleared(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        sync = _callback(app, "color-by.value")
        with self.assertRaises(PreventUpdate):
            sync([])
        with self.assertRaises(PreventUpdate):
            sync(None)

    def test_follow_word_click_mirrors_a_single_word_as_a_list(self):
        app = self._app()
        follow = _callback(app, "scatter-word-select.value")
        self.assertEqual(follow("common"), ["common"])

    def test_follow_word_click_passes_through_a_list_and_clears_on_falsy(self):
        app = self._app()
        follow = _callback(app, "scatter-word-select.value")
        self.assertEqual(follow(["common", "rare"]), ["common", "rare"])
        self.assertEqual(follow(None), [])

    def test_word_filter_from_scatter_publishes_to_the_shared_store(self):
        app = self._app()
        publish = _callback(app, "word-click-filter.data@")
        self.assertEqual(publish(["common"]), ["common"])
        self.assertIsNone(publish([]))
        self.assertIsNone(publish(None))

    def test_toggle_word_select_visibility_follows_color_mode(self):
        app = self._app()
        toggle = _callback(app, "scatter-word-select.style")
        self.assertNotEqual(toggle("word_contribution").get("display"), "none")
        self.assertEqual(toggle("prediction")["display"], "none")

    def test_update_scatter_selection_clear_button_resets(self):
        from unittest import mock

        from shapash.webapp.nlp_components import scatter as scatter_module

        app = self._app()
        update = _callback(app, "scatter-selected-indices.data")
        with mock.patch.object(scatter_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "scatter-clear-btn.n_clicks"}]
            self.assertIsNone(update(None, None, 1))

    def test_update_scatter_selection_from_a_click(self):
        from unittest import mock

        from shapash.webapp.nlp_components import scatter as scatter_module

        app = self._app()
        update = _callback(app, "scatter-selected-indices.data")
        click_data = {"points": [{"customdata": [1]}]}
        with mock.patch.object(scatter_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "scatter-plot.clickData"}]
            self.assertEqual(update(None, click_data, None), [1])

    def test_update_scatter_selection_ignores_an_empty_click(self):
        from unittest import mock

        from shapash.webapp.nlp_components import scatter as scatter_module

        app = self._app()
        update = _callback(app, "scatter-selected-indices.data")
        with mock.patch.object(scatter_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "scatter-plot.clickData"}]
            self.assertIsNone(update(None, {"points": []}, None))
            self.assertIsNone(update(None, None, None))

    def test_update_scatter_selection_ignores_an_empty_box_select_as_a_recolor_echo(self):
        from dash.exceptions import PreventUpdate

        from unittest import mock

        from shapash.webapp.nlp_components import scatter as scatter_module

        app = self._app()
        update = _callback(app, "scatter-selected-indices.data")
        with mock.patch.object(scatter_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "scatter-plot.selectedData"}]
            with self.assertRaises(PreventUpdate):
                update({"points": []}, None, None)

    def test_update_scatter_selection_from_a_box_select(self):
        from unittest import mock

        from shapash.webapp.nlp_components import scatter as scatter_module

        app = self._app()
        update = _callback(app, "scatter-selected-indices.data")
        selected_data = {"points": [{"customdata": [0]}, {"customdata": [1]}]}
        with mock.patch.object(scatter_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "scatter-plot.selectedData"}]
            self.assertEqual(update(selected_data, None, None), [0, 1])

    def test_toggle_clear_button_visibility(self):
        app = self._app()
        toggle = _callback(app, "scatter-clear-btn.style")
        self.assertEqual(toggle(None)["display"], "none")
        self.assertEqual(toggle([0, 1])["display"], "inline")
