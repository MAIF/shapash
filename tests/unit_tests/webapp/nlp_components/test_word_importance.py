"""Unit tests for shapash.webapp.nlp_components.word_importance: the global Word Importance panel."""

import unittest

import numpy as np
import pandas as pd

from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import LABEL_NAMES, _callback, _callback_binding_ids, _collect_ids


class TestGlobalWordImportancePanel(unittest.TestCase):
    """The shell's Word Importance chart: rank-by, frequency floor, and its empty states."""

    @staticmethod
    def _explanation():
        # "rare" pulls hard once; "common" pulls mildly four times. The two rank-by modes must
        # disagree, which is the whole reason the control exists.
        token_strings = [["rare", "common"], ["common"], ["common", "common"], ["mild"]]
        values = [
            np.array([[0.9, -0.9], [0.3, -0.3]]),
            np.array([[0.3, -0.3]]),
            np.array([[0.3, -0.3], [0.3, -0.3]]),
            np.array([[-0.5, 0.5]]),
        ]
        texts = pd.Series(["rare common", "common", "common common", "mild"])
        explanation = NlpExplanation(
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
        return explanation

    @classmethod
    def _app(cls):
        app = NlpWebApp(cls._explanation(), engine=None)
        for key, spec in app.app.callback_map.items():
            if "global-importance-graph.figure" in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError("global-importance-graph")

    @staticmethod
    def _words(fig):
        # Bars are drawn bottom-to-top, so reverse back into rank order.
        return list(fig.data[0].y)[::-1]

    def setUp(self):
        self.graph = self._app()

    def _call(
        self, rank_by="mean", floor=1, sign="all", topk=10, indices=None, cell=None, errors=False, label=0, words=None
    ):
        return self.graph(label, topk, sign, [], rank_by, floor, indices, cell, errors, words)

    @staticmethod
    def _values(fig):
        return list(fig.data[0].x)[::-1]

    def test_mean_and_sum_produce_different_rankings(self):
        self.assertEqual(self._words(self._call(rank_by="mean"))[0], "rare")
        self.assertEqual(self._words(self._call(rank_by="sum"))[0], "common")

    def test_axis_names_the_statistic_drawn(self):
        self.assertEqual(self._call(rank_by="mean").layout.xaxis.title.text, "Mean SHAP contribution")
        self.assertEqual(self._call(rank_by="sum").layout.xaxis.title.text, "Total SHAP contribution")

    def test_frequency_floor_removes_rare_words(self):
        self.assertIn("rare", self._words(self._call(floor=1)))
        self.assertNotIn("rare", self._words(self._call(floor=2)))

    def test_panel_has_no_title(self):
        # The tab is already labelled "Word Importance" and the floor/class are visible in the
        # filter row above the chart, so the figure itself carries no title band — reclaiming that
        # vertical space for word rows instead.
        self.assertIsNone(self._call(floor=2).layout.title.text)
        self.assertIsNone(self._call(floor=1).layout.title.text)

    def test_sign_filter_still_works_in_both_modes(self):
        for rank_by in ("mean", "sum"):
            self.assertEqual(self._words(self._call(rank_by=rank_by, sign="negative")), ["mild"])
            self.assertNotIn("mild", self._words(self._call(rank_by=rank_by, sign="positive")))

    def test_default_floor_is_two(self):
        from shapash.webapp.nlp_components.word_importance import _DEFAULT_MIN_OCCURRENCES

        # A mean over a single observation is not a mean; the panel must not open on one.
        self.assertEqual(_DEFAULT_MIN_OCCURRENCES, 2)

    def test_cleared_floor_input_means_no_filter(self):
        # An emptied number box arrives as None and must not silently restore the default.
        self.assertIn("rare", self._words(self._call(floor=None)))

    def test_floor_below_one_is_clamped(self):
        self.assertIn("rare", self._words(self._call(floor=0)))

    def test_impossible_floor_explains_itself(self):
        fig = self._call(floor=99)
        self.assertEqual(len(fig.data), 0)
        self.assertIn("at least 99 time(s) in these 4 sample(s)", fig.layout.annotations[0].text)

    def test_empty_sign_filter_explains_itself_differently(self):
        # Only "mild" is negative; excluding it by scope leaves the sign filter with nothing, and
        # the message must point at the filter rather than at the frequency floor.
        fig = self._call(sign="negative", indices=[0, 1])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("negative", fig.layout.annotations[0].text)

    def test_floor_counts_within_the_selection(self):
        # "common" occurs 4x overall but once in sample 1, so a floor of 2 must exclude it there.
        fig = self._call(floor=2, indices=[1])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("in these 1 sample(s)", fig.layout.annotations[0].text)

    def test_hover_carries_the_occurrence_count(self):
        fig = self._call(floor=1)
        drawn = dict(zip(self._words(fig), [int(c[0]) for c in fig.data[0].customdata][::-1]))
        self.assertEqual(drawn, {"rare": 1, "common": 4, "mild": 1})
        self.assertIn("Occurrences:", fig.data[0].hovertemplate)

    def test_hover_counts_follow_the_selection(self):
        # A count that ignored the scope would contradict the floor applied right beside it:
        # "common" occurs 4x in the corpus but twice in samples 0-1.
        fig = self._call(floor=1, indices=[0, 1])
        drawn = dict(zip(self._words(fig), [int(c[0]) for c in fig.data[0].customdata][::-1]))
        self.assertEqual(drawn["common"], 2)

    def test_all_classes_ranks_on_the_strongest_class_magnitude(self):
        # "mild" pulls -0.5 on class 0 and +0.5 on class 1: across classes it is a magnitude of
        # 0.5, not the -0.5 the single-class view shows.
        fig = self._call(label="all")
        drawn = dict(zip(self._words(fig), self._values(fig)))
        self.assertEqual(drawn, {"rare": 0.9, "mild": 0.5, "common": 0.3})
        self.assertEqual(dict(zip(self._words(self._call()), self._values(self._call())))["mild"], -0.5)

    def test_all_classes_names_its_statistic_on_the_axis(self):
        self.assertEqual(self._call(label="all").layout.xaxis.title.text, "Largest |mean SHAP| across classes")
        self.assertEqual(
            self._call(label="all", rank_by="sum").layout.xaxis.title.text,
            "Largest |total SHAP| across classes",
        )

    def test_all_classes_ignores_a_stale_sign_filter(self):
        # Every value is a magnitude, so honouring "negative" would blank the chart on a control
        # the user cannot even see is active.
        self.assertEqual(self._words(self._call(label="all", sign="negative")), ["rare", "mild", "common"])

    def test_all_classes_still_honours_the_frequency_floor(self):
        self.assertNotIn("rare", self._words(self._call(label="all", floor=2)))

    def test_sign_filter_is_greyed_out_under_all_classes(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = None
        for key, spec in app.app.callback_map.items():
            if "sign-filter.options" in key:
                fn = getattr(spec["callback"], "__wrapped__", spec["callback"])
        self.assertIsNotNone(fn, "sign-filter gating callback not registered")
        options, value = fn("all")
        self.assertEqual([o["value"] for o in options if o.get("disabled")], ["positive", "negative"])
        self.assertEqual(value, "all")
        options, _ = fn(0)
        self.assertFalse(any(o.get("disabled") for o in options))

    def test_scatter_colouring_survives_all_classes(self):
        # The scatter reads the same dropdown, where int("all") would raise.
        app = NlpWebApp(self._explanation(), engine=None, projection=np.zeros((4, 2)))
        fn = None
        for key, spec in app.app.callback_map.items():
            if "scatter-plot.figure" in key:
                fn = getattr(spec["callback"], "__wrapped__", spec["callback"])
        self.assertIsNotNone(fn, "scatter callback not registered")
        self.assertIsNotNone(fn("word_contribution", ["common"], "all", False))

    def test_number_boxes_commit_without_needing_a_blur(self):
        # debounce=True commits only on Enter or blur, which leaves the spinner arrows looking
        # dead: clicking one keeps focus in the box, so the chart never updates. A numeric
        # (seconds) debounce commits after a pause instead.
        layout = NlpWebApp(self._explanation(), engine=None).app.layout
        for box_id in ("min-occurrences", "topk-input"):
            debounce = layout[box_id].debounce
            self.assertNotIsInstance(debounce, bool, f"{box_id} commits only on blur")
            self.assertGreater(debounce, 0)

    def test_topk_input_replaces_the_slider(self):
        found = set()
        _collect_ids(NlpWebApp(self._explanation(), engine=None).app.layout, found)
        self.assertIn("topk-input", found)
        self.assertNotIn("topk-slider", found)

    def test_topk_controls_the_bar_count(self):
        self.assertEqual(len(self._words(self._call(topk=2))), 2)

    def test_typed_topk_is_clamped_to_the_boxs_range(self):
        from shapash.webapp.nlp_components.word_importance import _MAX_TOPK, _MIN_TOPK

        # The browser enforces min/max on the spinner but not on typed input.
        self.assertEqual(len(self._words(self._call(topk=999))), 3)  # only 3 words exist
        self.assertEqual(len(self._words(self._call(topk=0))), _MIN_TOPK)
        self.assertEqual(len(self._words(self._call(topk=-5))), _MIN_TOPK)
        self.assertLessEqual(_MAX_TOPK, 50)

    def test_cleared_topk_box_falls_back_to_the_default(self):
        from shapash.webapp.nlp_components.word_importance import _DEFAULT_TOPK

        self.assertEqual(_DEFAULT_TOPK, 20)
        # Only None (an emptied box) restores the default — a typed 0 clamps instead.
        self.assertEqual(len(self._words(self._call(topk=None))), 3)  # all 3 words, under the cap

    def test_chart_keeps_its_computed_height_so_labels_survive(self):
        # The panel scrolls; squeezing 50 words into it is what made plotly drop word labels.
        fig = self._call(topk=50)
        self.assertIsNotNone(fig.layout.height)

    def test_missing_class_prevents_update(self):
        from dash.exceptions import PreventUpdate

        with self.assertRaises(PreventUpdate):
            self.graph(None, 10, "all", [], "mean", 1, None, None, False, None)

    def test_generic_empty_explains_itself_when_neither_sign_nor_floor_is_the_cause(self):
        # Excluding every word in the corpus empties the ranking without any sign filter or floor
        # active — the fallback message names neither, unlike the two more specific reasons above.
        fig = self.graph(0, 10, "all", ["rare", "common", "mild"], "mean", 1, None, None, False, None)
        self.assertEqual(len(fig.data), 0)
        self.assertIn("No word passes these filters in these 4 sample(s)", fig.layout.annotations[0].text)


class TestWordImportanceClickCallbacks(unittest.TestCase):
    """Word-click store sync: a bar click or the clear button, and the clear button's own visibility."""

    @staticmethod
    def _app():
        return NlpWebApp(TestGlobalWordImportancePanel._explanation(), engine=None)

    def test_bar_click_publishes_the_word_and_resets_clickdata(self):
        from unittest import mock

        from shapash.webapp.nlp_components import word_importance as word_importance_module

        app = self._app()
        update = _callback(app, "data...global-importance-graph.clickData")
        click_data = {"points": [{"y": "common"}]}
        with mock.patch.object(word_importance_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "global-importance-graph.clickData"}]
            words, reset = update(click_data, None)
        self.assertEqual(words, ["common"])
        self.assertIsNone(reset)

    def test_clear_button_resets_the_store(self):
        from unittest import mock

        from shapash.webapp.nlp_components import word_importance as word_importance_module

        app = self._app()
        update = _callback(app, "data...global-importance-graph.clickData")
        with mock.patch.object(word_importance_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "word-filter-clear-btn.n_clicks"}]
            words, reset = update(None, 1)
        self.assertIsNone(words)
        self.assertIsNone(reset)

    def test_empty_click_prevents_update(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate
        from shapash.webapp.nlp_components import word_importance as word_importance_module

        app = self._app()
        update = _callback(app, "data...global-importance-graph.clickData")
        with mock.patch.object(word_importance_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "global-importance-graph.clickData"}]
            with self.assertRaises(PreventUpdate):
                update({"points": []}, None)
            with self.assertRaises(PreventUpdate):
                update(None, None)

    def test_toggle_clear_button_visibility(self):
        app = self._app()
        toggle = _callback(app, "word-filter-clear-btn.style")
        self.assertEqual(toggle(None)["display"], "none")
        self.assertEqual(toggle(["common"])["display"], "inline")


class TestWordImportanceScatterSync(unittest.TestCase):
    """Word Importance and Scatter communicate only through the shared stores (Phase A extraction).

    A bar click always writes ``word-click-filter`` directly now, whether or not a scatter is
    mounted (no more branching on the scatter's presence), and the scatter's own word dropdown
    mirrors that store bidirectionally instead of being the "source of truth" only when it exists.
    """

    @staticmethod
    def _explanation():
        return TestGlobalWordImportancePanel._explanation()

    def test_bar_click_registered_without_a_scatter(self):
        app = NlpWebApp(self._explanation(), engine=None)
        pairs = _callback_binding_ids(app, "global-importance-graph.clickData")
        self.assertIn(("global-importance-graph", "clickData"), pairs)
        self.assertIn(("word-filter-clear-btn", "n_clicks"), pairs)

    def test_bar_click_registered_with_a_scatter_too(self):
        # Before Phase A, a mounted scatter changed which store a bar click targeted; now Word
        # Importance always registers this callback unconditionally, regardless of scatter presence.
        app = NlpWebApp(self._explanation(), engine=None, projection=np.zeros((4, 2)))
        pairs = _callback_binding_ids(app, "global-importance-graph.clickData")
        self.assertIn(("global-importance-graph", "clickData"), pairs)

    def test_scatter_word_select_mirrors_the_shared_store(self):
        app = NlpWebApp(self._explanation(), engine=None, projection=np.zeros((4, 2)))
        pairs = _callback_binding_ids(app, "scatter-word-select.value")
        self.assertIn(("word-click-filter", "data"), pairs)

    def test_class_selector_publishes_to_the_active_class_store(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "active-class-store.data")
        self.assertEqual(fn(1), 1)
        # "All classes" (the string sentinel) is not itself a class index — falls back to 0.
        self.assertEqual(fn("all"), 0)
