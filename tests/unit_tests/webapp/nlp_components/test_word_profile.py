"""Unit tests for shapash.webapp.nlp_components.word_profile: WordProfileComponent."""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from shapash.explainer.nlp_explanation import NlpExplanation

from tests.unit_tests.webapp.nlp_components._shared import LABEL_NAMES, _collect_ids, _ctx


class TestWordProfileComponent(unittest.TestCase):
    """The single-word profile panel: word × aggregation × global selection, then drill-down."""

    STORES = {
        "current": "current-datapoint",
        "selection": "scatter-selected-indices",
        "error_cell": "error-cell",
        "errors_only": "errors-only-switch",
        "word_click": "word-click-filter",
    }

    @staticmethod
    def _explanation():
        """Four samples, two classes. 'happy' pulls +0.4/+0.3 in two, -0.6 in a third."""
        token_strings = [
            ["so", "happy", "today"],
            ["happy", "and", "happy"],
            ["not", "happy", "at", "all"],
            ["nothing", "here"],
        ]
        values = [
            np.array([[0.1, -0.1], [0.4, -0.4], [0.05, -0.05]]),
            np.array([[0.2, -0.2], [0.0, 0.0], [0.1, -0.1]]),
            np.array([[-0.1, 0.1], [-0.6, 0.6], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ]
        texts = pd.Series(["so happy today", "happy and happy", "not happy at all", "nothing here"])
        return NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["pos", "pos", "neg", "neg"], index=texts.index, name="prediction"),
            y_prob=None,
            # Sample 1 is the only model error, so errors-only scopes to exactly that row.
            y_true=pd.Series(["pos", "neg", "neg", "neg"], index=texts.index, name="ground_truth"),
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )

    def _app(self):
        import dash

        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        app = dash.Dash(__name__)
        comp = WordProfileComponent()
        comp.register_callbacks(app, _ctx(explanation, None), self.STORES)
        return app, explanation

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    # ── mounting / layout ───────────────────────────────────────────────
    def test_mounts_without_an_engine(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        # Data-only panel: it must survive a loaded snapshot with no live model.
        self.assertTrue(WordProfileComponent.is_available(_ctx(self._explanation(), None)))

    def test_layout_declares_every_id_its_callbacks_bind(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        found = set()
        _collect_ids(WordProfileComponent().layout(_ctx(self._explanation(), None)), found)
        for suffix in ("select", "agg", "class", "order", "limit", "graph", "caption", "results", "store"):
            self.assertIn(f"word-profile-{suffix}", found)

    def test_word_dropdown_is_seeded_with_the_top_word(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        layout = WordProfileComponent().layout(_ctx(explanation, None))
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
        # Not empty on open, and the seed is a real corpus word.
        self.assertIn(found["word-profile-select"].value, explanation.vocabulary())

    def test_class_picker_hidden_for_a_single_output_column(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        binary = replace(
            explanation,
            values=[v[:, 0] for v in explanation.values],
            base_values=np.zeros(4),
            label_names=["score"],
        )
        layout = WordProfileComponent().layout(_ctx(binary, None))
        found = set()
        _collect_ids(layout, found)
        # Still present (callbacks bind it), just not shown.
        self.assertIn("word-profile-class", found)

    # ── the profile callback ────────────────────────────────────────────
    def test_profile_reports_counts_and_one_bar_per_class(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "strongest", 10, None, None, False
        )
        self.assertEqual(len(fig.data[0].x), 2)
        self.assertIn("4 occurrence(s) in 3 of 4 sample(s)", caption)
        self.assertIsNotNone(table)
        self.assertEqual(store, [2, 0, 1])  # |-0.6| > 0.4 > 0.3

    def test_mean_reports_its_error_bars_in_the_caption(self):
        app, _ = self._app()
        _, caption, _, _ = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 10, None, None, False)
        self.assertIn("std across occurrences", caption)

    def test_sum_has_no_error_bars(self):
        app, _ = self._app()
        fig, caption, _, _ = self._callback(app, "word-profile-graph")("happy", "sum", 0, "most", 10, None, None, False)
        self.assertIsNone(fig.data[0].error_x.array)
        self.assertNotIn("std", caption)

    def test_absolute_aggregation_surfaces_the_two_way_word(self):
        app, _ = self._app()
        graph = self._callback(app, "word-profile-graph")
        signed = graph("happy", "mean", 0, "most", 10, None, None, False)[0].data[0].x
        magnitude = graph("happy", "mean_abs", 0, "most", 10, None, None, False)[0].data[0].x
        # The signed mean nearly cancels; the magnitude does not. This gap is the panel's point.
        self.assertLess(abs(signed[-1]), 0.05)
        self.assertGreater(magnitude[-1], 0.3)

    def test_order_reorders_the_drill_down_only(self):
        app, _ = self._app()
        graph = self._callback(app, "word-profile-graph")
        self.assertEqual(graph("happy", "mean", 0, "most", 10, None, None, False)[3], [0, 1, 2])
        self.assertEqual(graph("happy", "mean", 0, "least", 10, None, None, False)[3][0], 2)

    def test_limit_truncates_the_drill_down(self):
        app, _ = self._app()
        store = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 1, None, None, False)[3]
        self.assertEqual(len(store), 1)

    def test_scatter_selection_scopes_the_aggregate(self):
        app, _ = self._app()
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, [0, 3], None, False
        )
        self.assertIn("scoped to 2 selected sample(s)", caption)
        self.assertEqual(store, [0])

    def test_errors_only_scopes_the_aggregate(self):
        app, _ = self._app()
        # Sample 1 is the only misclassified row, and it does contain the word.
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, None, None, True
        )
        self.assertIn("scoped to 1 selected sample(s)", caption)
        self.assertEqual(store, [1])

    def test_confusion_cell_scopes_the_aggregate(self):
        app, _ = self._app()
        _, caption, _, _ = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, None, {"pred": 1, "true": 0, "indices": [2]}, False
        )
        self.assertIn("scoped to 1 selected sample(s)", caption)

    def test_word_absent_from_the_scope_says_so(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, [3], None, False
        )
        self.assertIn("no occurrences", caption)
        self.assertIn("does not occur", fig.layout.annotations[0].text)
        self.assertIsNone(table)
        self.assertEqual(store, [])

    def test_no_word_selected_shows_a_prompt(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            None, "mean", 0, "most", 10, None, None, False
        )
        self.assertIn("Pick a word", fig.layout.annotations[0].text)
        self.assertEqual((caption, table, store), ("", None, []))

    def test_falls_back_to_defaults_on_cleared_controls(self):
        app, _ = self._app()
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", None, None, None, None, None, None, None
        )
        self.assertIn("Mean", caption)
        self.assertEqual(len(store), 3)

    # ── cross-panel wiring ──────────────────────────────────────────────
    def test_follows_a_word_bar_clicked_in_the_importance_panel(self):
        app, _ = self._app()
        follow = self._callback(app, "word-profile-select.value")
        self.assertEqual(follow("today"), "today")
        # With a scatter the store holds the multi-select list; the newest word wins.
        self.assertEqual(follow(["today", "happy"]), "happy")

    def test_cleared_word_filter_leaves_the_selection_alone(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._app()
        follow = self._callback(app, "word-profile-select.value")
        for cleared in (None, []):
            with self.assertRaises(PreventUpdate):
                follow(cleared)

    def test_inspect_packs_the_sample_into_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import word_profile as mod

        app, _ = self._app()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "word-profile-inspect", "index": 1}
            # The store holds the displayed rows' positions, so row 1 is sample 0.
            datapoint = inspect([None, 1], [2, 0])
        self.assertEqual(datapoint["orig_idx"], 0)
        self.assertEqual(datapoint["text"], "so happy today")
        self.assertEqual(datapoint["label"], "pos")

    def test_inspect_ignores_clicks_it_cannot_resolve(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import word_profile as mod

        app, _ = self._app()
        inspect = self._callback(app, "current-datapoint")
        with self.assertRaises(PreventUpdate):
            inspect([None, None], [0, 1])  # no button actually clicked
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = None
            with self.assertRaises(PreventUpdate):
                inspect([1], [0])
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "word-profile-inspect", "index": 5}
            with self.assertRaises(PreventUpdate):
                inspect([1], [0])  # index past the end of the displayed rows


class TestWordProfileControls(unittest.TestCase):
    """The picker's sort/labels, the restricted aggregation list, and bar-click class ranking."""

    def _layout(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        comp = WordProfileComponent()
        layout = comp.layout(_ctx(TestWordProfileComponent._explanation(), None))
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
        return comp, found

    def _app(self):
        import dash

        from shapash.webapp.nlp_components import WordProfileComponent

        comp = WordProfileComponent()
        explanation = TestWordProfileComponent._explanation()
        comp.layout(_ctx(explanation, None))  # builds the option lists the sort callback serves
        app = dash.Dash(__name__)
        comp.register_callbacks(app, _ctx(explanation, None), TestWordProfileComponent.STORES)
        return app

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_sum_aggregations_are_not_offered(self):
        # Per word, a sum is the mean rescaled by the occurrence count — same bars, no information.
        _, found = self._layout()
        self.assertEqual([o["value"] for o in found["word-profile-agg"].options], ["mean", "mean_abs"])

    def test_options_carry_the_occurrence_count(self):
        _, found = self._layout()
        labels = {o["value"]: o["label"] for o in found["word-profile-select"].options}
        self.assertEqual(labels["happy"], "happy (4)")

    def test_option_values_stay_bare_words(self):
        # The label carries the count; the value must not, or every consumer of it breaks.
        _, found = self._layout()
        self.assertIn("happy", [o["value"] for o in found["word-profile-select"].options])

    def test_sort_toggle_reorders_without_changing_values(self):
        app = self._app()
        reorder = self._callback(app, "word-profile-select.options")
        alpha = [o["value"] for o in reorder("alpha")]
        freq = [o["value"] for o in reorder("frequency")]
        self.assertEqual(alpha, sorted(alpha))
        self.assertEqual(freq[0], "happy")  # the most frequent word
        self.assertEqual(set(alpha), set(freq))

    def test_unknown_sort_falls_back_to_alphabetical(self):
        app = self._app()
        reorder = self._callback(app, "word-profile-select.options")
        self.assertEqual(reorder(None), reorder("alpha"))

    def test_bar_click_sets_the_drill_down_class(self):
        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        value, reset = click({"points": [{"customdata": [1], "y": "pos"}]})
        self.assertEqual(value, 1)
        # clickData is reset so clicking the same bar twice re-fires.
        self.assertIsNone(reset)

    def test_bar_click_reads_the_index_not_the_label(self):
        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        # Duplicate display names must not be able to mis-resolve the class.
        self.assertEqual(click({"points": [{"customdata": [0], "y": "neg"}]})[0], 0)

    def test_graph_carries_the_class_index_on_every_bar(self):
        app = self._app()
        fig = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 5, None, None, False)[0]
        self.assertEqual([c[0] for c in fig.data[0].customdata], [1, 0])  # reversed for drawing

    def test_empty_clicks_are_ignored(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        for bad in (None, {}, {"points": []}, {"points": [{}]}):
            with self.assertRaises(PreventUpdate):
                click(bad)
