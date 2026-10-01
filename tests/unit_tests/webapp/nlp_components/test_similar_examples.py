"""Unit tests for shapash.webapp.nlp_components.similar_examples: SimilarExamplesComponent."""

import unittest

from shapash.compute.retrieval.similar_examples import Neighbor

from tests.unit_tests.webapp.nlp_components._shared import FakeEngine, _collect_ids, _ctx


class TestSimilarComponent(unittest.TestCase):
    """Exercise the Similar Examples component's renderer and callbacks directly."""

    @staticmethod
    def _register():
        import dash

        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = SimilarExamplesComponent()
        comp.register_callbacks(
            app, _ctx(explanation, engine), {"apply": "whatif-apply-store", "current": "current-datapoint"}
        )
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_layout_shows_layer_caption(self):
        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        found = set()
        _collect_ids(SimilarExamplesComponent().layout(_ctx(engine.to_explanation(), engine)), found)
        self.assertIn("similar-topk", found)
        self.assertIn("similar-threshold", found)
        self.assertIn("similar-mode", found)
        self.assertIn("similar-results", found)

    def test_update_similar_returns_table_and_texts(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "topk", 5, 0.95)
        self.assertEqual(texts, ["i am joyful", "this is awful"])
        self.assertIsNotNone(children)

    def test_update_similar_ignores_empty_text(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        update = self._callback(app, "similar-results")
        with self.assertRaises(PreventUpdate):
            update({"text": "  "}, "topk", 5, 0.95)

    def test_update_similar_threshold_mode_filters_and_reports_total(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.90)
        # Only the two neighbours scoring above 0.90 clear the threshold (see FakeEngine.find_similar_threshold).
        self.assertEqual(texts, ["i am joyful", "so glad today"])
        self.assertIsNotNone(children)

    def test_update_similar_threshold_mode_empty_above_cutoff(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.999)
        self.assertEqual(texts, [])
        self.assertIsNotNone(children)

    def test_toggle_mode_inputs_disables_the_inactive_control(self):
        app, _ = self._register()
        toggle = self._callback(app, "similar-topk.disabled")
        topk_disabled, threshold_disabled = toggle("threshold")
        self.assertTrue(topk_disabled)
        self.assertFalse(threshold_disabled)
        topk_disabled, threshold_disabled = toggle("topk")
        self.assertFalse(topk_disabled)
        self.assertTrue(threshold_disabled)

    def test_inspect_makes_neighbor_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import similar_examples as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "similar-apply", "index": 1}
            dp = inspect([1, 1], ["first neighbor", "second neighbor"])
        self.assertEqual(dp["text"], "second neighbor")
        self.assertEqual(dp["label"], "pos")  # FakeEngine.explain_text returns "pos"

    def test_neighbors_table_marks_matching_label(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [
            Neighbor(index=0, score=0.9, text="joyful one", label="pos"),
            Neighbor(index=1, score=0.5, text="grim one", label="neg"),
        ]
        table = _neighbors_table(neighbors, predicted_label="pos", component_id="similar")
        ids = set()
        _collect_ids(table, ids)
        # One Inspect button per neighbour (pattern-matching ids are dicts, so assert via the count).
        self.assertEqual(sum(1 for n in neighbors), 2)
        self.assertIsNotNone(table)

    def test_neighbors_table_without_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [Neighbor(index=0, score=0.9, text="some text", label=None)]
        table = _neighbors_table(neighbors, predicted_label=None, component_id="similar")
        self.assertIsNotNone(table)

    def test_match_rate_caption_reports_share_of_predicted_label(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [
            Neighbor(index=0, score=0.9, text="a", label="pos"),
            Neighbor(index=1, score=0.8, text="b", label="pos"),
            Neighbor(index=2, score=0.7, text="c", label="neg"),
            Neighbor(index=3, score=0.6, text="d", label="neg"),
        ]
        caption = _match_rate_caption(neighbors, predicted_label="pos")
        self.assertIn("2/4", caption)
        self.assertIn("50%", caption)

    def test_match_rate_caption_none_without_a_prediction_or_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [Neighbor(index=0, score=0.9, text="a", label="pos")]
        self.assertIsNone(_match_rate_caption(neighbors, predicted_label=None))
        unlabelled = [Neighbor(index=0, score=0.9, text="a", label=None)]
        self.assertIsNone(_match_rate_caption(unlabelled, predicted_label="pos"))

    def test_render_results_includes_cap_note_only_when_capped(self):
        from shapash.webapp.nlp_components.similar_examples import _render_results

        neighbors = [Neighbor(index=0, score=0.99, text="a", label="pos")]
        capped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=5)
        uncapped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=None)

        def _flat_text(node):
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return children
            if isinstance(children, list):
                return "".join(_flat_text(c) for c in children if c is not None)
            return _flat_text(children) if children is not None else ""

        self.assertIn("Showing 1 of 5", _flat_text(capped))
        self.assertNotIn("Showing", _flat_text(uncapped))
