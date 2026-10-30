"""Unit tests for shapash.webapp.nlp_components.counterfactual: CounterfactualComponent."""

import unittest

from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import FakeEngine, _callback


class TestCounterfactualComponentCallbacks(unittest.TestCase):
    """The generate/apply wiring: method-group toggling, the Generate click, and per-row Apply."""

    def _app(self, engine=None):
        engine = engine or FakeEngine(can_edit=True, can_cf=True)
        return NlpWebApp(engine.to_explanation(), engine=engine), engine

    def test_toggle_controls_shows_only_the_selected_generators_group(self):
        app, _ = self._app()
        toggle = _callback(app, "counterfactual-cfg-group-hotflip.style")
        hotflip_style, ablation_style = toggle("hotflip")
        self.assertNotEqual(hotflip_style.get("display"), "none")
        self.assertEqual(ablation_style.get("display"), "none")

    # generate(n_clicks, datapoint, selected_gen, *config_values) — 6 states: (num_examples,
    # max_flips/max_ablations, tokens_to_ignore) for each of FakeEngine's two generators, in order.
    _CONFIG_VALUES = (5, 3, "", 5, 3, "")

    def test_generate_prevents_update_without_a_click(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._app()
        generate = _callback(app, "counterfactual-results.children")
        with self.assertRaises(PreventUpdate):
            generate(None, {"text": "i am happy"}, "hotflip", *self._CONFIG_VALUES)

    def test_generate_prevents_update_without_text(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._app()
        generate = _callback(app, "counterfactual-results.children")
        with self.assertRaises(PreventUpdate):
            generate(1, None, "hotflip", *self._CONFIG_VALUES)
        with self.assertRaises(PreventUpdate):
            generate(1, {"text": "   "}, "hotflip", *self._CONFIG_VALUES)

    def test_generate_returns_a_results_table_on_success(self):
        app, engine = self._app()
        generate = _callback(app, "counterfactual-results.children")
        children, texts = generate(1, {"text": "i am happy"}, "hotflip", *self._CONFIG_VALUES)
        self.assertIsNotNone(children)
        self.assertEqual(texts, [cf.new_text for cf in engine.generate_counterfactuals("i am happy")])

    def test_generate_reports_when_no_counterfactual_is_found(self):
        class _NoResultsEngine(FakeEngine):
            def generate_counterfactuals(self, text, config=None, generator=None):
                return []

        app, _ = self._app(_NoResultsEngine(can_edit=True, can_cf=True))
        generate = _callback(app, "counterfactual-results.children")
        children, texts = generate(1, {"text": "i am happy"}, "hotflip", *self._CONFIG_VALUES)
        self.assertIn("No counterfactual found", children.children)
        self.assertEqual(texts, [])

    def test_apply_prevents_update_without_any_click(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._app()
        apply_fn = _callback(app, "whatif-apply-store.data")
        with self.assertRaises(PreventUpdate):
            apply_fn([None], ["a new text"])

    def test_apply_prevents_update_on_out_of_range_index(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate
        from shapash.webapp.nlp_components import counterfactual as counterfactual_module

        app, _ = self._app()
        apply_fn = _callback(app, "whatif-apply-store.data")
        with mock.patch.object(counterfactual_module, "callback_context") as cc:
            cc.triggered_id = {"type": "counterfactual-apply", "index": 5}
            with self.assertRaises(PreventUpdate):
                apply_fn([1], ["only one text"])

    def test_apply_publishes_the_chosen_text_and_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import counterfactual as counterfactual_module

        app, engine = self._app()
        apply_fn = _callback(app, "whatif-apply-store.data")
        with mock.patch.object(counterfactual_module, "callback_context") as cc:
            cc.triggered_id = {"type": "counterfactual-apply", "index": 0}
            text, datapoint = apply_fn([1], ["a new counterfactual text"])
        self.assertEqual(text, "a new counterfactual text")
        self.assertEqual(datapoint["text"], "a new counterfactual text")
        self.assertEqual(datapoint["label"], "pos")  # FakeEngine.explain_text always returns "pos"
