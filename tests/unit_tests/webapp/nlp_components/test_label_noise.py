"""Unit tests for shapash.webapp.nlp_components.label_noise: LabelNoiseComponent."""

import unittest

import pandas as pd

from shapash.compute.diagnostics.label_noise import LabelIssue
from shapash.compute.diagnostics.label_probe import ProbeVerdict
from shapash.webapp.nlp_components import LabelNoiseComponent

from tests.unit_tests.webapp.nlp_components._shared import _PROBE_CORPUS, FakeEngine, _ctx


class TestLabelNoiseComponent(unittest.TestCase):
    """Drive the panel's renderer and callbacks directly, without the app shell."""

    @staticmethod
    def _register(**kwargs):
        import dash

        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, **kwargs)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = LabelNoiseComponent()
        comp.register_callbacks(
            app, _ctx(explanation, engine), {"apply": "whatif-apply-store", "current": "current-datapoint"}
        )
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_detect_returns_a_view_and_the_flagged_indices(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [0])  # sample 0 is labelled neg but confidently predicted pos
        self.assertIsNotNone(children)

    def test_detect_forwards_the_controls_to_the_engine(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, 7, "normalized_margin")
        self.assertEqual(engine.detect_calls[-1], {"top_n": 7, "score": "normalized_margin", "probe": True})

    def test_blank_controls_fall_back_to_defaults(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, None, None)
        self.assertEqual(engine.detect_calls[-1], {"top_n": 50, "score": "self_confidence", "probe": True})

    def test_detect_renders_the_corpus_column_when_a_probe_corpus_is_bound(self):
        app, _ = self._register(probe_corpus=_PROBE_CORPUS)
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertEqual(headers, ["Score", "Given", "Probably", "Corpus", "Text", ""])
        # The flagged row is "i am happy" labelled "neg"; the corpus puts that vocabulary in "pos",
        # so the probe rejects the label too rather than defending it.
        cell = table.children[1].children[0].children[3].children
        self.assertEqual(cell.children, "0.14 → pos")

    def test_detect_omits_the_corpus_column_without_a_probe_corpus(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)

    def test_detect_does_nothing_before_the_button_is_clicked(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        with self.assertRaises(PreventUpdate):
            detect(None, 50, "self_confidence")

    def test_a_clean_corpus_renders_a_message_and_no_rows(self):
        app, engine = self._register()
        # Both classes labelled and both labels agreeing with the model, so there is no noise to find.
        engine.y_prob = pd.DataFrame({"neg": [0.1, 0.9], "pos": [0.9, 0.1]}, index=pd.RangeIndex(2))
        engine.y_true = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="ground_truth")
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [])
        self.assertIn("No label issues detected", children.children)

    def test_inspect_makes_the_flagged_sample_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            dp = inspect([1], [1])  # row 0 of the table points at compiled sample 1
        self.assertEqual(dp["text"], "this is bad")
        self.assertEqual(dp["orig_idx"], 1)
        self.assertEqual(dp["label"], "pos")  # the *prediction*, matching the dataset-row path
        self.assertEqual(len(dp["tokens"]), 3)

    def test_inspect_ignores_a_render_with_no_clicks(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with self.assertRaises(PreventUpdate):
            inspect([0], [1])

    def test_inspect_ignores_a_missing_trigger(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = None
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_an_out_of_range_row(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 5}
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_a_stale_index_beyond_the_corpus(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            with self.assertRaises(PreventUpdate):
                inspect([1], [99])


class TestLabelNoiseTable(unittest.TestCase):
    """The neighbour column is evidence *beside* the score, and must not appear when absent."""

    @staticmethod
    def _issue(**kwargs):
        base = dict(
            index=0,
            text="i am happy",
            given_label="neg",
            suggested_label="pos",
            given_prob=0.2,
            suggested_prob=0.8,
            score=0.2,
        )
        return LabelIssue(**{**base, **kwargs})

    def test_corpus_column_omitted_when_no_probe_ran(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        table = _issues_table([self._issue()], "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)
        self.assertEqual(headers, ["Score", "Given", "Probably", "Text", ""])

    def test_corpus_column_present_when_any_row_has_a_verdict(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        verdict = ProbeVerdict(given_prob=0.8, top_label="neg", backs_given=True)
        issues = [self._issue(probe=verdict), self._issue(index=1)]
        table = _issues_table(issues, "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertIn("Corpus", headers)

    def test_probe_summary_warns_when_the_corpus_backs_the_given_label(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        # The row the panel exists to catch: confident learning flagged it, but an independent
        # classifier defends the label — so it is the model that is wrong, not the corpus.
        verdict = ProbeVerdict(given_prob=0.83, top_label="neg", backs_given=True)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "⚠ 0.83")
        self.assertIn("text-warning", summary.className)
        self.assertIn("more likely", summary.title)

    def test_probe_summary_reports_a_corroborated_label_error_plainly(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        verdict = ProbeVerdict(given_prob=0.16, top_label="pos", backs_given=False)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "0.16 → pos")
        self.assertIn("text-muted", summary.className)

    def test_rows_without_a_verdict_render_a_dash(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        self.assertEqual(_probe_summary(self._issue()), "—")
