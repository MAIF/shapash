"""Unit tests for shapash.webapp.nlp_components.sentence_highlight: SentenceHighlightComponent."""

import unittest

from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import (
    LABEL_NAMES,
    _callback,
    _make_local_panel_datapoint,
    _make_local_panel_explanation,
)


class TestSentenceHighlightComponentCallbacks(unittest.TestCase):
    """The Sentence panel's own callbacks: class-picker sync to the prediction, and the render."""

    def _app(self):
        return NlpWebApp(_make_local_panel_explanation(), engine=None)

    def test_sync_local_class_prevents_update_without_a_datapoint(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        sync = _callback(app, "local-class-selector.value")
        with self.assertRaises(PreventUpdate):
            sync(None)
        with self.assertRaises(PreventUpdate):
            sync({"text": "i am happy", "label": None})

    def test_sync_local_class_prevents_update_for_an_unknown_label(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        sync = _callback(app, "local-class-selector.value")
        with self.assertRaises(PreventUpdate):
            sync(_make_local_panel_datapoint(label="not-a-real-class"))

    def test_sync_local_class_resolves_the_predicted_label_to_its_index(self):
        app = self._app()
        sync = _callback(app, "local-class-selector.value")
        self.assertEqual(sync(_make_local_panel_datapoint(label="pos")), LABEL_NAMES.index("pos"))

    def test_update_sentence_highlight_prevents_update_without_datapoint_or_class(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        update = _callback(app, "sentence-highlight.children")
        with self.assertRaises(PreventUpdate):
            update(None, 0)
        with self.assertRaises(PreventUpdate):
            update(_make_local_panel_datapoint(), None)

    def test_update_sentence_highlight_renders_for_a_valid_datapoint(self):
        from dash import html

        app = self._app()
        update = _callback(app, "sentence-highlight.children")
        result = update(_make_local_panel_datapoint(), 1)
        self.assertIsInstance(result, html.Div)
