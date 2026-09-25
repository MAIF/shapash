"""Unit tests for shapash.webapp.nlp_components.waterfall: WaterfallComponent."""

import unittest

from shapash.webapp.nlp_app import NlpWebApp

from tests.unit_tests.webapp.nlp_components._shared import (
    LABEL_NAMES,
    _callback,
    _make_local_panel_datapoint,
    _make_local_panel_explanation,
)


class TestWaterfallComponentCallbacks(unittest.TestCase):
    """The Waterfall panel's own callback: current datapoint + class + threshold -> figure."""

    def _app(self):
        return NlpWebApp(_make_local_panel_explanation(), engine=None)

    def test_update_waterfall_prevents_update_without_datapoint_or_class(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        update = _callback(app, "waterfall-graph.figure")
        with self.assertRaises(PreventUpdate):
            update(None, 0, 10)
        with self.assertRaises(PreventUpdate):
            update(_make_local_panel_datapoint(), None, 10)

    def test_update_waterfall_renders_and_titles_by_class(self):
        import plotly.graph_objs as go

        app = self._app()
        update = _callback(app, "waterfall-graph.figure")
        fig = update(_make_local_panel_datapoint(), 1, 10)
        self.assertIsInstance(fig, go.Figure)
        self.assertIn(LABEL_NAMES[1], fig.layout.title.text)

    def test_update_waterfall_defaults_the_threshold_when_missing(self):
        app = self._app()
        update = _callback(app, "waterfall-graph.figure")
        # None threshold (e.g. before the slider ever fires) must not raise — falls back to 10%.
        fig = update(_make_local_panel_datapoint(), 0, None)
        self.assertIsNotNone(fig)
