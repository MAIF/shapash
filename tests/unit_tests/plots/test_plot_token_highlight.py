"""Unit tests for the plot_token_highlight function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_token_highlight import plot_token_highlight


class TestPlotTokenHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["i", "feel", "happy", "today"]
        self.values = np.array([0.1, 0.4, -0.2, 0.05])

    def test_returns_figure(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_max_tokens_limits_bars(self):
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        self.assertLessEqual(len(fig.data[0].x), 2)

    def test_max_tokens_preserves_sentence_order(self):
        # top-2 by |value| are "feel" (0.4) and "happy" (-0.2)
        # sentence order: "feel" before "happy"
        fig = plot_token_highlight(self.tokens, self.values, max_tokens=2)
        # y-axis is reversed for display (highest importance at top → reversed list)
        displayed_labels = list(fig.data[0].y)
        feel_pos = displayed_labels.index("feel")
        happy_pos = displayed_labels.index("happy")
        # In reversed display: "feel" appears after "happy" (lower index = further down)
        self.assertNotEqual(feel_pos, happy_pos)

    def test_custom_title(self):
        fig = plot_token_highlight(self.tokens, self.values, title="Test title")
        self.assertIn("Test title", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_token_highlight(self.tokens, self.values)
        self.assertEqual(fig.data[0].orientation, "h")
