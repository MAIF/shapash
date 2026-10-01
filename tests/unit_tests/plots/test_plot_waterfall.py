"""Unit tests for the plot_waterfall function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_waterfall import plot_waterfall


class TestPlotWaterfall(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "so", "happy", "today", "[SEP]"]
        self.values = np.array([0.01, 0.08, 0.35, 0.02, -0.20, 0.05, 0.01])

    def test_returns_figure(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertIsInstance(fig, go.Figure)

    def test_has_waterfall_trace(self):
        fig = plot_waterfall(self.tokens, self.values)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Waterfall)

    def test_filters_special_tokens_by_default(self):
        fig = plot_waterfall(self.tokens, self.values)
        y_labels = list(fig.data[0].y)
        for label in y_labels:
            self.assertNotIn("[CLS]", label)
            self.assertNotIn("[SEP]", label)

    def test_keeps_special_tokens_when_disabled(self):
        # min_pct=0 disables grouping so every token gets its own bar
        fig = plot_waterfall(self.tokens, self.values, filter_special=False, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("[CLS]" in lbl for lbl in y_labels))

    def test_total_bar_present(self):
        fig = plot_waterfall(self.tokens, self.values)
        measures = list(fig.data[0].measure)
        self.assertIn("total", measures)

    def test_grouping_reduces_bar_count(self):
        # With min_pct=0.0 (no grouping), every non-special token is its own bar
        fig_no_group = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        # With min_pct=0.5, small tokens are lumped
        fig_grouped = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        self.assertLessEqual(len(fig_grouped.data[0].y), len(fig_no_group.data[0].y))

    def test_other_bar_present_when_grouping_active(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.5)
        y_labels = list(fig.data[0].y)
        self.assertTrue(any("other" in lbl for lbl in y_labels))

    def test_no_other_bar_when_min_pct_zero(self):
        fig = plot_waterfall(self.tokens, self.values, min_pct=0.0)
        y_labels = list(fig.data[0].y)
        self.assertFalse(any("other" in lbl for lbl in y_labels))

    def test_base_value_creates_absolute_bar(self):
        fig = plot_waterfall(self.tokens, self.values, base_value=0.20)
        measures = list(fig.data[0].measure)
        self.assertEqual(measures[0], "absolute")
        y_labels = list(fig.data[0].y)
        self.assertEqual(y_labels[0], "Base")

    def test_empty_tokens_returns_figure(self):
        fig = plot_waterfall([], np.array([]))
        self.assertIsInstance(fig, go.Figure)

    def test_all_special_tokens_returns_figure(self):
        fig = plot_waterfall(["[CLS]", "[SEP]"], np.array([0.01, -0.01]))
        self.assertIsInstance(fig, go.Figure)

    def test_custom_title(self):
        fig = plot_waterfall(self.tokens, self.values, title="joy waterfall")
        self.assertIn("joy waterfall", fig.layout.title.text)
