"""Unit tests for the plot_word_importance function — pure, no explainer or model required."""

import unittest

import pandas as pd
import plotly.graph_objs as go

from shapash.plots.plot_word_importance import plot_word_importance


class TestPlotWordImportance(unittest.TestCase):
    def setUp(self):
        self.word_imp = pd.Series(
            {"happy": 0.35, "terrible": -0.28, "wonderful": 0.20, "feel": -0.10},
        )

    def test_returns_figure(self):
        fig = plot_word_importance(self.word_imp)
        self.assertIsInstance(fig, go.Figure)

    def test_has_one_bar_trace(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data), 1)
        self.assertIsInstance(fig.data[0], go.Bar)

    def test_all_words_rendered(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(len(fig.data[0].x), len(self.word_imp))

    def test_custom_title(self):
        fig = plot_word_importance(self.word_imp, title="Joy importance")
        self.assertIn("Joy importance", fig.layout.title.text)

    def test_orientation_horizontal(self):
        fig = plot_word_importance(self.word_imp)
        self.assertEqual(fig.data[0].orientation, "h")
