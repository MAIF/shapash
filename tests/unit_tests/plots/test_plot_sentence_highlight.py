"""Unit tests for the plot_sentence_highlight function — pure, no explainer or model required."""

import unittest

import numpy as np
from dash import html

from shapash.plots.plot_sentence_highlight import plot_sentence_highlight


class TestPlotSentenceHighlight(unittest.TestCase):
    def setUp(self):
        self.tokens = ["[CLS]", "i", "feel", "happy", "[SEP]"]
        self.values = np.array([0.01, 0.05, 0.30, -0.20, 0.01])

    def test_returns_html_div(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsInstance(result, html.Div)

    def test_has_children(self):
        result = plot_sentence_highlight(self.tokens, self.values)
        self.assertIsNotNone(result.children)
        # legend + spans div + summary
        self.assertGreaterEqual(len(result.children), 3)

    def test_raises_on_2d_values(self):
        with self.assertRaises(ValueError):
            plot_sentence_highlight(self.tokens, np.zeros((5, 3)))

    def test_with_base_value_does_not_raise(self):
        result = plot_sentence_highlight(self.tokens, self.values, base_value=0.15)
        self.assertIsInstance(result, html.Div)

    def test_empty_tokens_does_not_raise(self):
        result = plot_sentence_highlight([], np.array([]))
        self.assertIsInstance(result, html.Div)
