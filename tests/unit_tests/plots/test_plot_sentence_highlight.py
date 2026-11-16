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

    def test_accepts_hex_and_named_colors(self):
        """Hex codes and color names are accepted; the strongest contribution takes the full color."""
        result = plot_sentence_highlight(self.tokens, self.values, color_positive="#123456", color_negative="blue")

        def backgrounds(component):
            style = getattr(component, "style", None) or {}
            children = getattr(component, "children", None)
            children = children if isinstance(children, list) else [children] if children is not None else []
            own = [style["backgroundColor"]] if "backgroundColor" in style else []
            return own + [bg for child in children for bg in backgrounds(child)]

        self.assertIn("rgb(18,52,86)", backgrounds(result))

    def test_raises_on_2d_values(self):
        with self.assertRaises(ValueError):
            plot_sentence_highlight(self.tokens, np.zeros((5, 3)))

    def test_with_base_value_does_not_raise(self):
        result = plot_sentence_highlight(self.tokens, self.values, base_value=0.15)
        self.assertIsInstance(result, html.Div)

    def test_empty_tokens_does_not_raise(self):
        result = plot_sentence_highlight([], np.array([]))
        self.assertIsInstance(result, html.Div)
