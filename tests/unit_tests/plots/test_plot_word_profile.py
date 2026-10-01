"""Unit tests for the plot_word_profile function — pure, no explainer or model required."""

import unittest

import pandas as pd
import plotly.graph_objs as go

from shapash.plots.plot_word_profile import plot_word_profile
from shapash.style.style_utils import DEFAULT_NLP_THEME


class TestPlotWordProfile(unittest.TestCase):
    def setUp(self):
        self.stats = pd.Series([0.2, -0.3], index=pd.Index([0, 1], name="class_idx"))

    def test_returns_figure_with_one_bar_per_class(self):
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data[0].x), 2)

    def test_class_zero_is_drawn_at_the_top(self):
        # Plotly draws the y-axis bottom-to-top, so the first class must be last in the array.
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        self.assertEqual(fig.data[0].y[-1], "pos")

    def test_sign_drives_colour(self):
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"])
        # Reversed alongside the values, so the last colour belongs to the positive class 0.
        self.assertEqual(fig.data[0].marker.color[-1], DEFAULT_NLP_THEME.xpl_positive)
        self.assertEqual(fig.data[0].marker.color[0], DEFAULT_NLP_THEME.xpl_negative)

    def test_falls_back_to_class_indices_without_names(self):
        fig = plot_word_profile(self.stats)
        self.assertEqual(set(fig.data[0].y), {"0", "1"})

    def test_spread_is_reversed_with_the_bars(self):
        spread = pd.Series([0.1, 0.9], index=pd.Index([0, 1], name="class_idx"))
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"], spread=spread)
        # Class 1's spread must sit on class 1's bar, which is drawn first.
        self.assertAlmostEqual(fig.data[0].error_x.array[0], 0.9)

    def test_misaligned_spread_does_not_shift_error_bars(self):
        # A spread indexed on classes the stats do not carry must not silently slide onto them.
        spread = pd.Series([0.1], index=pd.Index([5], name="class_idx"))
        fig = plot_word_profile(self.stats, label_names=["pos", "neg"], spread=spread)
        self.assertEqual(list(fig.data[0].error_x.array), [0.0, 0.0])

    def test_no_error_bars_without_spread(self):
        fig = plot_word_profile(self.stats)
        self.assertIsNone(fig.data[0].error_x.array)
