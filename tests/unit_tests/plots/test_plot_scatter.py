"""Unit tests for the plot_scatter function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_scatter import plot_scatter


class TestPlotScatter(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.texts = ["so happy today !", "Happy and happy again", "not happy at all", "nothing here"]

    def test_categorical_mode_draws_one_trace_per_label_name(self):
        fig = plot_scatter(self.xy, self.texts, labels=["pos", "pos", "neg", "neg"], label_names=["pos", "neg"])
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(fig.data[0].name, "pos")
        self.assertEqual([row[0] for row in fig.data[0].customdata], [0, 1])

    def test_categorical_mode_skips_empty_categories(self):
        fig = plot_scatter(self.xy, self.texts, labels=["pos", "pos", "pos", "pos"], label_names=["pos", "neg"])
        self.assertEqual(len(fig.data), 1)

    def test_word_contribution_mode_splits_present_and_absent(self):
        fig = plot_scatter(self.xy, self.texts, contributions=np.array([0.4, 0.0, -0.6, 0.0]))
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
        absent_trace, present_trace = fig.data
        self.assertEqual([row[0] for row in absent_trace.customdata], [1, 3])
        self.assertEqual([row[0] for row in present_trace.customdata], [0, 2])

    def test_error_mask_shrinks_correct_points_and_grows_wrong_ones(self):
        contributions = np.array([0.4, 0.0, -0.6, 0.0])
        error_mask = np.array([True, False, False, False])
        fig = plot_scatter(self.xy, self.texts, contributions=contributions, error_mask=error_mask)
        _, present_trace = fig.data
        # present_mask is [0, 2]; sample 0 is the error and should be emphasized over sample 2.
        sizes = present_trace.marker.size
        self.assertGreater(sizes[0], sizes[1])

    def test_requires_exactly_one_of_contributions_or_labels(self):
        with self.assertRaises(ValueError):
            plot_scatter(self.xy, self.texts)
        with self.assertRaises(ValueError):
            plot_scatter(
                self.xy,
                self.texts,
                labels=["pos"] * 4,
                label_names=["pos"],
                contributions=np.zeros(4),
            )

    def test_labels_without_label_names_raises(self):
        with self.assertRaises(ValueError):
            plot_scatter(self.xy, self.texts, labels=["pos"] * 4)
