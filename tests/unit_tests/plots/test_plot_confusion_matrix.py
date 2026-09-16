"""Unit tests for the plot_confusion_matrix function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_confusion_matrix import plot_confusion_matrix


class TestPlotConfusionMatrix(unittest.TestCase):
    def setUp(self):
        # 3-class matrix: rows = true, cols = predicted. Row 2 (index 2) is empty.
        self.cm = np.array([[5, 2, 0], [1, 4, 0], [0, 0, 0]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertIsInstance(fig, go.Figure)
        self.assertIsInstance(fig.data[0], go.Heatmap)

    def test_axes_are_labelled_true_and_predicted(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(list(fig.data[0].x), self.labels)
        self.assertEqual(list(fig.data[0].y), self.labels)

    def test_customdata_encodes_pred_then_true(self):
        # customdata[true][pred] must be [pred_idx, true_idx] for the click handler.
        fig = plot_confusion_matrix(self.cm, self.labels)
        cd = np.asarray(fig.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # true=0, pred=1
        self.assertEqual(list(cd[1, 2]), [2, 1])  # true=1, pred=2

    def test_counts_shown_as_text(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(fig.data[0].text[0][0], "5")

    def test_normalize_true_is_row_recall(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        np.testing.assert_allclose(z[0], [5 / 7, 2 / 7, 0.0])

    def test_normalize_true_handles_empty_row_without_nan(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        z = np.asarray(fig.data[0].z)
        self.assertFalse(np.isnan(z).any())
        np.testing.assert_array_equal(z[2], [0.0, 0.0, 0.0])

    def test_custom_title(self):
        fig = plot_confusion_matrix(self.cm, self.labels, title="Errors")
        self.assertIn("Errors", fig.layout.title.text)
