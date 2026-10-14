"""Unit tests for the plot_noise_matrix function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_noise_matrix import plot_noise_matrix


class TestPlotNoiseMatrix(unittest.TestCase):
    """The noise matrix is given-vs-true and mostly diagonal, so it needs its own axes and masking."""

    def setUp(self):
        # Typical shape: ~94% correctly labelled, one contaminated pair (A labelled, really B).
        self.joint = np.array([[0.50, 0.05, 0.0], [0.01, 0.30, 0.0], [0.0, 0.0, 0.14]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        self.assertIsInstance(plot_noise_matrix(self.joint, self.labels), go.Figure)

    def test_diagonal_is_masked_by_default(self):
        # Left in, the ~90% diagonal flattens every off-diagonal cell to the same near-white shade.
        z = plot_noise_matrix(self.joint, self.labels).data[0].z
        self.assertTrue(np.all(np.isnan(np.diag(z))))
        self.assertAlmostEqual(z[0][1], 0.05)

    def test_masked_cells_render_no_text(self):
        text = plot_noise_matrix(self.joint, self.labels).data[0].text
        self.assertEqual(text[0][0], "")
        self.assertEqual(text[0][1], "5.0%")

    def test_diagonal_can_be_kept(self):
        z = plot_noise_matrix(self.joint, self.labels, mask_diagonal=False).data[0].z
        np.testing.assert_allclose(np.diag(z), [0.50, 0.30, 0.14])

    def test_does_not_mutate_the_callers_matrix(self):
        original = self.joint.copy()
        plot_noise_matrix(self.joint, self.labels)
        np.testing.assert_array_equal(self.joint, original)

    def test_axes_name_the_label_semantics_not_the_prediction(self):
        # The whole reason this is not a plot_confusion_matrix mode: the columns are the estimated
        # *true* class, not the model's prediction.
        layout = plot_noise_matrix(self.joint, self.labels).layout
        self.assertEqual(layout.yaxis.title.text, "Given label")
        self.assertEqual(layout.xaxis.title.text, "Estimated true class")
