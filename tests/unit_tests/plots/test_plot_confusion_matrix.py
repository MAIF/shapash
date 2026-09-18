"""Unit tests for the plot_confusion_matrix function — pure, no explainer or model required."""

import unittest

import numpy as np
import plotly.graph_objs as go

from shapash.plots.plot_confusion_matrix import _emphasis_colorscale, plot_confusion_matrix


def _trace(fig, name):
    """The one trace called *name* — traces are named rather than positional on purpose."""
    return next(t for t in fig.data if t.name == name)


class TestPlotConfusionMatrix(unittest.TestCase):
    def setUp(self):
        # 3-class matrix: rows = true, cols = predicted. Row 2 (index 2) is empty.
        self.cm = np.array([[5, 2, 0], [1, 4, 0], [0, 0, 0]])
        self.labels = ["A", "B", "C"]

    def test_returns_heatmap_figure(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertIsInstance(fig, go.Figure)
        self.assertTrue(all(isinstance(t, go.Heatmap) for t in fig.data))

    def test_axes_are_labelled_true_and_predicted(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        for trace in fig.data:
            self.assertEqual(list(trace.x), self.labels)
            self.assertEqual(list(trace.y), self.labels)

    def test_customdata_encodes_pred_then_true(self):
        # customdata[true][pred] must be [pred_idx, true_idx] for the click handler — on *every*
        # trace, since a click can land on either the diagonal or an error cell.
        fig = plot_confusion_matrix(self.cm, self.labels)
        for trace in fig.data:
            cd = np.asarray(trace.customdata)
            self.assertEqual(list(cd[0, 1]), [1, 0])  # true=0, pred=1
            self.assertEqual(list(cd[1, 2]), [2, 1])  # true=1, pred=2

    def test_counts_shown_as_text(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual(_trace(fig, "correct").text[0][0], "5")  # diagonal
        self.assertEqual(_trace(fig, "errors").text[0][1], "2")  # off-diagonal
        # Each cell's number is written by exactly one trace, never both.
        self.assertEqual(_trace(fig, "errors").text[0][0], "")
        self.assertEqual(_trace(fig, "correct").text[0][1], "")

    def test_normalize_true_is_row_recall(self):
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        diag, errors = _trace(fig, "correct"), _trace(fig, "errors")
        self.assertAlmostEqual(np.asarray(diag.z)[0, 0], 5 / 7)
        np.testing.assert_allclose(np.asarray(errors.z)[0, 1:], [2 / 7, 0.0])

    def test_normalize_true_handles_empty_row_without_nan(self):
        # Row 2 has no samples: its cells must be 0, not NaN-from-division. Only the diagonal
        # masking may introduce NaNs, so check each trace on the half it owns.
        fig = plot_confusion_matrix(self.cm, self.labels, normalize="true")
        np.testing.assert_array_equal(np.asarray(_trace(fig, "errors").z)[2, :2], [0.0, 0.0])
        self.assertEqual(np.asarray(_trace(fig, "correct").z)[2, 2], 0.0)

    def test_custom_title(self):
        fig = plot_confusion_matrix(self.cm, self.labels, title="Errors")
        self.assertIn("Errors", fig.layout.title.text)


class TestOffDiagonalEmphasis(unittest.TestCase):
    """The errors get their own colour range so the diagonal cannot flatten them out."""

    def setUp(self):
        # A realistic shape: the diagonal dwarfs every error cell.
        self.cm = np.array([[500, 3, 0], [12, 480, 1], [0, 2, 450]])
        self.labels = ["A", "B", "C"]

    def test_diagonal_and_errors_are_separate_traces(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertEqual([t.name for t in fig.data], ["correct", "errors"])

    def test_diagonal_is_masked_out_of_the_error_trace(self):
        z = np.asarray(_trace(plot_confusion_matrix(self.cm, self.labels), "errors").z)
        self.assertTrue(np.isnan(np.diagonal(z)).all())
        np.testing.assert_array_equal(z[0, 1:], [3.0, 0.0])

    def test_error_scale_tops_out_at_the_worst_confusion_pair(self):
        # zmax = 12 (the B→A cell), not 500 — that is what keeps small errors visible.
        errors = _trace(plot_confusion_matrix(self.cm, self.labels), "errors")
        self.assertEqual(errors.zmin, 0)
        self.assertEqual(errors.zmax, 12.0)

    def test_only_the_error_trace_carries_the_colorbar(self):
        fig = plot_confusion_matrix(self.cm, self.labels)
        self.assertFalse(_trace(fig, "correct").showscale)
        self.assertIn("errors", _trace(fig, "errors").colorbar.title.text)

    def test_error_colorscale_defaults_to_the_palette_yellow_orange(self):
        errors = _trace(plot_confusion_matrix(self.cm, self.labels), "errors")
        stops = [color for _, color in errors.colorscale]
        self.assertEqual(stops[0], "rgb(255, 255, 220)")  # palest yellow
        self.assertEqual(stops[-1], "rgb(255, 77, 7)")  # deepest orange-red

    def test_error_colorscale_eases_the_pale_end(self):
        # Positions must be monotonic, span [0, 1], and be squeezed below the even spacing so a
        # small error cell already lands past the near-white stops.
        errors = _trace(plot_confusion_matrix(self.cm, self.labels), "errors")
        positions = [pos for pos, _ in errors.colorscale]
        self.assertEqual((positions[0], positions[-1]), (0.0, 1.0))
        self.assertTrue(all(b > a for a, b in zip(positions, positions[1:])))
        midpoint = positions[len(positions) // 2]
        self.assertLess(midpoint, 0.5)

    def test_custom_colorscale_is_honoured(self):
        fig = plot_confusion_matrix(self.cm, self.labels, colorscale=["rgb(0,0,0)", "rgb(255,255,255)"])
        self.assertEqual([color for _, color in _trace(fig, "errors").colorscale], ["rgb(0,0,0)", "rgb(255,255,255)"])

    def test_named_plotly_colorscale_is_passed_through(self):
        fig = plot_confusion_matrix(self.cm, self.labels, colorscale="Blues")
        self.assertTrue(len(_trace(fig, "errors").colorscale) > 1)

    def test_matrix_without_errors_keeps_a_valid_scale(self):
        perfect = np.diag([10, 20, 30])
        errors = _trace(plot_confusion_matrix(perfect, self.labels), "errors")
        self.assertGreater(errors.zmax, errors.zmin)

    def test_single_class_matrix_falls_back_to_one_trace(self):
        fig = plot_confusion_matrix(np.array([[7]]), ["A"])
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(np.asarray(fig.data[0].z)[0, 0], 7.0)

    def test_emphasis_can_be_switched_off_for_a_classic_matrix(self):
        fig = plot_confusion_matrix(self.cm, self.labels, emphasize_off_diagonal=False)
        self.assertEqual(len(fig.data), 1)
        z = np.asarray(fig.data[0].z)
        self.assertFalse(np.isnan(z).any())
        self.assertEqual(fig.data[0].text[0][0], "500")
        self.assertIn("count", fig.data[0].colorbar.title.text)


class TestEmphasisColorscale(unittest.TestCase):
    def test_single_colour_scale_is_expanded_to_a_valid_two_stop_scale(self):
        # Plotly rejects a one-stop colorscale, so a degenerate palette must still come back usable.
        self.assertEqual(_emphasis_colorscale(["rgb(1,2,3)"]), [[0.0, "rgb(1,2,3)"], [1.0, "rgb(1,2,3)"]])
