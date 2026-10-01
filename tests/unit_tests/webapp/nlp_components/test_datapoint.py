"""Unit tests for shapash.webapp.nlp_components.datapoint: unpack_datapoint."""

import unittest

import numpy as np

from shapash.webapp.nlp_components.datapoint import pack_datapoint, unpack_datapoint


class TestUnpackDatapoint(unittest.TestCase):
    """``unpack_datapoint`` slices a ``pack_datapoint`` payload down to one class for rendering."""

    def test_multiclass_slices_the_requested_column(self):
        dp = pack_datapoint(
            text="i am happy",
            orig_idx=2,
            tokens=["i", "am", "happy"],
            values=np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]]),
            base_values=np.array([0.5, -0.5]),
            label="joy",
        )
        tokens, vals, base_value, label = unpack_datapoint(dp, label_idx=1)
        self.assertEqual(tokens, ["i", "am", "happy"])
        np.testing.assert_allclose(vals, [-0.1, -0.2, -0.3])
        self.assertEqual(base_value, -0.5)
        self.assertEqual(label, "joy")

    def test_binary_1d_values_ignore_label_idx(self):
        dp = pack_datapoint(
            text="not great",
            orig_idx=None,
            tokens=["not", "great"],
            values=np.array([0.2, -0.4]),
            base_values=None,
        )
        tokens, vals, base_value, label = unpack_datapoint(dp, label_idx=1)
        np.testing.assert_allclose(vals, [0.2, -0.4])
        self.assertIsNone(base_value)
        self.assertIsNone(label)

    def test_single_scalar_base_value_is_shared_across_classes(self):
        dp = pack_datapoint(
            text="ok",
            orig_idx=0,
            tokens=["ok"],
            values=np.array([[0.1, 0.2, 0.3]]),
            base_values=np.array([0.7]),  # one shared scalar, not one per class
        )
        # label_idx=2 slices fine into values (3 classes) but has no matching entry in base_values.
        _, vals, base_value, _ = unpack_datapoint(dp, label_idx=2)
        np.testing.assert_allclose(vals, [0.3])
        self.assertEqual(base_value, 0.7)
