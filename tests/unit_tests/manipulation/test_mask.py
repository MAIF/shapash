"""
Unit test of mask
"""
import unittest

import pandas as pd

from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState
from shapash.manipulation.mask import compute_mask, compute_masked_contributions, init_mask


class TestMask(unittest.TestCase):
    """
    Class of Unit test for Mask
    TODO: Docstring
    """

    def test_compute_masked_contributions_1(self):
        """
        test of compute masked contributions 1
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame(
            [[True, False, False], [False, True, False], [False, False, False]], columns=column_name
        )
        output = compute_masked_contributions(xmatr, masktest)
        expected = pd.DataFrame([[-0.02, 0.43], [-1.08, 0.0], [-0.008, 1.02]], columns=["masked_neg", "masked_pos"])
        assert (xmatr.shape[0], 2) == output.shape
        assert output.equals(expected)

    def test_compute_masked_contributions_2(self):
        """
        test of compute masked contributions 2
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame([[True, False, False], [True, True, True], [False, False, False]], columns=column_name)
        output = compute_masked_contributions(xmatr, masktest)
        expected = pd.DataFrame([[-0.02, 0.43], [0.0, 0.0], [-0.008, 1.02]], columns=["masked_neg", "masked_pos"])
        assert (xmatr.shape[0], 2) == output.shape
        assert output.equals(expected)

    def test_compute_masked_contributions_3(self):
        """
        test of compute masked contributions 3
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame([[True, True, True], [True, True, True], [True, True, True]], columns=column_name)
        output = compute_masked_contributions(xmatr, masktest)
        expected = pd.DataFrame([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], columns=["masked_neg", "masked_pos"])
        assert (xmatr.shape[0], 2) == output.shape
        assert output.equals(expected)

    def test_init_mask(self):
        """
        test of initialization of mask
        """
        column_name = ["col1", "col2"]
        s_ord = pd.DataFrame([[0.1, 0.43], [-0.78, 0.002], [0.62, -0.008]], columns=column_name)
        expected = pd.DataFrame([[True, True], [True, True], [True, True]], columns=column_name)
        output = init_mask(s_ord)
        assert output.equals(expected)

    def test_compute_mask_is_pure(self):
        """
        compute_mask must not read or write any attribute on its `state` argument: it only
        derives its result from `data` and the filtering parameters, so it can be called
        repeatedly (e.g. once per plot) without accumulating state anywhere.
        """
        contrib_sorted = pd.DataFrame(
            data=[[0.5, 0.4, 0.3], [0.9, 0.8, 0.7]], columns=["contrib_1", "contrib_2", "contrib_3"]
        )
        data = {"var_dict": pd.DataFrame(), "contrib_sorted": contrib_sorted}
        state = SmartState()

        mask, masked_contributions, mask_params = compute_mask(state, data, threshold=0.5, max_contrib=2)

        assert vars(state) == {}
        expected_mask = pd.DataFrame(
            data=[[True, False, False], [True, True, False]], columns=["contrib_1", "contrib_2", "contrib_3"]
        )
        pd.testing.assert_frame_equal(expected_mask, mask)
        assert mask_params == {"features_to_hide": None, "threshold": 0.5, "positive": None, "max_contrib": 2}

        # calling it again with the same inputs gives the exact same result
        mask_2, masked_contributions_2, mask_params_2 = compute_mask(state, data, threshold=0.5, max_contrib=2)
        pd.testing.assert_frame_equal(mask, mask_2)
        pd.testing.assert_frame_equal(masked_contributions, masked_contributions_2)
        assert mask_params == mask_params_2

    def test_compute_mask_multiclass(self):
        """
        compute_mask must transparently support the multi-class case (a list of contribution
        matrices), matching what SmartExplainer.filter() computes via the same state/data.
        """
        contributions = [
            pd.DataFrame(data=[[0.5, 0.4, 0.3], [0.9, 0.8, 0.7]], columns=["Col1", "Col2", "Col3"]),
            pd.DataFrame(data=[[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]], columns=["Col1", "Col2", "Col3"]),
        ]
        data = {"var_dict": 1, "contrib_sorted": contributions}
        state = MultiDecorator(SmartState())

        mask, masked_contributions, mask_params = compute_mask(state, data, threshold=0.5, max_contrib=2)

        expected_mask = [
            pd.DataFrame(data=[[True, False, False], [True, True, False]], columns=["contrib_1", "contrib_2", "contrib_3"]),
            pd.DataFrame(data=[[False, False, False], [True, True, False]], columns=["contrib_1", "contrib_2", "contrib_3"]),
        ]
        assert len(expected_mask) == len(mask)
        for expected, actual in zip(expected_mask, mask):
            pd.testing.assert_frame_equal(expected, actual)
        assert mask_params == {"features_to_hide": None, "threshold": 0.5, "positive": None, "max_contrib": 2}
