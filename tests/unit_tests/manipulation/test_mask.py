"""
Unit test of mask
"""
import unittest

import pandas as pd

from shapash.manipulation.mask import compute_masked_contributions, init_mask


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
