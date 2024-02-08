"""
Unit test for filters
"""
import unittest

import numpy as np
import pandas as pd

from shapash.manipulation.filters import (
    cap_contributions,
    combine_masks,
    cutoff_contributions,
    cutoff_contributions_old,
    hide_contributions,
    sign_contributions,
)


class TestFilter(unittest.TestCase):
    """
    Unit test class for filters
    TODO: Docstring
    """

    def test_hide_contributions_1(self):
        """
        Unit test hide contributions 1
        """
        dataframe = pd.DataFrame([[2, 0, 1], [0, 1, 2], [2, 3, 1]], columns=["col1", "col2", "col3"])
        output = hide_contributions(dataframe, [0, 2])
        expected = pd.DataFrame(
            [[False, False, True], [False, True, False], [False, True, True]], columns=["col1", "col2", "col3"]
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_hide_contributions_2(self):
        """
        Unit test hide contributions 2
        """
        dataframe = pd.DataFrame([[2, 0, 1], [0, 1, 2], [2, 3, 1]], columns=["col1", "col2", "col3"])
        output = hide_contributions(dataframe, [-5])
        expected = pd.DataFrame(
            [[True, True, True], [True, True, True], [True, True, True]], columns=["col1", "col2", "col3"]
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_hide_contributions_3(self):
        """
        Unit test hide contributions 3
        """
        dataframe = pd.DataFrame([[2, 0, 1], [0, 1, 2], [2, 3, 1]], columns=["col1", "col2", "col3"])
        output = hide_contributions(dataframe, [])
        expected = pd.DataFrame(
            [[True, True, True], [True, True, True], [True, True, True]], columns=["col1", "col2", "col3"]
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_cap_contributions(self):
        """
        Unit test cap contributions
        """
        xmatr = pd.DataFrame(
            [[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=["c1", "c2", "c3"]
        )
        thresholdvalue = 0.3
        result = pd.DataFrame(
            [[False, True, False], [True, False, True], [True, False, True]], columns=["c1", "c2", "c3"]
        )
        output = cap_contributions(xmatr, thresholdvalue)
        assert xmatr.shape == output.shape
        assert output.equals(result)

    def test_sign_contributions_1(self):
        """
        Unit test sign contributions 1
        """
        dataframe = pd.DataFrame({"val1": [1, -1], "val2": [-2, -2], "val3": [0.5, -2]})
        output = sign_contributions(dataframe, positive=True)
        expected = pd.DataFrame({"val1": [True, False], "val2": [False, False], "val3": [True, False]})
        pd.testing.assert_frame_equal(output, expected)

    def test_sign_contributions_2(self):
        """
        Unit test sign contributions 2
        """
        dataframe = pd.DataFrame({"val1": [1, -1], "val2": [-2, -2], "val3": [0.5, -2]})
        output = sign_contributions(dataframe, positive=False)
        expected = pd.DataFrame({"val1": [False, True], "val2": [True, True], "val3": [False, True]})
        pd.testing.assert_frame_equal(output, expected)

    def test_cutoff_contributions_old(self):
        """
        Unit test cutoff contributions old
        """
        dataframe = pd.DataFrame(np.tile(np.array([1, 2, 3, 4]), (4, 2)))
        dataframe.columns = [f"col_{col}" for col in dataframe.columns]
        output = cutoff_contributions_old(dataframe, max_contrib=4)
        expected = pd.DataFrame(np.tile(np.array([True, True, True, True, False, False, False, False]), (4, 1)))
        expected.columns = [f"col_{col}" for col in expected.columns]
        assert output.equals(expected)

    def test_cutoff_contributions_0(self):
        """
        Unit test cutoff contributions 0
        """
        dataframe = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        output = cutoff_contributions(dataframe, 0)
        expected = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, False],
                "val3": [False, False, False],
                "val4": [False, False, False],
            }
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_cutoff_contributions_1(self):
        """
        Unit test cutoff contributions 1
        """
        dataframe = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        output = cutoff_contributions(dataframe, 1)
        expected = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, False],
            }
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_cutoff_contributions_2(self):
        """
        Unit test cutoff contributions 2
        """
        dataframe = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        output = cutoff_contributions(dataframe, 2)
        expected = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_cutoff_contributions_3(self):
        """
        Unit test cutoff contributions 3
        """
        dataframe = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        output = cutoff_contributions(dataframe)
        expected = pd.DataFrame(
            {
                "val1": [False, False, False],
                "val2": [False, False, True],
                "val3": [False, True, False],
                "val4": [False, False, True],
            }
        )
        pd.testing.assert_frame_equal(output, expected)

    def test_combine_masks_1(self):
        """
        Unit test combine mask 1
        """
        df1 = pd.DataFrame(
            [[True, False, True], [True, True, True], [False, False, False]], columns=["col1", "col2", "col3"]
        )
        df2 = pd.DataFrame([[True, False, True], [True, False, False]], columns=["col1", "col2", "col3"])
        df3 = pd.DataFrame([[False, False], [True, False], [True, False]], columns=["col1", "col2"])
        self.assertRaises(ValueError, combine_masks, [df1, df2])
        self.assertRaises(ValueError, combine_masks, [df1, df3])

    def test_combine_masks_2(self):
        """
        Unit test combine masks 2
        """
        df1 = pd.DataFrame(
            [[True, False, True], [True, True, True], [False, False, False]], columns=["col1", "col2", "col3"]
        )
        df2 = pd.DataFrame(
            [[False, False, True], [True, False, True], [True, False, False]],
            columns=["contrib_1", "contrib_2", "contrib_3"],
        )
        output = combine_masks([df1, df2])
        expected_output = pd.DataFrame(
            [[False, False, True], [True, False, True], [False, False, False]],
            columns=["contrib_1", "contrib_2", "contrib_3"],
        )
        pd.testing.assert_frame_equal(output, expected_output)
