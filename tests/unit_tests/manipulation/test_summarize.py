"""
Unit test of summarize
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from shapash.manipulation.summarize import (
    contribution_weighted_corr,
    contribution_weighted_corr_matrix,
    compute_corr,
    compute_features_import,
    group_contributions,
    summarize_el,
)


class TestSummarize(unittest.TestCase):
    """
    Unit test for summarize
    TODO: Docstring
    """

    def test_summarize_el_1(self):
        """
        Test summarize el 1
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame([[True, False, False], [False, True, False], [False, True, True]], columns=column_name)
        output = summarize_el(xmatr, masktest, "feat")
        expected = pd.DataFrame(
            [[0.1, np.nan], [0.002, np.nan], [-0.008, 0.4]], columns=["feat1", "feat2"], dtype=object
        )
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_2(self):
        """
        Test summarize el 2
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame([[True, False, False], [False, True, False], [False, False, True]], columns=column_name)
        output = summarize_el(xmatr, masktest, "feat")
        expected = pd.DataFrame([[0.1], [0.002], [0.4]], columns=["feat1"], dtype=object)
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_3(self):
        """
        Test summarize el 3
        """
        column_name = ["col1", "col2", "col3"]
        xmatr = pd.DataFrame(
            [["dfkj", "nfk", "bla"], ["Buble", "blue", "cool"], ["angry", "peace", "deep"]], columns=column_name
        )
        masktest = pd.DataFrame([[True, False, False], [False, True, False], [False, True, True]], columns=column_name)
        output = summarize_el(xmatr, masktest, "temp")
        expected = pd.DataFrame(
            [["dfkj", np.nan], ["blue", np.nan], ["peace", "deep"]], columns=["temp1", "temp2"], dtype=object
        )
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_4(self):
        """
        Test summarize el 4
        """
        column_name = ["col1", "col2", "col3"]
        index_list = ["A", "B", "C"]
        xmatr = pd.DataFrame([[0.1, 0.43, -0.02], [-0.78, 0.002, -0.3], [0.62, -0.008, 0.4]], columns=column_name)
        masktest = pd.DataFrame([[True, False, False], [False, True, False], [False, True, True]], columns=column_name)
        xmatr.index = index_list
        masktest.index = index_list
        output = summarize_el(xmatr, masktest, "temp")
        expected = pd.DataFrame(
            [[0.1, np.nan], [0.002, np.nan], [-0.008, 0.4]], columns=["temp1", "temp2"], dtype=object
        )
        expected.index = index_list
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_compute_features_import_1(self):
        """
        Test compute features import 1
        """
        column_name = ["col1", "col2", "col3"]
        index_list = ["A", "B", "C"]
        xmatr = pd.DataFrame(
            [[0.1, 0.4, -0.02], [-0.1, 0.2, -0.03], [0.2, -0.8, 0.4]], columns=column_name, index=index_list
        )
        output = compute_features_import(xmatr)
        expected = pd.Series([0.4, 1.4, 0.45], index=column_name)
        expected = expected / expected.sum()
        expected = expected.sort_values(ascending=True)
        assert output.equals(expected)

    def test_group_contributions_1(self):
        """
        Test compute contributions groups
        """
        column_name = ["col1", "col2", "col3"]
        index_list = ["A", "B", "C"]
        xmatr = pd.DataFrame(
            [[0.1, 0.4, -0.02], [-0.1, 0.2, -0.03], [0.2, -0.8, 0.4]], columns=column_name, index=index_list
        )

        features_groups = {"group1": ["col1", "col2"]}
        output = group_contributions(xmatr, features_groups)
        expected = pd.DataFrame(
            [[-0.02, 0.5], [-0.03, 0.1], [0.40, -0.6]], columns=["col3", "group1"], index=index_list
        )
        assert_frame_equal(output, expected)

    def test_group_contributions_2(self):
        """
        Test compute contributions groups
        """
        column_name = ["col1", "col2", "col3"]
        index_list = ["A", "B", "C"]
        xmatr = pd.DataFrame(
            [[0.1, 0.4, -0.02], [-0.1, 0.2, -0.03], [0.2, -0.8, 0.4]], columns=column_name, index=index_list
        )

        features_groups = {"group1": ["col1", "col2"], "group2": ["col3"]}
        output = group_contributions(xmatr, features_groups)
        expected = pd.DataFrame(
            [[0.5, -0.02], [0.1, -0.03], [-0.6, 0.4]], columns=["group1", "group2"], index=index_list
        )
        assert_frame_equal(output, expected)


class TestComputeCorr(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [2.0, 4.0, 6.0, 8.0, 10.0],
                "c": [5.0, 3.0, 1.0, 3.0, 5.0],
            }
        )

    def test_compute_corr_pearson(self):
        result = compute_corr(self.df, "pearson")
        assert_frame_equal(result, self.df.corr())

    def test_compute_corr_phik(self):
        result = compute_corr(self.df, "phik")
        assert result.shape == (len(self.df.columns), len(self.df.columns))
        assert list(result.columns) == list(self.df.columns)
        assert list(result.index) == list(self.df.columns)
        assert (result.values >= 0).all() and (result.values <= 1).all()

    def test_compute_corr_phik_fallback_to_pearson_when_not_installed(self):
        with patch("shapash.manipulation.summarize.import_optional_module", return_value=None):
            result = compute_corr(self.df, "phik")
        assert_frame_equal(result, self.df.corr())

    def test_compute_corr_unknown_method_raises(self):
        with self.assertRaises(NotImplementedError):
            compute_corr(self.df, "spearman")


class TestContributionWeightedCorr(unittest.TestCase):
    def test_contribution_weighted_corr_perfect_positive_corr(self):
        result = contribution_weighted_corr(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0, 6.0]))
        assert np.isclose(result, 1.0)

    def test_contribution_weighted_corr_treats_nan_as_zero(self):
        result = contribution_weighted_corr(pd.Series([1.0, np.nan, -1.0]), pd.Series([2.0, 0.0, -2.0]))
        assert np.isclose(result, 1.0)

    def test_contribution_weighted_corr_returns_zero_without_variation(self):
        result = contribution_weighted_corr(pd.Series([0.0, 0.0, 0.0]), pd.Series([1.0, -1.0, 2.0]))
        assert result == 0.0

    def test_contribution_weighted_corr_raises_on_invalid_input(self):
        with self.assertRaises(ValueError):
            contribution_weighted_corr([[1.0], [2.0]], [1.0, 2.0])

        with self.assertRaises(ValueError):
            contribution_weighted_corr([1.0, 2.0], [1.0])

        with self.assertRaises(ValueError):
            contribution_weighted_corr([1.0, np.inf], [1.0, 2.0])


class TestContributionWeightedCorrMatrix(unittest.TestCase):
    def test_contribution_weighted_corr_matrix_is_symmetric(self):
        contrib_values = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [2.0, 4.0, 6.0],
                "feature_3": [-1.0, 0.0, 1.0],
            }
        )

        result = contribution_weighted_corr_matrix(contrib_values)

        assert result.shape == (3, 3)
        assert list(result.index) == list(contrib_values.columns)
        assert list(result.columns) == list(contrib_values.columns)
        assert np.allclose(np.diag(result), 1.0)
        assert np.allclose(result, result.T)
        assert np.isclose(result.loc["feature_1", "feature_2"], 1.0)
        assert np.isclose(
            result.loc["feature_1", "feature_3"],
            contribution_weighted_corr(contrib_values["feature_1"], contrib_values["feature_3"]),
        )

    def test_contribution_weighted_corr_matrix_raises_on_invalid_input(self):
        with self.assertRaises(ValueError):
            contribution_weighted_corr_matrix([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            contribution_weighted_corr_matrix(np.empty((0, 2)))

        with self.assertRaises(ValueError):
            contribution_weighted_corr_matrix(np.array([[1.0, np.inf]]))
