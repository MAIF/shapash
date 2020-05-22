"""
Unit test of summarize
"""
import unittest
import pandas as pd
import numpy as np
from shapash.manipulation.summarize import summarize_el, compute_features_import


class TestSummarize(unittest.TestCase):
    """
    Unit test for summarize
    TODO: Docstring
    """
    def test_summarize_el_1(self):
        """
        Test summarize el 1
        """
        column_name = ['col1', 'col2', 'col3']
        xmatr = pd.DataFrame(
            [[0.1, 0.43, -0.02],
             [-0.78, 0.002, -0.3],
             [0.62, -0.008, 0.4]],
            columns=column_name
        )
        masktest = pd.DataFrame(
            [[True, False, False],
             [False, True, False],
             [False, True, True]],
            columns=column_name
        )
        output = summarize_el(xmatr, masktest, "feat")
        expected = pd.DataFrame(
            [[0.1, np.nan], [0.002, np.nan], [-0.008, 0.4]],
            columns=["feat1", "feat2"],
            dtype=object
        )
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_2(self):
        """
        Test summarize el 2
        """
        column_name = ['col1', 'col2', 'col3']
        xmatr = pd.DataFrame(
            [[0.1, 0.43, -0.02],
             [-0.78, 0.002, -0.3],
             [0.62, -0.008, 0.4]],
            columns=column_name
        )
        masktest = pd.DataFrame(
            [[True, False, False],
             [False, True, False],
             [False, False, True]],
            columns=column_name
        )
        output = summarize_el(xmatr, masktest, "feat")
        expected = pd.DataFrame(
            [[0.1], [0.002], [0.4]],
            columns=["feat1"],
            dtype=object
        )
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_3(self):
        """
        Test summarize el 3
        """
        column_name = ['col1', 'col2', 'col3']
        xmatr = pd.DataFrame(
            [["dfkj", "nfk", "bla"],
             ["Buble", "blue", "cool"],
             ["angry", "peace", "deep"]],
            columns=column_name
        )
        masktest = pd.DataFrame(
            [[True, False, False],
             [False, True, False],
             [False, True, True]],
            columns=column_name
        )
        output = summarize_el(xmatr, masktest, "temp")
        expected = pd.DataFrame(
            [['dfkj', np.nan], ['blue', np.nan], ['peace', 'deep']],
            columns=["temp1", "temp2"],
            dtype=object
        )
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_summarize_el_4(self):
        """
        Test summarize el 4
        """
        column_name = ['col1', 'col2', 'col3']
        index_list = ['A', 'B', 'C']
        xmatr = pd.DataFrame(
            [[0.1, 0.43, -0.02],
             [-0.78, 0.002, -0.3],
             [0.62, -0.008, 0.4]],
            columns=column_name
        )
        masktest = pd.DataFrame(
            [[True, False, False],
             [False, True, False],
             [False, True, True]],
            columns=column_name
        )
        xmatr.index = index_list
        masktest.index = index_list
        output = summarize_el(xmatr, masktest, "temp")
        expected = pd.DataFrame(
            [[0.1, np.nan], [0.002, np.nan], [-0.008, 0.4]],
            columns=["temp1", "temp2"],
            dtype=object
        )
        expected.index = index_list
        assert xmatr.shape[0] == output.shape[0]
        assert output.equals(expected)

    def test_compute_features_import_1(self):
        """
        Test compute features import 1
        """
        column_name = ['col1', 'col2', 'col3']
        index_list = ['A', 'B', 'C']
        xmatr = pd.DataFrame(
            [[0.1, 0.4, -0.02],
             [-0.1, 0.2, -0.03],
             [0.2, -0.8, 0.4]],
            columns=column_name,
            index=index_list
        )
        output = compute_features_import(xmatr)
        expected = pd.Series(
            [0.4, 1.4, 0.45],
            index=column_name
        )
        expected = expected / expected.sum()
        expected = expected.sort_values(ascending=True)
        assert output.equals(expected)
