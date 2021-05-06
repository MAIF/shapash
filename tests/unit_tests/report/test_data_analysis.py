import unittest
from unittest import TestCase
import pandas as pd
import numpy as np

from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.common import compute_col_types


class TestDataAnalysis(unittest.TestCase):

    def test_perform_global_dataframe_analysis_1(self):
        df = pd.DataFrame({
            "string_data": ["a", "b", "c", "d", "e", np.nan],
            "bool_data": [True, True, False, False, False, np.nan],
            "int_data": [10, 20, 30, 40, 50, 0],
        })

        d = perform_global_dataframe_analysis(df)
        expected_d = {
            'number of features': '3',
            'number of observations': '6',
            'missing values': '2',
            '% missing values': '0.111',
        }
        TestCase().assertDictEqual(d, expected_d)

    def test_perform_global_dataframe_analysis_2(self):
        df = None
        d = perform_global_dataframe_analysis(df)
        assert d == {}

    def test_perform_univariate_dataframe_analysis_1(self):
        df = pd.DataFrame({
            "string_data": ["a", "b", "c", "d", "e", np.nan]*10,
            "bool_data": [True, True, False, False, False, np.nan]*10,
            "int_continuous_data": list(range(60)),
            "float_continuous_data": np.linspace(0, 2, 60),
            "int_cat_data": [1, 1, 1, 2, 2, 2]*10,
            "float_cat_data": [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]*10
        })
        d = perform_univariate_dataframe_analysis(df, col_types=compute_col_types(df))
        expected_d = {
            'int_continuous_data': {
                'count': '60',
                'mean': '29.5',
                'std': '17.5',
                'min': '0',
                '25%': '14.8',
                '50%': '29.5',
                '75%': '44.2',
                'max': '59'
            },
            'float_continuous_data': {
                'count': '60',
                'mean': '1',
                'std': '0.592',
                'min': '0',
                '25%': '0.5',
                '50%': '1',
                '75%': '1.5',
                'max': '2'
            },
            'int_cat_data': {
                'distinct values': '2',
                'missing values': '0'
            },
            'float_cat_data': {
                'distinct values': '2',
                'missing values': '0'
            },
            'string_data': {
                'distinct values': '5',
                'missing values': '10'
            },
            'bool_data': {
                'distinct values': '2',
                'missing values': '10'
            }
        }
        TestCase().assertDictEqual(d, expected_d)

    def test_perform_univariate_dataframe_analysis_2(self):
        df = None
        d = perform_univariate_dataframe_analysis(df, col_types=compute_col_types(df))
        assert d == {}
