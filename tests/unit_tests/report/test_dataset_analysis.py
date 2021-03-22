import unittest
from unittest import TestCase
import pandas as pd
import numpy as np

from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis


class TestGeneration(unittest.TestCase):

    def test_perform_global_dataframe_analysis_1(self):
        df = pd.DataFrame({
            "string_data": ["a", "b", "c", "d", "e", np.nan],
            "bool_data": [True, True, False, False, False, np.nan],
            "int_data": [10, 20, 30, 40, 50, 0],
        })

        d = perform_global_dataframe_analysis(df)
        expected_d = {
            'number of features': 3,
            'number of observations': 6,
            'missing values': 2,
            '% missing values': 1 / 9,
        }
        TestCase().assertDictEqual(d, expected_d)

    def test_perform_global_dataframe_analysis_2(self):
        df = None
        d = perform_global_dataframe_analysis(df)
        assert d == {}

    def test_perform_univariate_dataframe_analysis_1(self):
        df = pd.DataFrame({
            "string_data": ["a", "b", "c", "d", "e", np.nan],
            "bool_data": [True, True, False, False, False, np.nan],
            "int_continuous_data": [10, 20, 30, 40, 50, 0],
            "float_continuous_data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "int_cat_data": [1, 1, 1, 2, 2, 2],
            "float_cat_data": [0.2, 0.2, 0.2, 0.6, 0.6, 0.6]
        })
        d = perform_univariate_dataframe_analysis(df)
        expected_d = {
            'int_continuous_data': {
                'count': 6,
                'mean': 25,
                'std': 18.71,
                'min': 0,
                '25%': 12.5,
                '50%': 25.0,
                '75%': 37.5,
                'max': 50
            },
            'float_continuous_data': {
                'count': 6,
                'mean': 0.35,
                'std': 0.19,
                'min': 0.1,
                '25%': 0.23,
                '50%': 0.35,
                '75%': 0.48,
                'max': 0.6
            },
            'int_cat_data': {
                'distinct values': 2,
                'missing values': 0
            },
            'float_cat_data': {
                'distinct values': 2,
                'missing values': 0
            },
            'string_data': {
                'distinct values': 5,
                'missing values': 1
            },
            'bool_data': {
                'distinct values': 2,
                'missing values': 1
            }
        }
        TestCase().assertDictEqual(d, expected_d)

    def test_perform_univariate_dataframe_analysis_2(self):
        df = None
        d = perform_univariate_dataframe_analysis(df)
        assert d == {}
