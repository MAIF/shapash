import unittest
import numpy as np
import pandas as pd

from shapash.report.common import VarType, series_dtype, numeric_is_continuous, get_callable


class TestGeneration(unittest.TestCase):

    def test_series_dtype_1(self):
        """
        Test string series
        """
        s = pd.Series(["a", "b", "c", "d", "e", np.nan])

        assert series_dtype(s) == VarType.TYPE_CAT

    def test_series_dtype_2(self):
        """
        Test bool series
        """
        s = pd.Series([True, True, False, False, False])

        assert series_dtype(s) == VarType.TYPE_CAT

    def test_series_dtype_3(self):
        """
        Test int and continuous series
        """
        s = pd.Series([10, 20, 30, 40, 50, 0])

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_series_dtype_4(self):
        """
        Test float and continuous series
        """
        s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_series_dtype_int_5(self):
        """
        Test int and categorical series
        """
        s = pd.Series([1, 1, 1, 2, 2, 2])

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_series_dtype_int_6(self):
        """
        Test float and categorical series
        """
        s = pd.Series([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_numeric_is_continuous_1(self):
        """
        Test int and continuous series
        """
        s = pd.Series(list(range(50)))

        assert numeric_is_continuous(s) is True

    def test_numeric_is_continuous_2(self):
        """
        Test float and continuous series
        """
        s = pd.Series(np.linspace(0, 1, 100))

        assert numeric_is_continuous(s) is True

    def test_numeric_is_continuous_3(self):
        """
        Test int and categorical series
        """
        s = pd.Series([1, 1, 1, 2, 2, 2])

        assert numeric_is_continuous(s) is False

    def test_numeric_is_continuous_4(self):
        """
        Test float and categorical series
        """
        s = pd.Series([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])

        assert numeric_is_continuous(s) is False

    def test_get_callable(self):
        fn = get_callable('sklearn.metrics.accuracy_score')

        from sklearn.metrics import accuracy_score
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 0]

        assert accuracy_score(y_true, y_pred) == fn(y_true, y_pred)
