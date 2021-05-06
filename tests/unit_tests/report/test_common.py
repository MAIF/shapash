import unittest
import numpy as np
import pandas as pd

from shapash.report.common import VarType, series_dtype, numeric_is_continuous, get_callable, \
    compute_top_correlations_features, display_value, replace_dict_values


class TestCommon(unittest.TestCase):

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
        s = pd.Series(list(range(50)))

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_series_dtype_4(self):
        """
        Test float and continuous series
        """
        s = pd.Series(np.linspace(0, 3, 50))

        assert series_dtype(s) == VarType.TYPE_NUM

    def test_series_dtype_int_5(self):
        """
        Test int and categorical series
        """
        s = pd.Series([1, 1, 1, 2, 2, 2])

        assert series_dtype(s) == VarType.TYPE_CAT

    def test_series_dtype_int_6(self):
        """
        Test float and categorical series
        """
        s = pd.Series([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])

        assert series_dtype(s) == VarType.TYPE_CAT

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

    def test_compute_top_correlations_features_1(self):
        """
        Test function with small number of features
        """
        df = pd.DataFrame(np.random.rand(10, 2))

        corr = df.corr()

        list_features = compute_top_correlations_features(corr=corr, max_features=20)
        assert len(list_features) == 2

    def test_compute_top_correlations_features_2(self):
        """
        Test function with high number of features
        """
        df = pd.DataFrame(np.random.rand(10, 30))

        corr = df.corr()

        list_features = compute_top_correlations_features(corr=corr, max_features=5)

        assert len(list_features) == 5

    def test_display_value_1(self):
        value = 123456.789
        expected_str = '123,456.789'
        assert display_value(value, ',', '.') == expected_str

    def test_display_value_2(self):
        value = 123456.789
        expected_str = '123 456,789'
        assert display_value(value, ' ', ',') == expected_str

    def test_display_value_3(self):
        value = 123456.789
        expected_str = '123.456,789'
        assert display_value(value, '.', ',') == expected_str

    def test_replace_dict_values_1(self):
        d = {'a': 1234, 'b': 0.1234, 'c': {'d': 123.456, 'e': 1234, 'f': {'g': 1234}}}
        expected_d = {'a': '1,234', 'b': '0.1234', 'c': {'d': '123.456', 'e': '1,234', 'f': {'g': '1,234'}}}
        res_d = replace_dict_values(d, display_value, ',', '.')
        assert d == expected_d
        assert res_d == expected_d  # The replace_dict_values function modify inplace but also returns the result dict
