"""
Unit test for multi decorator
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from shapash.explainer.multi_decorator import MultiDecorator


class DummyState:
    """
    Dummy State Class
    TODO: Docstring
    """

    def __init__(self):
        self.dummy_member = True

    def dummy_function(self, xparam, yparam):
        """
        Dummy function
        TODO: Docstring
        Parameters
        ----------
        xparam : [type]
            [description]
        yparam : [type]
            [description]
        Returns
        -------
        [type]
            [description]
        """
        return "+".join([str(xparam), str(yparam)])


class TestMultiDecorator(unittest.TestCase):
    """
    Unit test multi decorator Class
    TODO: Docstring
    """

    def test_delegate_1(self):
        """
        Unit test delegate 1
        """
        state = MultiDecorator(DummyState())
        output = state.delegate("dummy_function", [1, 2, 3], 9)
        expected = ["1+9", "2+9", "3+9"]
        self.assertListEqual(output, expected)

    def test_delegate_2(self):
        """
        Unit test delegate 2
        """
        state = MultiDecorator(DummyState())
        with self.assertRaises(ValueError):
            state.delegate("dummy_function")

    def test_delegate_3(self):
        """
        Unit test delegate 3
        """
        state = MultiDecorator(DummyState())
        with self.assertRaises(AttributeError):
            state.delegate("non_existing_function", [1, 2, 3], 9)

    def test_delegate_4(self):
        """
        Unit test delegate 4
        """
        state = MultiDecorator(DummyState())
        with self.assertRaises(ValueError):
            state.delegate("dummy_member", [1, 2, 3], 9)

    def test_delegate_5(self):
        """
        Unit test delegate 5
        """
        state = MultiDecorator(DummyState())
        with self.assertRaises(ValueError):
            state.delegate("dummy_function", 1, 9)

    def test_assign_contributions(self):
        """
        Unit test assign contributions
        """
        backend = Mock()
        backend.assign_contributions = Mock(
            side_effect=[
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
                {"a": 5, "b": 6},
            ]
        )
        state = MultiDecorator(backend)
        output = state.assign_contributions([1, 2, 3])
        expected = {"a": [1, 3, 5], "b": [2, 4, 6]}
        self.assertDictEqual(output, expected)

    def test_check_contributions_1(self):
        """
        Unit test check contributions 1
        """
        backend = Mock()
        backend.check_contributions = Mock(side_effect=[True, True, True])
        state = MultiDecorator(backend)
        output = state.check_contributions([1, 2, 3], Mock())
        assert output

    def test_check_contributions_2(self):
        """
        Unit test check contributions 2
        """
        backend = Mock()
        backend.check_contributions = Mock(side_effect=[True, False, True])
        state = MultiDecorator(backend)
        output = state.check_contributions([1, 2, 3], Mock())
        assert not output

    @patch("shapash.explainer.multi_decorator.MultiDecorator.delegate")
    def test_combine_masks(self, mock_delegate):
        """
        Unit test combine masks
        TODO: mock_delegate description
        Parameters
        ----------
        mock_delegate : [type]
            [description]
        """
        backend = Mock()
        df1 = np.array([[1, 2, 3], [4, 5, 6]])
        df2 = np.array([[-1, -2, -3], [-4, -5, -6]])
        df3 = np.array([[10, 20, 30], [40, 50, 60]])
        df4 = np.array([[-10, -20, -30], [-40, -50, -60]])
        df5 = np.array([[100, 200, 300], [400, 500, 600]])
        df6 = np.array([[-100, -200, -300], [-400, -500, -600]])
        masks = [[df1, df2], [df3, df4], [df5, df6]]
        transposed_masks = [[df1, df3, df5], [df2, df4, df6]]
        backend.combine_masks = Mock()
        state = MultiDecorator(backend)
        state.combine_masks(masks)
        mock_delegate.assert_called_with("combine_masks", transposed_masks)

    def test_compute_masked_contributions(self):
        """
        Unit test compute masked contributions
        """
        backend = Mock()
        state = MultiDecorator(backend)
        state.compute_masked_contributions([1, 2, 3], [1, 2, 3])
        assert backend.compute_masked_contributions.call_count == 3

    def test_init_mask(self):
        """
        Unit test init mask
        """
        backend = Mock()
        state = MultiDecorator(backend)
        state.init_mask([1, 2, 3])
        assert backend.init_mask.call_count == 3

    def test_summarize_1(self):
        """
        Unit test summarize 1
        """
        backend = Mock()
        contrib_sorted1 = pd.DataFrame([[-1, 2, -3, 4], [-5, 6, -7, 8]])
        contrib_sorted2 = pd.DataFrame([[1, -2, 3, -4], [5, -6, 7, -8]])
        contrib_sorted = [contrib_sorted1, contrib_sorted2]
        var_dict1 = pd.DataFrame([[1, 0, 2, 3], [1, 0, 3, 2]])
        var_dict2 = pd.DataFrame([[1, 0, 2, 3], [1, 0, 3, 2]])
        var_dict = [var_dict1, var_dict2]
        x_sorted1 = pd.DataFrame([[1.0, 3.0, 22.0, 1.0], [2.0, 1.0, 2.0, 38.0]])
        x_sorted2 = pd.DataFrame([[1.0, 3.0, 22.0, 1.0], [2.0, 1.0, 2.0, 38.0]])
        x_sorted = [x_sorted1, x_sorted2]
        mask = pd.DataFrame([[True, True, False, False], [True, True, False, False]])
        columns_dict = {0: "Pclass", 1: "Sex", 2: "Age", 3: "Embarked"}
        features_dict = {"Pclass": "Pclass", "Sex": "Sex", "Age": "Age (years)", "Embarked": "emb"}
        state = MultiDecorator(backend)
        state.summarize(contrib_sorted, var_dict, x_sorted, mask, columns_dict, features_dict)
        assert backend.summarize.call_count == 2

    def test_compute_features_import(self):
        """
        Unit test compute features import
        """
        backend = Mock()
        state = MultiDecorator(backend)
        contrib1 = pd.DataFrame([[-1, 2, -3, 4], [-5, 6, -7, 8]])
        contrib2 = pd.DataFrame([[1, -2, 3, -4], [5, -6, 7, -8]])
        state.compute_features_import([contrib1, contrib2])
        assert backend.compute_features_import.call_count == 2
