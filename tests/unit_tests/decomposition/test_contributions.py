"""
Unit test of contributions
"""
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import category_encoders as ce
from shapash.decomposition.contributions import inverse_transform_contributions
from shapash.decomposition.contributions import make_mapping_dict, get_origin_column_name
from shapash.decomposition.contributions import compute_contributions, rank_contributions


class TestContributions(unittest.TestCase):
    """
    Unit test class of contributions
    TODO: Docstring
    Parameters
    ----------
    unittest : [type]
        [description]
    """

    def test_compute_contributions_1(self):
        """
        Unit test compute contributions 1
        """
        mock_x = Mock()
        explainer = Mock()
        output = compute_contributions(mock_x, explainer)
        explainer.shap_values.assert_called()
        assert len(output) == 2

    def test_compute_contributions_2(self):
        """
        Unit test compute contributions 2
        """
        mock_x = Mock()
        explainer = Mock()
        preprocessing = Mock()
        output = compute_contributions(mock_x, explainer, preprocessing)
        preprocessing.transform.assert_called()
        explainer.shap_values.assert_called()
        assert len(output) == 2

    def test_compute_contribution_3(self):
        """
        Unit test compute contributions 3
        """
        explainer = Mock()
        explainer.shap_values.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        x_pred = pd.DataFrame({'a': [2, 8], 'b': [3, 9], 'c': [4, 10]})
        expected_output = pd.DataFrame(
            data=np.array([[1, 2, 3], [4, 5, 6]]),
            columns=x_pred.columns,
            index=x_pred.index
        )
        s, b = compute_contributions(x_pred, explainer)
        pd.testing.assert_frame_equal(expected_output, s)

    def test_compute_contribution_4(self):
        """
        Unit test compute contributions 4
        """
        explainer = Mock()
        explainer.shap_values.return_value = [
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1.4, -2, 3], [2.4, 5.5, -6]])
        ]
        x_pred = pd.DataFrame({'a': [2, 8], 'b': [3, 9], 'c': [4, 10]})
        expected_output = [
            pd.DataFrame(
                data=np.array([[1, 2, 3], [4, 5, 6]]),
                columns=x_pred.columns,
                index=x_pred.index
            ),
            pd.DataFrame(
                data=np.array([[1.4, -2, 3], [2.4, 5.5, -6]]),
                columns=x_pred.columns,
                index=x_pred.index
            )
        ]
        s, b = compute_contributions(x_pred, explainer)
        for i in range(2):
            pd.testing.assert_frame_equal(expected_output[i], s[i])

    def test_rank_contributions_1(self):
        """
        Unit test rank contributions 1
        """
        dataframe_s = pd.DataFrame(
            [[3.4, 1, -9, 4],
             [-45, 3, 43, -9]],
            columns=["Phi_" + str(i) for i in range(4)],
            index=['raw_1', 'raw_2']
        )

        dataframe_x = pd.DataFrame(
            [['Male', 'House', 'Married', 'PhD'],
             ['Female', 'Flat', 'Married', 'Master']],
            columns=["X" + str(i) for i in range(4)],
            index=['raw_1', 'raw_2']
        )

        expected_s_ord = pd.DataFrame(
            data=[[-9, 4, 3.4, 1],
                  [-45, 43, -9, 3]],
            columns=['contribution_' + str(i) for i in range(4)],
            index=['raw_1', 'raw_2']
        )

        expected_x_ord = pd.DataFrame(
            data=[['Married', 'PhD', 'Male', 'House'],
                  ['Female', 'Married', 'Master', 'Flat']],
            columns=['feature_' + str(i) for i in range(4)],
            index=['raw_1', 'raw_2']
        )

        expected_s_dict = pd.DataFrame(
            data=[[2, 3, 0, 1],
                  [0, 2, 3, 1]],
            columns=['feature_' + str(i) for i in range(4)],
            index=['raw_1', 'raw_2']
        )

        s_ord, x_ord, s_dict = rank_contributions(dataframe_s, dataframe_x)

        assert np.array_equal(s_ord.values, expected_s_ord.values)
        assert np.array_equal(x_ord.values, expected_x_ord.values)
        assert np.array_equal(s_dict.values, expected_s_dict.values)

        assert list(s_ord.columns) == list(expected_s_ord.columns)
        assert list(x_ord.columns) == list(expected_x_ord.columns)
        assert list(s_dict.columns) == list(expected_s_dict.columns)

        assert pd.Index.equals(s_ord.index, expected_s_ord.index)
        assert pd.Index.equals(x_ord.index, expected_x_ord.index)
        assert pd.Index.equals(s_dict.index, expected_s_dict.index)

    def test_make_mapping_dict_1(self):
        """
        Unit test make mapping dict 1
        """
        mapping = [
            {'col': 'Sector',
             'mapping': pd.DataFrame(
                 data=np.array([[1, 2], [3, 4]]),
                 index=[0, 1],
                 columns=['Sector_0', 'Sector_1']
             )
            },
            {'col': 'Game',
             'mapping': pd.DataFrame(
                 data=np.array([[5, 6, 7], [8, 9, 10]]),
                 index=[0, 1],
                 columns=['Game_0', 'Game_1', 'Game_3'])
             },
        ]
        output = make_mapping_dict(mapping)
        expected = {
            'Sector': ['Sector_0', 'Sector_1'],
            'Game': ['Game_0', 'Game_1', 'Game_3']
        }
        self.assertDictEqual(output, expected)

    def test_get_origin_column_name_1(self):
        """
        Unit test get origin column name 1
        """
        encoded_column_name = 'Game_1'
        mapping_dict = {
            'Sector': ['Sector_0', 'Sector_1'],
            'Game': ['Game_0', 'Game_1', 'Game_3']
        }
        output = get_origin_column_name(encoded_column_name, mapping_dict)
        assert output == 'Game'

    def test_inverse_transform_contributions_1(self):
        """
        Unit test inverse transform contributions 1
        """
        contributions = pd.DataFrame(np.random.rand(10, 15))
        preprocessing = None
        output = inverse_transform_contributions(contributions, preprocessing)
        pd.testing.assert_frame_equal(output, contributions)

    @patch('shapash.decomposition.contributions.make_mapping_dict')
    def test_inverse_transform_contributions_2(self, mock_make_mapping_dict):
        """
        Unit test inverse transform contributions 2
        TODO: Docstring
        Parameters
        ----------
        mock_make_mapping_dict : [type]
            [description]
        """
        mock_make_mapping_dict.return_value = {
            'Sector': ['Sector_0', 'Sector_1'],
            'Game': ['Game_0', 'Game_1', 'Game_2']
        }
        contributions = pd.DataFrame(
            data=[[1, 2, 3], [4, 5, 0]],
            columns=['Sector_0', 'Sector_1', 'Game_0']
        )
        preprocessing = Mock(spec=ce.BaseNEncoder)
        preprocessing.mapping = []
        output = inverse_transform_contributions(contributions, preprocessing)
        expected_output = pd.DataFrame(
            data=[[3, 3], [9, 0]],
            columns=['Sector', 'Game']
        )
        pd.testing.assert_frame_equal(output, expected_output)
