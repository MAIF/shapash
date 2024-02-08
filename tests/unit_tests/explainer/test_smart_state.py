"""
Unit test smart state
"""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from shapash.explainer.smart_state import SmartState


class TestSmartState(unittest.TestCase):
    """
    Unit test Smart State Class
    TODO: Docstring
    """

    def test_validate_contributions_1(self):
        """
        Unit test validate contributions
        Parameters
        ----------
        unittest : [type]
            [description]
        """
        state = SmartState()
        x_init = Mock()
        contributions = pd.DataFrame([[2, 1], [8, 4]], columns=["Col1", "Col2"], index=["Id1", "Id2"])
        expected_output = contributions
        output = state.validate_contributions(contributions, x_init)
        assert not pd.testing.assert_frame_equal(expected_output, output)

    def test_validate_contributions_2(self):
        """
        Unit test validate contributions 2
        """
        state = SmartState()
        contributions = np.array([[2, 1], [8, 4]])
        x_init = pd.DataFrame([[1, 2], [3, 4]], columns=["Col1", "Col2"], index=["Id1", "Id2"])
        expected_output = pd.DataFrame([[2, 1], [8, 4]], columns=["Col1", "Col2"], index=["Id1", "Id2"])
        output = state.validate_contributions(contributions, x_init)
        assert not pd.testing.assert_frame_equal(expected_output, output)

    @patch("shapash.explainer.smart_state.inverse_transform_contributions")
    def test_inverse_transform_contributions(self, mock_inverse_transform_contributions):
        """
        Unit test inverse transform contributions
        Parameters
        ----------
        mock_inverse_transform_contributions : [type]
            [description]
        """
        state = SmartState()
        state.inverse_transform_contributions(Mock(), Mock())
        mock_inverse_transform_contributions.assert_called()

    def test_check_contributions_1(self):
        """
        Unit test check contributions 1
        """
        state = SmartState()
        contributions = pd.DataFrame(
            [[-0.2, 0.1], [0.8, -0.4], [0.5, -0.7]],
        )
        x_init = pd.DataFrame(
            [[1, 2], [3, 4]],
        )
        assert not state.check_contributions(contributions, x_init)

    def test_check_contributions_2(self):
        """
        Unit test check contributions 2
        """
        state = SmartState()
        contributions = pd.DataFrame([[-0.2, 0.1], [0.8, -0.4]], index=["row_1", "row_2"])
        x_init = pd.DataFrame([[1, 2], [3, 4]])
        assert not state.check_contributions(contributions, x_init)

    def test_check_contributions_3(self):
        """
        Unit test check contributions 3
        """
        state = SmartState()
        contributions = pd.DataFrame(
            [[-0.2, 0.1], [0.8, -0.4]],
            columns=["col_1", "col_2"],
        )
        x_init = pd.DataFrame(
            [[1, 2], [3, 4]],
        )
        assert not state.check_contributions(contributions, x_init)

    def test_check_contributions_4(self):
        """
        Unit test check contributions 4
        """
        state = SmartState()
        contributions = pd.DataFrame([[-0.2, 0.1], [0.8, -0.4]], columns=["col_1", "col_2"], index=["row_1", "row_2"])
        x_init = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"], index=["row_1", "row_2"])
        assert state.check_contributions(contributions, x_init)

    @patch("shapash.explainer.smart_state.rank_contributions")
    def test_rank_contributions(self, mock_rank_contributions):
        """
        Unit test rank contributions
        Parameters
        ----------
        mock_rank_contributions : [type]
            [description]
        """
        state = SmartState()
        state.rank_contributions(Mock(), Mock())
        mock_rank_contributions.assert_called()

    def test_assign_contributions_1(self):
        """
        Unit test assign contributions 1
        """
        state = SmartState()
        output = state.assign_contributions([1, 2, 3])
        expected = {"contrib_sorted": 1, "x_sorted": 2, "var_dict": 3}
        self.assertDictEqual(output, expected)

    def test_assign_contributions_2(self):
        """
        Unit test assign contributions 2
        """
        state = SmartState()
        ranked = [1, 2]
        with self.assertRaises(ValueError):
            state.assign_contributions(ranked)

    @patch("shapash.explainer.smart_state.cap_contributions")
    def test_cap_contributions(self, mock_cap_contributions):
        """
        Unit test cap contributions
        Parameters
        ----------
        mock_cap_contributions : [type]
            [description]
        """
        state = SmartState()
        state.cap_contributions(Mock(), Mock())
        mock_cap_contributions.assert_called()

    @patch("shapash.explainer.smart_state.hide_contributions")
    def test_hide_contributions(self, mock_hide_contributions):
        """
        Unit test hide contributions
        Parameters
        ----------
        mock_hide_contributions : [type]
            [description]
        """
        state = SmartState()
        state.hide_contributions(Mock(), Mock())
        mock_hide_contributions.assert_called()

    @patch("shapash.explainer.smart_state.sign_contributions")
    def test_sign_contributions(self, mock_sign_contributions):
        """
        Unit test sign contributions
        Parameters
        ----------
        mock_sign_contributions : [type]
            [description]
        """
        state = SmartState()
        state.sign_contributions(Mock(), Mock())
        mock_sign_contributions.assert_called()

    @patch("shapash.explainer.smart_state.cutoff_contributions")
    def test_cutoff_contributions(self, mock_cutoff_contributions):
        """
        Unit test cutoff contributions
        Parameters
        ----------
        mock_cutoff_contributions : [type]
            [description]
        """
        state = SmartState()
        state.cutoff_contributions(Mock(), Mock())
        mock_cutoff_contributions.assert_called()

    @patch("shapash.explainer.smart_state.combine_masks")
    def test_combine_masks(self, mock_combine_masks):
        """
        Unit test combine masks
        Parameters
        ----------
        mock_combine_masks : [type]
            [description]
        """
        state = SmartState()
        state.combine_masks(Mock())
        mock_combine_masks.assert_called()

    @patch("shapash.explainer.smart_state.compute_masked_contributions")
    def test_compute_masked_contributions(self, mock_compute_masked_contributions):
        """
        Unit test compute masked contributions
        Parameters
        ----------
        mock_compute_masked_contributions : [type]
            [description]
        """
        state = SmartState()
        state.compute_masked_contributions(Mock(), Mock())
        mock_compute_masked_contributions.assert_called()

    @patch("shapash.explainer.smart_state.init_mask")
    def test_init_mask(self, mock_init_mask):
        """
        Unit test init mask
        Parameters
        ----------
        mock_init_mask : [type]
            [description]
        """
        state = SmartState()
        state.init_mask(Mock())
        mock_init_mask.assert_called()

    def test_summarize_1(self):
        """
        Unit test summarize 1
        """
        state = SmartState()
        contrib_sorted = pd.DataFrame(
            [
                [0.32230754, 0.1550689, 0.10183475, 0.05471339],
                [-0.58547512, -0.37050409, -0.07249285, 0.00171975],
                [-0.48666675, 0.25507156, -0.16968889, 0.0757443],
            ],
            columns=["contribution_0", "contribution_1", "contribution_2", "contribution_3"],
            index=[0, 1, 2],
        )
        var_dict = pd.DataFrame(
            [[1, 0, 2, 3], [1, 0, 3, 2], [1, 0, 2, 3]],
            columns=["feature_0", "feature_1", "feature_2", "feature_3"],
            index=[0, 1, 2],
        )
        x_sorted = pd.DataFrame(
            [[1.0, 3.0, 22.0, 1.0], [2.0, 1.0, 2.0, 38.0], [2.0, 3.0, 26.0, 1.0]],
            columns=["feature_0", "feature_1", "feature_2", "feature_3"],
            index=[0, 1, 2],
        )
        mask = pd.DataFrame(
            [[True, True, False, False], [True, True, False, False], [True, True, False, False]],
            columns=["contribution_0", "contribution_1", "contribution_2", "contribution_3"],
            index=[0, 1, 2],
        )
        columns_dict = {0: "Pclass", 1: "Sex", 2: "Age", 3: "Embarked"}
        features_dict = {"Pclass": "Pclass", "Sex": "Sex", "Age": "Age", "Embarked": "Embarked"}
        output = state.summarize(contrib_sorted, var_dict, x_sorted, mask, columns_dict, features_dict)
        expected = pd.DataFrame(
            [
                ["Sex", 1.0, 0.32230754, "Pclass", 3.0, 0.1550689],
                ["Sex", 2.0, -0.58547512, "Pclass", 1.0, -0.37050409],
                ["Sex", 2.0, -0.48666675, "Pclass", 3.0, 0.25507156],
            ],
            columns=["feature_1", "value_1", "contribution_1", "feature_2", "value_2", "contribution_2"],
            index=[0, 1, 2],
            dtype=object,
        )
        assert not pd.testing.assert_frame_equal(expected, output)

    @patch("shapash.explainer.smart_state.compute_features_import")
    def test_compute_features_import(self, mock_compute_features_import):
        """
        Unit test compute features import
        """
        state = SmartState()
        state.compute_features_import(Mock())
        mock_compute_features_import.assert_called()

    @patch("shapash.explainer.smart_state.group_contributions")
    def test_compute_grouped_contributions(self, mock_group_contributions):
        """
        Unit test compute features groups contributions
        """
        state = SmartState()
        state.compute_grouped_contributions(Mock(), {})
        mock_group_contributions.assert_called()
