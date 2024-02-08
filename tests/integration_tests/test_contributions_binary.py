"""
Unit test for contributions binary
"""
import unittest
from os.path import abspath, dirname, join

import category_encoders as ce
import numpy as np
import pandas as pd

from shapash.decomposition.contributions import inverse_transform_contributions


class TestContributions(unittest.TestCase):
    """
    Unit test Contributions Class
    TODO: Docstring
    Parameters
    ----------
    unittest : [type]
        [description]
    """

    def setUp(self):
        """
        Setup
        Parameters
        ----------
        unittest : [type]
            [description]
        """
        data_path = dirname(dirname(abspath(__file__)))
        self.ds_titanic_clean = pd.read_pickle(join(data_path, "data", "clean_titanic.pkl"))

    def test_inverse_transform_contributions_ce_basen(self):
        """
        Unit test inverse transform contributions ce base n
        """
        preprocessing = ce.BaseNEncoder(cols=["Age", "Sex"], return_df=True, base=3)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        contributions = pd.DataFrame(
            data=np.random.rand(fitted_dataset.shape[0], fitted_dataset.shape[1]),
            columns=fitted_dataset.columns,
            index=self.ds_titanic_clean.index,
        )
        output = inverse_transform_contributions(contributions, preprocessing)
        assert isinstance(output, pd.DataFrame)
        assert self.ds_titanic_clean.shape == output.shape
        np.testing.assert_almost_equal(contributions.values.sum(axis=1), output.values.sum(axis=1))

    def test_inverse_transform_contributions_ce_binary(self):
        """
        Unit test inverse transform contributions ce binary
        """
        preprocessing = ce.BinaryEncoder(cols=["Pclass", "Age", "Sex"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        contributions = pd.DataFrame(
            data=np.random.rand(fitted_dataset.shape[0], fitted_dataset.shape[1]),
            columns=fitted_dataset.columns,
            index=self.ds_titanic_clean.index,
        )
        output = inverse_transform_contributions(contributions, preprocessing)
        assert isinstance(output, pd.DataFrame)
        assert self.ds_titanic_clean.shape == output.shape
        np.testing.assert_almost_equal(contributions.values.sum(axis=1), output.values.sum(axis=1))

    def test_inverse_transform_contributions_ce_onehot(self):
        """
        Unit test inverse transform contributions ce onehot
        """
        preprocessing = ce.OneHotEncoder(cols=["Pclass", "Sex"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        contributions = pd.DataFrame(
            data=np.random.rand(fitted_dataset.shape[0], fitted_dataset.shape[1]),
            columns=fitted_dataset.columns,
            index=self.ds_titanic_clean.index,
        )
        output = inverse_transform_contributions(contributions, preprocessing)
        assert isinstance(output, pd.DataFrame)
        assert self.ds_titanic_clean.shape == output.shape
        np.testing.assert_almost_equal(contributions.values.sum(axis=1), output.values.sum(axis=1))

    def test_inverse_transform_contributions_ce_ordinal(self):
        """
        Unit test inverse transform contributions ce ordinal
        """
        preprocessing = ce.OrdinalEncoder(cols=["Pclass", "Age"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        contributions = pd.DataFrame(
            data=np.random.rand(fitted_dataset.shape[0], fitted_dataset.shape[1]),
            columns=fitted_dataset.columns,
            index=self.ds_titanic_clean.index,
        )
        output = inverse_transform_contributions(contributions, preprocessing)
        assert isinstance(output, pd.DataFrame)
        assert self.ds_titanic_clean.shape == output.shape
        np.testing.assert_almost_equal(contributions.values.sum(axis=1), output.values.sum(axis=1))
