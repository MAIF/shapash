"""
Unit test inverse transform
"""
import unittest
from os.path import abspath, dirname, join

import category_encoders as ce
import pandas as pd

from shapash.utils.transform import inverse_transform


class TestInverseTranform(unittest.TestCase):
    """
    Unit test invers transform class
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
        # The "python"-storage titanic fixture below was intentionally built pyarrow-free (see git
        # history on the .pkl). With pyarrow installed, pandas' default "auto" string storage infers
        # pyarrow-backed strings for any newly-built column (e.g. category_encoders' inverse-transform
        # output), which then fails assert_frame_equal's exact-dtype check against the fixture even
        # though the values match. Pin "python" storage for the duration of this test only.
        previous_storage = pd.options.mode.string_storage
        pd.options.mode.string_storage = "python"
        self.addCleanup(lambda: setattr(pd.options.mode, "string_storage", previous_storage))

        data_path = dirname(dirname(abspath(__file__)))
        if int(pd.__version__.split(".")[0]) >= 3:
            self.ds_titanic_clean = pd.read_pickle(join(data_path, "data", "clean_titanic_pandas_3.pkl"))
        else:
            self.ds_titanic_clean = pd.read_pickle(join(data_path, "data", "clean_titanic.pkl"))

    def test_inverse_transform_ce_basen(self):
        """
        Unit test inverse transform base n
        """
        preprocessing = ce.BaseNEncoder(cols=["Age", "Sex"], return_df=True, base=3)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        output = inverse_transform(fitted_dataset, preprocessing)
        pd.testing.assert_frame_equal(output, self.ds_titanic_clean)

    def test_inverse_transform_ce_onehot(self):
        """
        Unit test inverse transform ce onehot
        """
        preprocessing = ce.OneHotEncoder(cols=["Age", "Sex"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        output = inverse_transform(fitted_dataset, preprocessing)
        pd.testing.assert_frame_equal(output, self.ds_titanic_clean)

    def test_inverse_transform_ce_binary(self):
        """
        Unit test inverse transform ce binary
        """
        preprocessing = ce.BinaryEncoder(cols=["Age", "Sex"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        output = inverse_transform(fitted_dataset, preprocessing)
        pd.testing.assert_frame_equal(output, self.ds_titanic_clean)

    def test_inverse_transform_ce_ordinal(self):
        """
        Unit test inverse transform ce ordinal
        """
        preprocessing = ce.OrdinalEncoder(cols=["Age", "Sex"], return_df=True)
        fitted_dataset = preprocessing.fit_transform(self.ds_titanic_clean)
        output = inverse_transform(fitted_dataset, preprocessing)
        pd.testing.assert_frame_equal(output, self.ds_titanic_clean)
