"""
Unit test contributions multiclass
"""
import unittest
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from shapash.decomposition.contributions import compute_contributions, rank_contributions


class TestContributions(unittest.TestCase):
    """
    Unit test Contributions Class
    TODO: Docstring
    Parameters
    ----------
    unittest : [type]
        [description]
    Returns
    -------
    [type]
        [description]
    """
    def setUp(self):
        """
        Setup
        """
        iris = load_iris()
        x_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        y_df = pd.DataFrame(data=iris.target, columns=["target"])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_df,
            y_df,
            random_state=1
        )

    def check_compute_contributions(self, slist, blist, x_test):
        """
        Unit test compute contributions
        Parameters
        ----------
        slist : list
            [description]
        blist : list
            [description]
        x_test : pandas.DataFrame
            [description]
        """
        assert len(slist) == 3
        assert len(blist) == 3
        for i in range(3):
            assert slist[i].shape == x_test.shape
            assert slist[i].index.equals(x_test.index)
            np.testing.assert_array_equal(np.array(x_test.columns), np.array(x_test.columns))
            assert slist[i].isnull().sum().sum() == 0

    def check_sum_contributions(self, slist, blist, y_dataframe):
        """
        Unit test check sum contributions
        Parameters
        ----------
        slist : list
            [description]
        blist : list
            [description]
        y_dataframe : [type]
            [description]
        """
        for i in range(3):
            np.testing.assert_almost_equal(slist[i].sum(axis=1) + blist[i], y_dataframe[:, i], decimal=5)

    def get_predictions(self, model, **args):
        """
        Method to get predictions
        Parameters
        ----------
        model : [type]
            [description]
        Returns
        -------
        [type]
            [description]
        """
        model.fit(self.x_train, self.y_train)
        if args:
            return model.predict(self.x_test, **args)
        else:
            return model.predict(self.x_test)

    def test_compute_contributions_multiclass_rf(self):
        """
        Unit test to compute contributions multiclass RF
        """
        model = RandomForestClassifier(n_estimators=3)
        model.fit(self.x_train, self.y_train)
        y_dataframe = model.predict_proba(self.x_test)

        explainer = shap.TreeExplainer(model)
        slist, blist = compute_contributions(self.x_test, explainer)
        self.check_compute_contributions(slist, blist, self.x_test)
        self.check_sum_contributions(slist, blist, y_dataframe)

    def test_compute_contributions_multiclass_xgbc(self):
        """
        Unit test to compute contributions multiclass XGBC
        """
        model = xgb.XGBClassifier()
        model.fit(self.x_train, self.y_train)
        y_dataframe = model.predict(self.x_test, output_margin=True)

        explainer = shap.TreeExplainer(model)
        slist, blist = compute_contributions(self.x_test, explainer)
        self.check_compute_contributions(slist, blist, self.x_test)
        self.check_sum_contributions(slist, blist, y_dataframe)

    def test_compute_contributions_multiclass_lgbmc(self):
        """
        Unit test to compute contributions multiclass LGBMC
        """
        model = lgb.LGBMClassifier(
            n_estimators=3,
            objective='multiclass'
        )
        model.fit(self.x_train, self.y_train)
        y_dataframe = model.predict(self.x_test, raw_score=True)

        explainer = shap.TreeExplainer(model)
        slist, blist = compute_contributions(self.x_test, explainer)
        self.check_compute_contributions(slist, blist, self.x_test)
        self.check_sum_contributions(slist, blist, y_dataframe)

    def test_compute_contributions_multiclass_svc(self):
        """
        Unit test to compute contributions multiclass SVC
        """
        model = SVC(kernel='rbf', probability=True)
        model.fit(self.x_train, self.y_train)
        y_dataframe = model.predict_proba(self.x_test)

        explainer = shap.KernelExplainer(model.predict_proba, self.x_train)
        slist, blist = compute_contributions(self.x_test, explainer)
        self.check_compute_contributions(slist, blist, self.x_test)
        self.check_sum_contributions(slist, blist, y_dataframe)

    def test_rank_contributions_1(self):
        """
        Unit test rank contributions 1
        """
        model = RandomForestClassifier(n_estimators=3)
        model.fit(self.x_train, self.y_train)
        explainer = shap.TreeExplainer(model)
        slist, blist = compute_contributions(self.x_test, explainer)

        assert isinstance(slist, list)
        assert isinstance(blist, list)
        for i in range(3):
            s_ord, x_ord, s_dict = rank_contributions(slist[i], pd.DataFrame(data=self.x_test))
            assert np.all(np.diff(np.abs(s_ord), axis=1) <= 0) == 1
            assert np.array_equal(
                x_ord.values,
                np.take_along_axis(self.x_test.values, s_dict.values, axis=1)
            )
