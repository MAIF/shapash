"""
Unit test shap_backend
"""
import unittest
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import sklearn.svm as svm
import sklearn.linear_model as skl
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from shapash.utils.shap_backend import shap_contributions, check_explainer, get_shap_interaction_values
from shapash.utils.model_synoptic import simple_tree_model,catboost_model,linear_model,svm_model
import shap

class TestShapBackend(unittest.TestCase):
    """
    Unit test shap backend transform class
    Ensure the proper functioning of the backend for a set of models

    Parameters
    ----------
    unittest.TestCase
    """
    def setUp(self):
        self.modellist = [
            lgb.LGBMRegressor(n_estimators=1), lgb.LGBMClassifier(n_estimators=1),
            xgb.XGBRegressor(n_estimators=1), xgb.XGBClassifier(n_estimators=1),
            svm.SVR(kernel='linear'), svm.SVC(kernel='linear'),
            cb.CatBoostRegressor(n_estimators=1), cb.CatBoostClassifier(n_estimators=1),
            ske.GradientBoostingRegressor(n_estimators=1), ske.GradientBoostingClassifier(n_estimators=1),
            ske.ExtraTreesRegressor(n_estimators=1), ske.ExtraTreesClassifier(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1), ske.RandomForestClassifier(n_estimators=1),
            skl.LogisticRegression() , skl.LinearRegression()
        ]

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df = df[['x1','x2']]
        self.y_df = df['y'].to_frame()

    def test_shap_contributions_0(self):
        """
        test shap_backend
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            shap_contributions(model, self.x_df)

    def test_shap_contributions_1(self):
        """
        test shap_backend with explainer pre-compute
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            if str(type(model)) in simple_tree_model:
                explainer = shap.TreeExplainer(model)

            elif str(type(model)) in catboost_model:
                explainer = shap.TreeExplainer(model)

            elif str(type(model)) in linear_model:
                explainer = shap.LinearExplainer(model, self.x_df)

            elif str(type(model)) in svm_model:
                explainer = shap.KernelExplainer(model.predict, self.x_df)

            shap_contributions(model, self.x_df, explainer)

    def test_check_explainer_1(self):
        """
        Unit test check explainer 1
        """
        explainer = None
        assert check_explainer(explainer) == None

    def test_check_explainer_2(self):
        """
        Unit test check explainer 2
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        explainer = shap.TreeExplainer(clf)
        assert explainer.__class__.__base__.__name__ == "Explainer"
        check_explainer(explainer)

    def test_shap_interaction_values_1(self):
        """
        Unit test assert shap_interaction_values function raises an error when not dealing with TreeExplainer object
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            if str(type(model)) in linear_model:
                explainer = shap.LinearExplainer(model, self.x_df)
            elif str(type(model)) in svm_model:
                explainer = shap.KernelExplainer(model.predict, self.x_df)
            else:
                explainer = None

            if explainer:
                self.assertRaises(ValueError, get_shap_interaction_values, self.x_df, explainer)

    def test_shap_interaction_values_2(self):
        """
        Unit test shap_interaction_values function
        """
        for model in [ske.RandomForestRegressor(n_estimators=1), ske.RandomForestClassifier(n_estimators=1)]:
            self.x_df = self.x_df.astype(float)
            model.fit(self.x_df, self.y_df)
            print(model)
            explainer = shap.TreeExplainer(model)
            interaction_values = get_shap_interaction_values(self.x_df, explainer)
            assert interaction_values.shape[0] == self.x_df.shape[0]
            assert interaction_values.shape[1] == self.x_df.shape[1]
            assert interaction_values.shape[2] == self.x_df.shape[1]
