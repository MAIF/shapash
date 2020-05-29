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
from shapash.utils.shap_backend import shap_contributions

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
            xgb.XGBRegressor(n_estimators=1), xgb.XGBRegressor(n_estimators=1),
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

    def test_shap_contributions(self):
        """
        test shap_backend
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df,self.y_df)
            shap_contributions(model,self.x_df)