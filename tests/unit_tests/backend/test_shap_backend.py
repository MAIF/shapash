import unittest

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import xgboost as xgb

from shapash.backend.shap_backend import ShapBackend


class TestShapBackend(unittest.TestCase):
    def setUp(self):
        self.model_list = [
            lgb.LGBMRegressor(n_estimators=1),
            lgb.LGBMClassifier(n_estimators=1),
            xgb.XGBRegressor(n_estimators=1),
            xgb.XGBClassifier(n_estimators=1),
            cb.CatBoostRegressor(n_estimators=1),
            cb.CatBoostClassifier(n_estimators=1),
            ske.GradientBoostingRegressor(n_estimators=1),
            ske.GradientBoostingClassifier(n_estimators=1),
            ske.ExtraTreesRegressor(n_estimators=1),
            ske.ExtraTreesClassifier(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1),
            ske.RandomForestClassifier(n_estimators=1),
        ]

        df = pd.DataFrame(range(0, 21), columns=["id"])
        df["y"] = df["id"].apply(lambda x: 1 if x < 10 else 0)
        df["x1"] = np.random.randint(1, 123, df.shape[0])
        df["x2"] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index("id")
        self.x_df = df[["x1", "x2"]]
        self.y_df = df["y"].to_frame()

    def test_shap_backend_init(self):
        """
        test shap_backend
        """
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            backend_xpl = ShapBackend(model)
            assert hasattr(backend_xpl, "explainer")

    def test_run_explainer(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            backend_xpl = ShapBackend(model)
            explain_data = backend_xpl.run_explainer(self.x_df)
            assert explain_data is not None

    def test_get_local_contributions(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            backend_xpl = ShapBackend(model)
            explain_data = backend_xpl.run_explainer(self.x_df)
            contributions = backend_xpl.get_local_contributions(self.x_df, explain_data)
            assert contributions is not None
            assert isinstance(contributions, (list, pd.DataFrame, np.ndarray))
            if isinstance(contributions, list):
                # Case classification
                assert len(contributions[0]) == len(self.x_df)
            else:
                assert len(contributions) == len(self.x_df)

    def test_get_global_contributions(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            backend_xpl = ShapBackend(model)
            explain_data = backend_xpl.run_explainer(self.x_df)
            contributions = backend_xpl.get_local_contributions(self.x_df, explain_data)
            features_imp = backend_xpl.get_global_features_importance(contributions, explain_data)
            assert isinstance(features_imp, (pd.Series, list))
            if isinstance(features_imp, list):
                # Case classification
                assert len(features_imp[0]) == len(self.x_df.columns)
            else:
                assert len(features_imp) == len(self.x_df.columns)
