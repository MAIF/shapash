"""
Unit tests lime backend.
"""

import unittest

import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import xgboost as xgb

from shapash.backend.lime_backend import LimeBackend


class TestLimeBackend(unittest.TestCase):
    def setUp(self):
        self.model_list = [
            xgb.XGBClassifier(n_estimators=1),
            ske.RandomForestClassifier(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1),
            ske.GradientBoostingRegressor(n_estimators=1),
        ]

        df = pd.DataFrame(range(0, 4), columns=["id"])
        df["y"] = df["id"].apply(lambda x: 1 if x < 3 else 0)
        df["x1"] = np.random.randint(1, 123, df.shape[0])
        df["x2"] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index("id")
        self.x_df = df[["x1", "x2"]]
        self.y_df = df["y"].to_frame()

    def test_init(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            backend_xpl = LimeBackend(model)
            assert hasattr(backend_xpl, "explainer")

            backend_xpl = LimeBackend(model, data=self.x_df)
            assert hasattr(backend_xpl, "data")

            backend_xpl = LimeBackend(model, preprocessing=ce.OrdinalEncoder())
            assert hasattr(backend_xpl, "preprocessing")
            assert isinstance(backend_xpl.preprocessing, ce.OrdinalEncoder)

    def test_get_global_contributions(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df.values, self.y_df)
            backend_xpl = LimeBackend(model, data=self.x_df)
            explain_data = backend_xpl.run_explainer(self.x_df)
            contributions = backend_xpl.get_local_contributions(self.x_df, explain_data)

            assert contributions is not None
            assert isinstance(contributions, (list, pd.DataFrame, np.ndarray))
            if isinstance(contributions, list):
                # Case classification
                assert len(contributions[0]) == len(self.x_df)
            else:
                assert len(contributions) == len(self.x_df)

            features_imp = backend_xpl.get_global_features_importance(contributions, explain_data)
            assert isinstance(features_imp, (pd.Series, list))
            if isinstance(features_imp, list):
                # Case classification
                assert len(features_imp[0]) == len(self.x_df.columns)
            else:
                assert len(features_imp) == len(self.x_df.columns)

    def test_run_explainer_multiclass_returns_list_contributions(self):
        """Test multiclass (>2) branch returns a list-like contributions payload."""
        x_multi = pd.DataFrame(
            {
                "x1": [10, 11, 20, 21, 30, 31],
                "x2": [1, 2, 1, 2, 1, 2],
            },
            index=[0, 1, 2, 3, 4, 5],
        )
        y_multi = pd.Series([0, 0, 1, 1, 2, 2], index=x_multi.index)

        model = ske.RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(x_multi.values, y_multi)

        backend_xpl = LimeBackend(model, data=x_multi)
        explain_data = backend_xpl.run_explainer(x_multi)

        assert isinstance(explain_data, dict)
        assert "contributions" in explain_data

        contributions = explain_data["contributions"]
        assert isinstance(contributions, list)
        assert len(contributions) == 3
        for class_contrib in contributions:
            assert isinstance(class_contrib, pd.DataFrame)
            assert class_contrib.shape == (len(x_multi), x_multi.shape[1])

        local_contrib = backend_xpl.get_local_contributions(x_multi, explain_data)
        assert isinstance(local_contrib, list)
        assert len(local_contrib) == 3
        for class_contrib_df in local_contrib:
            assert isinstance(class_contrib_df, pd.DataFrame)
            assert class_contrib_df.shape == (len(x_multi), x_multi.shape[1])
