import unittest
from unittest.mock import patch

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap
import sklearn.ensemble as ske
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from shapash.backend.shap_backend import ShapBackend, get_shap_interaction_values


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

    def test_get_shap_interaction_values_sum_4d_output(self):
        class DummyTreeExplainer:
            def shap_interaction_values(self, x_df):
                return np.ones((len(x_df), x_df.shape[1], x_df.shape[1], 2))

        x_df = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "c"])

        with patch("shapash.backend.shap_backend.shap.TreeExplainer", DummyTreeExplainer):
            out = get_shap_interaction_values(x_df, DummyTreeExplainer())

        assert out.shape == (4, 3, 3)
        assert np.all(out == 2)

    def test_get_shap_interaction_values_select_label_from_list_output(self):
        class DummyTreeExplainer:
            def shap_interaction_values(self, x_df):
                return [
                    np.ones((len(x_df), x_df.shape[1], x_df.shape[1])),
                    np.full((len(x_df), x_df.shape[1], x_df.shape[1]), 3.0),
                ]

        x_df = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "c"])

        with patch("shapash.backend.shap_backend.shap.TreeExplainer", DummyTreeExplainer):
            out = get_shap_interaction_values(x_df, DummyTreeExplainer(), class_index=1)

        assert out.shape == (4, 3, 3)
        assert np.all(out == 3)

    def test_get_shap_interaction_values_select_label_from_4d_output(self):
        class DummyTreeExplainer:
            def shap_interaction_values(self, x_df):
                out = np.zeros((len(x_df), x_df.shape[1], x_df.shape[1], 2))
                out[..., 0] = 2.0
                out[..., 1] = 5.0
                return out

        x_df = pd.DataFrame(np.zeros((4, 3)), columns=["a", "b", "c"])

        with patch("shapash.backend.shap_backend.shap.TreeExplainer", DummyTreeExplainer):
            out = get_shap_interaction_values(x_df, DummyTreeExplainer(), class_index=1)

        assert out.shape == (4, 3, 3)
        assert np.all(out == 5)


class TestShapBackendExplainerSelection(unittest.TestCase):
    def setUp(self):
        self.x_df = pd.DataFrame(
            {"x1": np.arange(20).astype(float), "x2": np.arange(20)[::-1].astype(float)}
        )
        self.y = (self.x_df["x1"] > 10).astype(int)

    def test_explainer_args_without_explicit_explainer_class(self):
        model = lgb.LGBMRegressor(n_estimators=1).fit(self.x_df, self.y)
        backend_xpl = ShapBackend(model, explainer_args={"algorithm": "tree"})
        assert isinstance(backend_xpl.explainer, shap.explainers.Tree)

    def test_linear_model_uses_linear_explainer(self):
        model = LinearRegression().fit(self.x_df, self.y)
        masker = shap.maskers.Independent(self.x_df)
        backend_xpl = ShapBackend(model, masker=masker)
        assert isinstance(backend_xpl.explainer, shap.explainers.Linear)

    def test_falls_back_to_predict_proba(self):
        model = KNeighborsClassifier(n_neighbors=3).fit(self.x_df, self.y)
        backend_xpl = ShapBackend(model, masker=self.x_df)
        assert hasattr(backend_xpl, "explainer")

    def test_falls_back_to_predict(self):
        model = KNeighborsRegressor(n_neighbors=3).fit(self.x_df, self.y)
        backend_xpl = ShapBackend(model, masker=self.x_df)
        assert hasattr(backend_xpl, "explainer")


class TestGetShapInteractionValues(unittest.TestCase):
    def test_raises_when_explainer_is_not_tree_explainer(self):
        x_df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})

        class FakeExplainer:
            pass

        with pytest.raises(ValueError, match="not a TreeExplainer"):
            get_shap_interaction_values(x_df, FakeExplainer())
