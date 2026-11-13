"""
Unit tests lime backend.
"""

import unittest
import pickle
from types import SimpleNamespace

import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble as ske
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from shapash import SmartExplainer
from shapash.backend.lime_backend import LimeBackend


@pytest.mark.parametrize(
    "model",
    [
        LinearSVC(),
        RidgeClassifier(),
        Perceptron(random_state=0),
        SGDClassifier(random_state=0),
        NuSVC(),
        ske.VotingClassifier(estimators=[("linear", LogisticRegression())]),
        ske.StackingClassifier(estimators=[("linear", LogisticRegression())], final_estimator=LinearSVC(), cv=2),
    ],
    ids=lambda model: type(model).__name__,
)
def test_classifiers_without_probabilities_fail_explicitly(model):
    x = pd.DataFrame({"first": np.arange(20, dtype=float), "second": np.arange(20, dtype=float) ** 2})
    model.fit(x, np.arange(20) % 2)
    with pytest.raises(ValueError, match=f"LIME backend requires predict_proba.*{type(model).__name__}") as error:
        LimeBackend(model)
    assert "backend='shap'" in str(error.value)
    assert "calibrated classifier" in str(error.value)
    assert "does not switch backends automatically" in str(error.value)
    explainer = SmartExplainer(model=model, backend="lime")
    with pytest.raises(ValueError, match="LIME backend requires predict_proba"):
        explainer.compile(x=x)


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(),
        SGDClassifier(loss="log_loss", random_state=0),
        NuSVC(probability=True, random_state=0),
        ske.VotingClassifier(estimators=[("linear", LogisticRegression())], voting="soft"),
        ske.RandomForestRegressor(n_estimators=1, random_state=0),
    ],
    ids=lambda model: type(model).__name__,
)
def test_probability_classifiers_and_regressors_remain_accepted(model):
    x = pd.DataFrame({"first": np.arange(20, dtype=float), "second": np.arange(20, dtype=float) ** 2})
    model.fit(x, np.arange(20) % 2)
    backend = LimeBackend(model, data=x)
    result = backend.run_explainer(x.iloc[:1])
    assert backend.get_local_contributions(x.iloc[:1], result) is not None


def test_noncallable_predict_proba_is_rejected():
    class InvalidClassifier:
        classes_ = [0, 1]
        predict_proba = None

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    with pytest.raises(ValueError, match="LIME backend requires predict_proba"):
        LimeBackend(InvalidClassifier())


@pytest.mark.parametrize("explicit_data", [False, True])
def test_lime_pickle_rebuilds_explainer_with_original_background(explicit_data):
    x = pd.DataFrame({"first": np.arange(20, dtype=float), "second": np.arange(20, dtype=float) ** 2})
    model = LogisticRegression().fit(x, np.arange(20) % 2)
    backend = LimeBackend(model, data=x if explicit_data else None)
    backend.run_explainer(x)
    restored = pickle.loads(pickle.dumps(backend))
    assert restored.explainer is None
    pd.testing.assert_frame_equal(restored.data, x)
    result = restored.run_explainer(x.iloc[:1])
    assert restored.get_local_contributions(x.iloc[:1], result) is not None
    assert restored.explainer.discretizer.means == backend.explainer.discretizer.means


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

    def test_extract_intercept_supports_lime_formats_and_fallbacks(self):
        cases = [
            ({0: 1.5, 1: 2.5}, 1, 2.5),
            ({"0": "1.5", "1": "2.5"}, 1, 2.5),
            ({0: 1.5, 1: 2.5}, 3, 1.5),
            ([1.5, 2.5], 1, 2.5),
            ((1.5, 2.5), 3, 1.5),
            ("2.5", None, 2.5),
            (2.5, None, 2.5),
            ({}, 1, 0.0),
            ([], 1, 0.0),
            ("not a number", None, 0.0),
            (object(), None, 0.0),
            (None, None, 0.0),
        ]
        for intercept, class_idx, expected in cases:
            with self.subTest(intercept=intercept, class_idx=class_idx):
                explanation = SimpleNamespace(intercept=intercept)
                self.assertEqual(LimeBackend._extract_intercept(explanation, class_idx), expected)

        self.assertEqual(LimeBackend._extract_intercept(SimpleNamespace()), 0.0)

    def test_get_global_contributions(self):
        for model in self.model_list:
            print(type(model))
            model.fit(self.x_df.values, self.y_df)
            backend_xpl = LimeBackend(model, data=self.x_df)
            explain_data = backend_xpl.run_explainer(self.x_df)
            contributions = backend_xpl.get_local_contributions(self.x_df, explain_data)

            assert contributions is not None
            assert "base_values" in explain_data
            assert isinstance(contributions, (list, pd.DataFrame, np.ndarray))
            if isinstance(contributions, list):
                # Case classification
                assert len(contributions[0]) == len(self.x_df)
            else:
                assert len(contributions) == len(self.x_df)

            base_values = explain_data["base_values"]
            assert isinstance(base_values, np.ndarray)
            if backend_xpl._case == "classification":
                assert base_values.shape[0] == len(self.x_df)
                assert base_values.shape[1] == len(backend_xpl._classes)
            else:
                assert base_values.shape[0] == len(self.x_df)

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
        assert "base_values" in explain_data

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

        base_values = explain_data["base_values"]
        assert isinstance(base_values, np.ndarray)
        assert base_values.shape == (len(x_multi), 3)
