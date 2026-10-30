import unittest
from unittest.mock import MagicMock, patch

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import shap
import sklearn.ensemble as ske
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import (
    GammaRegressor,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import NuSVC, SVC

from shapash.backend.shap_backend import ShapBackend, get_shap_interaction_values
from shapash.model import TensorFlowModelAdapter


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

    def test_get_shap_interaction_values_4d_sample_first_with_same_sample_feature_size(self):
        class DummyTreeExplainer:
            def shap_interaction_values(self, x_df):
                out = np.zeros((len(x_df), x_df.shape[1], x_df.shape[1], 2))
                out[..., 0] = 1.0
                out[..., 1] = 4.0
                return out

        x_df = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "c"])

        with patch("shapash.backend.shap_backend.shap.TreeExplainer", DummyTreeExplainer):
            out = get_shap_interaction_values(x_df, DummyTreeExplainer(), class_index=1)

        assert out.shape == (3, 3, 3)
        assert np.all(out == 4)

    def test_get_shap_interaction_values_4d_output_first_with_same_sample_feature_size(self):
        class DummyTreeExplainer:
            def shap_interaction_values(self, x_df):
                out = np.zeros((2, len(x_df), x_df.shape[1], x_df.shape[1]))
                out[0, ...] = 2.0
                out[1, ...] = 6.0
                return out

        x_df = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "c"])

        with patch("shapash.backend.shap_backend.shap.TreeExplainer", DummyTreeExplainer):
            out = get_shap_interaction_values(x_df, DummyTreeExplainer(), class_index=1)

        assert out.shape == (3, 3, 3)
        assert np.all(out == 6)


class TestShapBackendExplainerSelection(unittest.TestCase):
    def setUp(self):
        self.x_df = pd.DataFrame({"x1": np.arange(20).astype(float), "x2": np.arange(20)[::-1].astype(float)})
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


@pytest.mark.parametrize("model_class", [PoissonRegressor, GammaRegressor, TweedieRegressor])
@pytest.mark.parametrize("explainer_args", [None, {"algorithm": "auto"}], ids=["default", "explicit-auto"])
@pytest.mark.parametrize("as_masker", [False, True], ids=["dataframe", "independent-masker"])
def test_glm_explanations_reconstruct_predictions(model_class, explainer_args, as_masker):
    x = pd.DataFrame({"first": np.linspace(-2, 2, 20), "second": np.linspace(0, 1, 20) ** 2})
    model = model_class(link="log") if model_class is TweedieRegressor else model_class()
    model.fit(x, np.exp(x["first"]))
    masker = shap.maskers.Independent(x) if as_masker else x
    backend = ShapBackend(model, masker=masker, explainer_args=explainer_args)
    assert not isinstance(backend.explainer, shap.explainers.Linear)
    data = backend.run_explainer(x.iloc[:4])
    np.testing.assert_allclose(
        data["base_values"] + data["contributions"].sum(axis=1), model.predict(x.iloc[:4]), atol=1e-6
    )


@pytest.mark.parametrize("n_classes", [2, 3])
def test_hard_voting_classifier_requires_soft_voting(n_classes):
    x = pd.DataFrame({"first": np.arange(30, dtype=float), "second": np.arange(30, dtype=float) ** 2})
    model = VotingClassifier(estimators=[("linear", LogisticRegression(max_iter=1000))]).fit(
        x, np.arange(30) % n_classes
    )
    with pytest.raises(ValueError, match="VotingClassifier requires voting='soft'"):
        ShapBackend(model, masker=x)


@pytest.mark.parametrize("model_class", [SVC, NuSVC])
def test_pairwise_svm_scores_require_per_class_outputs(model_class):
    x = pd.DataFrame({"first": np.arange(30, dtype=float), "second": np.arange(30, dtype=float) ** 2})
    model = model_class(decision_function_shape="ovo").fit(x, np.arange(30) % 3)
    with pytest.raises(ValueError, match="decision_function_shape='ovr' or probability=True"):
        ShapBackend(model, masker=x)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"neural_explainer": "unknown"}, "neural_explainer must"),
        ({"neural_background": pd.DataFrame([[1]])}, "neural_background requires"),
        ({"neural_explainer": "deep"}, "nonempty encoded background"),
        ({"neural_explainer": "deep", "masker": pd.DataFrame()}, "nonempty encoded background"),
        ({"neural_explainer": "deep", "explainer_args": {"algorithm": "auto"}}, "native constructor options"),
        ({"neural_explainer": "gradient", "explainer_compute_args": {"ranked_outputs": 1}}, "all outputs"),
    ],
)
def test_invalid_neural_configuration(options, message):
    x = pd.DataFrame({"first": [0.0, 1.0, 2.0], "second": [2.0, 1.0, 0.0]})
    model = LinearRegression().fit(x, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match=message):
        ShapBackend(model, **options)


def test_neural_method_requires_framework_adapter():
    x = pd.DataFrame({"first": [0.0, 1.0, 2.0]})
    model = LinearRegression().fit(x, [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="requires KerasModelAdapter"):
        ShapBackend(model, neural_explainer="deep", masker=x)


@pytest.fixture
def neural_adapter():
    with (
        patch.object(TensorFlowModelAdapter, "_validate_model"),
        patch.object(TensorFlowModelAdapter, "_forward", lambda self, values: values.sum(axis=1)),
        patch.object(TensorFlowModelAdapter, "_to_numpy", lambda self, values: np.asarray(values)),
    ):
        yield TensorFlowModelAdapter(None, n_features=2, task="regression")


@pytest.mark.parametrize("stage", ["construction", "computation"])
def test_auto_neural_falls_back_to_gradient(neural_adapter, stage, capsys):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    deep, gradient = MagicMock(), MagicMock()
    deep.explainer.__str__.return_value = "DeepExplainer"
    gradient.explainer.__str__.return_value = "GradientExplainer"
    result = {"contributions": x.to_numpy(), "base_values": np.zeros(2)}
    gradient.explain.return_value = result
    error = NotImplementedError("unsupported operation")
    if stage == "computation":
        deep.explain.side_effect = error
    with patch(
        "shapash.backend.shap_backend.NeuralShap", side_effect=[error if stage == "construction" else deep, gradient]
    ) as bridge:
        with pytest.warns(UserWarning, match="Falling back to GradientExplainer"):
            backend = ShapBackend(neural_adapter, masker=x)
            assert backend.run_explainer(x) is result
        assert capsys.readouterr().out == "INFO: Shap explainer type - GradientExplainer\n"
        assert [call.args[2] for call in bridge.call_args_list] == ["deep", "gradient"]
        assert backend.explainer is gradient.explainer
        assert backend.neural_explainer == "auto"
        assert backend._selected_neural == "gradient"
        backend.run_explainer(x)
        assert bridge.call_count == 2


@pytest.mark.parametrize("stage", ["construction", "computation"])
def test_auto_neural_falls_back_to_agnostic(neural_adapter, stage, capsys):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    deep, gradient = MagicMock(), MagicMock()
    error = NotImplementedError("unsupported operation")
    for bridge in (deep, gradient):
        bridge.explain.side_effect = error
    bridges = [error, error] if stage == "construction" else [deep, gradient]
    with patch("shapash.backend.shap_backend.NeuralShap", side_effect=bridges):
        with pytest.warns(UserWarning) as recorded:
            backend = ShapBackend(neural_adapter, masker=x)
            data = backend.run_explainer(x)
        assert len(recorded) == 2
        assert "GradientExplainer" in str(recorded[0].message)
        assert "agnostic SHAP" in str(recorded[1].message)
        assert isinstance(backend.explainer, shap.ExactExplainer)
        output = capsys.readouterr().out
        assert output.count("INFO: Shap explainer type -") == 1
        assert "ExactExplainer" in output
        assert backend._selected_neural is None
        np.testing.assert_allclose(data["contributions"].sum(axis=1) + data["base_values"], x.sum(axis=1))
        assert backend.__getstate__()["explainer"] is backend.explainer


@pytest.mark.parametrize(
    "method, name", [("auto", "DeepExplainer"), ("deep", "DeepExplainer"), ("gradient", "GradientExplainer")]
)
def test_neural_reports_used_explainer(neural_adapter, method, name, capsys):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    bridge = MagicMock()
    bridge.explainer.__str__.return_value = name
    with patch("shapash.backend.shap_backend.NeuralShap", return_value=bridge):
        backend = ShapBackend(neural_adapter, masker=x, neural_explainer=method)
        assert backend.run_explainer(x) is bridge.explain.return_value
    assert capsys.readouterr().out == f"INFO: Shap explainer type - {name}\n"


@pytest.mark.parametrize("method", ["deep", "gradient"])
@pytest.mark.parametrize("stage", ["construction", "computation"])
def test_explicit_neural_method_never_falls_back(neural_adapter, method, stage):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    error = NotImplementedError("unsupported operation")
    bridge = MagicMock()
    bridge.explain.side_effect = error
    with patch(
        "shapash.backend.shap_backend.NeuralShap",
        side_effect=error if stage == "construction" else None,
        return_value=bridge,
    ) as constructor:
        with pytest.raises(NotImplementedError, match="unsupported operation"):
            ShapBackend(neural_adapter, masker=x, neural_explainer=method).run_explainer(x)
        assert constructor.call_count == 1


@pytest.mark.parametrize("stage", ["construction", "computation"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid input"),
        TypeError("unknown argument"),
        RuntimeError("out of memory"),
        AssertionError("incorrect shape"),
        KeyError("missing output"),
    ],
)
def test_auto_neural_does_not_hide_unrelated_errors(neural_adapter, stage, error):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    bridge = MagicMock()
    bridge.explain.side_effect = error
    with patch(
        "shapash.backend.shap_backend.NeuralShap",
        side_effect=error if stage == "construction" else None,
        return_value=bridge,
    ) as constructor:
        with pytest.raises(type(error), match=str(error)):
            ShapBackend(neural_adapter, masker=x).run_explainer(x)
        assert constructor.call_count == 1


def test_explicit_agnostic_skips_neural_explainers(neural_adapter):
    x = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], columns=["a", "b"])
    with patch("shapash.backend.shap_backend.NeuralShap") as bridge:
        backend = ShapBackend(neural_adapter, neural_explainer="agnostic", neural_background=x.iloc[:1])
        assert isinstance(backend.explainer, shap.ExactExplainer)
        data = backend.run_explainer(x)
        np.testing.assert_allclose(data["contributions"].sum(axis=1) + data["base_values"], x.sum(axis=1))
        bridge.assert_not_called()
