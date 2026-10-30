"""Exercise additional sklearn estimators through explanation and production prediction."""

import pickle

import numpy as np
import pandas as pd
import pytest
import shap
from sklearn import ensemble, linear_model, neighbors, neural_network, svm, tree
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shapash import SmartExplainer
from shapash.utils.model_synoptic import sklearn_model


CLASSIFIERS = [
    ensemble.HistGradientBoostingClassifier,
    tree.DecisionTreeClassifier,
    tree.ExtraTreeClassifier,
    linear_model.LogisticRegressionCV,
    svm.LinearSVC,
    linear_model.RidgeClassifier,
    linear_model.RidgeClassifierCV,
    linear_model.SGDClassifier,
    linear_model.Perceptron,
    linear_model.PassiveAggressiveClassifier,
    svm.NuSVC,
    neighbors.KNeighborsClassifier,
    neural_network.MLPClassifier,
    ensemble.AdaBoostClassifier,
    ensemble.BaggingClassifier,
    ensemble.VotingClassifier,
    ensemble.StackingClassifier,
]
REGRESSORS = [
    ensemble.HistGradientBoostingRegressor,
    tree.DecisionTreeRegressor,
    tree.ExtraTreeRegressor,
    linear_model.Ridge,
    linear_model.RidgeCV,
    linear_model.Lasso,
    linear_model.LassoCV,
    linear_model.ElasticNet,
    linear_model.ElasticNetCV,
    linear_model.BayesianRidge,
    linear_model.ARDRegression,
    linear_model.HuberRegressor,
    linear_model.TheilSenRegressor,
    linear_model.QuantileRegressor,
    linear_model.SGDRegressor,
    linear_model.PassiveAggressiveRegressor,
    linear_model.Lars,
    linear_model.LarsCV,
    linear_model.LassoLars,
    linear_model.LassoLarsCV,
    linear_model.LassoLarsIC,
    linear_model.OrthogonalMatchingPursuit,
    linear_model.OrthogonalMatchingPursuitCV,
    svm.LinearSVR,
    linear_model.PoissonRegressor,
    linear_model.GammaRegressor,
    linear_model.TweedieRegressor,
    svm.NuSVR,
    neighbors.KNeighborsRegressor,
    neural_network.MLPRegressor,
    ensemble.AdaBoostRegressor,
    ensemble.BaggingRegressor,
    ensemble.VotingRegressor,
    ensemble.StackingRegressor,
]


def _make_model(model_class):
    if model_class in (ensemble.VotingClassifier, ensemble.StackingClassifier):
        model = model_class(
            estimators=[
                ("linear", linear_model.LogisticRegression()),
                ("tree", tree.DecisionTreeClassifier(max_depth=3, random_state=0)),
            ]
        )
        if model_class is ensemble.VotingClassifier:
            model.set_params(voting="soft")
        return model
    if model_class in (ensemble.VotingRegressor, ensemble.StackingRegressor):
        return model_class(
            estimators=[
                ("linear", linear_model.Ridge()),
                ("tree", tree.DecisionTreeRegressor(max_depth=3, random_state=0)),
            ]
        )
    return model_class()


def _check_model_workflow(
    model_class, with_preprocessing, n_classes=None, model_options=None, nonconsecutive_labels=False
):
    if n_classes is None:
        values, target = make_regression(n_samples=72, n_features=3, noise=1, random_state=0)
        target = target / target.std()
        if model_class in (linear_model.PoissonRegressor, linear_model.GammaRegressor, linear_model.TweedieRegressor):
            target = np.exp(target / 2)
    else:
        values, target = make_classification(
            n_samples=72, n_features=3, n_informative=3, n_redundant=0, n_classes=n_classes, random_state=0
        )
        if nonconsecutive_labels:
            target = np.array([3, 8, 42])[target]
    x = pd.DataFrame(values, columns=["col_0", "col_1", "col_2"], index=np.arange(100, 172))
    preprocessing = None
    if with_preprocessing:
        x["col_2"] = np.where(x["col_2"] > 0, "positive", "negative")
        preprocessing = ColumnTransformer(
            [
                ("scale", StandardScaler(), ["col_0", "col_1"]),
                ("category", OneHotEncoder(sparse_output=False), ["col_2"]),
            ]
        ).fit(x)
        encoded = pd.DataFrame(preprocessing.transform(x), index=x.index)
        encoded.columns = [f"col_{i}" for i in range(encoded.shape[1])]
    else:
        encoded = x

    model = _make_model(model_class)
    params = model.get_params()
    if "random_state" in params:
        model.set_params(random_state=0)
    if "cv" in params:
        model.set_params(cv=3)
    if isinstance(model, (ensemble.HistGradientBoostingClassifier, ensemble.HistGradientBoostingRegressor)):
        model.set_params(max_iter=5, min_samples_leaf=5)
    if isinstance(model, (linear_model.SGDRegressor, svm.LinearSVR)):
        model.set_params(max_iter=10000)
    if isinstance(model, (neural_network.MLPClassifier, neural_network.MLPRegressor)):
        model.set_params(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000)
    if isinstance(
        model,
        (
            ensemble.AdaBoostClassifier,
            ensemble.AdaBoostRegressor,
            ensemble.BaggingClassifier,
            ensemble.BaggingRegressor,
        ),
    ):
        model.set_params(n_estimators=5)
    if model_options:
        model.set_params(**model_options)
    model.fit(encoded, target)

    explainer = SmartExplainer(model=model, preprocessing=preprocessing)
    sample = encoded.iloc[:6]
    explainer.compile(x=sample, y_target=pd.Series(target[:6], index=sample.index))
    compiled = explainer.explainer
    expected_shape = (len(sample), x.shape[1])
    contributions = compiled.contributions
    if n_classes is not None:
        assert compiled._case == "classification"
        assert len(contributions) == n_classes
        frames = contributions
    else:
        assert compiled._case == "regression"
        frames = [contributions]
    for frame in frames:
        assert frame.shape == expected_shape
        assert frame.index.equals(sample.index)
        assert np.isfinite(frame.to_numpy()).all()
    np.testing.assert_array_equal(compiled.y_pred.iloc[:, 0], model.predict(sample))

    explain_data = compiled.explain_data
    reconstructed = explain_data["contributions"].sum(axis=1) + explain_data["base_values"]
    if isinstance(compiled.backend.explainer, shap.explainers.Linear):
        expected = model.decision_function(sample) if n_classes is not None else model.predict(sample)
    elif isinstance(model, ensemble.HistGradientBoostingClassifier) and isinstance(
        compiled.backend.explainer, shap.explainers.Tree
    ):
        expected = model.decision_function(sample)
    elif hasattr(model, "predict_proba"):
        expected = model.predict_proba(sample)
    elif hasattr(model, "decision_function"):
        expected = model.decision_function(sample)
    else:
        expected = model.predict(sample)
    np.testing.assert_allclose(reconstructed, expected, atol=1e-5, rtol=1e-5)
    assert len(explainer.to_pandas()) == len(sample)
    assert len(explainer.plot.features_importance().data) > 0

    predictor = explainer.to_smartpredictor()
    predictor.add_input(x=x.iloc[:6])
    np.testing.assert_allclose(predictor.data["x_preprocessed"], sample)
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(sample))
    assert predictor.data["contributions"].shape == expected_shape
    assert np.isfinite(predictor.data["contributions"].to_numpy(dtype=float)).all()
    for i, label in enumerate(model.predict(sample)):
        expected_contributions = contributions[list(model.classes_).index(label)] if n_classes else contributions
        np.testing.assert_allclose(
            predictor.data["contributions"].iloc[i].to_numpy(dtype=float),
            expected_contributions.iloc[i],
            atol=1e-5,
            rtol=1e-5,
        )
    assert len(predictor.detail_contributions()) == len(sample)
    if n_classes is not None and hasattr(model, "predict_proba"):
        np.testing.assert_allclose(predictor.predict_proba(), model.predict_proba(sample))
        assert "proba" in predictor.data["ypred"]
    elif n_classes is not None:
        assert "proba" not in predictor.data["ypred"]
        with pytest.raises(ValueError, match="model has no predict_proba method"):
            predictor.predict_proba()
    restored = pickle.loads(pickle.dumps(predictor))
    restored.add_input(x=x.iloc[:6])
    pd.testing.assert_frame_equal(restored.data["contributions"], predictor.data["contributions"])


@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize("model_class", REGRESSORS, ids=lambda cls: cls.__name__)
def test_sklearn_regressor_workflow(model_class, with_preprocessing):
    _check_model_workflow(model_class, with_preprocessing)


@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
@pytest.mark.parametrize("model_class", CLASSIFIERS, ids=lambda cls: cls.__name__)
def test_sklearn_classifier_workflow(model_class, with_preprocessing, n_classes):
    _check_model_workflow(model_class, with_preprocessing, n_classes)


@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize(
    "model_class",
    [
        linear_model.RidgeClassifier,
        linear_model.RidgeClassifierCV,
        linear_model.SGDClassifier,
        linear_model.Perceptron,
        linear_model.PassiveAggressiveClassifier,
        svm.NuSVC,
        neighbors.KNeighborsClassifier,
        neural_network.MLPClassifier,
        ensemble.AdaBoostClassifier,
        ensemble.BaggingClassifier,
        ensemble.VotingClassifier,
        ensemble.StackingClassifier,
    ],
    ids=lambda cls: cls.__name__,
)
def test_additional_classifier_nonconsecutive_labels(model_class, with_preprocessing):
    _check_model_workflow(model_class, with_preprocessing, n_classes=3, nonconsecutive_labels=True)


@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
@pytest.mark.parametrize(
    "model_class, model_options",
    [
        pytest.param(svm.NuSVC, {"probability": True}, id="NuSVC-probabilities"),
        pytest.param(svm.NuSVC, {"probability": True, "decision_function_shape": "ovo"}, id="NuSVC-ovo-probabilities"),
        pytest.param(linear_model.SGDClassifier, {"loss": "log_loss"}, id="SGDClassifier-log-loss"),
        pytest.param(ensemble.StackingClassifier, {"final_estimator": svm.LinearSVC()}, id="StackingClassifier-scores"),
    ],
)
def test_additional_classifier_output_variants(model_class, model_options, n_classes, with_preprocessing):
    _check_model_workflow(model_class, with_preprocessing, n_classes, model_options)


@pytest.mark.parametrize("with_preprocessing", [False, True], ids=["raw", "column_transformer"])
@pytest.mark.parametrize(
    "model_options",
    [
        {"power": 0, "link": "identity"},
        {"power": 0, "link": "log"},
        {"power": 1, "link": "auto"},
        {"power": 1.5, "link": "auto"},
        {"power": 2, "link": "auto"},
        {"power": 3, "link": "auto"},
    ],
    ids=["normal", "normal-log", "poisson", "compound-poisson", "gamma", "inverse-gaussian"],
)
def test_tweedie_link_variants(model_options, with_preprocessing):
    _check_model_workflow(linear_model.TweedieRegressor, with_preprocessing, model_options=model_options)


def test_sklearn_feature_metadata_covers_registered_estimators():
    from shapash.utils.model_synoptic import dict_model_feature

    for model_type in sklearn_model:
        assert dict_model_feature[model_type] == ["length"]


@pytest.mark.parametrize("remainder", ["drop", "passthrough"])
@pytest.mark.parametrize("as_list", [False, True])
def test_column_transformer_export_raw_schema(remainder, as_list, tmp_path):
    from shapash.utils.io import load_pickle

    x = pd.DataFrame(
        {
            "with_underscore": np.linspace(-2, 2, 30),
            "other": np.linspace(1, 5, 30),
            "category": ["A", "B", "C"] * 10,
            "unused": np.arange(30, dtype=float),
        },
        index=np.arange(100, 130),
    )
    encoder = ColumnTransformer(
        [
            ("scale_with_underscore", StandardScaler(), ["other", "with_underscore"]),
            ("category", OneHotEncoder(sparse_output=False), ["category"]),
        ],
        remainder=remainder,
    ).fit(x)
    encoded = pd.DataFrame(encoder.transform(x), index=x.index)
    encoded.columns = [f"col_{i}" for i in range(encoded.shape[1])]
    model = tree.DecisionTreeRegressor(random_state=0).fit(encoded, np.arange(len(x)))
    explainer = SmartExplainer(model=model, preprocessing=[encoder] if as_list else encoder)
    explainer.compile(x=encoded)
    original_columns = explainer.explainer.x_init.columns.copy()
    predictor = explainer.to_smartpredictor()
    assert explainer.explainer.x_init.columns.equals(original_columns)

    raw = x.iloc[:5, ::-1]
    predictor.add_input(x=raw)
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[:5]))
    pd.testing.assert_frame_equal(predictor.data["x_preprocessed"], encoded.iloc[:5])
    expected_contributions = predictor.data["contributions"].copy()

    predictor.add_input(x=explainer.explainer.x_init.iloc[:5])
    pd.testing.assert_frame_equal(predictor.data["contributions"], expected_contributions)
    predictor.add_input(x=raw.iloc[0].to_dict())
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[:1]))

    if remainder == "drop":
        predictor.add_input(x=raw.drop(columns="unused"))
        np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[:5]))
    with pytest.raises(ValueError, match="not declared"):
        predictor.add_input(x=raw.assign(unexpected=1))
    with pytest.raises(ValueError, match="missing raw input columns"):
        predictor.add_input(x=raw.drop(columns="category"))

    path = str(tmp_path / "predictor.pkl")
    predictor.save(path)
    restored = load_pickle(path)
    restored.add_input(x=raw)
    pd.testing.assert_frame_equal(restored.data["contributions"], expected_contributions)


def test_column_transformer_export_repeated_feature_and_metadata():
    x = pd.DataFrame({"value": np.linspace(-2, 2, 30)})
    encoder = ColumnTransformer([("first", StandardScaler(), ["value"]), ("second", StandardScaler(), ["value"])]).fit(
        x
    )
    encoded = pd.DataFrame(encoder.transform(x), columns=["col_0", "col_1"])
    model = tree.DecisionTreeRegressor(random_state=0).fit(encoded, np.arange(len(x)))
    explainer = SmartExplainer(
        model=model,
        preprocessing=encoder,
        features_dict={"first_value": "First value", "second_value": "Second value"},
        features_groups={"both": ["first_value", "second_value"]},
    )
    explainer.compile(x=encoded)
    predictor = explainer.to_smartpredictor()
    predictor.add_input(x=x.iloc[:5])
    assert predictor.features_dict["first_value"] == "First value"
    assert predictor.features_groups == {"both": ["first_value", "second_value"]}
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[:5]))
    assert len(predictor.detail_contributions()) == 5
