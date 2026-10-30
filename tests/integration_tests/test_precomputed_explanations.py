"""Model-free explanation workflows with explicit task and output metadata."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shapash import SmartExplainer
from shapash.backend._precomputed_backend import PrecomputedBackend


@pytest.fixture
def data():
    return pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "category": ["a", "b", "a", "b"]}, index=[10, 20, 30, 40])


@pytest.mark.parametrize("classes", [None, [3, 8], [3, 8, 42], ["no", "yes"]])
@pytest.mark.parametrize("preprocess", [False, True])
@pytest.mark.parametrize("as_array", [False, True])
def test_model_free_workflow(data, classes, preprocess, as_array, tmp_path):
    task = "classification" if classes else "regression"
    x = data.copy()
    x["category"] = [0.0, 1.0, 0.0, 1.0]
    preprocessing = None
    if preprocess:
        preprocessing = ColumnTransformer(
            [
                ("scale", StandardScaler(), ["value"]),
                ("onehot", OneHotEncoder(sparse_output=False), ["category"]),
            ]
        ).fit(data)
        x = pd.DataFrame(preprocessing.transform(data), columns=["col_0", "col_1", "col_2"], index=data.index)
    values = np.random.default_rng(42).normal(size=x.shape)
    frame = pd.DataFrame(values, index=x.index, columns=x.columns)
    contributions = values if as_array else frame
    proba = None
    base = 0.5
    if classes:
        contributions = [contributions * (i + 1) for i in range(len(classes))]
        proba = pd.DataFrame(np.full((len(x), len(classes)), 1 / len(classes)), index=x.index, columns=classes)
        prediction = pd.Series([classes[i % len(classes)] for i in range(len(x))], index=x.index, name="prediction")
        base = np.arange(len(classes), dtype=float) / 10
    else:
        prediction = frame.sum(axis=1) + base
    xpl = SmartExplainer(task=task, classes=classes, preprocessing=preprocessing)
    with patch("shapash.explainer.explainer.get_backend_cls_from_name") as backend_selection:
        xpl.compile(
            x=x,
            contributions={"contributions": contributions, "base_values": base},
            y_pred=prediction,
            proba_values=proba,
            y_target=prediction,
        )
        backend_selection.assert_not_called()
    compiled = xpl.explainer
    assert compiled.model is None
    assert isinstance(compiled.backend, PrecomputedBackend)
    assert compiled._classes == classes
    frames = compiled.contributions if classes else [compiled.contributions]
    assert len(frames) == (len(classes) if classes else 1)
    for i, result in enumerate(frames):
        assert result.shape == (len(x), 2)
        assert result.index.equals(x.index)
        np.testing.assert_allclose(result.sum(axis=1), values.sum(axis=1) * (i + 1))
    assert len(xpl.to_pandas(proba=bool(classes))) == len(x)
    assert len(xpl.plot.features_importance().data) > 0
    label = classes[-1] if classes else None
    assert len(xpl.plot.local_plot(index=10, label=label).data) > 0
    assert len(xpl.plot.local_plot(index=10, label=label, plot_type="waterfall").data) > 0
    assert len(xpl.plot.contribution_plot(col=compiled.x_init.columns[0], label=label).data) > 0
    path = str(tmp_path / "precomputed.pkl")
    xpl.save(path)
    restored = SmartExplainer.load(path)
    pd.testing.assert_frame_equal(restored.to_pandas(proba=bool(classes)), xpl.to_pandas(proba=bool(classes)))
    assert len(restored.plot.features_importance().data) > 0
    xpl.add(y_pred=prediction, proba_values=proba)
    with pytest.raises(ValueError, match="requires a model"):
        xpl.to_smartpredictor()
    with pytest.raises(ValueError, match="requires a model"):
        xpl.generate_report("unused.html")
    xpl.init_app()
    assert xpl.smartapp is not None


@pytest.mark.parametrize("format", ["frame", "array", "tensor"])
def test_model_free_binary_single_output_and_multiclass_tensor(format):
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    values = x * 0.1
    classes = [3, 8, 42] if format == "tensor" else [3, 8]
    contributions = np.stack([values * (i + 1) for i in range(3)], axis=-1) if format == "tensor" else values
    if format == "array":
        contributions = values.to_numpy()
    xpl = SmartExplainer(task="classification", classes=classes)
    xpl.compile(x=x, contributions=contributions, y_pred=pd.Series([3, 8]))
    frames = xpl.explainer.contributions
    np.testing.assert_allclose(frames[-1], values * (3 if format == "tensor" else 1))
    if format != "tensor":
        np.testing.assert_allclose(frames[0], -values)
    assert len(xpl.to_pandas()) == 2
    with pytest.raises(ValueError, match="supply proba_values"):
        xpl.to_pandas(proba=True)
    with pytest.raises(ValueError, match="require explicit base_values"):
        xpl.plot.local_plot(index=0, label=8, plot_type="waterfall")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "task must"),
        ({"task": "invalid"}, "task must"),
        ({"task": "classification"}, "at least two classes"),
        ({"task": "classification", "classes": [1]}, "at least two classes"),
        ({"task": "classification", "classes": [1, 1]}, "unique"),
        ({"task": "classification", "classes": [1, "x"]}, "mix numeric"),
        ({"task": "classification", "classes": [0, np.nan]}, "finite"),
        ({"task": "regression", "classes": [0, 1]}, "does not accept classes"),
        ({"task": "regression", "backend": "lime"}, "backend configuration"),
        ({"task": "regression", "neural_explainer": "auto"}, "backend configuration"),
    ],
)
def test_invalid_model_free_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SmartExplainer(**kwargs)


@pytest.mark.parametrize("missing", ["contributions", "y_pred"])
def test_model_free_requires_contributions_and_predictions(missing):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    options = {"x": x, "contributions": x, "y_pred": x.iloc[:, 0]}
    del options[missing]
    with pytest.raises(ValueError, match="requires both"):
        SmartExplainer(task="regression").compile(**options)


@pytest.mark.parametrize(
    "output, message",
    [
        (pd.Series([0, 9]), "not present in classes"),
        (pd.Series([0, 1], index=[2, 3]), "same index"),
        (pd.DataFrame([[0, 1], [1, 0]]), "exactly one column"),
    ],
)
def test_model_free_invalid_labels(output, message):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match=message):
        SmartExplainer(task="classification", classes=[0, 1]).compile(x=x, contributions=x, y_pred=output)


@pytest.mark.parametrize(
    "probabilities, message",
    [
        (pd.DataFrame({1: [0.5, 0.5], 0: [0.5, 0.5]}), "columns must match"),
        (pd.DataFrame({0: [0.5, 0.5], 1: [0.5, 0.5]}, index=[2, 3]), "same index"),
        (pd.DataFrame({0: [1.5, 0.5], 1: [-0.5, 0.5]}), "finite probabilities"),
        (pd.DataFrame({0: [0.5, np.nan], 1: [0.5, 0.5]}), "finite probabilities"),
        (pd.DataFrame({0: [0.1, 0.2], 1: [0.5, 0.5]}), "summing to one"),
    ],
)
def test_model_free_invalid_probabilities(probabilities, message):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    xpl = SmartExplainer(task="classification", classes=[0, 1])
    with pytest.raises(ValueError, match=message):
        xpl.compile(x=x, contributions=x, y_pred=pd.Series([0, 1]), proba_values=probabilities)


@pytest.mark.parametrize("operation", ["predict", "compute_features_stability", "get_interaction_values"])
def test_model_free_unavailable_computation(operation):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    xpl = SmartExplainer(task="regression")
    xpl.compile(x=x, contributions=x, y_pred=x.iloc[:, 0])
    options = {"selection": [0]} if operation == "compute_features_stability" else {}
    with pytest.raises(ValueError, match="requires a model"):
        getattr(xpl.explainer, operation)(**options)


@pytest.mark.parametrize("base", [np.nan, [1.0, 2.0, 3.0], pd.Series([1.0, 2.0], index=[2, 3])])
def test_model_free_invalid_baselines(base):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="base_values"):
        SmartExplainer(task="regression").compile(
            x=x, contributions={"contributions": x, "base_values": base}, y_pred=x.iloc[:, 0]
        )


@pytest.mark.parametrize("values", [np.array([[np.inf], [1.0]]), np.array([["a"], ["b"]])])
def test_model_free_invalid_contributions(values):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="finite numeric"):
        SmartExplainer(task="regression").compile(x=x, contributions=values, y_pred=x.iloc[:, 0])


@pytest.mark.parametrize("kind", ["index", "columns", "shape", "classes"])
def test_model_free_rejects_misaligned_contributions(kind):
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    contributions = x.copy()
    options = {"task": "regression"}
    if kind == "index":
        contributions.index = [2, 3]
    elif kind == "columns":
        contributions = contributions[["b", "a"]]
    elif kind == "shape":
        contributions = contributions.iloc[:1]
    else:
        options = {"task": "classification", "classes": [0, 1, 2]}
    with pytest.raises(ValueError):
        SmartExplainer(**options).compile(x=x, contributions=contributions, y_pred=pd.Series([0, 1]))


@pytest.mark.parametrize(
    "x",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": [1.0, 2.0]}, index=[0, 0]),
        pd.DataFrame([[1.0, 2.0]], columns=["a", "a"]),
    ],
)
def test_model_free_requires_nonempty_unambiguous_input(x):
    with pytest.raises(ValueError, match="nonempty DataFrame"):
        SmartExplainer(task="regression").compile(x=x, contributions=x, y_pred=pd.Series(np.zeros(len(x))))


def test_model_free_groups_filter_updates_and_baselines():
    x = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}, index=["first", "second"])
    classes = ["no", "yes"]
    prediction = pd.Series(["yes", "no"], index=x.index)
    baselines = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=x.index, columns=classes)
    xpl = SmartExplainer(task="classification", classes=classes, features_groups={"ab": ["a", "b"]})
    xpl.compile(x=x, contributions={"contributions": [-x, x], "base_values": baselines}, y_pred=prediction)
    np.testing.assert_array_equal(xpl.explainer.contributions_groups[1]["ab"], x["a"] + x["b"])
    assert xpl.plot._get_waterfall_base_value(["second"], label_num=1) == 0.4
    xpl.filter(max_contrib=1)
    assert len(xpl.to_pandas()) == 2
    probabilities = pd.DataFrame({"no": [0.2, 0.8], "yes": [0.8, 0.2]}, index=x.index)
    xpl.add(proba_values=probabilities, y_target=prediction)
    assert len(xpl.to_pandas(proba=True)) == 2
    np.testing.assert_allclose(xpl.explainer.prediction_error.iloc[:, 0], [0.2, 0.2])
    with pytest.raises(ValueError, match="not present in classes"):
        xpl.add(y_pred=pd.Series(["missing", "no"], index=x.index))
    with pytest.raises(ValueError, match="columns must match"):
        xpl.add(proba_values=probabilities[["yes", "no"]])
    pd.testing.assert_frame_equal(xpl.explainer.proba_values, probabilities)


@pytest.mark.parametrize("base", [np.array([0.1]), np.zeros((2, 3)), pd.DataFrame({1: [0.1, 0.2], 0: [0.3, 0.4]})])
def test_model_free_classification_rejects_invalid_baseline_shape_or_order(base):
    x = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="base_values"):
        SmartExplainer(task="classification", classes=[0, 1]).compile(
            x=x, contributions={"contributions": x, "base_values": base}, y_pred=pd.Series([0, 1])
        )


def test_model_free_regression_does_not_invent_probabilities_or_baselines():
    x = pd.DataFrame({"a": [1.0, 2.0]})
    xpl = SmartExplainer(task="regression")
    xpl.compile(x=x, contributions=x, y_pred=x.iloc[:, 0])
    assert len(xpl.plot.local_plot(index=0).data) > 0
    with pytest.raises(ValueError, match="require explicit base_values"):
        xpl.plot.local_plot(index=0, plot_type="waterfall")
    with pytest.raises(ValueError, match="only accepted for classification"):
        xpl.add(proba_values=x)
    with pytest.raises(ValueError, match="finite numeric"):
        xpl.add(y_pred=pd.Series([np.nan, 2.0]))


def test_model_metadata_remains_authoritative():
    from sklearn.linear_model import LinearRegression

    x = pd.DataFrame({"a": [1.0, 2.0]})
    model = LinearRegression().fit(x, x["a"])
    for metadata in ({"task": "regression"}, {"classes": [0, 1]}):
        with pytest.raises(ValueError, match="only accepted without a model"):
            SmartExplainer(model=model, **metadata)
