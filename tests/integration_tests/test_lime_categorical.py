import pickle
from unittest.mock import patch

import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from shapash import SmartExplainer
from shapash.utils.columntransformer_backend import get_onehot_feature_counts
from shapash.utils.transform import inverse_transform


pytest.importorskip("lime")


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("drop", [None, "first"])
@pytest.mark.parametrize("infrequent", [False, True])
def test_grouped_onehot_perturbations_export_and_serialization(task, drop, infrequent):
    x = pd.DataFrame(
        {
            "value": np.linspace(-2, 2, 48),
            "ordinal": ["low", "medium", "high"] * 16,
            "color": ["red", "red", "red", "blue", "blue", "green", "yellow", "black"] * 6,
            "flag": ["yes", "no"] * 24,
            "unchanged": np.linspace(10, 30, 48),
        },
        index=np.arange(100, 148),
    )
    preprocessing = ColumnTransformer(
        [
            ("scale", StandardScaler(), ["value"]),
            ("ordinal", OrdinalEncoder(), ["ordinal"]),
            (
                "onehot",
                OneHotEncoder(
                    sparse_output=False,
                    drop=drop,
                    handle_unknown="ignore",
                    min_frequency=7 if infrequent else None,
                ),
                ["color", "flag"],
            ),
        ],
        remainder="passthrough",
    ).fit(x)
    encoded = pd.DataFrame(preprocessing.transform(x), index=x.index)
    encoded.columns = [f"col_{i}" for i in range(encoded.shape[1])]
    if task == "regression":
        model = Ridge().fit(encoded, np.sin(x["value"]) + np.arange(len(x)) % 3)
        predict = model.predict
        method = "predict"
    else:
        target = np.array([3, 8, 42])[np.arange(len(x)) % (2 if task == "binary" else 3)]
        model = LogisticRegression().fit(encoded, target)
        predict = model.predict_proba
        method = "predict_proba"

    start = preprocessing.output_indices_["onehot"].start
    valid_groups = []
    for count in get_onehot_feature_counts(preprocessing.named_transformers_["onehot"]):
        valid_groups.append(
            (slice(start, start + count), set(map(tuple, encoded.iloc[:, start : start + count].values)))
        )
        start += count
    evaluated = []

    def checked_prediction(values):
        assert isinstance(values, pd.DataFrame)
        assert values.columns.equals(encoded.columns)
        for indices, valid in valid_groups:
            assert set(map(tuple, values.iloc[:, indices].values)) <= valid
        evaluated.append(len(values))
        return predict(values)

    explainer = SmartExplainer(
        model=model, preprocessing=preprocessing, backend="lime", categorical_features=["ordinal"]
    )
    sample = encoded.iloc[:6]
    with patch.object(model, method, checked_prediction):
        explainer.compile(x=sample)
        predictor = explainer.to_smartpredictor()
        predictor.add_input(x=x.iloc[:6])
    assert max(evaluated) > len(sample)
    assert explainer.explainer.backend._lime_input.categorical_indices == [1, 2, 3]
    frames = explainer.explainer.contributions
    for frame in frames if isinstance(frames, list) else [frames]:
        assert frame.columns.equals(inverse_transform(sample, preprocessing).columns)
        assert frame.shape == x.iloc[:6].shape
        assert np.isfinite(frame.to_numpy()).all()
    subset = sample.index[[0, 2]].tolist()
    selected = explainer.explainer.backend.get_local_contributions(
        sample, explainer.explainer.explain_data, subset=subset
    )
    for actual, expected in zip(
        selected if isinstance(selected, list) else [selected],
        frames if isinstance(frames, list) else [frames],
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual, expected.loc[subset])
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(sample))
    pd.testing.assert_frame_equal(predictor.data["x_preprocessed"], sample)
    assert len(explainer.to_pandas()) == len(sample)
    assert len(explainer.plot.features_importance().data) > 0
    assert len(predictor.detail_contributions()) == len(sample)

    restored = pickle.loads(pickle.dumps(predictor))
    newer = x.iloc[6:10]
    restored.add_input(x=newer)
    np.testing.assert_array_equal(restored.predict().iloc[:, 0], model.predict(encoded.iloc[6:10]))
    assert restored.backend.categorical_features == ["ordinal"]
    assert restored.data["contributions"].shape == newer.shape
    assert np.isfinite(restored.data["contributions"].to_numpy(dtype=float)).all()

    supplied = SmartExplainer(
        model=model, preprocessing=preprocessing, backend="lime", categorical_features=["ordinal"]
    )
    supplied.compile(x=sample, contributions=explainer.explainer.explain_data)
    supplied_frames = supplied.explainer.contributions
    for actual, expected in zip(
        supplied_frames if isinstance(supplied_frames, list) else [supplied_frames],
        frames if isinstance(frames, list) else [frames],
        strict=True,
    ):
        pd.testing.assert_frame_equal(actual, expected)


def test_unknown_onehot_category_keeps_valid_all_zero_block_after_reload():
    x = pd.DataFrame({"category": ["red", "blue", "green"] * 8, "value": np.arange(24, dtype=float)})
    preprocessing = ColumnTransformer(
        [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["category"])],
        remainder="passthrough",
    ).fit(x)
    encoded = pd.DataFrame(preprocessing.transform(x), columns=["col_0", "col_1", "col_2", "col_3"])
    model = LogisticRegression().fit(encoded, np.arange(len(x)) % 2)
    explainer = SmartExplainer(model=model, preprocessing=preprocessing, backend="lime", categorical_features=[])
    explainer.compile(x=encoded.iloc[:4])
    predictor = pickle.loads(pickle.dumps(explainer.to_smartpredictor()))
    newer = pd.DataFrame({"category": ["unknown"], "value": [5.0]})
    predictor.add_input(x=newer)
    np.testing.assert_array_equal(predictor.data["x_preprocessed"].iloc[:, :3], np.zeros((1, 3)))
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(predictor.data["x_preprocessed"]))
    assert np.isfinite(predictor.data["contributions"].to_numpy(dtype=float)).all()


@pytest.mark.parametrize(
    "encoder_class", [ce.OrdinalEncoder, ce.OneHotEncoder, ce.BinaryEncoder, ce.BaseNEncoder, ce.TargetEncoder]
)
@pytest.mark.parametrize("nested", [False, True])
def test_category_encoder_inference_samples_observed_encoded_blocks(encoder_class, nested):
    x = pd.DataFrame({"category": ["red", "blue", "green"] * 12, "value": np.arange(36, dtype=float)})
    target = np.array([0, 1, 3])[np.arange(len(x)) % 3]
    encoder = encoder_class(cols=["category"]).fit(x, target)
    if nested:
        encoder = ColumnTransformer([("encoding", encoder, x.columns.tolist())]).fit(x, target)
        values = encoder.transform(x)
        encoded = pd.DataFrame(values, columns=[f"col_{i}" for i in range(values.shape[1])])
    else:
        encoded = encoder.transform(x)
    model = LogisticRegression().fit(encoded, target)
    category_columns = encoded.columns[:-1].tolist()
    valid = set(map(tuple, encoded[category_columns].values))
    predict = model.predict_proba
    evaluated = []

    def checked_prediction(values):
        assert set(map(tuple, values[category_columns].values)) <= valid
        evaluated.append(len(values))
        return predict(values)

    explainer = SmartExplainer(model=model, preprocessing=encoder, backend="lime")
    with patch.object(model, "predict_proba", checked_prediction):
        explainer.compile(x=encoded.iloc[:6])
    assert max(evaluated) > 6
    assert explainer.explainer.backend._lime_input.categorical_indices == [0]
    for frame in explainer.explainer.contributions:
        assert frame.columns.equals(inverse_transform(encoded, encoder).columns)
    predictor = pickle.loads(pickle.dumps(explainer.to_smartpredictor()))
    predictor.add_input(x=x.iloc[6:9])
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[6:9]))


def test_constant_onehot_group_has_zero_contribution_and_survives_export():
    x = pd.DataFrame(
        {"constant": ["a"] * 24, "constant_number": [1] * 24, "value": np.linspace(-1, 1, 24), "unused": np.arange(24)}
    )
    preprocessing = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(sparse_output=False, drop="first"), ["constant", "constant_number"]),
            ("empty", OrdinalEncoder(), []),
            ("value", StandardScaler(), ["value"]),
            ("unused", "drop", ["unused"]),
        ]
    ).fit(x)
    encoded = pd.DataFrame(preprocessing.transform(x), columns=["col_0"])
    model = Ridge().fit(encoded, x["value"] ** 2)
    explainer = SmartExplainer(model=model, preprocessing=preprocessing, backend="lime", categorical_features=[])
    explainer.compile(x=encoded.iloc[:6])
    assert explainer.explainer.contributions["onehot_constant"].eq(0).all()
    assert explainer.explainer.contributions["onehot_constant_number"].eq(0).all()
    assert inverse_transform(encoded, preprocessing)["onehot_constant_number"].eq(1).all()
    predictor = pickle.loads(pickle.dumps(explainer.to_smartpredictor()))
    predictor.add_input(x=x.iloc[6:9].drop(columns="unused"))
    np.testing.assert_array_equal(predictor.predict().iloc[:, 0], model.predict(encoded.iloc[6:9]))
    assert predictor.data["contributions"]["onehot_constant"].eq(0).all()


def test_explicit_categorical_numeric_column_without_preprocessing():
    x = pd.DataFrame({"category=name": [0.25, 4.5, 9.75] * 8, "value": np.linspace(-1, 1, 24)})
    model = LogisticRegression().fit(x, np.arange(len(x)) % 2)
    predict = model.predict_proba
    evaluated = []

    def checked_prediction(values):
        assert np.isin(values["category=name"], x["category=name"]).all()
        evaluated.append(len(values))
        return predict(values)

    explainer = SmartExplainer(model=model, backend="lime", categorical_features=["category=name"])
    with patch.object(model, "predict_proba", checked_prediction):
        explainer.compile(x=x.iloc[:6])
    assert max(evaluated) > 6
    assert explainer.explainer.backend._lime_input.categorical_indices == [0]
    restored = pickle.loads(pickle.dumps(explainer.to_smartpredictor()))
    restored.add_input(x=x.iloc[6:9])
    np.testing.assert_array_equal(restored.predict().iloc[:, 0], model.predict(x.iloc[6:9]))
