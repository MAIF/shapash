import category_encoders as ce
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from shapash.backend._lime_input import LimeInput
from shapash.backend.lime_backend import LimeBackend
from shapash.utils.columntransformer_backend import get_col_mapping_ct, get_onehot_feature_counts
from shapash.utils.transform import inverse_transform


def _encoded(preprocessing, x):
    return pd.DataFrame(
        preprocessing.transform(x),
        index=x.index,
        columns=[f"encoded_{i}" for i in range(preprocessing.transform(x).shape[1])],
    )


def test_explicit_categorical_values_round_trip_and_extend_background():
    x = pd.DataFrame({"value": [1.5, 2.5, 3.5], "category=name": [0.25, 4.5, 9.75]}, index=[10, 20, 30])
    layout = LimeInput(x, None, ["category=name"], False)
    assert layout.categorical_indices == [1]
    pd.testing.assert_frame_equal(layout.decode(layout.background).set_axis(x.index), x)
    newer = pd.DataFrame({"value": [4.5], "category=name": [42.25]}, index=[50])
    pd.testing.assert_frame_equal(layout.decode(layout.encode(newer)).set_axis(newer.index), newer)
    np.testing.assert_array_equal(layout.background["category=name"], [0, 1, 2])


def test_nonnumeric_categories_are_inferred_without_dtype_coercion():
    x = pd.DataFrame({"value": [1.0, 2.0, 3.0], "category": ["red", "blue", None]})
    layout = LimeInput(x, None, None, False)
    assert layout.categorical_indices == [1]
    pd.testing.assert_frame_equal(layout.decode(layout.background), x)
    with pytest.raises(ValueError, match="finite numeric"):
        LimeInput(x, None, [], False)


@pytest.mark.parametrize("preprocessing", [None, []])
def test_pandas_categorical_dtype_survives_prediction_decoding(preprocessing):
    x = pd.DataFrame({"category": pd.Categorical(["red", "blue", None]), "value": [1.0, 2.0, 3.0]})
    layout = LimeInput(x, preprocessing, None, False)
    pd.testing.assert_frame_equal(layout.decode(layout.background), x)


@pytest.mark.parametrize("drop", [None, "first", "if_binary"])
@pytest.mark.parametrize("infrequent", [False, True])
def test_onehot_groups_follow_fitted_output_width(drop, infrequent):
    x = pd.DataFrame(
        {"category": ["red", "red", "red", "blue", "green", "yellow"], "flag": ["a", "b"] * 3, "value": range(6)}
    )
    onehot = OneHotEncoder(
        sparse_output=False, drop=drop, handle_unknown="ignore", min_frequency=2 if infrequent else None
    )
    preprocessing = ColumnTransformer(
        [("category", onehot, ["category", "flag"]), ("value", StandardScaler(), ["value"])]
    ).fit(x)
    encoded = _encoded(preprocessing, x)
    layout = LimeInput(encoded, preprocessing, [], False)
    assert layout.categorical_indices == [0, 1]
    assert len(layout.features) == 3
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)
    counts = get_onehot_feature_counts(preprocessing.named_transformers_["category"])
    assert sum(counts) == preprocessing.output_indices_["category"].stop
    mapping = get_col_mapping_ct(preprocessing, encoded)
    assert [len(mapping["category_category"]), len(mapping["category_flag"])] == counts
    assert mapping["value_value"] == [encoded.columns[-1]]
    np.testing.assert_array_equal(layout.background.columns, inverse_transform(encoded, preprocessing).columns)


@pytest.mark.parametrize(
    "explicit, expected",
    [(None, [1, 2]), ([], [2]), (["value"], [0, 2]), (["category"], [1, 2])],
)
def test_explicit_list_replaces_encoder_inference_using_original_names(explicit, expected):
    x = pd.DataFrame({"value": [1.0, 2.0, 3.0], "category": ["a", "b", "c"], "onehot": ["x", "y", "x"]})
    preprocessing = ColumnTransformer(
        [
            ("scale", StandardScaler(), ["value"]),
            ("ordinal", OrdinalEncoder(), ["category"]),
            ("onehot", OneHotEncoder(sparse_output=False), ["onehot"]),
        ]
    ).fit(x)
    encoded = _encoded(preprocessing, x)
    layout = LimeInput(encoded, preprocessing, explicit, False)
    assert layout.categorical_indices == expected
    with pytest.raises(ValueError, match="Unknown categorical"):
        LimeInput(encoded, preprocessing, ["ordinal_category"], False)


@pytest.mark.parametrize(
    "encoder_class", [ce.OrdinalEncoder, ce.OneHotEncoder, ce.BinaryEncoder, ce.BaseNEncoder, ce.TargetEncoder]
)
@pytest.mark.parametrize("nested", [False, True])
def test_category_encoders_preserve_blocks_and_numeric_passthrough(encoder_class, nested):
    x = pd.DataFrame({"category": ["a", "b", "c"] * 4, "value": np.arange(12, dtype=float)})
    encoder = encoder_class(cols=["category"])
    target = np.arange(len(x)) % 2
    if nested:
        preprocessing = ColumnTransformer([("mixed", encoder, ["category", "value"])]).fit(x, target)
        encoded = _encoded(preprocessing, x)
    else:
        preprocessing = encoder.fit(x, target)
        encoded = preprocessing.transform(x)
    layout = LimeInput(encoded, preprocessing, None, False)
    assert layout.categorical_indices == [0]
    assert len(layout.features) == 2
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)


def test_sequential_category_encoders_compose_feature_mappings():
    x = pd.DataFrame({"first": ["a", "b", "c"] * 3, "second": ["x", "y", "z"] * 3, "value": range(9)})
    first = ce.OneHotEncoder(cols=["first"]).fit(x)
    second = ce.OrdinalEncoder(cols=["second"]).fit(first.transform(x))
    encoded = second.transform(first.transform(x))
    layout = LimeInput(encoded, [first, second], None, False)
    assert layout.categorical_indices == [0, 1]
    assert layout.background.columns.tolist() == x.columns.tolist()
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)


@pytest.mark.parametrize("selector", [["category"], [False, True], [1], slice(1, 2), slice("category", "category")])
def test_column_transformer_selectors_and_remainder(selector):
    x = pd.DataFrame({"value": [1.0, 2.0, 3.0], "category": ["a", "b", "c"]})
    preprocessing = ColumnTransformer(
        [("onehot", OneHotEncoder(sparse_output=False), selector)], remainder="passthrough"
    ).fit(x)
    encoded = _encoded(preprocessing, x)
    layout = LimeInput(encoded, preprocessing, ["category"], False)
    assert layout.categorical_indices == [0]
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)


@pytest.mark.parametrize("as_list", [False, True])
def test_mapping_preprocessing_infers_categories_and_can_be_overridden(as_list):
    x = pd.DataFrame({"category": [0.25, 4.5, 9.75], "value": [1.0, 2.0, 3.0]})
    mapping = {
        "col": "category",
        "mapping": pd.Series([0.25, 4.5, 9.75], index=["red", "blue", "green"]),
        "data_type": "object",
    }
    preprocessing = [mapping] if as_list else mapping
    layout = LimeInput(x, preprocessing, None, False)
    assert layout.categorical_indices == [0]
    pd.testing.assert_frame_equal(layout.decode(layout.background), x)
    assert LimeInput(x, preprocessing, [], False).categorical_indices == []


def test_invalid_onehot_values_are_rejected_but_dropped_baseline_is_valid():
    x = pd.DataFrame({"category": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
    preprocessing = ColumnTransformer(
        [("onehot", OneHotEncoder(sparse_output=False), ["category"])], remainder="passthrough"
    ).fit(x)
    encoded = _encoded(preprocessing, x)
    for values in ([1, 1, 0], [0, 0, 0], [0.5, 0.5, 0]):
        invalid = encoded.copy()
        invalid.iloc[0, :3] = values
        with pytest.raises(ValueError, match="Invalid one-hot"):
            LimeInput(invalid, preprocessing, [], False)
    dropped = ColumnTransformer(
        [("onehot", OneHotEncoder(sparse_output=False, drop="first"), ["category"])], remainder="passthrough"
    ).fit(x)
    encoded = _encoded(dropped, x)
    layout = LimeInput(encoded, dropped, [], False)
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)


def test_schema_and_categorical_model_constraints_are_explicit():
    x = pd.DataFrame({"first": [0, 1], "second": [1, 0]})
    with pytest.raises(ValueError, match="every input feature"):
        LimeInput(x, None, ["first"], True)
    with pytest.raises(ValueError, match="Unknown categorical"):
        LimeInput(x, None, ["missing"], False)
    layout = LimeInput(x, None, ["first"], False)
    with pytest.raises(ValueError, match="same encoded columns"):
        layout.encode(x.iloc[:, ::-1])
    with pytest.raises(ValueError, match="integer category indices"):
        layout.decode(np.array([[0.5, 1]]))
    with pytest.raises(ValueError, match="outside the observed"):
        layout.decode(np.array([[42, 1]]))


def test_fully_dropped_constant_and_unused_branches():
    x = pd.DataFrame({"constant": ["a"] * 4, "value": [1.0, 2.0, 3.0, 4.0], "unused": range(4)})
    preprocessing = ColumnTransformer(
        [
            ("constant", OneHotEncoder(sparse_output=False, drop="first"), ["constant"]),
            ("empty", OrdinalEncoder(), []),
            ("value", StandardScaler(), ["value"]),
            ("unused", "drop", ["unused"]),
        ]
    ).fit(x)
    encoded = _encoded(preprocessing, x)
    layout = LimeInput(encoded, preprocessing, [], False)
    assert layout.background.columns.tolist() == ["constant_constant", "value_value"]
    assert layout.features[0].columns == []
    assert layout.background["constant_constant"].eq(0).all()
    pd.testing.assert_frame_equal(layout.decode(layout.background), encoded)
    restored = inverse_transform(encoded, preprocessing)
    assert restored["constant_constant"].tolist() == ["a"] * 4
    assert get_col_mapping_ct(preprocessing, encoded)["constant_constant"] == []
    with pytest.raises(ValueError, match="Unknown categorical"):
        LimeInput(encoded, preprocessing, ["unused"], False)


def test_repeated_categorical_source_branches_are_rejected():
    x = pd.DataFrame({"category": ["red", "blue", "green"]})
    preprocessing = ColumnTransformer(
        [
            ("ordinal", OrdinalEncoder(), ["category"]),
            ("onehot", OneHotEncoder(sparse_output=False), ["category"]),
        ]
    ).fit(x)
    with pytest.raises(ValueError, match="same categorical source"):
        LimeInput(_encoded(preprocessing, x), preprocessing, [], False)


def test_infrequent_unknown_categories_restore_a_complete_group():
    x = pd.DataFrame({"category": ["red"] * 5 + ["blue", "green"], "value": np.arange(7, dtype=float)})
    preprocessing = ColumnTransformer(
        [
            (
                "onehot",
                OneHotEncoder(sparse_output=False, min_frequency=2, handle_unknown="infrequent_if_exist"),
                ["category"],
            )
        ],
        remainder="passthrough",
    ).fit(x)
    layout = LimeInput(_encoded(preprocessing, x), preprocessing, None, False)
    newer = _encoded(preprocessing, pd.DataFrame({"category": ["unknown"], "value": [8.0]}))
    pd.testing.assert_frame_equal(layout.decode(layout.encode(newer)), newer)


@pytest.mark.parametrize("invalid", ["first", [0], ["first", "first"]])
def test_categorical_feature_argument_is_validated(invalid):
    class Model:
        def predict(self, x):
            return np.zeros(len(x))

    with pytest.raises(ValueError, match="list of unique feature names"):
        LimeBackend(Model(), categorical_features=invalid)
