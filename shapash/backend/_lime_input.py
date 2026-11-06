"""Categorical and grouped representations of encoded inputs for tabular LIME."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from shapash.utils.category_encoder_backend import (
    dummies_category_encoder,
    get_col_mapping_ce,
    supported_category_encoder,
)
from shapash.utils.columntransformer_backend import (
    get_column_transformer_columns,
    get_onehot_feature_counts,
    no_dummies_sklearn,
)
from shapash.utils.transform import preprocessing_tolist


@dataclass
class _Feature:
    name: Any
    source: Any
    columns: list[Any]
    categorical: bool = False
    structural: bool = False
    allow_zero: bool | None = None
    values: pd.DataFrame | None = None


def _encoder_columns(encoder: Any, inputs: list[Any]) -> dict[Any, list[Any]]:
    outputs = list(encoder.get_feature_names_out())
    mapping = get_col_mapping_ce(encoder)
    return {name: [column for column in mapping.get(name, [name]) if column in outputs] for name in inputs}


def _column_transformer_features(
    encoder: ColumnTransformer,
    columns: pd.Index,
) -> list[_Feature]:
    features = []
    for branch, transformer, selector in encoder.transformers_:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        inputs = get_column_transformer_columns(encoder, selector)
        if not inputs:
            continue
        output = encoder.output_indices_[branch]
        branch_columns = columns[output].tolist()
        if isinstance(transformer, OneHotEncoder):
            position = 0
            for i, (source, count) in enumerate(zip(inputs, get_onehot_feature_counts(transformer), strict=True)):
                dropped = transformer.drop_idx_ is not None and transformer.drop_idx_[i] is not None
                infrequent = (
                    transformer.infrequent_categories_[i]
                    if transformer.min_frequency is not None or transformer.max_categories is not None
                    else None
                )
                allow_zero = (
                    dropped
                    or transformer.handle_unknown == "ignore"
                    or (transformer.handle_unknown in ("infrequent_if_exist", "warn") and infrequent is None)
                )
                features.append(
                    _Feature(
                        f"{branch}_{source}",
                        source,
                        branch_columns[position : position + count],
                        categorical=True,
                        structural=True,
                        allow_zero=allow_zero,
                    )
                )
                position += count
            if position != len(branch_columns):
                raise ValueError("OneHotEncoder metadata does not match the encoded LIME input columns.")
        elif str(type(transformer)) in supported_category_encoder:
            outputs = list(transformer.get_feature_names_out())
            positions = dict(zip(outputs, branch_columns, strict=True))
            mapping = _encoder_columns(transformer, inputs)
            selected = get_col_mapping_ce(transformer)
            for source, encoded in mapping.items():
                features.append(
                    _Feature(
                        f"{branch}_{source}",
                        source,
                        [positions[column] for column in encoded],
                        categorical=source in selected,
                        structural=source in selected and str(type(transformer)) in dummies_category_encoder,
                    )
                )
        elif isinstance(transformer, FunctionTransformer) or str(type(transformer)) in no_dummies_sklearn:
            if len(inputs) != len(branch_columns):
                raise ValueError("LIME cannot infer a feature mapping for this ColumnTransformer branch.")
            for source, encoded in zip(inputs, branch_columns, strict=True):
                name = encoded if branch == "remainder" else f"{branch}_{source}"
                features.append(
                    _Feature(
                        name,
                        source,
                        [encoded],
                        categorical=isinstance(transformer, OrdinalEncoder),
                    )
                )
        else:
            raise ValueError(f"Grouped LIME does not support preprocessing metadata for {type(transformer).__name__}.")
    return features


def _features(background: pd.DataFrame, preprocessing: Any) -> list[_Feature]:
    if preprocessing is None or (isinstance(preprocessing, list) and not preprocessing):
        return [
            _Feature(name, name, [name], categorical=not is_numeric_dtype(background[name].dtype))
            for name in background.columns
        ]
    steps = preprocessing_tolist(preprocessing)
    transformers = [step for step in steps if isinstance(step, ColumnTransformer)]
    if transformers:
        if len(steps) != 1:
            raise ValueError("Grouped LIME requires a single ColumnTransformer preprocessing step.")
        return _column_transformer_features(transformers[0], background.columns)

    encoders = [step for step in steps if str(type(step)) in supported_category_encoder]
    inputs = list(encoders[0].feature_names_in_) if encoders else background.columns.tolist()
    features = [_Feature(name, name, [name]) for name in inputs]
    for step in steps:
        if str(type(step)) in supported_category_encoder:
            mapping = _encoder_columns(step, list(step.feature_names_in_))
            selected = get_col_mapping_ce(step)
            for feature in features:
                categorical = any(column in selected for column in feature.columns)
                feature.categorical |= categorical
                feature.structural |= categorical and str(type(step)) in dummies_category_encoder
                feature.columns = [output for column in feature.columns for output in mapping.get(column, [column])]
        elif isinstance(step, list):
            mapped_sources = {mapping["col"] for mapping in step}
            for feature in features:
                feature.categorical |= feature.source in mapped_sources
        else:
            raise ValueError(f"Grouped LIME cannot infer the preprocessing metadata of {type(step).__name__}.")
    return features


class LimeInput:
    """Represent each encoded categorical block as one categorical LIME feature."""

    def __init__(
        self,
        background: pd.DataFrame,
        preprocessing: Any,
        categorical_features: list[str] | None,
        all_categorical: bool,
    ) -> None:
        if not isinstance(background, pd.DataFrame) or background.empty or not background.columns.is_unique:
            raise ValueError("LIME background must be a nonempty DataFrame with unique columns.")
        self.columns = background.columns.copy()
        self.features = _features(background, preprocessing)
        by_name = {feature.name: feature for feature in self.features}
        if len(by_name) != len(self.features):
            raise ValueError("LIME preprocessing metadata contains duplicate original feature names.")
        encoded = [column for feature in self.features for column in feature.columns]
        if len(encoded) != len(set(encoded)) or set(encoded) != set(self.columns):
            raise ValueError("LIME preprocessing metadata must cover every encoded column exactly once.")
        sources = {feature.source for feature in self.features}
        if categorical_features is not None:
            unknown = set(categorical_features) - sources
            if unknown:
                raise ValueError(f"Unknown categorical features before preprocessing: {sorted(unknown)}.")
        for feature in self.features:
            if categorical_features is not None:
                feature.categorical = feature.source in categorical_features
                if all_categorical and not feature.categorical and not feature.structural:
                    raise ValueError("CategoricalNB requires every input feature in categorical_features.")
            else:
                feature.categorical |= all_categorical
            feature.categorical |= feature.structural
        repeated = {
            feature.source
            for feature in self.features
            if feature.categorical and sum(other.source == feature.source for other in self.features) > 1
        }
        if repeated:
            raise ValueError(
                "Grouped LIME cannot independently transform the same categorical source in multiple branches: "
                f"{sorted(repeated)}."
            )
        self.categorical_indices = [i for i, feature in enumerate(self.features) if feature.categorical]
        self.background = self.encode(background)

    def encode(self, x: pd.DataFrame) -> pd.DataFrame:
        """Convert observed model inputs to the LIME feature space without losing category values."""
        if not x.columns.equals(self.columns):
            raise ValueError("LIME inputs and background must have the same encoded columns in the same order.")
        result = {}
        for feature in self.features:
            values = x.loc[:, feature.columns]
            if feature.allow_zero is not None:
                array = values.to_numpy()
                sums = array.sum(axis=1)
                valid_sum = np.isin(sums, [0, 1]) if feature.allow_zero else sums == 1
                if not np.isin(array, [0, 1]).all() or not valid_sum.all():
                    raise ValueError(f"Invalid one-hot values for feature {feature.source!r}.")
            if feature.categorical:
                if not feature.columns:
                    result[feature.name] = np.zeros(len(x), dtype=int)
                    continue
                feature.values = (
                    values.drop_duplicates(ignore_index=True)
                    if feature.values is None
                    else pd.concat([feature.values, values], ignore_index=True).drop_duplicates(ignore_index=True)
                )
                categories = pd.MultiIndex.from_frame(feature.values)
                result[feature.name] = categories.get_indexer(pd.MultiIndex.from_frame(values))
            else:
                if len(feature.columns) != 1:
                    raise ValueError(f"LIME requires categorical grouping for feature {feature.source!r}.")
                column = values.iloc[:, 0]
                if (
                    not is_numeric_dtype(column.dtype)
                    or not np.isfinite(column.to_numpy(dtype=float, na_value=np.nan)).all()
                ):
                    raise ValueError(f"Continuous LIME feature {feature.source!r} must contain finite numeric values.")
                result[feature.name] = column.to_numpy()
        return pd.DataFrame(result, index=x.index)

    def decode(self, values: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """Restore complete encoded blocks before calling the prediction model."""
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != len(self.features):
            raise ValueError("Unexpected LIME perturbation shape.")
        result = {}
        for i, feature in enumerate(self.features):
            if feature.categorical:
                codes = array[:, i]
                if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all():
                    raise ValueError("LIME categorical perturbations must be integer category indices.")
                if not feature.columns:
                    if not np.equal(codes, 0).all():
                        raise ValueError("Invalid LIME category for a dropped constant feature.")
                    continue
                if feature.values is None or (codes < 0).any() or (codes >= len(feature.values)).any():
                    raise ValueError("LIME categorical perturbation is outside the observed category mapping.")
                restored = feature.values.iloc[codes.astype(int)]
                for column in feature.columns:
                    result[column] = restored[column].reset_index(drop=True)
            else:
                result[feature.columns[0]] = array[:, i]
        return pd.DataFrame(result, columns=self.columns)
