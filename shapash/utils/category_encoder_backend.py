"""
Category_encoder
"""

from typing import Any, Literal, TypeAlias

import category_encoders as ce
import numpy as np
import pandas as pd

EncoderMapping: TypeAlias = dict[str, Any]
CategoryEncoder: TypeAlias = (
    ce.OneHotEncoder | ce.OrdinalEncoder | ce.BaseNEncoder | ce.BinaryEncoder | ce.TargetEncoder
)
CategoryEncoding: TypeAlias = CategoryEncoder | list[EncoderMapping]

category_encoder_onehot: str = "<class 'category_encoders.one_hot.OneHotEncoder'>"
category_encoder_ordinal: str = "<class 'category_encoders.ordinal.OrdinalEncoder'>"
category_encoder_basen: str = "<class 'category_encoders.basen.BaseNEncoder'>"
category_encoder_binary: str = "<class 'category_encoders.binary.BinaryEncoder'>"
category_encoder_targetencoder: str = "<class 'category_encoders.target_encoder.TargetEncoder'>"

dummies_category_encoder: tuple[str, ...] = (category_encoder_onehot, category_encoder_binary, category_encoder_basen)

no_dummies_category_encoder: tuple[str, ...] = (category_encoder_ordinal, category_encoder_targetencoder)

supported_category_encoder: tuple[str, ...] = (
    category_encoder_onehot,
    category_encoder_binary,
    category_encoder_basen,
    category_encoder_ordinal,
    category_encoder_targetencoder,
)


def inv_transform_ce(x_in: pd.DataFrame, encoding: CategoryEncoding) -> pd.DataFrame:
    """
    Choose and apply the reversed transformation for the given encoding.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : CategoryEncoder or list of dict
        A category encoder (OrdinalEncoder/OneHotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
        or a list of mappings.

    Returns
    -------
    pandas.DataFrame
        The reversed transformation for the given encoding.
    """
    if isinstance(encoding, list):
        return inv_transform_ordinal(x_in, encoding)

    if str(type(encoding)) == category_encoder_ordinal:
        rst = inv_transform_ordinal(x_in, encoding.mapping)

    elif str(type(encoding)) == category_encoder_onehot:
        x = encoding.reverse_dummies(x_in, encoding.mapping)
        rst = inv_transform_ordinal(x, encoding.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_basen:
        x = reverse_basen(x_in, encoding)
        rst = inv_transform_ordinal(x, encoding.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_binary:
        x = reverse_basen(x_in, encoding)
        rst = inv_transform_ordinal(x, encoding.ordinal_encoder.mapping)

    elif str(type(encoding)) == category_encoder_targetencoder:
        rst = inv_transform_target(x_in, encoding)

    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no inverse done.")

    return rst


def inv_transform_target(x_in: pd.DataFrame, enc_target: ce.TargetEncoder) -> pd.DataFrame:
    """
    Reversed transformation for target encoded data using target encoded value.

    If a targeted value is linked to multiple label, we raise an exception.

    example of multi label :
    original mapping :
        target   value
           0.5    A
           0.5    B
           0.3    C
    new mapping :
        target value
          0.5   A / B
          0.3   C

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    enc_target : category_encoders.TargetEncoder
        TargetEncoder instance from category_encoders.

    Returns
    -------
    pandas.DataFrame
        The reversed dataframe.
    """
    for tgt_enc in enc_target.ordinal_encoder.mapping:
        name_target = tgt_enc.get("col")
        mapping_ordinal = enc_target.mapping[name_target]
        mapping_target = tgt_enc.get("mapping")
        if mapping_target is None:
            raise ValueError(f"Missing target mapping for column {name_target}.")
        reverse_target = pd.Series(mapping_target.index.values, index=mapping_target)
        rst_target = pd.concat([reverse_target, mapping_ordinal], axis=1, join="inner").fillna(value="NaN")
        aggregate = rst_target.groupby(1)[0].apply(lambda x: " / ".join(map(str, x)))
        if aggregate.shape[0] != rst_target.shape[0]:
            raise Exception("Multiple label found for the same value in TargetEncoder on col " + str(name_target) + ".")
            # print("Warning in inverse TargetEncoder - col " + str(name_target) + ": Multiple label for the same value, "
            #                                                                   "each label will be separate using : / ")

        data_type = tgt_enc.get("data_type") or "object"
        transco = {
            "col": name_target,
            "mapping": pd.Series(data=aggregate.index, index=aggregate.values),
            "data_type": data_type,
        }
        x_in = inv_transform_ordinal(x_in, [transco])
    return x_in


def inv_transform_ordinal(x_in: pd.DataFrame, encoding: list[EncoderMapping]) -> pd.DataFrame:
    """
    Reversed transformation based on ordinal category encoder.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : list of dict
        Mappings containing ``col``, ``mapping`` and optionally ``data_type`` for inverse transformation.

    Returns
    -------
    pandas.DataFrame
        The reversed dataframe.
    """
    for switch in encoding:
        col_name = switch.get("col")
        if col_name not in x_in.columns:
            raise Exception(f"Columns {col_name} not in dataframe.")
        column_mapping = switch.get("mapping")
        if column_mapping is None:
            raise ValueError(f"Missing mapping for column {col_name}.")
        if isinstance(column_mapping, dict):
            inverse = pd.Series(data=column_mapping.keys(), index=column_mapping.values())
        else:
            inverse = pd.Series(data=column_mapping.index, index=column_mapping.values)
        x_in[col_name] = x_in[col_name].map(inverse).astype(switch.get("data_type"))
    return x_in


def reverse_basen(x_in: pd.DataFrame, encoding: ce.BaseNEncoder | ce.BinaryEncoder) -> pd.DataFrame:
    """
    Reversed dummies based on baseN category encoder.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : category_encoders.BaseNEncoder or category_encoders.BinaryEncoder
        Fitted encoder whose mapping is used to reverse the encoded columns.

    Returns
    -------
    pandas.DataFrame
        The reversed dummies dataframe for a given baseN encoding encoding.
    """
    x = x_in.copy(deep=True)
    out_cols = x.columns.values.tolist()
    for ind_enc in range(len(encoding.mapping)):
        col_list = encoding.mapping[ind_enc].get("mapping").columns.tolist()
        insert_at = out_cols.index(col_list[0])

        if encoding.base == 1:
            value_array = np.array([int(col0.split("_")[-1]) for col0 in col_list])
        else:
            len0 = len(col_list)
            value_array = np.array([encoding.base ** (len0 - 1 - i) for i in range(len0)])
        x.insert(insert_at, encoding.cols[ind_enc], np.dot(x[col_list].values, value_array.T))
        x.drop(col_list, axis=1, inplace=True)
        out_cols = x.columns.values.tolist()
    return x


def calc_inv_contrib_ce(
    x_contrib: pd.DataFrame,
    encoding: CategoryEncoder | EncoderMapping | list[EncoderMapping],
    agg_columns: Literal["sum", "first"],
) -> pd.DataFrame:
    """
    Reversed contribution when category encoder and/or a dict is used.
    If category encoder create multiple columns, we use the mapping to find which contribution columns to sum.
    Else we return the contribution without aggregate.

    Parameters
    ----------
    x_contrib : pandas.DataFrame
        Contributions set.
    encoding : CategoryEncoder, list of dict, or dict
        The processing apply to the original data.
    agg_columns : str
        Type of aggregation performed. For Shap we want so sum contributions of one hot encoded variables.

    Returns
    -------
    pandas.DataFrame
        The aggregate contributions depending on which processing is apply.
    """
    if isinstance(encoding, (dict, list)):
        return x_contrib

    if str(type(encoding)) in dummies_category_encoder:
        drop_col = []
        for switch in encoding.mapping:
            col_in = switch.get("col")
            mod = switch.get("mapping").columns.tolist()
            insert_at = x_contrib.columns.tolist().index(mod[0])
            if agg_columns == "first":
                x_contrib.insert(insert_at, col_in, x_contrib[mod[0]])
            else:
                x_contrib.insert(insert_at, col_in, x_contrib[mod].sum(axis=1))
            drop_col += mod
        x_contrib.drop(drop_col, axis=1, inplace=True)
        return x_contrib
    else:
        return x_contrib


def transform_ce(x_in: pd.DataFrame, encoding: CategoryEncoding) -> pd.DataFrame:
    """
    Choose and apply the transformation for the given encoding.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Raw dataset to apply preprocessing.
    encoding : CategoryEncoder or list of dict
        A category encoder (OrdinalEncoder/OneHotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
        or a list of mappings.

    Returns
    -------
    pandas.DataFrame
        The dataset preprocessed with the given encoding.
    """
    encoder = [
        category_encoder_ordinal,
        category_encoder_onehot,
        category_encoder_basen,
        category_encoder_binary,
        category_encoder_targetencoder,
    ]

    if isinstance(encoding, list):
        return transform_ordinal(x_in, encoding)

    if str(type(encoding)) in encoder:
        rst = encoding.transform(x_in)
    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no preprocessing done.")

    return rst


def transform_ordinal(x_in: pd.DataFrame, encoding: list[EncoderMapping]) -> pd.DataFrame:
    """
    Transformation based on ordinal category encoder.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Raw dataset to apply preprocessing.
    encoding : list of dict
        Mappings containing ``col`` and ``mapping`` for transformation.

    Returns
    -------
    pandas.DataFrame
        The dataframe preprocessed.
    """
    for switch in encoding:
        col_name = switch.get("col")
        if col_name not in x_in.columns:
            raise Exception(f"Columns {col_name} not in dataframe.")
        column_mapping = switch.get("mapping")
        if column_mapping is None:
            raise ValueError(f"Missing mapping for column {col_name}.")
        if isinstance(column_mapping, dict):
            transform = pd.Series(data=column_mapping.values(), index=column_mapping.keys())
        else:
            transform = pd.Series(data=column_mapping.values, index=column_mapping.index)
        x_in[col_name] = x_in[col_name].map(transform).astype(transform.dtype)
    return x_in


def get_col_mapping_ce(encoder: CategoryEncoder) -> dict[str, list[str]]:
    """
    Get the columns mapping of a category encoder list.

    Parameters
    ----------
    encoder : CategoryEncoder
        Fitted category_encoders encoder.

    Returns
    -------
    dict[str, list[str]]
        Dict of mapping between dataframe columns before and after encoding.
    """
    if str(type(encoder)) in [
        category_encoder_ordinal,
        category_encoder_onehot,
        category_encoder_basen,
        category_encoder_targetencoder,
    ]:
        encoder_mapping = encoder.mapping
    elif str(type(encoder)) == category_encoder_binary:
        encoder_mapping = encoder.mapping
    else:
        raise NotImplementedError(f"{encoder} not supported.")

    dict_col_mapping = dict()
    if isinstance(encoder_mapping, dict):
        for col in encoder_mapping.keys():
            dict_col_mapping[col] = [col]
    elif isinstance(encoder_mapping, list):
        for col_enc in encoder_mapping:
            if isinstance(col_enc.get("mapping"), pd.DataFrame):
                dict_col_mapping[col_enc.get("col")] = col_enc.get("mapping").columns.to_list()
            else:
                dict_col_mapping[col_enc.get("col")] = [col_enc.get("col")]
    else:
        raise NotImplementedError
    return dict_col_mapping
