"""
Transform Module
"""
import re

import numpy as np
import pandas as pd

from shapash.utils.category_encoder_backend import (
    get_col_mapping_ce,
    inv_transform_ce,
    supported_category_encoder,
    transform_ce,
)
from shapash.utils.columntransformer_backend import (
    columntransformer,
    get_col_mapping_ct,
    inv_transform_ct,
    supported_sklearn,
    transform_ct,
)

# TODO
# encode targeted variable ? from sklearn.preprocessing import LabelEncoder
# make an easy version for dict, not writing all mapping


def inverse_transform(x_init, preprocessing=None):
    """
    Reverse transformation giving a preprocessing.

    Preprocessing could be :
        - a single category_encoders
        - a single ColumnTransformer
        - list with multiple category_encoders with optional (dict, list of dict)
        - list with a single ColumnTransformer with optional (dict, list of dict)
        - dict
        - list of dict

    Preprocessing return an error when using ColumnTransformer and category encoding in the same preprocessing,
    the category encoding must be used in the ColumnTransformer.

    If ColumnTransformer is used, there is an inverse transformation for each transformers,
    so a column could be multi-processed, that's why inverse colnames are prefixed by the transformers names.

    Parameters
    ----------
    x_init : pandas.DataFrame
        Prediction set.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data

    Returns
    -------
    pandas.Dataframe
        return the dataframe before preprocessing.
    """

    if preprocessing is None:
        return x_init
    else:
        # Transform preprocessing into a list
        list_encoding = preprocessing_tolist(preprocessing)

        # Check encoding are supported
        use_ct, use_ce = check_transformers(list_encoding)

        # Apply Inverse Transform
        x_inverse = x_init.copy()

        for encoding in list_encoding:
            if use_ct:
                x_inverse = inv_transform_ct(x_inverse, encoding)
            else:
                x_inverse = inv_transform_ce(x_inverse, encoding)
        return x_inverse


def apply_preprocessing(x_init, model, preprocessing=None):
    """
    Apply preprocessing on a raw dataset giving a preprocessing.

    Preprocessing could be :
        - a single category_encoders
        - a single ColumnTransformer
        - list with multiple category_encoders with optional (dict, list of dict)
        - list with a single ColumnTransformer with optional (dict, list of dict)
        - dict
        - list of dict

    Preprocessing return an error when using ColumnTransformer and category encoding in the same preprocessing,
    the category encoding must be used in the ColumnTransformer.

    If ColumnTransformer is used, there is an inverse transformation for each transformers,
    so a column could be multi-processed, that's why inverse colnames are prefixed by the transformers names.

    Parameters
    ----------
    x_init : pandas.DataFrame
        Raw dataset to apply preprocessing.
    model: model object
        model used to check the different values of target estimate predict_proba
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing to apply to the original data

    Returns
    -------
    pandas.Dataframe
        return the dataframe with preprocessing.
    """

    if preprocessing is None:
        return x_init
    else:
        # Transform preprocessing into a list
        list_encoding = preprocessing_tolist(preprocessing)
        # Check encoding are supported
        use_ct, use_ce = check_transformers(list_encoding)
        # Apply Transform
        for encoding in list_encoding:
            if use_ct:
                x_init = transform_ct(x_init, model, encoding)
            else:
                x_init = transform_ce(x_init, encoding)
        return x_init


def preprocessing_tolist(preprocess):
    """
    Transform preprocess into a list, if preprocess contains a dict, transform the dict into a list of dict.

    Parameters
    ----------
    preprocess : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data

    Returns
    -------
    List
        A list containing all preprocessing.
    """
    list_encoding = preprocess if isinstance(preprocess, list) else [preprocess]
    list_encoding = [[x] if isinstance(x, dict) else x for x in list_encoding]
    return list_encoding


def check_transformers(list_encoding):
    """
    Check that all transformation are supported.
        - a single category encoders transformer
        - a single columns transformers
        - list of multiple category encoder with optional (dict, list of dict)
        - list with a single transformer with optional (dict, list of dict)
        - dict
        - list of dict

    If a dict is used, the dict must contain, a col, a mapping and a data_type.
    Example :
        input_dict1['col'] = 'my_col'
        input_dict1['mapping'] = pd.Series(data=['C', 'D', np.nan], index=['C', 'D', 'missing'])
        input_dict1['data_type'] = 'object'

    Parameters
    ----------
    list_encoding : list
        A list containing at least one transformation

    Returns
    -------
        use_ct : boolean
            true if column transformer is used
        use_ce : boolean
            true if category encoder is used

    """

    use_ct = False
    use_ce = False

    for enc in list_encoding:
        if str(type(enc)) in columntransformer:
            use_ct = True
            for encoding in enc.transformers_:
                ct_encoding = encoding[1]
                if (str(type(ct_encoding)) not in supported_sklearn) and (
                    str(type(ct_encoding)) not in supported_category_encoder
                ):
                    if str(type(ct_encoding)) != "<class 'str'>":
                        raise ValueError("One of the encoders used in ColumnTransformers isn't supported.")

        elif str(type(enc)) in supported_category_encoder:
            use_ce = True

        # Check we have a list of dict
        elif isinstance(enc, list):
            for enc_dict in enc:
                if isinstance(enc_dict, dict):
                    # Check dict structure : col - mapping - data_type
                    if not all(struct in enc_dict for struct in ("col", "mapping", "data_type")):
                        raise Exception(f"{enc_dict} should have col, mapping and data_type as keys.")
                else:
                    raise Exception(f"{enc} is not a list of dict.")
        else:
            raise Exception(f"{enc} is not supported yet.")

    # check that encoding don't use ColumnTransformer and Category encoding at the same time
    if use_ct and use_ce:
        raise Exception(
            "Can't support ColumnTransformer and Category encoding at the same time. "
            "Use Category encoding in ColumnTransformer"
        )

    # check that Category encoding is apply on different columns
    col = []
    for enc in list_encoding:
        if not str(type(enc)) in ("<class 'list'>", "<class 'dict'>", columntransformer):
            col += enc.cols
    duplicate = {x for x in col if col.count(x) > 1}
    if duplicate:
        raise Exception("Columns " + str(duplicate) + " is used in multiple category encoding")

    return use_ct, use_ce


def apply_postprocessing(x_init, postprocessing):
    """
    Transforms x_init depending on postprocessing parameters.

    Parameters
    ----------
    x_init: pandas.Dataframe
        Dataframe that needs to be modified
    postprocessing: dict
        Modifications to apply in x_init dataframe.

    Returns
    -------
    pandas.Dataframe
        Modified DataFrame.
    """
    new_preds = x_init.copy()
    for feature_name in postprocessing.keys():
        dict_postprocessing = postprocessing[feature_name]
        data_modif = new_preds[feature_name]
        new_datai = list()

        if dict_postprocessing["type"] == "prefix":
            for value in data_modif.values:
                new_datai.append(dict_postprocessing["rule"] + str(value))
            new_preds[feature_name] = new_datai

        elif dict_postprocessing["type"] == "suffix":
            for value in data_modif.values:
                new_datai.append(str(value) + dict_postprocessing["rule"])
            new_preds[feature_name] = new_datai

        elif dict_postprocessing["type"] == "transcoding":
            unique_values = x_init[feature_name].unique().tolist()
            unique_values = [value for value in unique_values if value not in dict_postprocessing["rule"].keys()]
            for value in unique_values:
                dict_postprocessing["rule"][value] = value
            new_preds[feature_name] = new_preds[feature_name].map(dict_postprocessing["rule"])

        elif dict_postprocessing["type"] == "regex":
            new_preds[feature_name] = new_preds[feature_name].apply(
                lambda x, d=dict_postprocessing: re.sub(d["rule"]["in"], d["rule"]["out"], x)
            )

        elif dict_postprocessing["type"] == "case":
            if dict_postprocessing["rule"] == "lower":
                new_preds[feature_name] = new_preds[feature_name].apply(lambda x: x.lower())
            elif dict_postprocessing["rule"] == "upper":
                new_preds[feature_name] = new_preds[feature_name].apply(lambda x: x.upper())

    return new_preds


def adapt_contributions(case, contributions):
    """
    If _case is "classification" and contributions a np.array or pd.DataFrame
    this function transform contributions matrix in a list of 2 contributions
    matrices: Opposite contributions and contributions matrices.

    Parameters
    ----------
    case: string
        String which precised if it's a regression problem or a classification one.
    contributions: pandas.DataFrame, np.ndarray or list
        Contribution of each feature to the predicted value.

    Returns
    -------
        pandas.DataFrame, np.ndarray or list
        contributions object modified
    """
    # For classification with numpy arrays, we first transform the last dimension in lists
    # So that we have the following format : [contributions_class_0, contributions_class_1, ...]
    if isinstance(contributions, np.ndarray) and contributions.ndim == 3:
        contributions = [contributions[:, :, i] for i in range(contributions.shape[-1])]
    if (isinstance(contributions, pd.DataFrame) and case == "classification") or (
        isinstance(contributions, (np.ndarray, list)) and case == "classification" and np.array(contributions).ndim == 2
    ):
        return [contributions * -1, contributions]
    else:
        return contributions


def get_preprocessing_mapping(x_encoded, preprocessing=None):
    """
    Get the columns mapping from preprocessing.

    Parameters
    ----------
    x_encoded : pd.DataFrame
        Pandas dataframe after encoder transformations
    preprocessing : category_encoders or ColumnTransformer or list or dict or list of dict
        The processing apply to the original data

    Returns
    -------
    dict
        the mapping between columns names before and after preprocessing.
    """
    if preprocessing is None:
        return {}

    # To avoid recursion error as dict are converted to list of dict when using preprocessing_tolist function
    if isinstance(preprocessing, dict):
        return {}  # The names of the columns are not changing when using dict

    # Transform preprocessing into a list
    list_encoding = preprocessing_tolist(preprocessing)

    # Check encoding are supported
    check_transformers(list_encoding)

    dict_col_mapping = dict()
    for enc in list_encoding:
        if str(type(enc)) == columntransformer:
            dict_col_mapping.update(get_col_mapping_ct(enc, x_encoded))

        elif str(type(enc)) in supported_category_encoder:
            dict_col_mapping.update(get_col_mapping_ce(enc))

        elif isinstance(enc, dict):
            pass  # The names of the columns are not changing when using dict

        elif isinstance(enc, list):
            for sub_enc in enc:
                # Recursive call
                dict_col_mapping.update(get_preprocessing_mapping(preprocessing=sub_enc, x_encoded=x_encoded))

    return dict_col_mapping


def get_features_transform_mapping(x_init, x_encoded, preprocessing=None):
    """
    Get the columns mapping from preprocessing and add missing columns that are not used or changed in preprocessing.

    Parameters
    ----------
    x_init : pd.DataFrame
        Pandas dataframe before preprocessing transformations
    x_encoded : pd.DataFrame
        Pandas dataframe after preprocessing transformations
    preprocessing : category_encoders or ColumnTransformer or list or dict or list of dict
        The processing apply to the original data

    Returns
    -------
    dict
        the mapping between columns names before and after preprocessing.
    """
    dict_all_cols_mapping = dict()
    dict_all_cols_mapping.update(get_preprocessing_mapping(x_encoded=x_encoded, preprocessing=preprocessing))
    # Adding columns which name was not changed during preprocessing
    for col_name in x_init.columns:
        if col_name not in dict_all_cols_mapping.keys():
            dict_all_cols_mapping[col_name] = [col_name]
    return dict_all_cols_mapping


def handle_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing values for categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe on which we will replace the missing values
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df_handle_missing = df.copy()
    df_handle_missing[categorical_cols] = df_handle_missing[categorical_cols].fillna("missing")
    return df_handle_missing
