"""
sklearn columntransformer
"""

import numpy as np
import pandas as pd

from shapash.utils.category_encoder_backend import (
    category_encoder_binary,
    dummies_category_encoder,
    get_col_mapping_ce,
    inv_transform_ce,
    inv_transform_ordinal,
    supported_category_encoder,
    transform_ordinal,
)
from shapash.utils.model import extract_features_model
from shapash.utils.model_synoptic import (
    catboost_model,
    dict_model_feature,
    lightgbm_model,
    linear_model,
    simple_tree_model_sklearn,
    svm_model,
    xgboost_model,
)

columntransformer = "<class 'sklearn.compose._column_transformer.ColumnTransformer'>"

sklearn_onehot = "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"
sklearn_ordinal = "<class 'sklearn.preprocessing._encoders.OrdinalEncoder'>"
sklearn_standardscaler = "<class 'sklearn.preprocessing._data.StandardScaler'>"
sklearn_quantiletransformer = "<class 'sklearn.preprocessing._data.QuantileTransformer'>"
sklearn_powertransformer = "<class 'sklearn.preprocessing._data.PowerTransformer'>"

sklearn_model = linear_model + svm_model + simple_tree_model_sklearn

other_model = xgboost_model + catboost_model + lightgbm_model

dummies_sklearn = sklearn_onehot

no_dummies_sklearn = (sklearn_ordinal, sklearn_standardscaler, sklearn_quantiletransformer, sklearn_powertransformer)

supported_sklearn = (
    sklearn_onehot,
    sklearn_ordinal,
    sklearn_standardscaler,
    sklearn_quantiletransformer,
    sklearn_powertransformer,
)


def inv_transform_ct(x_in, encoding):
    """
    Inverse transform when using a ColumnsTransformer.

    As ColumnsTransformer output hstack the result of transformers, if the TOP-preprocessed data are re-ordered
    after the ColumnTransformer the inverse transform must return false result.

    We successively inverse the transformers with columns position. That's why inverse colnames
    are prefixed by the transformers names.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Prediction set.
    encoding : list
        The list must contain a single ColumnsTransformer and an optional list of dict.

    Returns
    -------
    pandas.Dataframe
        The reversed transformation for the given list of encoding.
    """
    if str(type(encoding)) == columntransformer:
        # We use inverse tranform from the encoding method base on columns position
        init = 0
        rst = pd.DataFrame()

        for enc in encoding.transformers_:
            name_encoding = enc[0]
            ct_encoding = enc[1]
            col_encoding = enc[2]
            # For Scikit encoding we use the associated inverse transform method
            if str(type(ct_encoding)) in supported_sklearn:
                frame, init = inv_transform_sklearn_in_ct(x_in, init, name_encoding, col_encoding, ct_encoding)

            # For category encoding we use the mapping
            elif str(type(ct_encoding)) in supported_category_encoder:
                frame, init = inv_transform_ce_in_ct(x_in, init, name_encoding, col_encoding, ct_encoding)

            # columns not encode
            elif name_encoding == "remainder":
                if ct_encoding == "passthrough":
                    nb_col = len(col_encoding)
                    frame = x_in.iloc[:, init : init + nb_col]
                else:
                    frame = pd.DataFrame()

            else:
                raise Exception(f"{ct_encoding} is not supported yet.")

            rst = pd.concat([rst, frame], axis=1)

    elif str(type(encoding)) == "<class 'list'>":
        rst = inv_transform_ordinal(x_in, encoding)

    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no inverse done.")

    return rst


def inv_transform_ce_in_ct(x_in, init, name_encoding, col_encoding, ct_encoding):
    """
    Inverse transform when using category_encoder in ColumnsTransformer preprocessing.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Data processed.
    init : np.int
        Columns index that give the first column to look at.
    name_encoding : String
        Name of the encoding give by the user.
    col_encoding : list
        Processed features name.
    ct_encoding : category_encoder
        Type of encoding.

    Returns
    -------
    frame : pandas.Dataframe
        The reversed transformation for the given list of encoding.
    init : np.int
        Index of the last column use to make the transformation.
    """
    colname_output = [name_encoding + "_" + val for val in col_encoding]
    colname_input = ct_encoding.get_feature_names_out()
    nb_col = len(colname_input)
    x_to_inverse = x_in.iloc[:, init : init + nb_col].copy()
    x_to_inverse.columns = colname_input
    frame = inv_transform_ce(x_to_inverse, ct_encoding)
    frame.columns = colname_output
    init += nb_col
    return frame, init


def inv_transform_sklearn_in_ct(x_in, init, name_encoding, col_encoding, ct_encoding):
    """
    Inverse transform when using sklearn in ColumnsTransformer preprocessing.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Data processed.
    init : np.int
        Columns index that give the first column to look at.
    name_encoding : String
        Name of the encoding give by the user.
    col_encoding : list
        Processed features name.
    ct_encoding : sklearn, category_encoder
        Type of encoding.

    Returns
    -------
    frame : pandas.Dataframe
        The reversed transformation for the given list of encoding.
    init : np.int
        Index of the last column use to make the transformation.
    """
    colname_output = [name_encoding + "_" + val for val in col_encoding]
    if str(type(ct_encoding)) in dummies_sklearn:
        colname_input = ct_encoding.get_feature_names_out(col_encoding)
        nb_col = len(colname_input)
    else:
        nb_col = len(colname_output)
    x_inverse = ct_encoding.inverse_transform(x_in.iloc[:, init : init + nb_col])
    frame = pd.DataFrame(x_inverse, columns=colname_output, index=x_in.index)
    init += nb_col
    return frame, init


def calc_inv_contrib_ct(x_contrib, encoding, agg_columns):
    """
    Reversed contribution when ColumnTransformer is used.

    As columns transformers output hstack the result of transformers, if the TOP-preprocessed data are re-ordered
    after the ColumnTransformer the inverse transform must return false result.

    Parameters
    ----------
    x_contrib : pandas.DataFrame
        Contributions set.
    encoding : ColumnTransformer, list, dict
        The processing apply to the original data.
    agg_columns : str (default: 'sum')
        Type of aggregation performed. For Shap we want so sum contributions of one hot encoded variables.

    Returns
    -------
    pandas.Dataframe
        The aggregate contributions depending on which processing is apply.
    """

    if str(type(encoding)) == columntransformer:
        # We use inverse tranform from the encoding method base on columns position
        init = 0
        rst = pd.DataFrame()

        for enc in encoding.transformers_:
            name_encoding = enc[0]
            ct_encoding = enc[1]
            col_encoding = enc[2]

            if str(type(ct_encoding)) in supported_category_encoder + supported_sklearn:
                # We create new columns names depending on the name of the transformers and the name of the column.
                colname_output = [name_encoding + "_" + val for val in col_encoding]

                # If the processing create multiple columns we find the number of original categories and aggregate
                # the contribution.
                if str(type(ct_encoding)) in dummies_sklearn or str(type(ct_encoding)) in dummies_category_encoder:
                    for i_enc in range(len(colname_output)):
                        if str(type(ct_encoding)) == sklearn_onehot:
                            col_origin = ct_encoding.categories_[i_enc]
                        elif str(type(ct_encoding)) == category_encoder_binary:
                            try:
                                col_origin = ct_encoding.base_n_encoder.mapping[i_enc].get("mapping").columns.tolist()
                            except Exception:
                                col_origin = ct_encoding.mapping[i_enc].get("mapping").columns.tolist()
                        else:
                            col_origin = ct_encoding.mapping[i_enc].get("mapping").columns.tolist()
                        nb_col = len(col_origin)
                        if agg_columns == "first":
                            contrib_inverse = x_contrib.iloc[:, init]
                        else:
                            contrib_inverse = x_contrib.iloc[:, init : init + nb_col].sum(axis=1)
                        frame = pd.DataFrame(
                            contrib_inverse, columns=[colname_output[i_enc]], index=contrib_inverse.index
                        )
                        rst = pd.concat([rst, frame], axis=1)
                        init += nb_col
                else:
                    nb_col = len(colname_output)
                    frame = x_contrib.iloc[:, init : init + nb_col]
                    frame.columns = colname_output
                    rst = pd.concat([rst, frame], axis=1)
                    init += nb_col

            elif name_encoding == "remainder":
                if ct_encoding == "passthrough":
                    nb_col = len(col_encoding)
                    frame = x_contrib.iloc[:, init : init + nb_col]
                    rst = pd.concat([rst, frame], axis=1)
                    init += nb_col
            else:
                raise Exception(f"{encoding.__class__.__name__} not supported, no inverse done.")
        return rst
    else:
        return x_contrib


def transform_ct(x_in, model, encoding):
    """
    Transform when using a ColumnsTransformer.

    As ColumnsTransformer output hstack the result of transformers, if the TOP-preprocessed data are re-ordered
    after the ColumnTransformer the inverse transform must return false result.

    We successively apply the transformers with columns position. That's why colnames
    are prefixed by the transformers names.

    Parameters
    ----------
    x_in : pandas.DataFrame
        Raw dataset to apply preprocessing
    model: model object
        model used to check the different values of target estimate predict_proba
    encoding : list
        The list must contain a single ColumnsTransformer and an optional list of dict.

    Returns
    -------
    pandas.Dataframe
        The data preprocessed for the given list of encoding.
    """
    if str(type(encoding)) == columntransformer:
        # We use inverse tranform from the encoding method base on columns position
        if str(type(model)) in sklearn_model:
            rst = pd.DataFrame(encoding.transform(x_in), index=x_in.index)
            rst.columns = ["col_" + str(feature) for feature in rst.columns]

        elif str(type(model)) in other_model:
            rst = pd.DataFrame(
                encoding.transform(x_in),
                columns=extract_features_model(model, dict_model_feature[str(type(model))]),
                index=x_in.index,
            )
        else:
            raise ValueError("Model specified isn't supported by Shapash.")

    elif str(type(encoding)) == "<class 'list'>":
        rst = transform_ordinal(x_in, encoding)

    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no preprocessing done.")

    return rst


def get_names(name, trans, column, column_transformer):
    """
    Allow to extract features names from one encoder of the ColumnTransformer.
    If the right names aren't available, It creates a list with customized names.

    Parameters
    ----------
    name: string
        String which indicates the name of the transformer.
    trans: sklearn encoder or category_encoders
        One of the encoder fitted through the ColumnTransformer.
    column: list
        List of features impacted by the specific transformer.
    column_transformer: sklearn ColumnTransformer
        The fitted ColumnTransformer containing the specific encoder.

    Returns
    -------
    list:
        List of returned features when specific transformer is applied.
    """
    if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
        return []
    if trans == "passthrough":
        if hasattr(column_transformer, "_df_columns"):
            if (not isinstance(column, slice)) and all(isinstance(col, str) for col in column):
                return column
            else:
                return column_transformer._df_columns[column]
        else:
            indices = np.arange(column_transformer._n_features)
            return ["x%d" % i for i in indices[column]]
    if not hasattr(trans, "get_feature_names_out"):
        if column is None:
            return []
        else:
            return [name + "__" + f for f in column]

    return [name + "__" + f for f in trans.get_feature_names_out()]


def get_feature_names(column_transformer):
    """
    Allow to extract all features names from encoders of the ColumnTransformer once it has been applied.
    If the right names aren't available, It creates a list with customized names.

    Parameters
    ----------
    column_transformer: sklearn ColumnTransformer
        The fitted ColumnTransformer containing the specific encoder.

    Returns
    -------
    feature_names: list
        List of returned features names when ColumnTransformer is applied.
    """
    feature_names = []
    l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        feature_names.extend(get_names(name, trans, column, column_transformer))

    return feature_names


def get_list_features_names(list_preprocessing, columns_dict):
    """
    Allow to extract all features names from encoders when a list of preprocessing is uesd once it has been applied.
    If the right names aren't available, It creates a list with customized names.

    Parameters
    ----------
    list_preprocessing: list
        The fitted list_preprocessing containing the specific encoders.

    Returns
    -------
    feature_names: list
        List of returned features names when list_preprocessing is applied.

    """
    feature_expected = [value for value in columns_dict.values()]

    for enc in list_preprocessing:
        if str(type(enc)) in supported_category_encoder:
            feature_expected = enc.feature_names_out_

        elif str(type(enc)) in columntransformer:
            feature_expected = get_feature_names(enc)

        elif str(type(enc)) in ("<class 'list'>"):
            feature_expected = feature_expected

    return feature_expected


def get_feature_out(estimator, feature_in):
    """
    Returns estimator features out if it has get_feature_names_out method, else features_in
    """
    if hasattr(estimator, "get_feature_names_out") and hasattr(estimator, "categories_"):
        return estimator.get_feature_names_out(), estimator.categories_
    elif hasattr(estimator, "get_feature_names_out"):
        return estimator.get_feature_names_out(), []
    else:
        return feature_in, []


def get_col_mapping_ct(encoder, x_encoded):
    """
    Get the columns mapping of a column transformer encoder.

    Parameters
    ----------
    encoder : ColumnTransformer
        The encoder used.
    x_encoded : pd.DataFrame
        Pandas dataframe after encoder transformations

    Returns
    -------
    dict_col_mapping : dict
        Dict of mapping between dataframe columns before and after encoding.
    """
    dict_col_mapping = dict()
    idx_encoded = 0
    for name, estimator, features in encoder.transformers_:
        if name != "remainder":

            if str(type(estimator)) in dummies_sklearn:
                features_out, categories_out = get_feature_out(estimator, features)
                for i, f_name in enumerate(features):
                    dict_col_mapping[name + "_" + f_name] = list()
                    for _ in categories_out[i]:
                        dict_col_mapping[name + "_" + f_name].append(x_encoded.columns.to_list()[idx_encoded])
                        idx_encoded += 1

            elif str(type(estimator)) in no_dummies_sklearn:
                features_out, categories_out = get_feature_out(estimator, features)
                for f_name in features_out:
                    dict_col_mapping[name + "_" + f_name] = [x_encoded.columns.to_list()[idx_encoded]]
                    idx_encoded += 1

            elif str(type(estimator)) in supported_category_encoder:
                dict_mapping_ce = get_col_mapping_ce(estimator)
                for f_name in dict_mapping_ce.keys():
                    dict_col_mapping[name + "_" + f_name] = list()
                    for _ in dict_mapping_ce[f_name]:
                        dict_col_mapping[name + "_" + f_name].append(x_encoded.columns.to_list()[idx_encoded])
                        idx_encoded += 1

            else:
                raise NotImplementedError(f"Estimator not supported : {estimator}")

        elif estimator == "passthrough":
            try:
                features_out = encoder.feature_names_in_[features]
            except Exception:
                features_out = encoder._feature_names_in[features]  # for oldest sklearn version
            for f_name in features_out:
                dict_col_mapping[f_name] = [x_encoded.columns.to_list()[idx_encoded]]
                idx_encoded += 1

    return dict_col_mapping
