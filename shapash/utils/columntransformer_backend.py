"""
sklearn columntransformer
"""

import pandas as pd
from shapash.utils.category_encoder_backend import inv_transform_ordinal
from shapash.utils.category_encoder_backend import inv_transform_ce
from shapash.utils.category_encoder_backend import supported_category_encoder
from shapash.utils.category_encoder_backend import dummies_category_encoder
from shapash.utils.category_encoder_backend import category_encoder_binary
from shapash.utils.category_encoder_backend import transform_ordinal
from shapash.utils.model_synoptic import simple_tree_model_sklearn,catboost_model,\
    linear_model,svm_model, xgboost_model, lightgbm_model, dict_model_feature
from shapash.utils.model import extract_features_model

columntransformer = "<class 'sklearn.compose._column_transformer.ColumnTransformer'>"

sklearn_onehot = "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"
sklearn_ordinal = "<class 'sklearn.preprocessing._encoders.OrdinalEncoder'>"
sklearn_standardscaler = "<class 'sklearn.preprocessing._data.StandardScaler'>"
sklearn_quantiletransformer = "<class 'sklearn.preprocessing._data.QuantileTransformer'>"
sklearn_powertransformer = "<class 'sklearn.preprocessing._data.PowerTransformer'>"

sklearn_model = linear_model + svm_model + simple_tree_model_sklearn

other_model = xgboost_model + catboost_model + lightgbm_model

dummies_sklearn = (sklearn_onehot)

no_dummies_sklearn = (sklearn_ordinal,
                      sklearn_standardscaler,
                      sklearn_quantiletransformer,
                      sklearn_powertransformer)

supported_sklearn = (sklearn_onehot,
                     sklearn_ordinal,
                     sklearn_standardscaler,
                     sklearn_quantiletransformer,
                     sklearn_powertransformer)


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
                frame, init = inv_transform_sklearn_in_ct(x_in,
                                                          init,
                                                          name_encoding,
                                                          col_encoding,
                                                          ct_encoding)

            # For category encoding we use the mapping
            elif str(type(ct_encoding)) in supported_category_encoder:
                frame, init = inv_transform_ce_in_ct(x_in,
                                                     init,
                                                     name_encoding,
                                                     col_encoding,
                                                     ct_encoding)

            # columns not encode
            elif name_encoding == 'remainder':
                if ct_encoding == 'passthrough':
                    nb_col = len(col_encoding)
                    frame = x_in.iloc[:, init:init + nb_col]
                else:
                    frame = pd.DataFrame()

            else:
                raise Exception(f'{ct_encoding} is not supported yet.')

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
    colname_output = [name_encoding + '_' + val for val in col_encoding]
    colname_input = ct_encoding.get_feature_names()
    nb_col = len(colname_input)
    x_to_inverse = x_in.iloc[:, init:init + nb_col].copy()
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
    colname_output = [name_encoding + '_' + val for val in col_encoding]
    if str(type(ct_encoding)) in dummies_sklearn:
        colname_input = ct_encoding.get_feature_names(col_encoding)
        nb_col = len(colname_input)
    else:
        nb_col = len(colname_output)
    x_inverse = ct_encoding.inverse_transform(x_in.iloc[:, init:init + nb_col])
    frame = pd.DataFrame(x_inverse, columns=colname_output, index=x_in.index)
    init += nb_col
    return frame, init

def calc_inv_contrib_ct(x_contrib, encoding):
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

            if str(type(ct_encoding)) in supported_category_encoder+supported_sklearn:
                # We create new columns names depending on the name of the transformers and the name of the column.
                colname_output = [name_encoding + '_' + val for val in col_encoding]

                # If the processing create multiple columns we find the number of original categories and aggregate
                # the contribution.
                if str(type(ct_encoding)) in dummies_sklearn or str(type(ct_encoding)) in dummies_category_encoder:
                    for i_enc in range(len(colname_output)):
                        if str(type(ct_encoding)) == sklearn_onehot:
                            col_origin = ct_encoding.categories_[i_enc]
                        elif str(type(ct_encoding)) == category_encoder_binary:
                            col_origin = ct_encoding.base_n_encoder.mapping[i_enc].get('mapping').columns.tolist()
                        else:
                            col_origin = ct_encoding.mapping[i_enc].get('mapping').columns.tolist()
                        nb_col = len(col_origin)
                        contrib_inverse = x_contrib.iloc[:, init:init + nb_col].sum(axis=1)
                        frame = pd.DataFrame(contrib_inverse,
                                             columns=[colname_output[i_enc]],
                                             index=contrib_inverse.index)
                        rst = pd.concat([rst, frame], axis=1)
                        init += nb_col
                else:
                    nb_col = len(colname_output)
                    frame = x_contrib.iloc[:, init:init + nb_col]
                    frame.columns = colname_output
                    rst = pd.concat([rst, frame], axis=1)
                    init += nb_col

            elif name_encoding == 'remainder':
                if ct_encoding == 'passthrough':
                    nb_col = len(col_encoding)
                    frame = x_contrib.iloc[:, init:init + nb_col]
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
            rst = pd.DataFrame(encoding.transform(x_in),
                               index=x_in.index)
            rst.columns = ["col_" + str(feature) for feature in rst.columns]

        elif str(type(model)) in other_model:
            rst = pd.DataFrame(encoding.transform(x_in),
                                columns=extract_features_model(model, dict_model_feature[str(type(model))]),
                                index=x_in.index)
        else:
            raise ValueError("Model specified isn't supported by Shapash.")

    elif str(type(encoding)) == "<class 'list'>":
        rst = transform_ordinal(x_in, encoding)

    else:
        raise Exception(f"{encoding.__class__.__name__} not supported, no preprocessing done.")

    return rst