"""
Contributions
"""

import pandas as pd
import numpy as np
from shapash.utils.transform import preprocessing_tolist
from shapash.utils.transform import check_supported_inverse
from shapash.utils.category_encoder_backend import calc_inv_contrib_ce
from shapash.utils.columntransformer_backend import calc_inv_contrib_ct

def inverse_transform_contributions(contributions, preprocessing=None):
    """
    Reverse contribution giving a preprocessing.

    Preprocessing could be :
        - a single category_encoders
        - a single ColumnTransformer
        - list with multiple category_encoders with optional (dict, list of dict)
        - list with a single ColumnTransformer with optional (dict, list of dict)
        - dict
        - list of dict

    Parameters
    ----------
    contributions : pandas.DataFrame
        Contributions values.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        The processing apply to the original data.

    Returns
    -------
    pandas.Dataframe
        Return the aggregate contributions.

    """

    if not isinstance(contributions, pd.DataFrame):
        raise Exception('Shap values must be a pandas dataframe.')

    if preprocessing is None:
        return contributions
    else:
        #Transform preprocessing into a list
        list_encoding = preprocessing_tolist(preprocessing)

        # check supported inverse
        use_ct, use_ce = check_supported_inverse(list_encoding)

        # Apply Inverse Transform
        x_contrib_invers = contributions.copy()
        if use_ct:
            for encoding in list_encoding:
                x_contrib_invers = calc_inv_contrib_ct(x_contrib_invers, encoding)
        else:
            for encoding in list_encoding:
                x_contrib_invers = calc_inv_contrib_ce(x_contrib_invers, encoding)
        return x_contrib_invers

def compute_contributions(x_df, explainer, preprocessing=None):
    """
    Compute Shapley contributions of a prediction set for a certain model.

    Parameters
    ----------
    x_df: pandas.DataFrame
        Prediction set : features to be preprocessed before using SHAP.
    explainer: object
        Any SHAP explainer already initialized with a model.
    preprocessing: object, optional (default: None)
        A single transformer, from sklearn or category_encoders

    Returns
    -------
    pandas.DataFrame
        Shapley contributions of the model on the prediction set, as computed by the explainer.
    """
    if preprocessing:
        x_df = preprocessing.transform(x_df)
    shap_values = explainer.shap_values(x_df)
    res = pd.DataFrame()
    if isinstance(shap_values, list):
        res = [pd.DataFrame(data=tab, index=x_df.index, columns=x_df.columns) for tab in shap_values]
    elif isinstance(shap_values, np.ndarray):
        res = pd.DataFrame(data=shap_values, index=x_df.index, columns=x_df.columns)
    bias = explainer.expected_value
    return res, bias


def rank_contributions(s_df, x_df):
    """
    Function to sort contributions and input features
    by decreasing contribution absolute values

    Parameters
    ----------
    s_df: pandas.DataFrame
        Local contributions dataframe.
    x_df: pandas.DataFrame
        Input features.

    Returns
    -------
    pandas.DataFrame
        Local contributions sorted by decreasing absolute values.
    pandas.DataFrame
        Input features sorted by decreasing contributions absolute values.
    pandas.DataFrame
        Input features names sorted for each observation
        by decreasing contributions absolute values.
    """
    argsort = np.argsort(-np.abs(s_df.values), axis=1)
    sorted_contrib = np.take_along_axis(s_df.values, argsort, axis=1)
    sorted_features = np.take_along_axis(x_df.values, argsort, axis=1)

    contrib_col = ['contribution_' + str(i) for i in range(s_df.shape[1])]
    col = ['feature_' + str(i) for i in range(s_df.shape[1])]

    s_dict = pd.DataFrame(data=argsort, columns=col, index=x_df.index)
    s_ord = pd.DataFrame(data=sorted_contrib, columns=contrib_col, index=x_df.index)
    x_ord = pd.DataFrame(data=sorted_features, columns=col, index=x_df.index)
    return [s_ord, x_ord, s_dict]
