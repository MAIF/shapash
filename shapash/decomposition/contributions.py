"""
Contributions
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from shapash.utils.category_encoder_backend import calc_inv_contrib_ce
from shapash.utils.columntransformer_backend import calc_inv_contrib_ct
from shapash.utils.transform import check_transformers, preprocessing_tolist


def inverse_transform_contributions(
    contributions: pd.DataFrame,
    preprocessing: Any | None = None,
    agg_columns: Literal["sum", "first"] = "sum",
) -> pd.DataFrame:
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
    preprocessing : category_encoders, ColumnTransformer, list or dict or None, optional
        The processing apply to the original data.
    agg_columns : Literal["sum", "first"], optional
        Type of aggregation performed. For Shap we want so sum contributions of one hot encoded variables.

    Returns
    -------
    pd.DataFrame
        Return the aggregate contributions.

    """

    if not isinstance(contributions, pd.DataFrame):
        raise Exception("Shap values must be a pandas dataframe.")

    if preprocessing is None:
        return contributions
    else:
        # Transform preprocessing into a list
        list_encoding = preprocessing_tolist(preprocessing)

        # check supported inverse
        use_ct, use_ce = check_transformers(list_encoding)

        # Apply Inverse Transform
        x_contrib_invers = contributions.copy()
        if use_ct:
            for encoding in list_encoding:
                x_contrib_invers = calc_inv_contrib_ct(x_contrib_invers, encoding, agg_columns)
        else:
            for encoding in list_encoding:
                x_contrib_invers = calc_inv_contrib_ce(x_contrib_invers, encoding, agg_columns)
        return x_contrib_invers


def rank_contributions(s_df: pd.DataFrame, x_df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Function to sort contributions and input features
    by decreasing contribution absolute values

    Parameters
    ----------
    s_df : pd.DataFrame
        Local contributions dataframe.
    x_df : pd.DataFrame
        Input features.

    Returns
    -------
    list[pd.DataFrame]
        A three-item list containing sorted contributions, sorted features, and
        the feature indices in sorted order, respectively.
    """
    argsort = np.argsort(-np.abs(s_df.values), axis=1)
    sorted_contrib = np.take_along_axis(s_df.values, argsort, axis=1)
    sorted_features = np.take_along_axis(x_df.values, argsort, axis=1)

    contrib_col = ["contribution_" + str(i) for i in range(s_df.shape[1])]
    col = ["feature_" + str(i) for i in range(s_df.shape[1])]

    s_dict = pd.DataFrame(data=argsort, columns=col, index=x_df.index)
    s_ord = pd.DataFrame(data=sorted_contrib, columns=contrib_col, index=x_df.index)
    x_ord = pd.DataFrame(data=sorted_features, columns=col, index=x_df.index)
    return [s_ord, x_ord, s_dict]


def assign_contributions(ranked: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Turn a list of results into a dict.

    Parameters
    ----------
    ranked : list[pd.DataFrame]
        The three DataFrames returned by :func:`rank_contributions`.

    Returns
    -------
    dict[str, pd.DataFrame]
        The ranked contributions, features, and feature indices under explicit
        keys.

    Raises
    ------
    ValueError
        The output of rank_contributions should always be of length three.
    """
    if len(ranked) != 3:
        raise ValueError(
            f"Expected length : 3, observed length : {len(ranked)}, please check the outputs of rank_contributions."
        )
    return {"contrib_sorted": ranked[0], "x_sorted": ranked[1], "var_dict": ranked[2]}
