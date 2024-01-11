from typing import Optional

import pandas as pd

from shapash.report.common import VarType, display_value, replace_dict_values
from shapash.webapp.utils.utils import round_to_k


def perform_global_dataframe_analysis(df: Optional[pd.DataFrame]) -> dict:
    """
    Returns a python dict containing global information about a pandas DataFrame :
    Number of features, Number of observations, missing values...

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe that will be used to compute global information.

    Returns
    -------
    global_d : dict
        dictionary that contains an ensemble of global information about the input dataframe.
    """
    if df is None:
        return dict()
    missing_values = df.isna().sum().sum()
    global_d = {
        "number of features": len(df.columns),
        "number of observations": df.shape[0],
        "missing values": missing_values,
        "% missing values": missing_values / (df.shape[0] * df.shape[1]),
    }

    for stat in global_d.keys():
        if stat == "number of observations":
            global_d[stat] = int(global_d[stat])  # Keeping the exact number
        elif isinstance(global_d[stat], float):
            global_d[stat] = round_to_k(global_d[stat], 3)

    replace_dict_values(global_d, display_value, ",", ".")

    return global_d


def perform_univariate_dataframe_analysis(df: Optional[pd.DataFrame], col_types: dict) -> dict:
    """
    Returns a python dict containing information about each column of a pandas DataFrame.
    The computed information depends on the type of the column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe on which the analysis will be performed
    col_types : dict
        Dict of types for each column

    Returns
    -------
    d : dict
        A dict containing each column as keys and the corresponding dict of information for each column as values.
    """
    if df is None:
        return dict()
    d = df.describe().to_dict()
    for col in df.columns:
        if col_types[col] == VarType.TYPE_CAT:
            d[col] = {"distinct values": df[col].nunique(), "missing values": df[col].isna().sum()}

    for col in d.keys():
        for stat in d[col].keys():
            if stat in ["count", "distinct values"]:
                d[col][stat] = int(d[col][stat])  # Keeping the exact number here
            elif isinstance(d[col][stat], float):
                d[col][stat] = round_to_k(d[col][stat], 3)  # Rounding to 3 important figures

    replace_dict_values(d, display_value, ",", ".")

    return d
