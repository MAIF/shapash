from typing import Optional

import pandas as pd

from shapash.report.common import VarType, series_dtype, numeric_is_continuous


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
        'number of features': len(df.columns),
        'number of observations': df.shape[0],
        'missing values': missing_values,
        '% missing values': missing_values / (df.shape[0] * df.shape[1]),
    }

    return global_d


def perform_univariate_dataframe_analysis(df: Optional[pd.DataFrame]) -> dict:
    """
    Returns a python dict containing information about each column of a pandas DataFrame.
    The computed information depends on the type of the column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe on which the analysis will be performed

    Returns
    -------
    d : dict
        A dict containing each column as keys and the corresponding dict of information for each column as values.
    """
    if df is None:
        return dict()
    d = df.describe().round(2).to_dict()
    for col in df.columns:
        if series_dtype(df[col]) == VarType.TYPE_CAT \
                or (series_dtype(df[col]) == VarType.TYPE_NUM and not numeric_is_continuous(df[col])):
            d[col] = {
                'distinct values': df[col].nunique(),
                'missing values': df[col].isna().sum()
            }

    return d


