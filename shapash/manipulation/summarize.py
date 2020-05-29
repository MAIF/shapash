"""
Summarize Module
"""
import numpy as np
import pandas as pd


def summarize_el(dataframe, mask, prefix):
    """
    Compute a summarized Matrix.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Matrix containing contributions, label or feature names
        that will be summarized
    mask: pd.DataFrame
        Mask to apply during the summary step
    prefix: str
        prefix used for columns name

    Returns
    -------
    pd.DataFrame
        Result of the summarize step
    """
    matrix = dataframe.where(mask.to_numpy()).values.tolist()
    summarized_matrix = [[x for x in l if str(x) != 'nan'] for l in matrix]
    # Padding to create pd.DataFrame
    max_length = max(len(l) for l in summarized_matrix)
    for elem in summarized_matrix:
        elem.extend([np.nan] * (max_length - len(elem)))
    # Create DataFrame
    col_list = [prefix + str(x + 1) for x in list(range(max_length))]
    df_summarized_matrix = pd.DataFrame(summarized_matrix,
                                        index=list(dataframe.index),
                                        columns=col_list,
                                        dtype=object)

    return df_summarized_matrix


def compute_features_import(dataframe):
    """
    Compute a relative features importance, sum of absolute values
     ​​of the contributions for each
     features importance compute in base 100
    Parameters
    ----------
    dataframe: pd.DataFrame
        Matrix containing all contributions

    Returns
    -------
    pd.Series
        feature importance One row by feature,
        index of the serie = dataframe.columns
    """
    feat_imp = dataframe.abs().sum().sort_values(ascending=True)
    tot = feat_imp.sum()
    return feat_imp / tot
