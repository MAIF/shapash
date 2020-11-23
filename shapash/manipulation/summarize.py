"""
Summarize Module
"""
import numpy as np
import pandas as pd
from pandas.core.common import flatten


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

def summarize(s_contrib, var_dict, x_sorted, mask, columns_dict, features_dict):
    """
    Compute the summarized contributions of features.

    Parameters
    ----------
    s_contrib: pd.DataFrame
        Matrix containing contributions that will be summarized
    var_dict: pd.DataFrame
        Matrix of feature names that will be summarized
    x_sorted: pd.DataFrame
        Matrix containing the value of each feature
    mask: pd.DataFrame
        Mask to apply during the summary step
    columns_dict:
        Dict of column Names, matches column num with column name
    features_dict:
        Dict of column Label, matches column name with column label

    Returns
    -------
    pd.DataFrame
        Result of the summarize step
    """
    contrib_sum = summarize_el(s_contrib, mask, 'contribution_')
    var_dict_sum = summarize_el(var_dict, mask, 'feature_').applymap(
        lambda x: features_dict[columns_dict[x]] if not np.isnan(x) else x)
    x_sorted_sum = summarize_el(x_sorted, mask, 'value_')

    # Concatenate pd.DataFrame
    summary = pd.concat([contrib_sum, var_dict_sum, x_sorted_sum], axis=1)

    # Ordering columns
    ordered_columns = list(flatten(zip(var_dict_sum.columns, x_sorted_sum.columns, contrib_sum.columns)))
    summary = summary[ordered_columns]
    return summary
