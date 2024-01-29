"""
Filters module
"""
import numpy as np
import pandas as pd


def hide_contributions(var_dict, features_list):
    """
    Returns Boolean dataframe depending if the
    feature is present or not in the list of
    feature to hide.

    Parameters
    ----------
    var_dict: pd.DataFrame
        Dataframe with features indexes ordered
        by contribution.
    feature_list: List
        List of index, feature to hide.

    Returns
    -------
    pd.DataFrame
        Boolean dataframe depend on hidden features.
    """
    return ~var_dict.isin(features_list)


def cap_contributions(s_contrib, threshold=0.1):
    """
    The function is able to compute a mask indicating where the input matrix
    has values above a given threshold in absolute value.

    Parameters
    ----------
    s : pandas.DataFrame
        Local contributions, positive and negative values.
    threshold: float, optional (default: 0.1)
        User defined threshold above which local contributions are hidden.

    Returns
    -------
    pandas.DataFrame
        Mask with only True of False elements.
    """
    mask = s_contrib.abs() >= threshold
    return mask


def sign_contributions(dataframe, positive=True):
    """
    Returns Boolean values depending on
    the signs of local contributions
    stored in df and on the positive parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        Local contributions of the model.
    positive : boolean (default=True)
        True to evaluate positive value.
        False to evaluate negative value.

    Returns
    -------
    pandas.DataFrame
        Dataframe with boolean value.
    """
    if positive:
        return dataframe >= 0
    else:
        return dataframe < 0


def cutoff_contributions_old(dataframe, max_contrib):
    """
    The function cutoff_contributions computes a mask on a sorted contribution matrix.
    It outputs True everywhere the contribution is in the top-k,
    k being defined as an option by the user.

    Parameters
    ----------
    df : pd.Dataframe
        DataFrame is a sorted local contributions matrix.
    max_contrib: int
        The k most important contributions to keep.

    Returns
    -------
    pd.Dataframe
        Mask indicating where contributions should be considered.
    """
    mask = np.full_like(dataframe, False).astype(bool)
    mask[:, :max_contrib] = True
    return pd.DataFrame(mask, columns=dataframe.columns, index=dataframe.index)


def cutoff_contributions(mask, k=10):
    """
    Compute a mask that select for each raw the top-k True,
    k being defined as an option by the user.

    Parameters
    ----------
    mask : pd.Dataframe
        Boolean DataFrame indicating sorted contribution we want to hide/show.
    k: int (default: 10)
        The number of top feature we want to show.

    Returns
    -------
    pd.Dataframe
        Mask where only the k-top contributions are considered.
    """
    return mask.replace(False, np.nan).cumsum(axis=1).isin(range(1, k + 1))


def combine_masks(masks_list):
    """
    The function combine_masks computes a combined mask from a list of existing masks
    It outputs True everywhere the value is True for each mask in the list

    Parameters
    ----------
    masks_list : list of pandas dataframes
        List of masks used for filtering features et rows
        the shape of each mask must be equal to the initial dataset shape

    Returns
    -------
    pd.Dataframe of boolean
        combination of all masks.
    """

    if len(set(map(lambda x: x.shape, masks_list))) != 1:
        raise ValueError("Masks must have same dimensions.")

    masks_cube = np.dstack(masks_list)
    mask_final = np.min(masks_cube, axis=2)

    return pd.DataFrame(
        mask_final, columns=["contrib_{}".format(i + 1) for i in range(mask_final.shape[1])], index=masks_list[0].index
    )
