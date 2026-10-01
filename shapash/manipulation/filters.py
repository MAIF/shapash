"""
Filters module
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def hide_contributions(var_dict: pd.DataFrame, features_list: Sequence[object]) -> pd.DataFrame:
    """
    Returns Boolean dataframe depending if the
    feature is present or not in the list of
    feature to hide.

    Parameters
    ----------
    var_dict : pd.DataFrame
        DataFrame with feature indexes ordered by contribution.
    features_list : Sequence[object]
        Feature indexes to hide.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame indicating the features that are not hidden.
    """
    return ~var_dict.isin(features_list)


def cap_contributions(s_contrib: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    The function is able to compute a mask indicating where the input matrix
    has values above a given threshold in absolute value.

    Parameters
    ----------
    s_contrib : pandas.DataFrame
        Local contributions, positive and negative values.
    threshold: float, optional (default: 0.1)
        User defined threshold above which local contributions are hidden.

    Returns
    -------
    pandas.DataFrame
        Boolean mask indicating contributions whose absolute value is at least
        the threshold.
    """
    mask = s_contrib.abs() >= threshold
    return mask


def sign_contributions(dataframe: pd.DataFrame, positive: bool = True) -> pd.DataFrame:
    """
    Returns Boolean values depending on
    the signs of local contributions
    stored in dataframe and on the positive parameter.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Local contributions of the model.
    positive : bool, optional (default=True)
        If True, evaluate non-negative values. If False, evaluate negative
        values.

    Returns
    -------
    pandas.DataFrame
        Dataframe with boolean value.
    """
    if positive:
        return dataframe >= 0
    else:
        return dataframe < 0


def cutoff_contributions_old(dataframe: pd.DataFrame, max_contrib: int) -> pd.DataFrame:
    """
    The function cutoff_contributions computes a mask on a sorted contribution matrix.
    It outputs True everywhere the contribution is in the top-k,
    k being defined as an option by the user.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Sorted local contributions matrix.
    max_contrib : int
        The k most important contributions to keep.

    Returns
    -------
    pd.DataFrame
        Mask indicating where contributions should be considered.
    """
    mask = np.full_like(dataframe, False).astype(bool)
    mask[:, :max_contrib] = True
    return pd.DataFrame(mask, columns=dataframe.columns, index=dataframe.index)


def cutoff_contributions(mask: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Compute a mask that selects the top-k True values for each row,
    k being defined as an option by the user.

    Parameters
    ----------
    mask : pd.DataFrame
        Boolean DataFrame indicating sorted contribution we want to hide/show.
    k : int, optional (default=10)
        The number of top features to show.

    Returns
    -------
    pd.DataFrame
        Boolean mask where only the top-k contributions are considered.
    """
    # Convert False values to np.nan explicitly without changing data type
    mask_nan = mask.astype(float).replace(0, np.nan)

    # Compute the cumulative sum and check if the index is within the top-k
    return mask_nan.cumsum(axis=1).isin(range(1, k + 1))


def combine_masks(masks_list: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Compute a combined mask from a sequence of existing masks.
    The output is True only where every mask is True.

    Parameters
    ----------
    masks_list : Sequence[pd.DataFrame]
        Masks used for filtering features and rows. Each mask must have the
        same shape.

    Returns
    -------
    pd.DataFrame
        Boolean combination of all masks.
    """

    if len(set(map(lambda x: x.shape, masks_list))) != 1:
        raise ValueError("Masks must have same dimensions.")

    masks_cube = np.dstack(masks_list)
    mask_final = np.min(masks_cube, axis=2)

    return pd.DataFrame(
        mask_final, columns=[f"contrib_{i + 1}" for i in range(mask_final.shape[1])], index=masks_list[0].index
    )
