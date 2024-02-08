"""
Mask module
"""
import numpy as np
import numpy.ma as ma
import pandas as pd


def compute_masked_contributions(s_contrib, mask):
    """
    Compute the summed contributions of hidden features.

    Parameters
    ----------
    s_contrib: pd.DataFrame
        Matrix with both positive and negative values
    mask: pd.DataFrame
        Matrix with only True or False elements. False elements are the hidden elements.

    Returns
    -------
    pd.DataFrame
        Sum of contributions of hidden features.
    """
    colname = ["masked_neg", "masked_pos"]
    hidden_neg = np.sum(ma.array(s_contrib, mask=np.max(np.dstack([mask, (s_contrib > 0)]), axis=2)), axis=1)
    hidden_pos = np.sum(ma.array(s_contrib, mask=np.max(np.dstack([mask, (s_contrib < 0)]), axis=2)), axis=1)
    hidden_contrib = np.array([hidden_neg, hidden_pos])
    return pd.DataFrame(hidden_contrib.T, columns=colname, index=s_contrib.index)


def init_mask(s_contrib, value=True):
    """
    Compute True mask of dimensions corresponding to contributions matrix ones.

    Parameters
    ----------
    s_contrib: pd.DataFrame
        Matrix with both positive and negative values
    value: bool
        Value used for initialize the mask

    Returns
    -------
    pd.DataFrame
        mask of True values.
    """
    if value:
        mask = np.ones(s_contrib.shape, dtype=bool)
    else:
        mask = np.zeros(s_contrib.shape, dtype=bool)

    return pd.DataFrame(mask, columns=s_contrib.columns, index=s_contrib.index)
