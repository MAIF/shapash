"""
Mask module
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.ma as ma
import pandas as pd

if TYPE_CHECKING:
    from shapash.explainer.multi_decorator import MultiDecorator
    from shapash.explainer.smart_state import SmartState


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


def compute_mask(
    state: "SmartState | MultiDecorator",
    data: dict[str, Any],
    features_list: list[int] | None = None,
    threshold: float | None = None,
    positive: bool | None = None,
    max_contrib: int | None = None,
) -> tuple[pd.DataFrame | list[pd.DataFrame], pd.DataFrame | list[pd.DataFrame], dict[str, Any]]:
    """
    Apply filtering rules to a contributions matrix and return the result, without storing
    anything. This is the pure computation behind `SmartExplainer.filter()`: it derives the
    mask, the masked contributions and the parameters used from `data` alone, so callers that
    only need a one-off mask (e.g. drawing a plot) don't have to mutate an explainer to get it.

    Parameters
    ----------
    state: SmartState or MultiDecorator
        State object driving the mask computation (handles the single-dataframe and
        list-of-dataframes/multi-class cases transparently).
    data: dict
        Either `explainer.data` or `explainer.data_groups`, containing `contrib_sorted` and
        `var_dict`.
    features_list: list of int, optional
        Already-resolved column indexes to hide (see `SmartExplainer.check_features_name`).
    threshold: float, optional
        Absolute value threshold below which contributions are hidden.
    positive: bool, optional
        Hide negative (`True`) or positive (`False`) contributions. `None` shows all.
    max_contrib: int, optional
        Maximum number of contributions to keep.

    Returns
    -------
    mask: pd.DataFrame or list of pd.DataFrame
        Same shape as `data["contrib_sorted"]`, filled with booleans: `False` marks a hidden
        contribution.
    masked_contributions: pd.DataFrame or list of pd.DataFrame
        Summed contributions of the features hidden by `mask`.
    mask_params: dict
        `{"features_to_hide": features_list, "threshold": threshold, "positive": positive,
        "max_contrib": max_contrib}`
    """
    masks = [state.init_mask(data["contrib_sorted"], True)]
    if features_list:
        masks.append(state.hide_contributions(data["var_dict"], features_list=features_list))
    if threshold:
        masks.append(state.cap_contributions(data["contrib_sorted"], threshold=threshold))
    if positive is not None:
        masks.append(state.sign_contributions(data["contrib_sorted"], positive=positive))
    mask = state.combine_masks(masks)
    if max_contrib:
        mask = state.cutoff_contributions(mask, max_contrib=max_contrib)
    masked_contributions = state.compute_masked_contributions(data["contrib_sorted"], mask)
    mask_params: dict[str, Any] = {
        "features_to_hide": features_list,
        "threshold": threshold,
        "positive": positive,
        "max_contrib": max_contrib,
    }
    return mask, masked_contributions, mask_params
