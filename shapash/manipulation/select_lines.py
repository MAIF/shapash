"""
Select Lines Module
"""

from typing import Any, Literal

import pandas as pd
from pandas.core.common import flatten


def select_lines(dataframe: pd.DataFrame, condition: str | None = None) -> list[Any]:
    """
    Select lines of a pandas.DataFrame based
    on a boolean condition.
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Input dataframe used for the query.
    condition : str, optional
        A boolean condition expressed as a string. If None or empty, no lines are selected.
    Returns
    -------
    list[Any]
        Index labels of the lines to select. Labels can be tuples for a MultiIndex.
    """
    if condition:
        return dataframe.query(condition).index.values.tolist()
    else:
        return []


def keep_right_contributions(
    y_pred: pd.DataFrame,
    contributions: pd.DataFrame | list[pd.DataFrame],
    _case: Literal["classification", "regression"],
    _classes: list[int | float | str] | None,
    label_dict: dict[Any, Any] | None,
    proba_values: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep the right contributions/summary for the right ypred.

    Parameters
    ----------
    y_pred : pandas.DataFrame
            User-specified prediction values.
    contributions : pandas.DataFrame or list[pandas.DataFrame]
        Local contributions or summarized contributions. A DataFrame is expected for regression,
        and a list of per-class DataFrames for classification.
    _case : {"classification", "regression"}
        String that informs if the model used is for classification or regression problem.
    _classes : list[int or float or str] or None
        Class labels for classification, or None for regression.
    label_dict : dict[Any, Any] or None
        Optional mapping from class labels to domain names.
    proba_values : pandas.DataFrame or None, optional
        Probability values for each row, when available.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Predictions (with probabilities appended when provided) and matching contributions.

    """
    if _case == "classification":
        if _classes is None:
            raise ValueError("_classes must be provided for classification.")
        complete_sum = [list(x) for x in list(zip(*[df.values.tolist() for df in contributions], strict=False))]
        indexclas = [_classes.index(x) for x in list(flatten(y_pred.values))]
        summary = pd.DataFrame(
            [summar[ind] for ind, summar in zip(indexclas, complete_sum, strict=False)],
            columns=contributions[0].columns,
            index=contributions[0].index,
            dtype=object,
        )
        if label_dict is not None:
            y_pred = y_pred.map(lambda x: label_dict[x])
        if proba_values is not None:
            y_proba = pd.DataFrame(
                [proba[ind] for ind, proba in zip(indexclas, proba_values.values, strict=False)],
                columns=["proba"],
                index=y_pred.index,
            )
            y_pred = pd.concat([y_pred, y_proba], axis=1)

    else:
        summary = contributions

    return y_pred, summary
