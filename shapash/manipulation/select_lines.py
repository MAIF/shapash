"""
Select Lines Module
"""
import pandas as pd
from pandas.core.common import flatten


def select_lines(dataframe, condition=None):
    """
    Select lines of a pandas.DataFrame based
    on a boolean condition.
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe used for the query.
    condition : string
        A boolean condition expressed as string.
    Returns
    -------
    list
         list of indices, or lines, to select.
    """
    if condition:
        return dataframe.query(condition).index.values.tolist()
    else:
        return []


def keep_right_contributions(y_pred, contributions, _case, _classes, label_dict, proba_values=None):
    """
    Keep the right contributions/summary for the right ypred.

    Parameters
    ----------
    ypred: pandas.DataFrame (optional)
            User-specified prediction values.
    contributions: pandas.DataFrame (regression) or list (classification) (optional)
        local contributions aggregated if the preprocessing part requires it (e.g. one-hot encoding)
        or Result of the summarize step.
    _case: string
        String that informs if the model used is for classification or regression problem.
    _classes: list, None
        List of labels if the model used is for classification problem, None otherwise.
    label_dict: dict (optional)
        Dictionary mapping integer labels to domain names (classification - target values).
    proba_values: pandas.DataFrame
        the proba values for each row of the specified dataset

    """
    if _case == "classification":
        complete_sum = [list(x) for x in list(zip(*[df.values.tolist() for df in contributions]))]
        indexclas = [_classes.index(x) for x in list(flatten(y_pred.values))]
        summary = pd.DataFrame(
            [summar[ind] for ind, summar in zip(indexclas, complete_sum)],
            columns=contributions[0].columns,
            index=contributions[0].index,
            dtype=object,
        )
        if label_dict is not None:
            y_pred = y_pred.applymap(lambda x: label_dict[x])
        if proba_values is not None:
            y_proba = pd.DataFrame(
                [proba[ind] for ind, proba in zip(indexclas, proba_values.values)],
                columns=["proba"],
                index=y_pred.index,
            )
            y_pred = pd.concat([y_pred, y_proba], axis=1)

    else:
        summary = contributions

    return y_pred, summary
