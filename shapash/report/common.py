from typing import Union, Optional
from enum import Enum

import pandas as pd
import os
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype, is_categorical_dtype


class VarType(Enum):
    """
    Helper class to indicate the type of a variable.
    """
    TYPE_CAT = "Categorical"
    TYPE_NUM = "Numeric"
    TYPE_UNSUPPORTED = "Unsupported"

    def __str__(self):
        return str(self.value)


def series_dtype(s: pd.Series) -> VarType:
    """
    Computes the type of a pandas series.

    Parameters
    ----------
    s : pd.Series
        The series for which we wish to determine the type.

    Returns
    -------
    VarType
    """
    if is_bool_dtype(s):
        return VarType.TYPE_CAT
    elif is_string_dtype(s):
        return VarType.TYPE_CAT
    elif is_categorical_dtype(s):
        return VarType.TYPE_CAT
    elif is_numeric_dtype(s):
        if numeric_is_continuous(s):
            return VarType.TYPE_NUM
        else:
            return VarType.TYPE_CAT
    else:
        return VarType.TYPE_UNSUPPORTED


def numeric_is_continuous(s: pd.Series):
    """
    Function that returns True if a numeric pandas series is continuous and False if it is categorical.

    Parameters
    ----------
    s : pd.Series

    Returns
    -------
    bool
    """
    # This test could probably be improved
    n_unique = s.nunique()
    return True if n_unique > 15 else False


def compute_col_types(df_all: Optional[pd.DataFrame]) -> Optional[dict]:
    """
    Computes the type of each column and stores the result in a dict.

    Parameters
    ----------
    df_all : pd.DataFrame, optional

    Returns
    -------
    col_types : dict
        The types of each column
    """
    if df_all is None:
        return {}
    return {col: series_dtype(df_all[col]) for col in df_all.columns}


def get_callable(path: str):
    """
    This function is similar to the _locate function in Hydra library
    Locate an object by name or dotted path, importing as necessary.
    """
    if path == "":
        raise ImportError("Empty path")
    import builtins
    from importlib import import_module

    parts = [part for part in path.split(".") if part]
    module = None
    for n in reversed(range(len(parts))):
        try:
            mod = ".".join(parts[:n])
            module = import_module(mod)
        except Exception as e:
            if n == 0:
                raise ImportError(f"Error loading module '{path}'") from e
            continue
        if module:
            break
    if module:
        obj = module
    else:
        obj = builtins
    for part in parts[n:]:
        mod = mod + "." + part
        if not hasattr(obj, part):
            try:
                import_module(mod)
            except Exception as e:
                raise ImportError(
                    f"Encountered error: `{e}` when loading module '{path}'"
                ) from e
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj_type: type = obj
        return obj_type
    elif callable(obj):
        obj_callable = obj
        return obj_callable
    else:
        # dummy case
        raise ValueError(f"Invalid type ({type(obj)}) found for {path}")


def load_saved_df(path: str) -> Union[pd.DataFrame, None]:
    """
    Loads a pandas DataFrame that was saved using pd.to_csv method.

    Parameters
    ----------
    path : str
        Path to the dataframe object

    Returns
    -------
    pd.DataFrame or None
    """
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)

    else:
        return None


def compute_top_correlations_features(corr: pd.DataFrame, max_features: int) -> list:
    """
    Returns the max_features features having top correlations.

    Parameters
    ----------
    corr: pd.DataFrame
    max_features : int

    Returns
    -------
    list
    """
    sorted_corr = corr.abs().unstack().sort_values(kind="quicksort")[::-1]
    set_features = set()
    i = 0
    while len(set_features) < max_features and i < len(sorted_corr):
        if sorted_corr.index[i][0] != sorted_corr.index[i][1]:
            set_features.add(sorted_corr.index[i][0])
            # Last iteration can add one more feature otherwise
            if len(set_features) != max_features:
                set_features.add(sorted_corr.index[i][1])
        i += 1
    return list(set_features)
