from typing import Union
from enum import Enum

import pandas as pd
import os
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype, is_categorical_dtype


class VarType(Enum):
    TYPE_CAT = "Categorical"
    TYPE_NUM = "Numeric"
    TYPE_UNSUPPORTED = "Unsupported"

    def __str__(self):
        return str(self.value)


def series_dtype(s: pd.Series) -> VarType:
    if is_bool_dtype(s):
        return VarType.TYPE_CAT
    elif is_string_dtype(s):
        return VarType.TYPE_CAT
    elif is_categorical_dtype(s):
        return VarType.TYPE_CAT
    elif is_numeric_dtype(s):
        return VarType.TYPE_NUM
    else:
        return VarType.TYPE_UNSUPPORTED


def numeric_is_continuous(s: pd.Series):
    # This test could probably be improved
    n_unique = s.nunique()
    return True if n_unique > 5 else False


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
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)

    else:
        return None
