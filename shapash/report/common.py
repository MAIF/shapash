from enum import Enum

import pandas as pd
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


