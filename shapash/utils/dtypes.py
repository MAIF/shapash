import pandas as pd
from pandas.api.types import is_string_dtype


def is_text_like(series: pd.Series, strict_object: bool = False) -> bool:
    """Return whether a series should be treated as text-like/categorical.

    Parameters
    ----------
    series : pd.Series
        Series to evaluate.
    strict_object : bool, default=False
        If True, ``object`` dtype is accepted only when inferred values are
        textual or empty. If False, all ``object`` dtype columns are accepted.
    """
    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return True

    if dtype.name == "object":
        if not strict_object:
            return True
        inferred_dtype = pd.api.types.infer_dtype(series, skipna=True)
        return inferred_dtype in ("string", "empty")

    return is_string_dtype(dtype)


def text_like_columns(df: pd.DataFrame, strict_object: bool = False) -> list[str]:
    """Return dataframe columns considered text-like by ``is_text_like``."""
    return [col for col in df.columns if is_text_like(df[col], strict_object=strict_object)]
