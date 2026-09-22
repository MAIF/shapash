import random
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def subset_sampling(
    df: pd.DataFrame,
    selection: list[Any] | None = None,
    max_points: int = 2000,
    col: str | tuple[str, str] | list[str] | None = None,
    col_value_count: int | tuple[int, int] = 0,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Samples a subset of indices for plotting, optionally creating a note for the plot subtitle.

    Parameters
    ----------
    selection : list, optional
        Explicit row indices specifying a subset of the DataFrame for plotting.
        If None, sampling is performed over the full DataFrame.
    max_points : int, optional
        The maximum number of points to plot. Defaults to 2000.
    col : str or tuple(str, str) or list[str], optional
        Column name, crossed pair of column names, or list of column names used
        to drive intelligent sampling.
    col_value_count : int or tuple(int, int), optional
        Number of unique values in the specified column, or per-column counts
        when sampling from a crossed pair of features.

    Returns
    -------
    tuple
        A tuple containing the selected indices and an optional note.
    """
    random_seed = 79
    random.seed(random_seed)

    # Determine the sampling strategy
    selected_indices, additional_note = _determine_sampling_strategy(
        df, selection, max_points, col, col_value_count, random_seed
    )

    # Format the additional note
    if additional_note is not None:
        additional_note = _format_additional_note(df, selected_indices, additional_note)

    return selected_indices, additional_note


def _determine_sampling_strategy(
    df: pd.DataFrame,
    selection: list[Any] | None,
    max_points: int,
    col: str | tuple[str, str] | list[str] | None,
    col_value_count: int | tuple[int, int],
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Determine the sampling strategy based on the input parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe used for sampling.
    selection : list, optional
        Explicit row indices to keep. If None, sampling is performed on the
        full dataframe.
    max_points : int
        Maximum number of rows to return.
    col : str or tuple(str, str) or list[str], optional
        Column name or pair of column names used to drive intelligent sampling.
    col_value_count : int or tuple(int, int)
        Cardinality information associated with ``col``.
    random_seed : int
        Seed used for deterministic sampling.

    Returns
    -------
    tuple
        Selected indices and an optional note describing the strategy.
    """
    if selection is None:
        return _no_selection_sampling(df, max_points, col, col_value_count, random_seed)
    elif isinstance(selection, list):
        return _list_selection_sampling(df, selection, max_points, col, col_value_count, random_seed)
    else:
        raise ValueError("Parameter 'selection' must be a list.")


def _no_selection_sampling(
    df: pd.DataFrame,
    max_points: int,
    col: str | tuple[str, str] | list[str] | None,
    col_value_count: int | tuple[int, int],
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Handle sampling when no explicit selection is provided.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe used for sampling.
    max_points : int
        Maximum number of rows to return.
    col : str or tuple(str, str) or list[str], optional
        Column name or pair of column names used to drive intelligent sampling.
    col_value_count : int or tuple(int, int)
        Cardinality information associated with ``col``.
    random_seed : int
        Seed used for deterministic sampling.

    Returns
    -------
    tuple
        Selected indices and an optional note describing the strategy.
    """
    if df.shape[0] <= max_points:
        return df.index.tolist(), None
    elif col is None:
        selected_indices = random.sample(df.index.tolist(), max_points)
        return selected_indices, "Length of random Subset: "
    else:
        return _intelligent_sampling(df, max_points, col, col_value_count, random_seed), "Length of smart Subset: "


def _list_selection_sampling(
    df: pd.DataFrame,
    selection: list[Any],
    max_points: int,
    col: str | tuple[str, str] | list[str] | None,
    col_value_count: int | tuple[int, int],
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Handle sampling when explicit indices are provided.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe used for sampling.
    selection : list
        Explicit row indices to keep before optional downsampling.
    max_points : int
        Maximum number of rows to return.
    col : str or tuple(str, str) or list[str], optional
        Column name or pair of column names used to drive intelligent sampling.
    col_value_count : int or tuple(int, int)
        Cardinality information associated with ``col``.
    random_seed : int
        Seed used for deterministic sampling.

    Returns
    -------
    tuple
        Selected indices and an optional note describing the strategy.
    """
    if len(selection) <= max_points:
        return selection, "Length of user-defined Subset: "
    elif col is None:
        selected_indices = random.sample(selection, max_points)
        return selected_indices, "Length of random Subset: "
    else:
        subset = df.loc[selection]
        return (
            _intelligent_sampling(subset, max_points, col, col_value_count, random_seed),
            "Length of smart Subset: ",
        )


def _intelligent_sampling(
    data: pd.DataFrame,
    max_points: int,
    col: str | tuple[str, str] | list[str] | None,
    col_value_count: int | tuple[int, int],
    random_seed: int,
) -> list[Any] | np.ndarray:
    """
    Perform intelligent sampling based on the distribution of values in the specified column.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe from which rows are sampled.
    max_points : int
        Maximum number of rows to return.
    col : str or tuple(str, str) or list[str], optional
        Column name or pair of column names used to define sampling groups.
    col_value_count : int or tuple(int, int)
        Cardinality information associated with ``col``.
    random_seed : int
        Seed used for deterministic sampling.

    Returns
    -------
    list or numpy.ndarray
        Selected row indices.
    """
    rng = np.random.default_rng(seed=random_seed)

    if isinstance(col, (tuple, list)) and len(col) == 2:
        return _intelligent_sampling_pair(data, max_points, col, random_seed, rng)

    if isinstance(col_value_count, tuple):
        scalar_col_value_count = max(col_value_count)
    else:
        scalar_col_value_count = int(col_value_count)

    is_col_str = True
    if data[col].dtype.kind in "fc":
        try:
            if data[col].str.isnumeric().all():
                is_col_str = False
        except AttributeError:
            is_col_str = False

    if (scalar_col_value_count < len(data[col]) / 20) or is_col_str:
        cluster_labels = data[col]
        cluster_counts = cluster_labels.value_counts()
    else:
        n_clusters = min(100, len(data[col]) // 20)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
        cluster_labels = pd.Series(kmeans.fit_predict(data[col].values.reshape(-1, 1)))
        cluster_counts = cluster_labels.value_counts()

    weights = cluster_counts.apply(lambda x: (x**0.5) / x).to_dict()
    selection_weights = cluster_labels.apply(lambda x: weights[x])
    selection_weights /= selection_weights.sum()
    selected_indices = rng.choice(data.index.tolist(), max_points, p=selection_weights, replace=False)
    return selected_indices


def _intelligent_sampling_pair(
    data: pd.DataFrame,
    max_points: int,
    col: tuple[str, str] | list[str],
    random_seed: int,
    rng: np.random.Generator,
) -> list[Any] | np.ndarray:
    """
    Perform intelligent sampling on a crossed pair of variables.

    For categorical-like pairs, sampling is balanced across joint modalities.
    For two numeric variables with enough variability, 2D KMeans clusters are used.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe from which rows are sampled.
    max_points : int
        Maximum number of rows to return.
    col : tuple(str, str) or list[str]
        Pair of column names used to build the crossed sampling space.
    random_seed : int
        Seed used for deterministic clustering.
    rng : numpy.random.Generator
        Random generator used for weighted row sampling.

    Returns
    -------
    list or numpy.ndarray
        Selected row indices.
    """
    col1, col2 = col

    joint_labels = _build_joint_labels(data[col1], data[col2])
    both_numeric = _is_numeric_like(data[col1]) and _is_numeric_like(data[col2])

    # Keep a categorical-like strategy when the crossed space has few modalities.
    low_joint_cardinality = joint_labels.nunique(dropna=False) < len(joint_labels) / 20

    if both_numeric and not low_joint_cardinality:
        n_clusters = min(100, len(data) // 20)
        if n_clusters < 2:
            return rng.choice(data.index.tolist(), max_points, replace=False)

        numeric_df = data[[col1, col2]].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.fillna(numeric_df.median()).fillna(0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto")
        cluster_labels = pd.Series(kmeans.fit_predict(numeric_df.values), index=data.index)
    else:
        cluster_labels = joint_labels

    cluster_counts = cluster_labels.value_counts()
    weights = cluster_counts.apply(lambda x: (x**0.5) / x).to_dict()
    selection_weights = cluster_labels.apply(lambda x: weights[x])
    selection_weights /= selection_weights.sum()
    selected_indices = rng.choice(data.index.tolist(), max_points, p=selection_weights, replace=False)
    return selected_indices


def _build_joint_labels(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Build crossed string labels for two feature series.

    Missing values are replaced by the literal ``"missing"`` before the two
    values are concatenated with ``"||"``.

    Parameters
    ----------
    series1 : pd.Series
        First feature series.
    series2 : pd.Series
        Second feature series.

    Returns
    -------
    pd.Series
        Crossed labels combining both series values.
    """
    left = series1.astype(object).where(~series1.isna(), "missing")
    right = series2.astype(object).where(~series2.isna(), "missing")
    return left.astype(str) + "||" + right.astype(str)


def _is_numeric_like(series: pd.Series) -> bool:
    """
    Return whether a series can be treated as numeric for sampling.

    Parameters
    ----------
    series : pd.Series
        Series to inspect.

    Returns
    -------
    bool
        True if the series is already numeric or can be fully coerced to
        numeric values.
    """
    if series.dtype.kind in "biufc":
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().all()


def _format_additional_note(df: pd.DataFrame, selected_indices: list[Any] | np.ndarray, additional_note: str) -> str:
    """
    Format the additional note with the length and percentage of the selected subset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe before sampling.
    selected_indices : list or numpy.ndarray
        Selected row indices.
    additional_note : str
        Sampling note prefix.

    Returns
    -------
    str
        Human-readable sampling note including row count and percentage.
    """
    percentage = int(np.round(100 * len(selected_indices) / df.shape[0]))
    return f"{additional_note}{len(selected_indices)} ({percentage}%)"
