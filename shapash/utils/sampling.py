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
    col_value_count: int = 0,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Samples a subset of indices for plotting, optionally creating a note for the plot subtitle.

    Parameters
    ----------
    selection : list, optional
        A list of indices specifying a subset of the DataFrame for plotting.
    max_points : int, optional
        The maximum number of points to plot. Defaults to 2000.
    col : str, optional
        The column name based on which intelligent sampling is performed.
    col_value_count : int, optional
        The count of unique values in the specified column. Used for determining sampling strategy.

    Returns
    -------
    tuple
        A tuple containing the selected indices and an additional note.
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
    col_value_count: int,
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Determines the sampling strategy based on the input parameters.
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
    col_value_count: int,
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Handles sampling when no specific selection is made.
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
    col_value_count: int,
    random_seed: int,
) -> tuple[list[Any] | np.ndarray, str | None]:
    """
    Handles sampling when a specific list of indices is provided.
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
    col_value_count: int,
    random_seed: int,
) -> list[Any] | np.ndarray:
    """
    Performs intelligent sampling based on the distribution of values in the specified column.
    """
    rng = np.random.default_rng(seed=random_seed)

    if isinstance(col, (tuple, list)) and len(col) == 2:
        return _intelligent_sampling_pair(data, max_points, col, random_seed, rng)

    is_col_str = True
    if data[col].dtype.kind in "fc":
        try:
            if data[col].str.isnumeric().all():
                is_col_str = False
        except AttributeError:
            is_col_str = False

    if (col_value_count < len(data[col]) / 20) or is_col_str:
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
    Performs intelligent sampling on a crossed pair of variables.

    For categorical-like pairs, sampling is balanced across joint modalities.
    For two numeric variables with enough variability, 2D KMeans clusters are used.
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
    left = series1.astype(object).where(~series1.isna(), "missing")
    right = series2.astype(object).where(~series2.isna(), "missing")
    return left.astype(str) + "||" + right.astype(str)


def _is_numeric_like(series: pd.Series) -> bool:
    if series.dtype.kind in "biufc":
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().all()


def _format_additional_note(df: pd.DataFrame, selected_indices: list[Any] | np.ndarray, additional_note: str) -> str:
    """
    Formats the additional note with the length and percentage of the selected subset.
    """
    percentage = int(np.round(100 * len(selected_indices) / df.shape[0]))
    return f"{additional_note}{len(selected_indices)} ({percentage}%)"
