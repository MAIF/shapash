import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def subset_sampling(df, selection=None, max_points=2000, col=None, col_value_count=0):
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


def _determine_sampling_strategy(df, selection, max_points, col, col_value_count, random_seed):
    """
    Determines the sampling strategy based on the input parameters.
    """
    if selection is None:
        return _no_selection_sampling(df, max_points, col, col_value_count, random_seed)
    elif isinstance(selection, list):
        return _list_selection_sampling(df, selection, max_points, col, col_value_count, random_seed)
    else:
        raise ValueError("Parameter 'selection' must be a list.")


def _no_selection_sampling(df, max_points, col, col_value_count, random_seed):
    """
    Handles sampling when no specific selection is made.
    """
    if df.shape[0] <= max_points:
        return df.index.tolist(), None
    elif col is None:
        selected_indices = random.sample(df.index.tolist(), max_points)
        return selected_indices, "Length of random Subset: "
    else:
        selected_indices = _intelligent_sampling(df, max_points, col, col_value_count, random_seed)
        return selected_indices, "Length of smart Subset: "


def _list_selection_sampling(df, selection, max_points, col, col_value_count, random_seed):
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
        selected_indices = _intelligent_sampling(subset, max_points, col, col_value_count, random_seed)
        return selected_indices, "Length of smart Subset: "


def _intelligent_sampling(data, max_points, col, col_value_count, random_seed):
    """
    Performs intelligent sampling based on the distribution of values in the specified column.
    """
    rng = np.random.default_rng(seed=random_seed)
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


def _format_additional_note(df, selected_indices, additional_note):
    """
    Formats the additional note with the length and percentage of the selected subset.
    """
    percentage = int(np.round(100 * len(selected_indices) / df.shape[0]))
    return f"{additional_note}{len(selected_indices)} ({percentage}%)"
