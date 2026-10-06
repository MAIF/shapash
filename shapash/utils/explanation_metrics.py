from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def _df_to_array(instances: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """
    Transform inputs into arrays

    Parameters
    ----------
    instances : pandas.DataFrame, pandas.Series or numpy.ndarray
        Input data

    Returns
    -------
    numpy.ndarray
        Transformed features
    """
    if isinstance(instances, pd.DataFrame):
        return instances.values
    elif isinstance(instances, pd.Series):
        return np.array([instances.values])
    else:
        return instances


def _compute_distance(x1: np.ndarray, x2: np.ndarray, mean_vector: np.ndarray, epsilon: float = 0.0000001) -> float:
    """
    Compute distances between data points by using L1 on normalized data : sum(abs(x1-x2)/(mean_vector+epsilon))

    Parameters
    ----------
    x1 : numpy.ndarray
        First vector
    x2 : numpy.ndarray
        Second vector
    mean_vector : numpy.ndarray
        Each value of this vector is the std.dev for each feature in dataset
    epsilon : float, optional
        Small value added to the standard deviation to avoid division by zero, by default 0.0000001

    Returns
    -------
    diff : float
        Returns :math:`\\sum(\\frac{|x1-x2|}{mean\\_vector+epsilon})`
    """
    diff = np.sum(np.abs(x1 - x2) / (mean_vector + epsilon))
    return diff


def _compute_similarities(instance: np.ndarray, dataset: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between an instance and all other data points

    Parameters
    ----------
    instance : 1D numpy.ndarray
        Reference data point
    dataset : 2D numpy.ndarray
        Entire dataset used to identify neighbors

    Returns
    -------
    similarity_distance : array
        V[j] == distance between actual instance and instance j
    """
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    similarity_distance = np.zeros(dataset.shape[0])

    for j in range(0, dataset.shape[0]):
        # Calculate distance between point and instance j
        dist = _compute_distance(instance, dataset[j], mean_vector)
        similarity_distance[j] = dist

    return similarity_distance


def _get_radius(dataset: np.ndarray, n_neighbors: int, sample_size: int = 500, percentile: float = 95) -> float:
    """
    Calculate the maximum allowed distance between points to be considered as neighbors

    Parameters
    ----------
    dataset : numpy.ndarray
        Pool to sample from and calculate a radius
    n_neighbors : int
        Maximum number of neighbors considered per instance
    sample_size : int, optional
        Number of data points to sample from dataset, by default 500
    percentile : float, optional
        Percentile used to calculate the distance threshold, by default 95

    Returns
    -------
    radius : float
        Distance threshold
    """
    # Select 500 points max to sample
    size = min([dataset.shape[0], sample_size])
    # Randomly sample points from dataset
    rng = np.random.default_rng(seed=79)
    sampled_instances = dataset[rng.integers(0, dataset.shape[0], size), :]
    # Define normalization vector
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    # Initialize the similarity matrix
    similarity_distance = np.zeros((size, size))
    # Calculate pairwise distance between instances
    for i in range(size):
        for j in range(i, size):
            dist = _compute_distance(sampled_instances[i], sampled_instances[j], mean_vector)
            similarity_distance[i, j] = dist
            similarity_distance[j, i] = dist
    # Select top n_neighbors
    ordered_x = np.sort(similarity_distance)[:, 1 : n_neighbors + 1]
    # Select the value of the distance that captures XX% of all distances (percentile)
    return np.percentile(ordered_x.flatten(), percentile)


def find_neighbors(
    selection: list[Any],
    dataset: pd.DataFrame,
    model: Any,
    mode: Literal["classification", "regression"],
    n_neighbors: int = 10,
    return_positions: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
    """
    For each instance, select neighbors based on 3 criteria:

    1. First pick top N closest neighbors (L1 Norm + st. dev normalization)
    2. Filter neighbors whose model output is too different from instance (see condition below)
    3. Filter neighbors whose distance is too big compared to a certain threshold

    Parameters
    ----------
    selection : list[Any]
        Row labels to be displayed on the stability plot
    dataset : pandas.DataFrame
        Entire dataset used to identify neighbors
    model : Any
        ML model with a ``predict`` method for regression or a ``predict_proba`` method for classification
    mode : {"classification", "regression"}
        "classification" or "regression"
    n_neighbors : int, optional
        Top N neighbors initially allowed, by default 10
    return_positions : bool, optional
        Also return the dataset row positions in the same order as each neighborhood

    Returns
    -------
    list of numpy.ndarray
        Wrap all instances with corresponding neighbors in a list with length (#instances).
        Each array has shape (#neighbors, #features + 2), including the instance, its distance, and its prediction.
    all_positions : list of numpy.ndarray, optional
        Dataset row positions for each neighborhood, returned when ``return_positions`` is True.
    """
    instances = dataset.loc[selection].values
    selected_positions = dataset.index.get_indexer_for(selection)

    neighbor_rows = np.empty((0, instances.shape[1] + 1), float)
    all_positions = []
    """Filter 1 : Pick top N closest neighbors"""
    for selected_position, instance in zip(selected_positions, instances, strict=True):
        c = _compute_similarities(instance, dataset.values)
        # Pick indices of the closest neighbors (and include instance itself)
        neighbors_indices = np.argsort(c)[: n_neighbors + 1]
        if selected_position not in neighbors_indices:
            neighbors_indices[-1] = selected_position
        neighbors_indices = np.r_[selected_position, neighbors_indices[neighbors_indices != selected_position]]
        # Return instance with its neighbors
        neighbors = dataset.values[neighbors_indices]
        # Add distance column
        neighbors = np.append(neighbors, c[neighbors_indices].reshape(n_neighbors + 1, 1), axis=1)
        neighbor_rows = np.append(neighbor_rows, neighbors, axis=0)
        all_positions.append(neighbors_indices)

    # Calculate predictions for all instances and corresponding neighbors
    if mode == "regression":
        # For XGB it is necessary to add columns in df, otherwise columns mismatch
        predictions = model.predict(pd.DataFrame(neighbor_rows[:, :-1], columns=dataset.columns))
    elif mode == "classification":
        predictions = model.predict_proba(pd.DataFrame(neighbor_rows[:, :-1], columns=dataset.columns))[:, 1]

    # Add prediction column
    neighbor_rows = np.append(neighbor_rows, predictions.reshape(neighbor_rows.shape[0], 1), axis=1)
    # Split back into original chunks (1 chunck = instance + neighbors)
    all_neighbors = np.split(neighbor_rows, instances.shape[0])

    """Filter 2 : neighbors with similar blackbox output"""
    # Remove points if prediction is far away from instance prediction
    if mode == "regression":
        # Trick : use enumerate to allow the modifcation directly on the iterator
        for i, neighbors in enumerate(all_neighbors):
            keep = abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1 * abs(neighbors[0, -1])
            all_neighbors[i] = neighbors[keep]
            all_positions[i] = all_positions[i][keep]
    elif mode == "classification":
        for i, neighbors in enumerate(all_neighbors):
            keep = abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1
            all_neighbors[i] = neighbors[keep]
            all_positions[i] = all_positions[i][keep]

    """Filter 3 : neighbors below a distance threshold"""
    # Remove points if distance is bigger than radius
    radius = _get_radius(dataset.values, n_neighbors)

    for i, neighbors in enumerate(all_neighbors):
        # -2 indicates the distance column
        keep = neighbors[:, -2] < radius
        all_neighbors[i] = neighbors[keep]
        all_positions[i] = all_positions[i][keep]
    if return_positions:
        return all_neighbors, all_positions
    return all_neighbors


def shap_neighbors(
    instance: np.ndarray,
    x_encoded: pd.DataFrame,
    contributions: pd.DataFrame | list[pd.DataFrame],
    mode: Literal["classification", "regression"],
    neighbor_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For an instance and corresponding neighbors, calculate various
    metrics (described below) that are useful to evaluate local stability

    Parameters
    ----------
    instance : numpy.ndarray
        Instance + neighbours with corresponding features
    x_encoded : pandas.DataFrame
        Entire dataset used to identify neighbors
    contributions : pandas.DataFrame or list of pandas.DataFrame
        Calculated contribution values for the dataset, optionally one DataFrame per class
    mode : {"classification", "regression"}
        Prediction task. For binary classification, contributions for the positive class are used.
    neighbor_positions : numpy.ndarray, optional
        Dataset row positions returned by ``find_neighbors``. Required to distinguish rows
        with identical feature values.

    Returns
    -------
    tuple of numpy.ndarray
        ``(norm_shap_values, average_diff, norm_abs_shap_values[0, :])``

        norm_shap_values : numpy.ndarray
        Normalized SHAP values (with corresponding sign) in neighborhood order: the selected
        instance first, followed by its neighbors in distance order.
        average_diff : numpy.ndarray
        Variability (stddev / mean) of normalized SHAP values (using L1) across neighbors for each feature
        norm_abs_shap_values[0, :] : numpy.ndarray
        Normalized absolute SHAP value of the instance

    Raises
    ------
    ValueError
        If a list of contributions is incompatible with the selected mode, or if
        ``neighbor_positions`` is omitted and duplicate feature rows make the
        neighborhood row identities ambiguous.
    """
    # Extract SHAP values for instance and neighbors
    # :-2 indicates that two columns are disregarded : distance to instance and model output
    # If classification, select contrbutions of one class only
    if isinstance(contributions, list):
        if mode == "classification" and len(contributions) == 2:
            contributions = contributions[1]
        else:
            raise ValueError("Expected a single contribution DataFrame for the selected mode")
    if neighbor_positions is None:
        ind = (
            pd.merge(pd.DataFrame(instance[:, :-2], columns=x_encoded.columns), x_encoded.reset_index(), how="inner")
            .set_index(x_encoded.index.name if x_encoded.index.name is not None else "index")
            .index
        )
        if len(ind) != len(instance):
            raise ValueError("Feature rows are not unique; pass neighbor_positions to identify the neighbors")
        shap_values = contributions.loc[ind]
    else:
        shap_values = contributions.iloc[neighbor_positions]
    # For neighbors comparison, the sign of SHAP values is taken into account
    norm_shap_values = normalize(shap_values, axis=1, norm="l1")
    # But not for the average impact of the features across the dataset
    norm_abs_shap_values = normalize(np.abs(shap_values), axis=1, norm="l1")
    # Compute the average difference between the instance and its neighbors
    # And replace NaN with 0
    average_diff = np.divide(
        norm_shap_values.std(axis=0),
        norm_abs_shap_values.mean(axis=0),
        out=np.zeros(norm_abs_shap_values.shape[1]),
        where=norm_abs_shap_values.mean(axis=0) != 0,
    )

    return norm_shap_values, average_diff, norm_abs_shap_values[0, :]


def get_min_nb_features(
    selection: list[Any],
    contributions: pd.DataFrame | list[pd.DataFrame],
    mode: Literal["classification", "regression"],
    distance: float,
) -> list[int]:
    """
    Determine the minimum number of features needed for the prediction \
    of the interpretability method to be *close enough* \
    to the one obtained with all features.

    The closeness is defined via the following distances:

    * For regression:

        .. math::

            distance = \\frac{|output_{allFeatures} - output_{currentFeatures}|}{|output_{allFeatures}|}

    * For classification:

        .. math::

            distance = |output_{allFeatures} - output_{currentFeatures}|

    Parameters
    ----------
    selection : list[Any]
        Row labels to analyze
    contributions : pandas.DataFrame or list of pandas.DataFrame
        Calculated contribution values for the dataset, optionally one DataFrame per class
    mode : {"classification", "regression"}
        "classification" or "regression"
    distance : float
        How close we want to be from the model with all features, between 0 and 1

    Returns
    -------
    list of int
        List of minimum number of required features (for each instance) to be close enough to the prediction (ex: [4, 7, 8...])
    """
    if not (0 <= distance <= 1):
        raise ValueError("Distance should be between 0 and 1")

    if isinstance(contributions, list):
        if mode == "classification" and len(contributions) == 2:
            contributions = contributions[1]
        else:
            raise ValueError("Expected a single contribution DataFrame for the selected mode")
    contributions = contributions.loc[selection].values
    features_needed = []
    # For each instance, add features one by one (ordered by SHAP) until we get close enough
    for i in range(contributions.shape[0]):
        ids = np.flip(np.argsort(np.abs(contributions[i, :])))
        output_value = np.sum(contributions[i, :])

        score = 0
        for j, idx in enumerate(ids):  # noqa: B007
            # j : number of features needed
            # idx : positions of the j top shap values
            score += contributions[i, idx]
            # CLOSE_ENOUGH
            if mode == "regression":
                if abs(score - output_value) < distance * abs(output_value):
                    break
            elif mode == "classification":
                if abs(score - output_value) < distance:
                    break
        features_needed.append(j + 1)
    return features_needed


def get_distance(
    selection: list[Any],
    contributions: pd.DataFrame | list[pd.DataFrame],
    mode: Literal["classification", "regression"],
    nb_features: int,
) -> np.ndarray:
    """
    Determine how close we get to the output with all features by using only a subset of them

    Parameters
    ----------
    selection : list[Any]
        Row labels to analyze
    contributions : pandas.DataFrame or list of pandas.DataFrame
        Calculated contribution values for the dataset, optionally one DataFrame per class
    mode : {"classification", "regression"}
        "classification" or "regression"
    nb_features : int
        Number of features used

    Returns
    -------
    numpy.ndarray
        List of distances for each instance by using top selected features (ex: np.array([0.12, 0.16...])).

        * For regression:

            * normalized distance between the output of current model and output of full model

        * For classification:

            * distance between probability outputs (absolute value)
    """
    if isinstance(contributions, list):
        if mode == "classification" and len(contributions) == 2:
            contributions = contributions[1]
        else:
            raise ValueError("Expected a single contribution DataFrame for the selected mode")
    if nb_features > contributions.shape[1]:
        raise ValueError(
            f"nb_features ({nb_features}) exceeds the number of available features ({contributions.shape[1]})"
        )
    contributions = contributions.loc[selection].values
    top_features = np.array([sorted(row, key=abs, reverse=True) for row in contributions])[:, :nb_features]
    output_top_features = np.sum(top_features[:, :], axis=1)
    output_all_features = np.sum(contributions[:, :], axis=1)

    if mode == "regression":
        distance = abs(output_top_features - output_all_features) / abs(output_all_features)
    elif mode == "classification":
        distance = abs(output_top_features - output_all_features)
    return distance
