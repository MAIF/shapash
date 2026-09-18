"""
Summarize Module
"""

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.core.common import flatten
from sklearn.manifold import TSNE

from shapash._optional import import_optional_module
from shapash.utils.transform import get_features_transform_mapping


def summarize_el(dataframe: pd.DataFrame, mask: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute a summarized Matrix.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Matrix containing contributions, label or feature names
        that will be summarized
    mask: pd.DataFrame
        Mask to apply during the summary step
    prefix: str
        prefix used for columns name

    Returns
    -------
    pd.DataFrame
        Result of the summarize step
    """
    matrix = dataframe.where(mask.to_numpy()).values.tolist()
    summarized_matrix = [[x for x in ll if str(x) != "nan"] for ll in matrix]
    # Padding to create pd.DataFrame
    max_length = max(len(ll) for ll in summarized_matrix)
    for elem in summarized_matrix:
        elem.extend([np.nan] * (max_length - len(elem)))
    # Create DataFrame
    col_list = [prefix + str(x + 1) for x in list(range(max_length))]
    df_summarized_matrix = pd.DataFrame(summarized_matrix, index=list(dataframe.index), columns=col_list, dtype=object)

    return df_summarized_matrix


def compute_features_import(dataframe: pd.DataFrame, norm: int | float = 1) -> pd.Series:
    """
    Compute a relative features importance, sum of absolute values
    of the contributions for each
    features importance compute in base 100
    Parameters
    ----------
    dataframe: pd.DataFrame
        Matrix containing all contributions

    Returns
    -------
    pd.Series
        feature importance One row by feature,
        index of the serie = dataframe.columns
    """
    feat_imp = (((dataframe.abs() ** norm).sum()) ** (1 / norm)).sort_values(ascending=True)
    tot = feat_imp.sum()
    return feat_imp / tot


def summarize(
    s_contrib: pd.DataFrame,
    var_dict: pd.DataFrame,
    x_sorted: pd.DataFrame,
    mask: pd.DataFrame,
    columns_dict: dict[Any, str],
    features_dict: dict[str, str],
) -> pd.DataFrame:
    """
    Compute the summarized contributions of features.

    Parameters
    ----------
    s_contrib: pd.DataFrame
        Matrix containing contributions that will be summarized
    var_dict: pd.DataFrame
        Matrix of feature names that will be summarized
    x_sorted: pd.DataFrame
        Matrix containing the value of each feature
    mask: pd.DataFrame
        Mask to apply during the summary step
    columns_dict:
        Dict of column Names, matches column num with column name
    features_dict:
        Dict of column Label, matches column name with column label

    Returns
    -------
    pd.DataFrame
        Result of the summarize step
    """
    contrib_sum = summarize_el(s_contrib, mask, "contribution_")
    var_dict_sum = summarize_el(var_dict, mask, "feature_").map(
        lambda x: features_dict[columns_dict[x]] if not np.isnan(x) else x
    )
    x_sorted_sum = summarize_el(x_sorted, mask, "value_")

    # Concatenate pd.DataFrame
    summary = pd.concat([contrib_sum, var_dict_sum, x_sorted_sum], axis=1)

    # Ordering columns
    ordered_columns = list(flatten(zip(var_dict_sum.columns, x_sorted_sum.columns, contrib_sum.columns, strict=False)))
    summary = summary[ordered_columns]
    return summary


def group_contributions(contributions: pd.DataFrame, features_groups: dict[str, list[str]]) -> pd.DataFrame:
    """
    Regroup contributions according to features_groups parameter

    Parameters
    ----------
    contributions : pd.DataFrame
        Contributions of each unique feature.
    features_groups : dict
        Python dict that inform which features to regroup.

    Returns
    -------
    contributions : pd.DataFrame
        Contributions with grouped features.
    """
    new_contributions = contributions.copy()
    # Computing features groups that are the sum of their corresponding features contributions
    for group_name, grouped_features in features_groups.items():
        new_contributions[group_name] = new_contributions[grouped_features].sum(axis=1)

    # Dropping features that are part of the group of features
    for features_grouped in features_groups.values():
        new_contributions = new_contributions.drop(features_grouped, axis=1)

    return new_contributions


def project_feature_values_1d(
    feature_values: pd.DataFrame,
    col: str,
    x_init: pd.DataFrame,
    x_encoded: pd.DataFrame,
    preprocessing: Any,
    features_dict: dict[str, str] | None,
    how: Literal["tsne", "dict_of_values"] = "tsne",
) -> pd.Series:
    """
    Project feature values of a group of features in 1 dimension.
    If feature_values contains categorical features, use preprocessing to get
    the corresponding encoded variables.

    Parameters
    ----------
    feature_values : pd.DataFrame
        DataFrame that contains the feature values
    col : str
        Name of the group of features.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional
        Preprocessing used to encode categorical variables.
    x_init : pd.DataFrame
        Pandas dataframe before preprocessing transformations
    x_encoded : pd.DataFrame
        Pandas dataframe after preprocessing transformations
    preprocessing : category_encoders or ColumnTransformer or list or dict or list of dict
        The processing apply to the original data
    features_dict: dict, optional (default: None)
        Dictionary mapping technical feature names to domain names.
    how : str
        Method used to compute groups of features values in one column. Options: "tsne", "dict_of_values"

    Returns
    -------
    feature_values : pd.Series
        Series containing the projected feature values.
    """
    # Getting mapping of variables to transform categorical features with corresponding encoded variables
    encoding_mapping = get_features_transform_mapping(x_init, x_encoded, preprocessing)
    col_names_in_xinit = list()
    for c in feature_values.columns:
        col_names_in_xinit.extend(encoding_mapping.get(c, [c]))
    feature_values = x_encoded.loc[feature_values.index, col_names_in_xinit]

    # Project in 1D the feature values
    if how == "tsne":
        try:
            n_samples = feature_values.shape[0]
            perplexity = min(30, max(2, n_samples // 3))
            feature_values_proj_1d = TSNE(
                n_components=1,
                perplexity=perplexity,
                learning_rate="auto",
                init="random",
                random_state=79,
            ).fit_transform(feature_values)
            feature_values = pd.Series(feature_values_proj_1d[:, 0], name=col, index=feature_values.index)
        except Exception as e:
            warnings.warn(
                f"Could not project group features values with t-SNE: {e}",
                UserWarning,
                stacklevel=2,
            )
            feature_values = pd.Series(feature_values.iloc[:, 0], name=col, index=feature_values.index)

    elif how == "dict_of_values":
        features_dict = features_dict or {}
        feature_values.columns = [features_dict.get(x, x) for x in feature_values.columns]
        feature_values = pd.Series(
            feature_values.apply(lambda x: x.to_dict(), axis=1), name=col, index=feature_values.index
        )

    else:
        raise NotImplementedError(f"Unknown method: {how}")

    return feature_values


def compute_corr(df: pd.DataFrame, compute_method: Literal["phik", "pearson"]) -> pd.DataFrame:
    """
    Compute correlations between features of given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame used to compute correlations.
    compute_method : str
        Method used to compute correlations ('phik' or 'pearson').

    Returns
    -------
    pd.DataFrame
    """
    # Remove user warnings (when not enough values to compute correlation).
    warnings.filterwarnings("ignore")
    if compute_method == "phik":
        phik = import_optional_module(
            "phik",
            extra="Falling back to pearson. Install with: pip install phik",
            errors="warn",
        )
        if phik is None:
            return df.corr(numeric_only=True)
        return phik.phik_matrix(df, verbose=False)
    elif compute_method == "pearson":
        return df.corr(numeric_only=True)
    else:
        raise NotImplementedError(f"Not implemented correlation method : {compute_method}")


def create_grouped_features_values(
    x_init: pd.DataFrame,
    x_encoded: pd.DataFrame,
    preprocessing: Any,
    features_groups: dict[str, list[str]],
    features_dict: dict[str, str] | None,
    how: Literal["tsne", "dict_of_values"] = "tsne",
) -> pd.DataFrame:
    """
    Compute projections of groups of features using t-sne.

    Parameters
    ----------
    x_init : pd.DataFrame
        x_encoded dataset with inverse transformation with eventual postprocessing modifications.
    x_encoded : pd.DataFrame
        preprocessed dataset used by the model to perform the prediction.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional
        Preprocessing used to encode categorical variables.
    features_groups : dict
        Groups names and corresponding list of features
    features_dict: dict, optional (default: None)
        Dictionary mapping technical feature names to domain names.
    how : str
        Method used to compute groups of features values in one column.

    Returns
    -------
    df : pd.DataFrame
        features values with projection used for groups of features
    """
    df = x_init.copy()
    for group, grouped_features in features_groups.items():
        if not isinstance(grouped_features, list):
            raise ValueError(f"features_groups[{group}] should be a list of features")
        features_values = x_init[grouped_features]
        df[group] = project_feature_values_1d(
            features_values,
            col=group,
            x_init=x_init,
            x_encoded=x_encoded,
            preprocessing=preprocessing,
            features_dict=features_dict,
            how=how,
        )
        for f in grouped_features:
            if f in df.columns:
                df.drop(f, axis=1, inplace=True)

    return df


def contribution_weighted_corr(contrib_i: pd.Series, contrib_j: pd.Series) -> float:
    """Compute a contribution-weighted correlation between two features.

    The correlation is a weighted Pearson correlation where each
    observation is weighted using the maximum absolute contribution
    of the two features:

        weight_k = max(abs(contrib_i_k), abs(contrib_j_k))

    This gives more importance to observations where at least one
    of the two features has a strong contribution to the prediction.

    Missing contribution values (NaN) are considered as zero contributions.

    Parameters
    ----------
    contrib_i : pd.Series
        Contribution values of the first feature, with shape (n_samples,).
    contrib_j : pd.Series
        Contribution values of the second feature, with shape (n_samples,).

    Returns
    -------
    float
        Contribution-weighted correlation between -1 and 1.
        Returns 0 if the correlation cannot be computed because
        at least one feature has no variation in its contribution values.

    Raises
    ------
    ValueError
        If inputs are not one-dimensional, have different lengths,
        are empty, or contain infinite values.
    """
    contrib_i_array = np.asarray(contrib_i, dtype=np.float64)
    contrib_j_array = np.asarray(contrib_j, dtype=np.float64)

    if contrib_i_array.ndim != 1 or contrib_j_array.ndim != 1:
        raise ValueError("Contribution values must be one-dimensional arrays.")

    if contrib_i_array.shape[0] != contrib_j_array.shape[0]:
        raise ValueError("Contribution arrays must contain the same number of observations.")

    if contrib_i_array.size == 0:
        raise ValueError("Contribution arrays must not be empty.")

    if np.any(np.isinf(contrib_i_array)) or np.any(np.isinf(contrib_j_array)):
        raise ValueError("Contribution values must not contain infinite values.")

    # Missing contribution values are considered as zero contributions.
    contrib_i_array = np.nan_to_num(contrib_i_array, nan=0.0)
    contrib_j_array = np.nan_to_num(contrib_j_array, nan=0.0)

    # Give more importance to observations where at least one
    # feature has a strong contribution.
    weights = np.maximum(np.abs(contrib_i_array), np.abs(contrib_j_array))

    weight_sum = float(np.sum(weights))

    # Both features have zero contribution values for every observation.
    if weight_sum == 0.0:
        return 0.0

    mean_i = float(np.average(contrib_i_array, weights=weights))
    mean_j = float(np.average(contrib_j_array, weights=weights))

    centered_i = contrib_i_array - mean_i
    centered_j = contrib_j_array - mean_j

    covariance = float(np.sum(weights * centered_i * centered_j))
    variance_i = float(np.sum(weights * centered_i**2))
    variance_j = float(np.sum(weights * centered_j**2))
    denominator = float(np.sqrt(variance_i * variance_j))

    # At least one feature has no variation in its contribution values.
    if denominator == 0.0:
        return 0.0

    return float(np.clip(covariance / denominator, -1.0, 1.0))


def contribution_weighted_corr_matrix(contrib_values: pd.DataFrame) -> pd.DataFrame:
    """Compute a contribution-weighted correlation matrix.

    For each pair of features, the correlation is computed using
    `contribution_weighted_corr`.

    The resulting matrix is symmetric and its diagonal is set to 1.

    Missing contribution values (NaN) are considered as zero contributions.

    Parameters
    ----------
    contrib_values : pd.DataFrame
        Contribution values with shape (n_samples, n_features).

    Returns
    -------
    pd.DataFrame
        Symmetric contribution-weighted correlation matrix with shape
        (n_features, n_features).

    Raises
    ------
    ValueError
        If contrib_values is not a non-empty two-dimensional array
        or contains infinite values.
    """
    contrib_array = np.asarray(contrib_values, dtype=np.float64)

    if contrib_array.ndim != 2:
        raise ValueError("contrib_values must have shape (n_samples, n_features).")

    if contrib_array.shape[0] == 0:
        raise ValueError("contrib_values must contain at least one observation.")

    if contrib_array.shape[1] == 0:
        raise ValueError("contrib_values must contain at least one feature.")

    if np.any(np.isinf(contrib_array)):
        raise ValueError("Contribution values must not contain infinite values.")

    # Missing contribution values are considered as zero contributions.
    contrib_array = np.nan_to_num(contrib_array, nan=0.0)

    n_features = contrib_array.shape[1]

    # Initialize the matrix with 1 on the diagonal.
    corr_matrix = np.eye(n_features, dtype=np.float64)

    # Only compute the upper triangular part of the matrix.
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = contribution_weighted_corr(contrib_array[:, i], contrib_array[:, j])

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return pd.DataFrame(corr_matrix, index=contrib_values.columns, columns=contrib_values.columns)
