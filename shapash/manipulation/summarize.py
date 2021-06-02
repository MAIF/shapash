"""
Summarize Module
"""
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from sklearn.manifold import TSNE
from shapash.utils.transform import get_features_transform_mapping


def summarize_el(dataframe, mask, prefix):
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
    summarized_matrix = [[x for x in l if str(x) != 'nan'] for l in matrix]
    # Padding to create pd.DataFrame
    max_length = max(len(l) for l in summarized_matrix)
    for elem in summarized_matrix:
        elem.extend([np.nan] * (max_length - len(elem)))
    # Create DataFrame
    col_list = [prefix + str(x + 1) for x in list(range(max_length))]
    df_summarized_matrix = pd.DataFrame(summarized_matrix,
                                        index=list(dataframe.index),
                                        columns=col_list,
                                        dtype=object)

    return df_summarized_matrix


def compute_features_import(dataframe):
    """
    Compute a relative features importance, sum of absolute values
     ​​of the contributions for each
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
    feat_imp = dataframe.abs().sum().sort_values(ascending=True)
    tot = feat_imp.sum()
    return feat_imp / tot

def summarize(s_contrib, var_dict, x_sorted, mask, columns_dict, features_dict):
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
    contrib_sum = summarize_el(s_contrib, mask, 'contribution_')
    var_dict_sum = summarize_el(var_dict, mask, 'feature_').applymap(
        lambda x: features_dict[columns_dict[x]] if not np.isnan(x) else x)
    x_sorted_sum = summarize_el(x_sorted, mask, 'value_')

    # Concatenate pd.DataFrame
    summary = pd.concat([contrib_sum, var_dict_sum, x_sorted_sum], axis=1)

    # Ordering columns
    ordered_columns = list(flatten(zip(var_dict_sum.columns, x_sorted_sum.columns, contrib_sum.columns)))
    summary = summary[ordered_columns]
    return summary


def group_contributions(contributions, features_groups):
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
    for group_name in features_groups.keys():
        new_contributions[group_name] = new_contributions[features_groups[group_name]].sum(axis=1)

    # Dropping features that are part of the group of features
    for features_grouped in features_groups.values():
        new_contributions = new_contributions.drop(features_grouped, axis=1)

    return new_contributions


def project_feature_values_1d(feature_values, col, x_pred, x_init, preprocessing):
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
    x_pred : pd.DataFrame
        Pandas dataframe before preprocessing transformations
    x_init : pd.DataFrame
        Pandas dataframe after preprocessing transformations
    preprocessing : category_encoders or ColumnTransformer or list or dict or list of dict
        The processing apply to the original data

    Returns
    -------
    feature_values : pd.Series
        Series containing the projected feature values.
    """
    # Getting mapping of variables to transform categorical features with corresponding encoded variables
    encoding_mapping = get_features_transform_mapping(x_pred, x_init, preprocessing)
    col_names_in_xinit = list()
    for c in feature_values.columns:
        col_names_in_xinit.extend(encoding_mapping.get(c, [c]))
    feature_values = x_init.loc[feature_values.index, col_names_in_xinit]
    # Project in 1D the feature values
    feature_values_proj_1d = TSNE(n_components=1, random_state=1).fit_transform(feature_values)
    feature_values = pd.Series(feature_values_proj_1d[:, 0], name=col, index=feature_values.index)
    return feature_values


def create_grouped_features_values(
        x_pred,
        x_init,
        preprocessing,
        features_groups,
        how='tsne'
) -> pd.DataFrame:
    """
    Compute projections of groups of features using t-sne.

    Parameters
    ----------
    x_pred : pd.DataFrame
        x_init dataset with inverse transformation with eventual postprocessing modifications.
    x_init : pd.DataFrame
        preprocessed dataset used by the model to perform the prediction.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional
        Preprocessing used to encode categorical variables.
    features_groups : dict
        Groups names and corresponding list of features
    how : str
        Method used to project groups of features in 1D (only t-sne available for now).

    Returns
    -------

    """
    df = x_pred.copy()
    if how != 'tsne':
        raise NotImplementedError('Projecting features in 1D only supports t-sne for now.')
    for group in features_groups.keys():
        if not isinstance(features_groups[group], list):
            raise ValueError(f'features_groups[{group}] should be a list of features')
        features_values = x_pred[features_groups[group]]
        df[group] = project_feature_values_1d(
            features_values,
            col=group,
            x_pred=x_pred,
            x_init=x_init,
            preprocessing=preprocessing
        )
        for f in features_groups[group]:
            if f in df.columns:
                df.drop(f, axis=1, inplace=True)

    return df
