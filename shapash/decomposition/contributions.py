"""
Contributions
TODO: Describe this code
"""
from itertools import compress
from collections import OrderedDict
import pandas as pd
import numpy as np
import category_encoders as ce


def compute_contributions(x_df, explainer, preprocessing=None):
    """
    Compute Shapley contributions of a prediction set for a certain model.

    Parameters
    ----------
    x_df: pandas.DataFrame
        Prediction set : features to be preprocessed before using SHAP.
    explainer: object
        Any SHAP explainer already initialized with a model.
    preprocessing: object, optional (default: None)
        A single transformer, from sklearn or category_encoders

    Returns
    -------
    pandas.DataFrame
        Shapley contributions of the model on the prediction set, as computed by the explainer.
    """
    if preprocessing:
        x_df = preprocessing.transform(x_df)
    shap_values = explainer.shap_values(x_df)
    res = pd.DataFrame()
    if isinstance(shap_values, list):
        res = [pd.DataFrame(data=tab, index=x_df.index, columns=x_df.columns) for tab in shap_values]
    elif isinstance(shap_values, np.ndarray):
        res = pd.DataFrame(data=shap_values, index=x_df.index, columns=x_df.columns)
    bias = explainer.expected_value
    return res, bias


def make_mapping_dict(ce_mapping):
    """
    creates a dict of relations between key (columns in orignal dataset)
    and values (columns in encoded dataset).
    Parameters
    ----------
    ce_mapping : list
        mapping from category encoder
        (contains mapping of transformation for each transformed column)
        for instance for binary encoder, ce_maaping would look like
            [{
            'col': 'Pclass',
            'mapping': pd.Dataframe(
                data = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]],
                columns = ['Pclass_0', 'Pclass_1', 'Pclass_2'])
            },
            {
            'col': 'Age',
            'mapping': pd.Dataframe(
                data = [[0, 1], [1, 0]],
                columns = ['Age_0', 'Age_1'])
            }]

    Returns
    -------
    dict
        key = original column name , values = transformed columns names
    """

    if not isinstance(ce_mapping, list):
        raise Exception(f'Mapping {type(ce_mapping)} must be a list.')

    return {switch.get('col'): switch.get('mapping').columns.tolist() for switch in ce_mapping}


def get_origin_column_name(transformed_column_name, mapping_dict: dict):
    """
    Returns the inverse transform column name in the original dataset for transformed_column_name.
    get_origin_column_name(
        transformed_column_name = 'Pclass_0',
        mapping_dict = [
            {
              'col': 'Pclass',
              'mapping': pd.Dataframe(data = [...],columns=['Pclass_0', 'Pclass_1'])
            }
        ]
    )
    will return 'Pclass'
    Parameters
    ----------
    transformed_column_name : str
        name of column in the encoded dataset
    mapping_dict : list of dict
        mapping of the used encoder

    Returns
    -------
    str
        name of the column in the original dataset
    """

    keys_list = list(mapping_dict.keys())
    check_column_in_values = [transformed_column_name in key for key in mapping_dict.values()]
    if sum(check_column_in_values) == 1:
        return list(compress(keys_list, check_column_in_values))[0]
    else:
        return transformed_column_name


def inverse_transform_contributions(contributions, preprocessing):
    """
    Computes the local contributions for the original dataset (before preprocesing)

    Parameters
    ----------
    contributions : pandas.DataFrame
        matrix of local contributions (# samples x # features)
    preprocessing : category_encoders or None
        unique encoder from category_encoders package within :
        One hot encoder, Ordinal encoder, Binary encoder, BaseN encoder.

    Returns
    -------
    pandas.Dataframe
        matrix of agregated shap values (# samples x # features)
    """

    # TODO : add sklearn encoders
    # TODO : add standard scaler (not in category encoder package)

    if not isinstance(contributions, pd.DataFrame):
        raise Exception(f'Shap values must be a pandas dataframe.')

    if preprocessing is None:
        contributions_agg = contributions

    else:
        mapping_dict = dict()
        encoded_columns = contributions.columns.to_list()
        need_column_factorisation = True

        if isinstance(preprocessing, ce.OneHotEncoder):
            mapping_dict = make_mapping_dict(preprocessing.mapping)

        elif isinstance(preprocessing, ce.BinaryEncoder):
            mapping_dict = make_mapping_dict(preprocessing.base_n_encoder.mapping)

        elif isinstance(preprocessing, ce.BaseNEncoder):
            mapping_dict = make_mapping_dict(preprocessing.mapping)

        else:
            need_column_factorisation = False

        if need_column_factorisation:
            all_origin_columns = [get_origin_column_name(x, mapping_dict) for x in encoded_columns]
            unique_origin_columns = list(OrderedDict.fromkeys(all_origin_columns).keys())
            # operator_matrix is a (n,p) matrix , with n = nb of columns in original dataset and
            #                                          p = nb of columns in encoded dataset
            # operator_matrix(i,j) contains 1 if encoded column j is associated to original column i
            #                               0 if not
            operator_matrix = [
                [(e == o) * 1 for o in unique_origin_columns]
                for e in all_origin_columns
            ]
            contributions_agg = pd.DataFrame(
                data=contributions.dot(operator_matrix).values,
                columns=unique_origin_columns,
                index=contributions.index
            )

        else:
            contributions_agg = contributions

    return contributions_agg


def rank_contributions(s_df, x_df):
    """
    Function to sort contributions and input features
    by decreasing contribution absolute values

    Parameters
    ----------
    s_df: pandas.DataFrame
        Local contributions dataframe.
    x_df: pandas.DataFrame
        Input features.

    Returns
    -------
    pandas.DataFrame
        Local contributions sorted by decreasing absolute values.
    pandas.DataFrame
        Input features sorted by decreasing contributions absolute values.
    pandas.DataFrame
        Input features names sorted for each observation
        by decreasing contributions absolute values.
    """
    argsort = np.argsort(-np.abs(s_df.values), axis=1)
    sorted_contrib = np.take_along_axis(s_df.values, argsort, axis=1)
    sorted_features = np.take_along_axis(x_df.values, argsort, axis=1)

    contrib_col = ['contribution_' + str(i) for i in range(s_df.shape[1])]
    col = ['feature_' + str(i) for i in range(s_df.shape[1])]

    s_dict = pd.DataFrame(data=argsort, columns=col, index=x_df.index)
    s_ord = pd.DataFrame(data=sorted_contrib, columns=contrib_col, index=x_df.index)
    x_ord = pd.DataFrame(data=sorted_features, columns=col, index=x_df.index)
    return [s_ord, x_ord, s_dict]
