try:
    from acv_explainers import ACVTree
    from acv_explainers.utils import get_null_coalition
    is_acv_available = True
except ImportError:
    is_acv_available = False

from typing import Any, Optional, List
import logging

import pandas as pd
import numpy as np
from shapash.utils.model_synoptic import simple_tree_model, catboost_model
from shapash.utils.transform import get_features_transform_mapping


def active_shapley_values(
        model: Any,
        x_pred: pd.DataFrame,
        x_init: pd.DataFrame,
        x_train: Optional[pd.DataFrame] = None,
        c: Optional[List[List[int]]] = None,
        explainer: Optional[Any] = None,
        preprocessing: Optional[Any] = None
):
    """
    Compute the active Shapley values using ACV package.
    Parameters
    ----------
    model: model object from sklearn, catboost, xgboost or lightgbm library
        this model is used to choose a shap explainer and to compute
        active shapley values
    x_init  : pd.DataFrame
        x_init dataset with inverse transformation with eventual postprocessing modifications.
    x_pred : pd.DataFrame
        preprocessed dataset used by the model to perform the prediction.
    x_train : pd.DataFrame, Optional
        Training dataset used as background.
    c : list
        list of coalitions used by acv. If not set, will try to get the list
        of one hot encoded list of variables using the encoder if they exist.
    explainer : explainer object from shap, optional (default: None)
        this explainer is used to compute shapley values
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
    Returns
    -------
    np.array or list of np.array
    """
    if is_acv_available is False:
        raise ValueError(
            """
            Active Shapley values requires the ACV package,
            which can be installed using 'pip install acv-exp'
            """
                    )
    # Using train set as background if available. If not, we will use test set.
    if x_train is not None:
        data = np.array(x_train.values, dtype=np.double)
    else:
        logging.warning("No train set passed. We recommend to pass the x_train parameter "
                        "in order to avoid errors.")
        data = np.array(x_init.values, dtype=np.double)

    if explainer is None:
        if str(type(model)) in simple_tree_model or str(type(model)) in catboost_model:
            explainer = ACVTree(model=model, data=data)
            print("Backend: ACV")
        else:
            raise NotImplementedError(
                """
                Model not supported for ACV backend.
                """
            )

    if c is None:
        c = _get_one_hot_encoded_cols(x_pred=x_pred, x_init=x_init, preprocessing=preprocessing)

    # Below we have following notations used in ACV (see ACV package for more information) :
    # S_star : index of variables in the Sufficient Coalition
    # N_star : index of the remaining variables
    # sdp_index[i, :size[i]] : index of the variables in $S^\star$
    # sdp[i] : SDP value of the $S^\star$ of observation i
    # sdp_importance : global sdp of each variable
    sdp_importance, sdp_index, size, sdp = explainer.importance_sdp_clf(
        X=x_init.values,
        data=data,
        C=c,
        pi_level=0.9
    )
    s_star, n_star = get_null_coalition(sdp_index, size)
    contributions = explainer.shap_values_acv_adap(
        X=x_init.values,
        C=c,
        S_star=s_star,
        N_star=n_star,
        size=size
    )
    if contributions.shape[-1] > 1:
        contributions = [pd.DataFrame(contributions[:, :, i], columns=x_init.columns, index=x_init.index)
                         for i in range(contributions.shape[-1])]
    else:
        contributions = pd.DataFrame(contributions[:, :, 0], columns=x_init.columns, index=x_init.index)

    return contributions, explainer, sdp_index, sdp


def _get_one_hot_encoded_cols(x_pred, x_init, preprocessing):
    """
    Returns list of list of one hot encoded variables, or empty list of list otherwise.
    """
    mapping_features = get_features_transform_mapping(x_pred, x_init, preprocessing)
    ohe_coalitions = []
    for col, encoded_col_list in mapping_features.items():
        if len(encoded_col_list) > 1:
            ohe_coalitions.append([x_init.columns.to_list().index(c) for c in encoded_col_list])

    if ohe_coalitions == list():
        return [[]]
    return ohe_coalitions


def compute_features_import_acv(sdp_index, sdp, init_columns, features_mapping):
    """
    Computes the features importance using acv package and notations.
    """
    count_cols = {i: 0 for i in range(len(sdp_index[0]))}

    for i, list_imp_feat in enumerate(sdp_index):
        for c in list_imp_feat:
            if c != -1 and sdp[i] > 0.9:
                count_cols[c] += 1

    features_cols = {init_columns[k]: v for k, v in count_cols.items()}

    features_imp = pd.Series({k: features_cols[v[0]] / len(sdp_index) for k, v in features_mapping.items()})

    return features_imp.sort_values(ascending=True)
