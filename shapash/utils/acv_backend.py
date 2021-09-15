try:
    from acv_explainers import ACVTree
    from acv_explainers.utils import get_null_coalition
    is_acv_available = True
except ImportError:
    is_acv_available = False

from typing import Any, Optional, List

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
    x_init: pd.DataFrame
    x_pred: pd.DataFrame
    x_train: pd.DataFrame; Optional
    explainer : explainer object from shap, optional (default: None)
        this explainer is used to compute shapley values
    preprocessing:
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
        c = get_one_hot_encoded_cols(x_pred=x_pred, x_init=x_init, preprocessing=preprocessing)

    sdp_importance, sdp_index, size, sdp = explainer.importance_sdp_clf(
        X=x_init,
        data=data,
        C=c,
        global_proba=0.9
    )
    s_star, n_star = get_null_coalition(sdp_index, size)
    contributions = explainer.shap_values_acv_adap(
        X=x_init,
        C=c,
        N_star=n_star,
        size=size,
        S_star=s_star
    )
    if contributions.shape[-1] > 1:
        contributions = [pd.DataFrame(contributions[:, :, i], columns=x_init.columns, index=x_init.index)
                         for i in range(contributions.shape[-1])]
    else:
        contributions = pd.DataFrame(contributions[:, :, 0], columns=x_init.columns, index=x_init.index)

    return contributions, sdp_importance, explainer


def get_one_hot_encoded_cols(x_pred, x_init, preprocessing):
    mapping_features = get_features_transform_mapping(x_pred, x_init, preprocessing)
    ohe_coalitions = []
    for col, encoded_col_list in mapping_features.items():
        if len(encoded_col_list) > 1:
            ohe_coalitions.append([x_init.columns.to_list().index(c) for c in encoded_col_list])

    if ohe_coalitions == list():
        return [[]]
    return ohe_coalitions
