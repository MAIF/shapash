import pandas as pd

from shapash.backend.base_backend import BaseBackend
from shapash.utils.transform import get_features_transform_mapping

try:
    from acv_explainers import ACVTree
    from acv_explainers.utils import get_null_coalition
    _is_acv_available = True
except ImportError:
    _is_acv_available = False


class AcvBackend(BaseBackend):
    # Coalitions should be grouped using one column value only and not the sum like shap
    column_aggregation = 'first'

    def __init__(
            self,
            model,
            data,
            active_sdp=True,
            features_mapping=None,
            explainer_args=None,
            explainer_compute_args=None
    ):
        if _is_acv_available is False:
            raise ValueError(
                """
                Active Shapley values requires the ACV package,
                which can be installed using 'pip install acv-exp'
                """
            )
        super(AcvBackend, self).__init__(model)
        self.active_sdp = active_sdp
        self.data = data
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        self.explainer = ACVTree(model=model, data=data, **self.explainer_args)
        self.sdp = None
        self.features_mapping = features_mapping

    def _get_local_contributions(self, X):

        # Coalitions used
        c = self.explainer_compute_args.get('c', [[]])

        sdp_importance, sdp_index, size, sdp = self.explainer.importance_sdp_clf(
            X=X.values,
            data=self.data
        )
        s_star, n_star = get_null_coalition(sdp_index, size)
        contributions = self.explainer.shap_values_acv_adap(
            X=X.values,
            C=c,
            S_star=s_star,
            N_star=n_star,
            size=size
        )
        if contributions.shape[-1] > 1:
            contributions = [pd.DataFrame(contributions[:, :, i], columns=X.columns, index=X.index)
                             for i in range(contributions.shape[-1])]
        else:
            contributions = pd.DataFrame(contributions[:, :, 0], columns=X.columns, index=X.index)

        self.sdp = sdp
        self.sdp_index = sdp_index
        self.init_columns = X.columns.to_list()
        self.contributions = contributions
        return contributions

    def _get_global_features_importance(self, subset=None):
        if self.sdp is None:
            raise ValueError("Local contributions should be computed first")
        count_cols = {i: 0 for i in range(len(self.sdp_index[0]))}

        for i, list_imp_feat in enumerate(self.sdp_index):
            for col in list_imp_feat:
                if col != -1 and self.sdp[i] > 0.9:
                    count_cols[col] += 1

        features_cols = {self.init_columns[k]: v for k, v in count_cols.items()}

        if self.features_mapping:
            features_imp = pd.Series(
                {k: features_cols[v[0]] / len(self.sdp_index) for k, v in self.features_mapping.items()}
            )
        else:
            features_imp = pd.Series(features_cols).sort_values(ascending=True)

        return features_imp.sort_values(ascending=True)


def get_one_hot_encoded_cols(x_pred, x_init, preprocessing):
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
