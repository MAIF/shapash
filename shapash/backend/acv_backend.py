from typing import Any, Optional, List, Union

import numpy as np
import pandas as pd

from shapash.backend.base_backend import BaseBackend
from shapash.utils.transform import get_preprocessing_mapping

try:
    from acv_explainers import ACVTree
    from acv_explainers.utils import get_null_coalition
    _is_acv_available = True
except ImportError:
    _is_acv_available = False


class AcvBackend(BaseBackend):
    # Coalitions should be grouped using one column value only and not the sum like shap
    column_aggregation = 'first'
    name = 'acv'
    supported_cases = ['classification']

    def __init__(
            self,
            model,
            data=None,
            preprocessing=None,
            active_sdp=True,
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
        super(AcvBackend, self).__init__(model, preprocessing)
        self.active_sdp = active_sdp
        self.data = data
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        if data is not None:
            self.explainer = ACVTree(model=model, data=data, **self.explainer_args)
        else:
            self.explainer = None

    def run_explainer(self, x: pd.DataFrame) -> dict:
        if self.data is None:
            # This is used to handle the case where data object was not definied 
            self.data = x
            self.explainer = ACVTree(model=self.model, data=self.data, **self.explainer_args)

        explain_data = {}

        mapping = get_preprocessing_mapping(x, self.preprocessing)
        c = []
        for col in mapping.keys():
            if len(mapping[col]) > 1:
                c.append([x.columns.to_list().index(col_i) for col_i in mapping[col]])
        if len(c) == 0:
            c = [[]]

        sdp_importance, sdp_index, size, sdp = self.explainer.importance_sdp_clf(
            X=x.values,
            data=np.asarray(self.data)
        )
        s_star, n_star = get_null_coalition(sdp_index, size)
        contributions = self.explainer.shap_values_acv_adap(
            X=x.values,
            C=c,
            S_star=s_star,
            N_star=n_star,
            size=size
        )
        if contributions.shape[-1] > 1:
            contributions = [pd.DataFrame(contributions[:, :, i], columns=x.columns, index=x.index)
                             for i in range(contributions.shape[-1])]
        else:
            contributions = pd.DataFrame(contributions[:, :, 0], columns=x.columns, index=x.index)

        explain_data['sdp'] = sdp
        explain_data['sdp_index'] = sdp_index
        explain_data['init_columns'] = x.columns.to_list()
        explain_data['contributions'] = contributions
        explain_data['features_mapping'] = mapping

        return explain_data

    def get_global_features_importance(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            explain_data: Any = None,
            subset: Optional[List[int]] = None
    ) -> Union[pd.Series, List[pd.Series]]:

        count_cols = {i: 0 for i in range(len(explain_data['sdp_index'][0]))}

        for i, list_imp_feat in enumerate(explain_data['sdp_index']):
            for col in list_imp_feat:
                if col != -1 and explain_data['sdp'][i] > 0.9:
                    count_cols[col] += 1

        features_cols = {explain_data['init_columns'][k]: v for k, v in count_cols.items()}

        mapping = explain_data['features_mapping']
        list_cols_ohe = [c for list_c in mapping.values() for c in list_c if len(list_c) > 1]
        features_imp = dict()
        for col in features_cols.keys():
            if col in list_cols_ohe:
                for col_mapping in mapping.keys():
                    if col in mapping[col_mapping]:
                        features_imp[col_mapping] = features_cols[col]
            else:
                features_imp[col] = features_cols[col]

        features_imp = pd.Series(features_imp).sort_values(ascending=True)
        if self._case == 'classification':
            features_imp = [features_imp for _ in range(len(contributions))]
        return features_imp

