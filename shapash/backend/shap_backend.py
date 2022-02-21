from typing import Any, Optional, List, Union

import pandas as pd
import shap

from shapash.backend.base_backend import BaseBackend


class ShapBackend(BaseBackend):
    # When grouping features contributions together, Shap uses the sum of the contributions
    # of the features that belong to the group
    column_aggregation = 'sum'

    def __init__(self, model, preprocessing=None, explainer_args=None, explainer_compute_args=None):
        super(ShapBackend, self).__init__(model, preprocessing)
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        self.explainer = shap.TreeExplainer(model=model, **self.explainer_args)

    def _run_explainer(self, x: pd.DataFrame):
        """
        Computes and returns local contributions using Shap explainer

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model

        Returns
        -------
        explain_data : pd.DataFrame or list of pd.DataFrame
            local contributions
        """
        explain_data = self.explainer(x, **self.explainer_compute_args)
        return explain_data

    def _get_local_contributions(
            self,
            x: pd.DataFrame,
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        contributions = explain_data.values
        if subset is None:
            return contributions
        else:
            return contributions.loc[subset]

    def _get_global_features_importance(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.Series, List[pd.Series]]:
        if subset is not None:
            return self._state.compute_features_import(contributions.loc[subset])
        else:
            return self._state.compute_features_import(contributions)
