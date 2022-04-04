import pandas as pd
import shap

from shapash.backend.base_backend import BaseBackend


class ShapBackend(BaseBackend):
    # When grouping features contributions together, Shap uses the sum of the contributions
    # of the features that belong to the group
    column_aggregation = 'sum'
    name = 'shap'

    def __init__(self, model, preprocessing=None, explainer_args=None, explainer_compute_args=None):
        super(ShapBackend, self).__init__(model, preprocessing)
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        self.explainer = shap.Explainer(model=model, **self.explainer_args)

    def run_explainer(self, x: pd.DataFrame) -> dict:
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
        contributions = self.explainer(x, **self.explainer_compute_args)
        explain_data = dict(contributions=contributions.values)
        return explain_data
