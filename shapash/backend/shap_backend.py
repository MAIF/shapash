import pandas as pd
import numpy as np
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


def get_shap_interaction_values(x_df, explainer):
    """
    Compute the shap interaction values for a given dataframe.
    Also checks if the explainer is a TreeExplainer.

    Parameters
    ----------
    x_df : pd.DataFrame
        DataFrame for which will be computed the interaction values using the explainer.
    explainer : shap.TreeExplainer
        explainer object used to compute the interaction values.

    Returns
    -------
    shap_interaction_values : np.ndarray
        Shap interaction values for each sample as an array of shape (# samples x # features x # features).
    """
    if not isinstance(explainer, shap.TreeExplainer):
        raise ValueError(f"Explainer type ({type(explainer)}) is not a TreeExplainer. "
                         f"Shap interaction values can only be computed for TreeExplainer types")

    shap_interaction_values = explainer.shap_interaction_values(x_df)

    # For models with vector outputs the previous function returns one array for each output.
    # We sum the contributions here.
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = np.sum(shap_interaction_values, axis=0)

    return shap_interaction_values
