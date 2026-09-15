import numpy as np
import pandas as pd
import shap

from shapash.backend.base_backend import BaseBackend


class ShapBackend(BaseBackend):
    """The Shap Backend"""

    # When grouping features contributions together, Shap uses the sum of the contributions
    # of the features that belong to the group
    column_aggregation = "sum"
    name = "shap"

    def __init__(self, model, preprocessing=None, masker=None, explainer_args=None, explainer_compute_args=None):
        super().__init__(model, preprocessing)
        self.masker = masker
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}

        shap_parameters = {"model": model, "masker": self.masker, **self.explainer_args}

        if self.explainer_args:
            if "explainer" in self.explainer_args.keys():
                # For explicit explainer classes, keep user-provided kwargs only.
                # Some SHAP explainers (e.g. TreeExplainer) do not accept `masker`.
                explainer_args = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
                self.explainer = self.explainer_args["explainer"](**explainer_args)
            else:
                self.explainer = shap.Explainer(**shap_parameters)
        else:
            if shap.explainers.Linear.supports_model_with_masker(model, self.masker):
                self.explainer = shap.Explainer(model=model, masker=self.masker)
            elif shap.explainers.Tree.supports_model_with_masker(model, None):
                self.explainer = shap.Explainer(model=model)
            elif shap.explainers.Additive.supports_model_with_masker(model, self.masker):
                self.explainer = shap.Explainer(model=model, masker=self.masker)
            # otherwise use a model agnostic method
            elif hasattr(model, "predict_proba"):
                self.explainer = shap.Explainer(model=model.predict_proba, masker=self.masker)
            elif hasattr(model, "predict"):
                self.explainer = shap.Explainer(model=model.predict, masker=self.masker)
            # if we get here then we don't know how to handle what was given to us
            else:
                raise ValueError("The model is not recognized by Shapash! Model: " + str(model))

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
        print("INFO: Shap explainer type -", self.explainer)
        contributions = self.explainer(x, **self.explainer_compute_args)
        explain_data = dict(contributions=contributions.values)
        return explain_data


def get_shap_interaction_values(x_df, explainer, class_index=None):
    """
    Compute the shap interaction values for a given dataframe.
    Also checks if the explainer is a TreeExplainer.

    Parameters
    ----------
    x_df : pd.DataFrame
        DataFrame for which will be computed the interaction values using the explainer.
    explainer : shap.TreeExplainer
        explainer object used to compute the interaction values.

    class_index : int, optional
        Class index to select in classification / multi-output settings.
        If None, outputs are aggregated by summing across classes.

    Returns
    -------
    shap_interaction_values : np.ndarray
        Shap interaction values for each sample as an array of shape (# samples x # features x # features).
    """
    if not isinstance(explainer, shap.TreeExplainer):
        raise ValueError(
            f"Explainer type ({type(explainer)}) is not a TreeExplainer. "
            f"Shap interaction values can only be computed for TreeExplainer types"
        )

    shap_interaction_values = explainer.shap_interaction_values(x_df)

    # For models with vector outputs the previous function may return one array for each output
    # (list of arrays) or an array with an extra output dimension.
    # We either select one class/output if requested, or sum across outputs to keep
    # shape (#samples, #features, #features).
    if isinstance(shap_interaction_values, list):
        if class_index is None:
            shap_interaction_values = np.sum(shap_interaction_values, axis=0)
        else:
            shap_interaction_values = shap_interaction_values[class_index]
    elif isinstance(shap_interaction_values, np.ndarray) and shap_interaction_values.ndim == 4:
        n_samples = len(x_df)
        if shap_interaction_values.shape[0] == n_samples:
            # shape: (#samples, #features, #features, #outputs)
            if class_index is None:
                shap_interaction_values = np.sum(shap_interaction_values, axis=-1)
            else:
                shap_interaction_values = shap_interaction_values[..., class_index]
        elif shap_interaction_values.shape[1] == n_samples:
            # shape: (#outputs, #samples, #features, #features)
            if class_index is None:
                shap_interaction_values = np.sum(shap_interaction_values, axis=0)
            else:
                shap_interaction_values = shap_interaction_values[class_index, ...]
        else:
            # Fallback: preserve previous behavior by aggregating on last axis.
            shap_interaction_values = np.sum(shap_interaction_values, axis=-1)

    return shap_interaction_values
