from typing import Any

import pandas as pd

try:
    from tabicl.shap import get_shap_values
except ImportError:
    raise ImportError(
        "tabicl.shap is not installed. Please install it to use the Tabicl backend. In your terminal, run 'pip install tabicl[shap]'"
    ) from None
from shapash.backend import BaseBackend


class TabiclBackend(BaseBackend):
    """The Tabicl backend for SHAP explanations."""

    name = "tabicl"

    def __init__(
        self,
        model,
        preprocessing: Any | None = None,
        masker: pd.DataFrame | None = None,
        **kwargs: Any,
    ):
        super().__init__(model, preprocessing)
        self.masker = masker
        self.additional_params = kwargs

    def run_explainer(self, x: pd.DataFrame) -> dict[str, Any]:
        """Compute SHAP contributions for the provided input data.

        Args:
            x: Input data to explain.

        Return:
            A dictionary containing the computed contributions.
        """
        feature_names = list(x.columns)
        data = self.masker if self.masker is not None else x
        result = get_shap_values(
            estimator=self.model, X_test=data, attribute_names=feature_names, **self.additional_params
        )

        explain_data = dict(contributions=result.values, base_values=getattr(result, "base_values", None))

        return explain_data
