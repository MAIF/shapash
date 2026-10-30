import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap

from shapash.backend._neural_shap import NeuralShap, NeuralShapUnsupportedError, is_neural_compatibility_error
from shapash.backend.base_backend import BaseBackend
from shapash.model.tabular import KerasModelAdapter, PyTorchModelAdapter, TensorFlowModelAdapter
from shapash.utils.model_synoptic import generalized_linear_model


class ShapBackend(BaseBackend):
    """The Shap Backend"""

    # When grouping features contributions together, Shap uses the sum of the contributions
    # of the features that belong to the group
    column_aggregation = "sum"
    name = "shap"

    def __init__(
        self,
        model: Any,
        preprocessing: Any | None = None,
        masker: shap.maskers.Masker | pd.DataFrame | np.ndarray | Callable[..., Any] | None = None,
        explainer_args: dict[str, Any] | None = None,
        explainer_compute_args: dict[str, Any] | None = None,
        neural_explainer: Literal["auto", "deep", "gradient", "agnostic"] = "auto",
        neural_background: pd.DataFrame | None = None,
    ) -> None:
        """Initialize the SHAP backend.

        Parameters
        ----------
        model : Any
            Model to explain.
        preprocessing : category_encoders, ColumnTransformer, list or dict or None, optional
            Preprocessing applied to the model inputs.
        masker : shap.maskers.Masker, pandas.DataFrame, numpy.ndarray, callable, or None, optional
            SHAP masking strategy, when supported by the selected explainer. A DataFrame
            or array is used as background data. A callable must accept a binary feature
            mask via ``mask=...`` and one sample in the model's input format, and return
            an iterable of masked samples. If None, the selected explainer applies its
            default behavior. Some specialized explainers, such as TreeExplainer, do
            not accept a masker.
        explainer_args : dict[str, Any] or None, optional
            Keyword arguments used to construct the SHAP explainer.
        explainer_compute_args : dict[str, Any] or None, optional
            Keyword arguments passed when computing SHAP explanations.
        neural_explainer : {"auto", "deep", "gradient", "agnostic"}, optional
            For tabular framework adapters, auto tries DeepExplainer, then
            GradientExplainer, then agnostic SHAP, warning on compatibility
            failures. Explicit choices never fall back. Other models retain
            their existing automatic selection.
        neural_background : pandas.DataFrame or None, optional
            Encoded background observations for neural SHAP. Defaults to the
            DataFrame passed as masker by SmartExplainer.compile.
        """
        super().__init__(model, preprocessing)
        self.masker = masker
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        self.neural_explainer = neural_explainer
        self.neural_background = neural_background
        self._neural: NeuralShap | None = None
        self._selected_neural: Literal["deep", "gradient"] | None = None
        if neural_explainer not in ("auto", "deep", "gradient", "agnostic"):
            raise ValueError("neural_explainer must be 'auto', 'deep', 'gradient' or 'agnostic'.")
        is_adapter = isinstance(model, (KerasModelAdapter, TensorFlowModelAdapter, PyTorchModelAdapter))
        use_neural = neural_explainer in ("deep", "gradient") or (neural_explainer == "auto" and is_adapter)
        if use_neural:
            if any(key in self.explainer_args for key in ("model", "data", "explainer", "algorithm", "masker")):
                raise ValueError("Neural SHAP selects its model and explainer; supply only native constructor options.")
            if any(key in self.explainer_compute_args for key in ("ranked_outputs", "return_variances")):
                raise ValueError("Neural SHAP requires all outputs without return_variances.")
            background = neural_background if neural_background is not None else masker
            if not isinstance(background, pd.DataFrame) or background.empty:
                raise ValueError("Neural SHAP requires a nonempty encoded background DataFrame.")
            self.neural_background = background.copy()
            self._init_neural("gradient" if neural_explainer == "gradient" else "deep")
            return
        if neural_explainer == "agnostic" and is_adapter:
            background = neural_background if neural_background is not None else masker
            if neural_background is not None and (not isinstance(background, pd.DataFrame) or background.empty):
                raise ValueError("Neural SHAP requires a nonempty encoded background DataFrame.")
            self._init_agnostic()
            return
        if neural_background is not None:
            raise ValueError("neural_background requires a tabular framework adapter or an explicit neural method.")

        if str(type(model)) == "<class 'sklearn.ensemble._voting.VotingClassifier'>" and model.voting != "soft":
            raise ValueError(
                "VotingClassifier requires voting='soft' for SHAP explanations; "
                "hard voting does not expose per-class probabilities or decision scores."
            )
        if (
            str(type(model)) in ("<class 'sklearn.svm._classes.NuSVC'>", "<class 'sklearn.svm._classes.SVC'>")
            and not hasattr(model, "predict_proba")
            and model.decision_function_shape == "ovo"
            and len(self._classes or []) > 2
        ):
            raise ValueError(
                "SVM classification requires decision_function_shape='ovr' or probability=True "
                "for per-class SHAP explanations."
            )
        is_glm = str(type(model)) in generalized_linear_model
        shap_parameters = {"model": model.predict if is_glm else model, "masker": self.masker, **self.explainer_args}

        if self.explainer_args:
            if "explainer" in self.explainer_args.keys():
                # For explicit explainer classes, keep user-provided kwargs only.
                # Some SHAP explainers (e.g. TreeExplainer) do not accept `masker`.
                explainer_args = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
                self.explainer = self.explainer_args["explainer"](**explainer_args)
            else:
                self.explainer = shap.Explainer(**shap_parameters)
        else:
            if is_glm:
                # Explain predictions after the inverse link, not the linear predictor.
                self.explainer = shap.Explainer(model=model.predict, masker=self.masker)
            elif shap.explainers.Linear.supports_model_with_masker(model, self.masker):
                self.explainer = shap.Explainer(model=model, masker=self.masker)
            elif shap.explainers.Tree.supports_model_with_masker(model, None):
                self.explainer = shap.Explainer(model=model)
            elif shap.explainers.Additive.supports_model_with_masker(model, self.masker):
                self.explainer = shap.Explainer(model=model, masker=self.masker)
            # otherwise use a model agnostic method
            elif hasattr(model, "predict_proba"):
                self.explainer = shap.Explainer(model=model.predict_proba, masker=self.masker)
            elif hasattr(model, "decision_function"):
                self.explainer = shap.Explainer(model=model.decision_function, masker=self.masker)
            elif hasattr(model, "predict"):
                self.explainer = shap.Explainer(model=model.predict, masker=self.masker)
            # if we get here then we don't know how to handle what was given to us
            else:
                raise ValueError("The model is not recognized by Shapash! Model: " + str(model))

    def _init_neural(self, method: Literal["deep", "gradient"]) -> None:
        if self.neural_background is None:
            raise ValueError("Neural SHAP requires a nonempty encoded background DataFrame.")
        self._selected_neural = method
        try:
            self._neural = NeuralShap(self.model, self.neural_background, method, self.explainer_args)
        except (NeuralShapUnsupportedError, NotImplementedError, LookupError, AssertionError, RuntimeError) as error:
            self._fallback_neural(error)
            return
        self.explainer = self._neural.explainer

    def _fallback_neural(self, error: Exception) -> None:
        if self.neural_explainer != "auto" or not is_neural_compatibility_error(error):
            raise error
        next_method = "GradientExplainer" if self._selected_neural == "deep" else "agnostic SHAP"
        warnings.warn(
            f"Neural SHAP {self._selected_neural} is incompatible: {error}. Falling back to {next_method}.",
            UserWarning,
            stacklevel=3,
        )
        self._neural = None
        if self._selected_neural == "deep":
            self._init_neural("gradient")
        else:
            self._selected_neural = None
            self._init_agnostic()

    def _init_agnostic(self) -> None:
        predict = self.model.predict_proba if self._case == "classification" else self.model.predict
        background = self.neural_background if self.neural_background is not None else self.masker
        self.explainer = shap.Explainer(model=predict, masker=background, **self.explainer_args)

    def __getstate__(self) -> dict[str, Any]:
        """Rebuild gradient hooks and framework graphs after deserialization."""
        state = self.__dict__.copy()
        if getattr(self, "_selected_neural", None) is not None:
            state["_neural"] = None
            state["explainer"] = None
        return state

    def run_explainer(self, x: pd.DataFrame) -> dict[str, Any]:
        """
        Computes and returns local contributions using Shap explainer

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model

        Returns
        -------
        explain_data : dict[str, Any]
            Mapping containing SHAP contributions and base values.
        """
        while getattr(self, "_selected_neural", None) is not None:
            if self._neural is None:
                method = self._selected_neural
                if method is None:
                    break
                self._init_neural(method)
                continue
            try:
                explain_data = self._neural.explain(x, self.explainer_compute_args)
            except (
                NeuralShapUnsupportedError,
                NotImplementedError,
                LookupError,
                AssertionError,
                RuntimeError,
            ) as error:
                self._fallback_neural(error)
            else:
                print("INFO: Shap explainer type -", self.explainer)
                return explain_data
        print("INFO: Shap explainer type -", self.explainer)
        contributions = self.explainer(x, **self.explainer_compute_args)
        explain_data = dict(contributions=contributions.values, base_values=getattr(contributions, "base_values", None))
        return explain_data


def get_shap_interaction_values(
    x_df: pd.DataFrame, explainer: shap.TreeExplainer, class_index: int | None = None
) -> np.ndarray:
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
        n_features = x_df.shape[1]

        is_sample_first_layout = (
            shap_interaction_values.shape[0] == n_samples
            and shap_interaction_values.shape[1] == n_features
            and shap_interaction_values.shape[2] == n_features
        )
        is_output_first_layout = (
            shap_interaction_values.shape[1] == n_samples
            and shap_interaction_values.shape[2] == n_features
            and shap_interaction_values.shape[3] == n_features
        )

        if is_sample_first_layout and not is_output_first_layout:
            # shape: (#samples, #features, #features, #outputs)
            if class_index is None:
                shap_interaction_values = np.sum(shap_interaction_values, axis=-1)
            else:
                shap_interaction_values = shap_interaction_values[..., class_index]
        elif is_output_first_layout and not is_sample_first_layout:
            # shape: (#outputs, #samples, #features, #features)
            if class_index is None:
                shap_interaction_values = np.sum(shap_interaction_values, axis=0)
            else:
                shap_interaction_values = shap_interaction_values[class_index, ...]
        elif is_sample_first_layout and is_output_first_layout:
            # Ambiguous rare case where sample, feature and output dimensions have
            # identical sizes. Preserve the previous sample-first interpretation.
            if class_index is None:
                shap_interaction_values = np.sum(shap_interaction_values, axis=-1)
            else:
                shap_interaction_values = shap_interaction_values[..., class_index]
        else:
            # Fallback: preserve previous behavior by aggregating on last axis.
            shap_interaction_values = np.sum(shap_interaction_values, axis=-1)

    return shap_interaction_values
