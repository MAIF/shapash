from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.naive_bayes import CategoricalNB

try:
    from lime import lime_tabular

    is_lime_available = True
except ImportError:
    is_lime_available = False

import pandas as pd

from shapash.backend._lime_input import LimeInput
from shapash.backend.base_backend import BaseBackend
from shapash.utils.check import check_contribution_object
from shapash.utils.transform import adapt_contributions
from shapash.utils.utils import choose_state


def _with_feature_names(
    values: np.ndarray | pd.DataFrame,
    predict_fn: Callable[[Any], Any],
    feature_names: list[str],
) -> Any:
    """Wrap a prediction function to pass feature names when possible.

    Parameters
    ----------
    values : np.ndarray or pd.DataFrame
        Input values to pass to the prediction function.
    predict_fn : Callable[[Any], Any]
        Model batch-prediction function: ``predict_proba`` for classification or
        ``predict`` for regression. It must accept multiple rows at once and return
        one prediction row per input row. DataFrame inputs are passed through;
        NumPy inputs are first wrapped in a DataFrame with ``feature_names`` as
        columns, then retried as a NumPy array if that call raises ``TypeError`` or
        ``ValueError``.
    feature_names : list[str]
        Explicit column names — no implicit closure over outer scope.

    Returns
    -------
    Any
        The result returned by ``predict_fn``.
    """
    if isinstance(values, pd.DataFrame):
        return predict_fn(values)
    try:
        return predict_fn(pd.DataFrame(values, columns=feature_names))
    except (TypeError, ValueError):
        return predict_fn(values)


class LimeBackend(BaseBackend):
    """The Lime Backend."""

    column_aggregation = "sum"
    name = "lime"
    support_groups = False

    def __init__(
        self,
        model: Any,
        preprocessing: Any | None = None,
        data: pd.DataFrame | None = None,
        categorical_features: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LIME backend.

        Parameters
        ----------
        model : Any
            Model to explain.
        preprocessing : category_encoders, ColumnTransformer, list or dict or None, optional
            Preprocessing applied to the model inputs.
        data : pd.DataFrame or None, optional
            Data used to initialize the LIME explainer. Defaults to the data
            passed to :meth:`run_explainer`.
        categorical_features : list of str or None, optional
            Complete list of categorical feature names before preprocessing.
            None infers categories from fitted encoders, nonnumeric input columns
            and CategoricalNB. An explicit list, including [], replaces inference.
            Encoded categorical blocks such as one-hot groups always remain
            grouped, regardless of this list.
        **kwargs : Any
            Additional keyword arguments accepted by the constructor.

        Raises
        ------
        ValueError
            If a classification model does not expose a callable predict_proba.
        """
        super().__init__(model, preprocessing)
        if self._case == "classification" and not callable(getattr(model, "predict_proba", None)):
            raise ValueError(
                f"The LIME backend requires predict_proba for classification, but {type(model).__name__} "
                "does not provide it. Use backend='shap' if the model is supported by SHAP, "
                "or a calibrated classifier exposing predict_proba. Shapash does not switch backends automatically."
            )
        self.explainer: Any = None
        if data is not None and not isinstance(data, pd.DataFrame):
            raise ValueError("LIME data must be an encoded background DataFrame.")
        self.data = data
        if categorical_features is not None and (
            not isinstance(categorical_features, list)
            or not all(isinstance(name, str) for name in categorical_features)
            or len(categorical_features) != len(set(categorical_features))
        ):
            raise ValueError("categorical_features must be None or a list of unique feature names.")
        self.categorical_features = None if categorical_features is None else categorical_features.copy()
        self._lime_input: LimeInput | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Rebuild LIME's non-pickleable explainer from its reference data on reload."""
        state = self.__dict__.copy()
        state["explainer"] = None
        state["_lime_input"] = None
        if state["data"] is None:
            state["data"] = state.get("_explainer_data")
        return state

    def run_explainer(self, x: pd.DataFrame) -> dict[str, Any]:
        """
        Computes local contributions using the Lime explainer.

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model.

        Returns
        -------
        dict[str, pd.DataFrame | list[pd.DataFrame] | np.ndarray]
            A dict with keys:

            - 'contributions':

              - pd.DataFrame of shape (n_samples, n_features)
                for binary classification or regression.
              - List[pd.DataFrame] of length n_classes
                for multiclass classification.

            - 'base_values': local intercepts by individual:

              - np.ndarray of shape (n_samples, n_classes) for classification.
              - np.ndarray of shape (n_samples,) for regression.
            - 'contributions_space': 'original', because encoded categorical
              blocks have already been represented by their original feature.
        """
        feature_names = list(x.columns)
        data = self.data if self.data is not None else x

        if self.explainer is None or self._lime_input is None or not self._lime_input.columns.equals(x.columns):
            self._explainer_data = data.copy()
            self._lime_input = LimeInput(
                data, self.preprocessing, self.categorical_features, isinstance(self.model, CategoricalNB)
            )
            self.explainer = lime_tabular.LimeTabularExplainer(
                self._lime_input.background.to_numpy(),
                feature_names=list(self._lime_input.background.columns),
                mode=self._case,
                categorical_features=self._lime_input.categorical_indices,
            )

        lime_input = self._lime_input
        lime_x = lime_input.encode(x)
        lime_feature_names = list(lime_x.columns)
        model_predict = self.model.predict_proba if self._case == "classification" else self.model.predict

        def predict_fn(values: np.ndarray | pd.DataFrame) -> Any:
            return _with_feature_names(lime_input.decode(values), model_predict, feature_names)

        if self._case == "classification":
            if self._classes is None:
                raise ValueError("Classification models must expose their classes.")
            num_classes = len(self._classes)
            if num_classes > 2:
                contributions, base_values = self._explain_multiclass(
                    lime_x, lime_feature_names, predict_fn, num_classes
                )
            else:
                contributions, base_values = self._explain_binary_or_regression(
                    lime_x, lime_feature_names, predict_fn, num_classes=2
                )
        else:
            contributions, base_values = self._explain_binary_or_regression(lime_x, lime_feature_names, predict_fn)

        return dict(contributions=contributions, base_values=base_values, contributions_space="original")

    def get_local_contributions(
        self, x: pd.DataFrame, explain_data: Any, subset: list[int] | None = None
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Validate original-space LIME contributions without aggregating encoded blocks a second time."""
        if not isinstance(explain_data, dict) or explain_data.get("contributions_space") != "original":
            return super().get_local_contributions(x, explain_data, subset)
        if "contributions" not in explain_data:
            raise ValueError("The LIME explanation must contain contributions.")
        if self._lime_input is None:
            data = self.data if self.data is not None else x
            self._lime_input = LimeInput(
                data, self.preprocessing, self.categorical_features, isinstance(self.model, CategoricalNB)
            )
        original = self._lime_input.encode(x)
        contributions = adapt_contributions(self._case, explain_data["contributions"])
        self.state = choose_state(contributions)
        check_contribution_object(self._case, self._classes, contributions)
        local = self.state.validate_contributions(contributions, original)
        if subset is not None:
            original = original.loc[subset]
            local = [frame.loc[subset] for frame in local] if isinstance(local, list) else local.loc[subset]
        self.state.check_contributions(local, original)
        return local

    @staticmethod
    def _contributions_for_label(explanation: Any, feature_names: list[str], label: int) -> dict[str, float]:
        return {feature_names[i]: float(value) for i, value in explanation.as_map()[label]}

    @staticmethod
    def _extract_intercept(exp: Any, class_idx: int | None = None) -> float:
        """Extract a numeric intercept from a LIME explanation object."""
        intercept = getattr(exp, "intercept", None)

        if isinstance(intercept, dict):
            if class_idx is not None and class_idx in intercept:
                return float(intercept[class_idx])
            if class_idx is not None and str(class_idx) in intercept:
                return float(intercept[str(class_idx)])
            if len(intercept) > 0:
                return float(next(iter(intercept.values())))

        if isinstance(intercept, (list, tuple)):
            if class_idx is not None and class_idx < len(intercept):
                return float(intercept[class_idx])
            if len(intercept) > 0:
                return float(intercept[0])

        if intercept is not None:
            try:
                return float(intercept)
            except (TypeError, ValueError):
                pass

        return 0.0

    def _explain_multiclass(
        self,
        x: pd.DataFrame,
        feature_names: list[str],
        predict_fn: Callable[[Any], Any],
        num_classes: int,
    ) -> tuple[list[pd.DataFrame], np.ndarray]:
        """
        Compute LIME contributions for multiclass classification.

        explain_instance is called once per sample; with top_labels=num_classes
        it returns explanations for every class in a single call, so there is
        no need to loop over classes in the outer dimension.

        Parameters
        ----------
        x : pd.DataFrame
            Input observations to explain. Each row is explained once with
            :meth:`lime.lime_tabular.LimeTabularExplainer.explain_instance`.
        feature_names : list[str]
            Ordered list of feature names used to build each per-class
            contributions dataframe and preserve the original column order.
        predict_fn : Callable
            Batch prediction callable passed to LIME. It must accept multiple
            perturbed rows at once and return class probabilities, with one output
            row per input row (shape ``(n_samples, n_classes)``).
        num_classes : int
            Number of target classes to explain.

        Returns
        -------
        tuple[list[pd.DataFrame], np.ndarray]
            - One DataFrame of shape (n_samples, n_features) per class,
              ordered by class index.
            - Local intercepts of shape (n_samples, n_classes).
        """
        # One explain_instance call per sample — O(n_samples)
        explanations = []
        for idx in x.index:
            exp = self.explainer.explain_instance(
                x.loc[idx].to_numpy(),
                predict_fn,
                top_labels=num_classes,
                num_features=x.shape[1],
            )
            explanations.append(exp)

        contribution = []
        base_values = pd.DataFrame(index=x.index, columns=list(range(num_classes)), dtype=float)
        for j in range(num_classes):
            class_contrib = [self._contributions_for_label(exp, feature_names, j) for exp in explanations]
            contribution.append(pd.DataFrame(class_contrib, index=x.index)[feature_names])
            for idx, exp in zip(x.index, explanations, strict=False):
                base_values.at[idx, j] = self._extract_intercept(exp, class_idx=j)

        return contribution, base_values.to_numpy(dtype=float)

    def _explain_binary_or_regression(
        self,
        x: pd.DataFrame,
        feature_names: list[str],
        predict_fn: Callable[[Any], Any],
        num_classes: int | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Compute LIME contributions for binary classification or regression.

        Parameters
        ----------
        predict_fn : Callable
            Batch prediction callable passed to LIME. It must accept multiple
            perturbed rows at once and return one output row per input row. For
            classification, it returns class probabilities with shape
            ``(n_samples, n_classes)``; for regression, it returns one prediction
            per row with shape ``(n_samples,)``.

        Returns
        -------
        tuple[pd.DataFrame, np.ndarray]
            - Contributions of shape (n_samples, n_features).
            - Local intercepts by individual:
              - classification: shape (n_samples, 2)
              - regression: shape (n_samples,)
        """
        lime_contrib = []
        base_values_cls: list[list[float]] = []
        base_values_reg: list[float] = []
        for i in x.index:
            exp = self.explainer.explain_instance(
                x.loc[i].to_numpy(),
                predict_fn,
                num_features=x.shape[1],
            )
            lime_contrib.append(self._contributions_for_label(exp, feature_names, 1))
            if self._case == "classification":
                # Binary classification contributions are expanded later as [-contrib, +contrib].
                # Keep base values coherent with this convention.
                pos_intercept = self._extract_intercept(exp, class_idx=1)
                base_values_cls.append([-pos_intercept, pos_intercept])
            else:
                base_values_reg.append(self._extract_intercept(exp, class_idx=0))

        contrib_df = pd.DataFrame(lime_contrib, index=x.index)[feature_names]
        if self._case == "classification":
            base_values_df = pd.DataFrame(base_values_cls, index=x.index, columns=list(range(num_classes or 2)))
            return contrib_df, base_values_df.to_numpy(dtype=float)

        base_values_series = pd.Series(base_values_reg, index=x.index)
        return contrib_df, base_values_series.to_numpy(dtype=float)
