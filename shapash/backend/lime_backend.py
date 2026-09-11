from collections.abc import Callable

try:
    from lime import lime_tabular

    is_lime_available = True
except ImportError:
    is_lime_available = False

import pandas as pd

from shapash.backend.base_backend import BaseBackend


def _with_feature_names(values, predict_fn, feature_names: list):
    """
    Wraps a predict function to ensure feature names are passed when possible.

    Parameters
    ----------
    values : array-like or pd.DataFrame
    predict_fn : callable
    feature_names : list
        Explicit column names — no implicit closure over outer scope.
    """
    if isinstance(values, pd.DataFrame):
        return predict_fn(values)
    try:
        return predict_fn(pd.DataFrame(values, columns=feature_names))
    except (TypeError, ValueError):
        return predict_fn(values)


def _transform_name(var_name: str, x_df: pd.DataFrame) -> str:
    """Transform a LIME contribution feature string to a comprehensive column name.

    Parameters
    ----------
    var_name : str
        Feature name returned by LIME, often containing additional formatting.
    x_df : pd.DataFrame
        DataFrame used to match the LIME feature string against actual columns.

    Returns
    -------
    str
        The matching column name from ``x_df``.

    Raises
    ------
    ValueError
        If no matching column name can be found.
    """
    for colname in x_df.columns:
        if f" {colname} " in f" {var_name} ":
            return colname
    raise ValueError(f"Could not match LIME feature string {var_name!r} to any column in {list(x_df.columns)}")


class LimeBackend(BaseBackend):
    """The Lime Backend"""

    column_aggregation = "sum"
    name = "lime"
    support_groups = False

    def __init__(self, model, preprocessing=None, data=None, **kwargs):
        super().__init__(model, preprocessing)
        self.explainer = None
        self.data = data

    def run_explainer(self, x: pd.DataFrame) -> dict:
        """
        Computes local contributions using the Lime explainer.

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model.

                Returns
                -------
                dict
                        A dict with keys:
                        - 'contributions':
                            - pd.DataFrame of shape (n_samples, n_features)
                                for binary classification or regression.
                            - List[pd.DataFrame] of length n_classes
                                for multiclass classification.
                        - 'base_values': local intercepts by individual:
                            - np.ndarray of shape (n_samples, n_classes) for classification.
                            - np.ndarray of shape (n_samples,) for regression.
        """
        feature_names = list(x.columns)
        data = self.data if self.data is not None else x

        # added condition to reinitialise the explainer
        # whenever the feature names differ from what it was built with
        if self.explainer is None or self.explainer.feature_names != feature_names:
            self.explainer = lime_tabular.LimeTabularExplainer(
                data.to_numpy(), feature_names=feature_names, mode=self._case
            )

        model_predict = self.model.predict_proba if self._case == "classification" else self.model.predict

        def predict_fn(values):
            return _with_feature_names(values, model_predict, feature_names)

        if self._case == "classification":
            num_classes = len(self._classes)
            if num_classes > 2:
                contributions, base_values = self._explain_multiclass(x, feature_names, predict_fn, num_classes)
            else:
                contributions, base_values = self._explain_binary_or_regression(
                    x, feature_names, predict_fn, num_classes=2
                )
        else:
            contributions, base_values = self._explain_binary_or_regression(x, feature_names, predict_fn)

        return dict(contributions=contributions, base_values=base_values)

    @staticmethod
    def _extract_intercept(exp, class_idx=None) -> float:
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
        feature_names: list,
        predict_fn: Callable,
        num_classes: int,
    ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """
        Compute LIME contributions for multiclass classification.

        explain_instance is called once per sample; with top_labels=num_classes
        it returns explanations for every class in a single call, so there is
        no need to loop over classes in the outer dimension.

        Returns
        -------
        tuple[list[pd.DataFrame], pd.DataFrame]
            - One DataFrame of shape (n_samples, n_features) per class.
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
            class_contrib = [{_transform_name(feat, x): val for feat, val in exp.as_list(j)} for exp in explanations]
            contribution.append(pd.DataFrame(class_contrib, index=x.index)[feature_names])
            for idx, exp in zip(x.index, explanations, strict=False):
                base_values.at[idx, j] = self._extract_intercept(exp, class_idx=j)

        return contribution, base_values.to_numpy(dtype=float)

    def _explain_binary_or_regression(
        self,
        x: pd.DataFrame,
        feature_names: list,
        predict_fn: Callable,
        num_classes: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
        """
        Compute LIME contributions for binary classification or regression.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]
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
            lime_contrib.append({_transform_name(feat, x): val for feat, val in exp.as_list()})
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
