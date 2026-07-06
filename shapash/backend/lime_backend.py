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
            A dict with key 'contributions':
            - pd.DataFrame of shape (n_samples, n_features)
              for binary classification or regression.
            - List[pd.DataFrame] of length n_classes
              for multiclass classification.
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
                contributions = self._explain_multiclass(x, feature_names, predict_fn, num_classes)
            else:
                contributions = self._explain_binary_or_regression(x, feature_names, predict_fn)
        else:
            contributions = self._explain_binary_or_regression(x, feature_names, predict_fn)

        return dict(contributions=contributions)

    def _explain_multiclass(
        self,
        x: pd.DataFrame,
        feature_names: list,
        predict_fn: callable,
        num_classes: int,
    ) -> list[pd.DataFrame]:
        """
        Compute LIME contributions for multiclass classification.

        explain_instance is called once per sample; with top_labels=num_classes
        it returns explanations for every class in a single call, so there is
        no need to loop over classes in the outer dimension.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame of shape (n_samples, n_features) per class.
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
        for j in range(num_classes):
            class_contrib = [{_transform_name(feat, x): val for feat, val in exp.as_list(j)} for exp in explanations]
            contribution.append(pd.DataFrame(class_contrib, index=x.index)[feature_names])

        return contribution

    def _explain_binary_or_regression(
        self,
        x: pd.DataFrame,
        feature_names: list,
        predict_fn: callable,
    ) -> pd.DataFrame:
        """
        Compute LIME contributions for binary classification or regression.

        Returns
        -------
        pd.DataFrame
            Contributions of shape (n_samples, n_features).
        """
        lime_contrib = []
        for i in x.index:
            exp = self.explainer.explain_instance(
                x.loc[i].to_numpy(),
                predict_fn,
                num_features=x.shape[1],
            )
            lime_contrib.append({_transform_name(feat, x): val for feat, val in exp.as_list()})

        return pd.DataFrame(lime_contrib, index=x.index)[feature_names]
