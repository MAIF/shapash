try:
    from lime import lime_tabular

    is_lime_available = True
except ImportError:
    is_lime_available = False

import pandas as pd

from shapash.backend.base_backend import BaseBackend


class LimeBackend(BaseBackend):
    """The Lime Backend"""

    column_aggregation = "sum"
    name = "lime"
    support_groups = False

    def __init__(self, model, preprocessing=None, data=None, **kwargs):
        super().__init__(model, preprocessing)
        self.explainer = None
        self.data = data

    def run_explainer(self, x: pd.DataFrame):
        """
        Computes local contributions using Lime explainer

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model

        Returns
        -------
        explain_data : dict
            dict containing local contributions
        """
        data = self.data if self.data is not None else x
        explainer = lime_tabular.LimeTabularExplainer(data.values, feature_names=x.columns, mode=self._case)

        def _with_feature_names(values, predict_fn):
            if isinstance(values, pd.DataFrame):
                return predict_fn(values)
            try:
                return predict_fn(pd.DataFrame(values, columns=x.columns))
            except (TypeError, ValueError):
                return predict_fn(values)

        lime_contrib = []
        if self._case == "classification":
            num_classes = len(self._classes)
            if num_classes > 2:
                contribution = []
                for j in range(num_classes):
                    list_contrib = []
                    for idx in x.index:
                        exp = explainer.explain_instance(
                            x.loc[idx].to_numpy(),
                            lambda values: _with_feature_names(values, self.model.predict_proba),
                            top_labels=num_classes,
                            num_features=x.shape[1],
                        )
                        list_contrib.append(
                            {_transform_name(var_name[0], x): var_name[1] for var_name in exp.as_list(j)}
                        )
                    df_contrib = pd.DataFrame(list_contrib)
                    df_contrib = df_contrib[list(x.columns)]
                    contribution.append(df_contrib.values)
                return dict(contributions=contribution)

        for i in x.index:
            if self._case == "classification":
                exp = explainer.explain_instance(
                    x.loc[i].to_numpy(),
                    lambda values: _with_feature_names(values, self.model.predict_proba),
                    num_features=x.shape[1],
                )
                lime_contrib.append({_transform_name(var_name[0], x): var_name[1] for var_name in exp.as_list()})
            else:
                exp = explainer.explain_instance(
                    x.loc[i].to_numpy(),
                    lambda values: _with_feature_names(values, self.model.predict),
                    num_features=x.shape[1],
                )
                lime_contrib.append({_transform_name(var_name[0], x): var_name[1] for var_name in exp.as_list()})

        contributions = pd.DataFrame(lime_contrib, index=x.index)
        contributions = contributions[list(x.columns)]

        explain_data = dict(contributions=contributions)

        return explain_data


def _transform_name(var_name, x_df):
    """Function for transform name of LIME contribution shape to a comprehensive name"""
    for colname in list(x_df.columns):
        if f" {colname} " in f" {var_name} ":
            col_rename = colname
    return col_rename
