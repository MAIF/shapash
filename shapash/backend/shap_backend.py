import shap

from shapash.backend.base_backend import BaseBackend


class ShapBackend(BaseBackend):
    # When grouping features contributions together, Shap uses the sum of the contributions
    # of the features that belong to the group
    column_aggregation = 'sum'

    def __init__(self, model, explainer_args=None, explainer_compute_args=None):
        super(ShapBackend, self).__init__(model)
        self.explainer_args = explainer_args if explainer_args else {}
        self.explainer_compute_args = explainer_compute_args if explainer_compute_args else {}
        self.explainer = shap.TreeExplainer(model=model, **self.explainer_args)

    def _get_global_features_importance(self, subset=None):
        if self.contributions is None:
            raise AssertionError('Local contributions should be computed first')

        if subset is not None:
            return self._state.compute_features_import(self.contributions.loc[subset])
        else:
            return self._state.compute_features_import(self.contributions)

    def _get_local_contributions(self, X):
        contributions = self.explainer(X, **self.explainer_compute_args).values

        return contributions
