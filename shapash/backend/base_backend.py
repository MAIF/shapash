from abc import ABC, abstractmethod

from shapash.utils.check import check_model, check_contribution_object
from shapash.utils.transform import adapt_contributions
from shapash.explainer.smart_state import SmartState
from shapash.explainer.multi_decorator import MultiDecorator


class BaseBackend(ABC):

    # class properties
    # --------------------
    # `column_aggregation` defines a way to aggregate local contributions.
    # Default is sum, possible values are 'sum' or 'first'.
    # It allows to compute (column-wise) aggregation of local contributions.
    column_aggregation = 'sum'

    # `name` defines the string name of the backend allowing to identify and
    # construct the backend from it.
    name = 'base'

    def __init__(self, model):
        self.model = model
        self._state = None
        self.contributions = None
        self._case, self._classes = check_model(model)

    def get_global_features_importance(self, subset=None):
        return self._get_global_features_importance(subset)

    def get_local_contributions(self, X, preprocessing=None):
        # Compute local contributions using inherited backend method
        contributions = self._get_local_contributions(X)

        # Put contributions in the right format and perform aggregations
        contributions = self.format_and_aggregate_contributions(X, contributions, preprocessing)

        self.contributions = contributions
        return contributions

    @abstractmethod
    def _get_global_features_importance(self, subset=None):
        raise NotImplementedError

    @abstractmethod
    def _get_local_contributions(self, X):
        raise NotImplementedError

    def format_and_aggregate_contributions(self, X, contributions, preprocessing):
        contributions = adapt_contributions(self._case, contributions)
        self._state = self.choose_state(contributions)
        check_contribution_object(self._case, self._classes, contributions)
        contributions = self._state.validate_contributions(contributions, X)
        # TODO : check if we need to do that for all backends. Sometimes the user may want
        #  to apply its custom preprocessing for his case to his contributions
        contributions = self.apply_preprocessing(contributions, preprocessing)
        return contributions

    def choose_state(self, contributions):
        """
        Select implementation of the smart explainer. Typically check if it is a
        multi-class problem, in which case the implementation should be adapted
        to lists of contributions.

        Parameters
        ----------
        contributions : object
            Local contributions. Could also be a list of local contributions.

        Returns
        -------
        object
            SmartState or SmartMultiState, depending on the nature of the input.
        """
        if isinstance(contributions, list):
            return MultiDecorator(SmartState())
        else:
            return SmartState()

    def apply_preprocessing(self, contributions, preprocessing=None):
        """
        Reconstruct contributions for original features, taken into account a preprocessing.

        Parameters
        ----------
        contributions : object
            Local contributions, or list of local contributions.
        preprocessing : object
            Encoder taken from scikit-learn or category_encoders

        Returns
        -------
        object
            Reconstructed local contributions in the original space. Can be a list.
        """
        if preprocessing:
            return self._state.inverse_transform_contributions(
                contributions,
                preprocessing,
                agg_columns=self.column_aggregation
            )
        else:
            return contributions
