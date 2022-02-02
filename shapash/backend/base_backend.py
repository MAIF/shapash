from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import pandas as pd

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

    def __init__(self, model: Any):
        self.model = model
        self.explain_data: Any = None
        self._state = None
        self._case, self._classes = check_model(model)
        self._has_run = False

    @abstractmethod
    def _run_explainer(self, x: pd.DataFrame) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _get_global_features_importance(self, explain_data: Any, subset: Optional[List[int]] = None):
        raise NotImplementedError

    @abstractmethod
    def _get_local_contributions(self, explain_data: Any, subset: Optional[List[int]] = None):
        raise NotImplementedError

    def run_explainer(self, x: pd.DataFrame):
        self.explain_data = self._run_explainer(x=x)
        self._has_run = True

    def get_local_contributions(
            self,
            x: pd.DataFrame,
            subset: Optional[List[int]] = None,
            preprocessing: Any = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:

        if self._has_run is False:
            self.run_explainer(x=x)

        # Get local contributions using inherited backend method
        contributions = self._get_local_contributions(explain_data=self.explain_data, subset=subset)

        # Put contributions in the right format and perform aggregations
        contributions = self.format_and_aggregate_contributions(x, contributions, preprocessing)

        return contributions

    def get_global_features_importance(
            self,
            subset: Optional[List[int]] = None,
            preprocessing: Any = None
    ) -> Union[pd.Series, List[pd.Series]]:
        return self._get_global_features_importance(subset)

    def format_and_aggregate_contributions(
            self,
            x: pd.DataFrame,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            preprocessing: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        contributions = adapt_contributions(self._case, contributions)
        self._state = self.choose_state(contributions)
        check_contribution_object(self._case, self._classes, contributions)
        contributions = self._state.validate_contributions(contributions, x)
        # TODO : check if we need to do that for all backends. Sometimes the user may want
        #  to apply its custom preprocessing for his case to his contributions
        contributions = self.apply_preprocessing(contributions, preprocessing)
        return contributions

    def choose_state(self, contributions: Union[pd.DataFrame, List[pd.DataFrame]]):
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

    def apply_preprocessing(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            preprocessing: Any = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
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
