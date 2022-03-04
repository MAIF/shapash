from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import pandas as pd

from shapash.utils.check import check_model, check_contribution_object
from shapash.utils.transform import adapt_contributions, get_preprocessing_mapping
from shapash.utils.utils import choose_state


class BaseBackend(ABC):
    """``BaseBackend`` is the base class for all backends.
    All explainability implementations should extend this abstract class
    and implement the methods marked as abstract.
    """

    # class properties
    # --------------------
    # `column_aggregation` defines a way to aggregate local contributions.
    # Default is sum, possible values are 'sum' or 'first'.
    # It allows to compute (column-wise) aggregation of local contributions.
    column_aggregation = 'sum'

    # `name` defines the string name of the backend allowing to identify and
    # construct the backend from it.
    name = 'base'

    def __init__(self, model: Any, preprocessing: Optional[Any] = None):
        """Create a backend instance using a given implementation.

        Parameters
        ----------
        model : any
            Model used.
        preprocessing: category_encoders, ColumnTransformer, list or dict
            The processing apply to the original data.
        """
        self.model = model
        self.preprocessing = preprocessing
        self.explain_data: Any = None
        self._state = None
        self._case, self._classes = check_model(model)

    @abstractmethod
    def _run_explainer(self, x: pd.DataFrame) -> Any:
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of BaseBackend and "
            f"must implement the `_run_explainer` method"
        )

    @abstractmethod
    def _get_local_contributions(
            self,
            x: pd.DataFrame,
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of BaseBackend and "
            f"must implement the `_get_local_contributions` method"
        )

    @abstractmethod
    def _get_global_features_importance(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.Series, List[pd.Series]]:
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of BaseBackend and "
            f"must implement the `_get_global_features_importance` method"
        )

    def run_explainer(self, x: pd.DataFrame) -> Any:
        """This method computes all the explainability data that can be computationally
        intensive in a dictionary.

        The result will then be used in the `get_local_contributions` and
        `get_global_features_importance` methods.

        Parameters
        ----------
        x : pd.DataFrame
            The dataframe of observations used by the model.

        Returns
        -------
        explainability_data : dict
            All the data required to get local and global explainability.
        """
        return self._run_explainer(x)

    def get_local_contributions(
            self,
            x: pd.DataFrame,
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Get local contributions using the explainer data computed in the `run_explainer`
        method.

        This method is based on the `_get_local_contributions`. It applies some aggregations
        and transformations to the result of the `_get_local_contributions` if needed. For
        example, if there are some one-hot-encoded columns, it automatically checks and applies
        aggregations depending on the result of `_get_local_contributions` and the `preprocessing`
        (encoder).

        Parameters
        ----------
        x : pd.DataFrame
            The dataframe of observations used by the model.
        explain_data : dict
            The data computed in the `run_explainer` method.
        subset : list
            list of indices on which to get local contributions.

        Returns
        -------
        local_contributions : pd.DataFrame
            The local contributions computed by the backend.
        """
        local_contributions = self._get_local_contributions(x, explain_data, subset)
        local_contributions = self.format_and_aggregate_local_contributions(x, local_contributions)
        return local_contributions

    def get_global_features_importance(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
            explain_data: Any,
            subset: Optional[List[int]] = None
    ) -> Union[pd.Series, List[pd.Series]]:
        """Get global contributions using the explainer data computed in the `run_explainer`
        method.

        This method is based on the `_get_global_features_importance`.

        Parameters
        ----------
        contributions : pd.DataFrame
            The dataframe of local contributions formatted and aggregated, result of
            the `get_local_contributions` method.
        explain_data : dict
            The data computed in the `run_explainer` method.
        subset : list
            list of indices on which to get local contributions.

        Returns
        -------
        pd.DataFrame
            The global features importance computed by the backend.
        """
        return self._get_global_features_importance(contributions, explain_data, subset)

    def format_and_aggregate_local_contributions(
            self,
            x: pd.DataFrame,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]],
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        contributions = adapt_contributions(self._case, contributions)
        self._state = choose_state(contributions)
        check_contribution_object(self._case, self._classes, contributions)
        contributions = self._state.validate_contributions(contributions, x)
        contributions_cols = (
            contributions.columns.to_list() if isinstance(contributions, pd.DataFrame)
            else contributions[0].columns.to_list()
        )
        if _needs_preprocessing(contributions_cols, x, self.preprocessing):
            contributions = self.apply_preprocessing(contributions)
        return contributions

    def apply_preprocessing(
            self,
            contributions: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Reconstruct contributions for original features, taken into account a preprocessing.

        Parameters
        ----------
        contributions : object
            Local contributions, or list of local contributions.

        Returns
        -------
        object
            Reconstructed local contributions in the original space. Can be a list.
        """
        if self.preprocessing:
            return self._state.inverse_transform_contributions(
                contributions,
                self.preprocessing,
                agg_columns=self.column_aggregation
            )
        else:
            return contributions


def _needs_preprocessing(result_cols, x, preprocessing):
    mapping = get_preprocessing_mapping(x, preprocessing)
    cols_after_preprocessing = [x for list_c in mapping.values() for x in list_c]
    for col in result_cols:
        if col in cols_after_preprocessing and col not in mapping.keys():
            return True
    return False
