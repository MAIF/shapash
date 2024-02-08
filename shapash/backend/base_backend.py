from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from shapash.utils.check import check_contribution_object, check_model
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
    column_aggregation = "sum"

    # `name` defines the string name of the backend allowing to identify and
    # construct the backend from it.
    name = "base"
    support_groups = True
    supported_cases = ["classification", "regression"]

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
        self.state = None
        self._case, self._classes = check_model(model)
        if self._case not in self.supported_cases:
            raise ValueError(f"Model not supported by the backend as it does not cover {self._case} case")

    @abstractmethod
    def run_explainer(self, x: pd.DataFrame) -> dict:
        """
        Computes local contributions.
        Must be implemented by a child class

        Parameters
        ----------
        x : pd.DataFrame
            The observations dataframe used by the model

        Returns
        -------
        explain_data : dict
            dict containing local contributions
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` is a subclass of BaseBackend and "
            f"must implement the `_run_explainer` method"
        )

    def get_local_contributions(
        self, x: pd.DataFrame, explain_data: Any, subset: Optional[List[int]] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """Get local contributions using the explainer data computed in the `run_explainer`
        method.

        It applies some aggregations and transformations to the result of the `run_explainer` method
        if needed. For example, if there are some one-hot-encoded columns, it automatically checks
        and applies aggregations depending on the result of `_run_explainer` and the `preprocessing`
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
        assert isinstance(explain_data, dict), "The _run_explainer method should return a dict"
        if "contributions" not in explain_data.keys():
            raise ValueError(
                "The _run_explainer method should return a dict"
                " with at least `contributions` key containing "
                "the local contributions"
            )

        local_contributions = explain_data["contributions"]
        if subset is not None:
            local_contributions = local_contributions.loc[subset]
        local_contributions = self.format_and_aggregate_local_contributions(x, local_contributions)
        return local_contributions

    def get_global_features_importance(
        self, contributions: pd.DataFrame, explain_data: Optional[dict] = None, subset: Optional[List[int]] = None
    ) -> Union[pd.Series, List[pd.Series]]:
        """Get global contributions using the explainer data computed in the `run_explainer`
        method.

        Parameters
        ----------
        contributions : pd.DataFrame or list of pd.DataFrame
            The local contributions computed and aggregated by the backend.
        explain_data : dict, optional
            The data computed in the `run_explainer` method.
        subset : list
            list of indices on which to get local contributions.

        Returns
        -------
        pd.Series or list of pd.Series
            The global features importance computed by the backend.
        """
        state = choose_state(contributions)
        if subset is not None:
            if isinstance(contributions, list):
                contributions = [c.loc[subset] for c in contributions]
            else:
                contributions = contributions.loc[subset]
        return state.compute_features_import(contributions)

    def format_and_aggregate_local_contributions(
        self,
        x: pd.DataFrame,
        contributions: Union[pd.DataFrame, np.array, List[pd.DataFrame], List[np.array]],
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        This function allows to format and aggregate contributions in the right format
        (pd.DataFrame or list of pd.DataFrame).

        If a preprocessing exists, it also uses it to change the contributions if needed.

        Parameters
        ----------
        x : pd.DataFrame
            The dataframe of observations used by the model.
        contributions : pd.DataFrame or np.array or list of pd.DataFrame or list of np.array
            Local contributions, or list of local contributions.

        Returns
        -------
        contributions : pd.DataFrame or list of pd.DataFrame
            Contributions formatted and aggregated
        """
        contributions = adapt_contributions(self._case, contributions)
        self.state = choose_state(contributions)
        check_contribution_object(self._case, self._classes, contributions)
        contributions = self.state.validate_contributions(contributions, x)
        contributions_cols = (
            contributions.columns.to_list()
            if isinstance(contributions, pd.DataFrame)
            else contributions[0].columns.to_list()
        )
        if _needs_preprocessing(contributions_cols, x, self.preprocessing):
            contributions = self._apply_preprocessing(contributions)
        return contributions

    def _apply_preprocessing(
        self, contributions: Union[pd.DataFrame, List[pd.DataFrame]]
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
            return self.state.inverse_transform_contributions(
                contributions, self.preprocessing, agg_columns=self.column_aggregation
            )
        else:
            return contributions


def _needs_preprocessing(result_cols, x, preprocessing):
    """
    Checks if preprocessing is needed depending on the preprocessing used.
    """
    mapping = get_preprocessing_mapping(x, preprocessing)
    cols_after_preprocessing = [x for list_c in mapping.values() for x in list_c]
    for col in result_cols:
        if col in cols_after_preprocessing and col not in mapping.keys():
            return True
    return False
