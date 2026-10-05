"""
Smart State Module
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from shapash.decomposition.contributions import (
    assign_contributions,
    inverse_transform_contributions,
    rank_contributions,
)
from shapash.manipulation.filters import (
    cap_contributions,
    combine_masks,
    cutoff_contributions,
    hide_contributions,
    sign_contributions,
)
from shapash.manipulation.mask import compute_masked_contributions, init_mask
from shapash.manipulation.summarize import compute_features_import, group_contributions, summarize


class SmartState:
    """
    State pattern attached to SmartExplainer. This is the base class that only handles matrices
    of local contributions. The multi-class case is tackled in SmartMultiState.
    """

    def validate_contributions(self, contributions: np.ndarray | pd.DataFrame, x_init: pd.DataFrame) -> pd.DataFrame:
        """
        Check type of contributions and transform into pd.Dataframe if necessary

        Parameters
        ----------
        contributions : pandas.DataFrame or numpy.ndarray
            Local contributions
        x_init : pandas.DataFrame
            Prediction set.

        Returns
        -------
        pandas.DataFrame
            Local contributions on the original feature space (no encoding).
        """
        if not isinstance(contributions, np.ndarray | pd.DataFrame):
            raise ValueError("Type of contributions must be pd.DataFrame or np.ndarray")
        if isinstance(contributions, np.ndarray):
            return pd.DataFrame(contributions, columns=x_init.columns, index=x_init.index)
        else:
            return contributions

    def inverse_transform_contributions(
        self,
        contributions: pd.DataFrame,
        preprocessing: Any | None,
        agg_columns: Literal["sum", "first"] = "sum",
    ) -> pd.DataFrame:
        """
        Compute local contributions in the original feature space, despite category encoding.

        Parameters
        ----------
        contributions : pandas.DataFrame
            Local contributions of a model on a prediction set.
        preprocessing : Any or None, optional
            Single step of preprocessing, typically a category encoder.
        agg_columns : Literal["sum", "first"], optional
            Type of aggregation performed. For SHAP, contributions of one-hot encoded variables are summed.

        Returns
        -------
        pandas.DataFrame
            Local contributions on the original feature space (no encoding).
        """
        return inverse_transform_contributions(contributions, preprocessing, agg_columns)

    def check_contributions(
        self, contributions: pd.DataFrame, x_init: pd.DataFrame, features_names: bool = True
    ) -> bool:
        """
        Check that contributions and prediction set match in terms of lines and columns.

        Parameters
        ----------
        contributions : pandas.DataFrame
            Local contributions to check.
        x_init : pandas.DataFrame
            Prediction set.
        features_names : bool, optional (default: True)
            Whether to check that contributions and x_init have the same feature names.
        Returns
        -------
        bool
            True if inputs share shape and index. False otherwise.
        """
        if x_init.shape != contributions.shape:
            return False
        if not x_init.index.equals(contributions.index):
            return False
        if features_names:
            if not x_init.columns.equals(contributions.columns):
                return False
        else:
            if not len(x_init.columns) == len(contributions.columns):
                return False
        return True

    def rank_contributions(self, contributions: pd.DataFrame, x_init: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Rank contributions line by line and build a reference dictionary to the prediction set.

        Parameters
        ----------
        contributions : pandas.DataFrame
            Local contributions to sort.
        x_init : pandas.DataFrame
            Prediction set.

        Returns
        -------
        pandas.DataFrame
            Local contributions sorted by decreasing absolute values.
        pandas.DataFrame
            Input features sorted by decreasing contributions absolute values.
        pandas.DataFrame
            Input features names sorted for each observation
            by decreasing contributions absolute values.
        """
        return rank_contributions(contributions, x_init)

    def assign_contributions(self, ranked: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Turn a list of results into a dict.

        Parameters
        ----------
        ranked : list
            The output of rank_contributions.

        Returns
        -------
        dict
            Same data but rearrange into a dict with explicit names.

        Raises
        ------
        ValueError
            The output of rank_contributions should always be of length three.
        """
        return assign_contributions(ranked)

    def hide_contributions(self, var_dict: pd.DataFrame, features_list: list[int]) -> pd.DataFrame:
        """
        Returns Boolean dataframe with True/False depending if the
        feature is present or not in the list of
        feature to hide.

        Parameters
        ----------
        var_dict: pd.DataFrame
            Dataframe with features indexes ordered
            by contribution.
        features_list : list of int
            List of feature indexes to hide.

        Returns
        -------
        pd.DataFrame
            Boolean dataframe depend on hidden features.
        """
        return hide_contributions(var_dict, features_list)

    def cap_contributions(self, s_contrib: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """
        Compute a mask indicating where the input matrix
        has values above a given threshold in absolute value.

        Parameters
        ----------
        s_contrib : pandas.DataFrame
            Local contributions, positive and negative values.
        threshold : float, optional (default: 0.1)
            User defined threshold above which local contributions are hidden.

        Returns
        -------
        pandas.DataFrame
            Mask with only True or False elements.
        """
        return cap_contributions(s_contrib, threshold=threshold)

    def sign_contributions(self, dataframe: pd.DataFrame, positive: bool = True) -> pd.DataFrame:
        """
        Returns Boolean values depending on
        the signs of local contributions
        stored in dataframe and on the positive parameter.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Local contributions of the model.
        positive : bool, optional (default: True)
            True to evaluate positive value.
            False to evaluate negative value.

        Returns
        -------
        pandas.DataFrame
            Dataframe with boolean value.
        """
        return sign_contributions(dataframe, positive=positive)

    def cutoff_contributions(self, dataframe: pd.DataFrame, max_contrib: int) -> pd.DataFrame:
        """
        The function cutoff_contributions computes a mask on a sorted contribution matrix.
        It outputs True everywhere the contribution is in the top-k,
        k being defined as an option by the user.

        Parameters
        ----------
        dataframe : pd.Dataframe
            DataFrame is a sorted shapley matrix.
        max_contrib : int
            The k most important contributions to keep.

        Returns
        -------
        pd.Dataframe
            Mask indicating where contributions should be considered.
        """
        return cutoff_contributions(dataframe, max_contrib)

    def combine_masks(self, masks: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine a list of masks with the AND operator.

        Parameters
        ----------
        masks : list of pandas.DataFrame
            List of boolean pandas.DataFrames.

        Returns
        -------
        pd.Dataframe
            Combination of all masks.
        """
        return combine_masks(masks)

    def compute_masked_contributions(self, s_contrib: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the summed contributions of hidden features.

        Parameters
        ----------
        s_contrib : pd.DataFrame
            Matrix with both positive and negative values
        masks : pd.DataFrame
            Matrix with only True or False elements. False elements are the hidden elements.

        Returns
        -------
        pd.DataFrame
            Sum of hidden negative and positive contributions.
        """
        return compute_masked_contributions(s_contrib, masks)

    def init_mask(self, s_contrib: pd.DataFrame, value: bool = True) -> pd.DataFrame:
        """
        Initialize a True mask for the dataset.

        Parameters
        ----------
        s_contrib : pd.DataFrame
            Matrix with both positive and negative values
        value : bool, optional (default: True)
            Value used for initialize the mask

        Returns
        -------
        pd.Dataframe
            mask initialized
        """
        return init_mask(s_contrib, value)

    def summarize(
        self,
        s_contrib: pd.DataFrame,
        var_dict: pd.DataFrame,
        x_sorted: pd.DataFrame,
        mask: pd.DataFrame,
        columns_dict: dict[Any, str],
        features_dict: dict[str, str],
    ) -> pd.DataFrame:
        """
        Compute the summarized contributions of features.

        Parameters
        ----------
        s_contrib: pd.DataFrame
            Matrix containing contributions that will be summarized
        var_dict: pd.DataFrame
            Matrix of feature names that will be summarized
        x_sorted: pd.DataFrame
            Matrix containing the value of each feature
        mask: pd.DataFrame
            Mask to apply during the summary step
        columns_dict : dict
            Mapping from column indexes to column names.
        features_dict : dict[str, str]
            Mapping from technical feature names to display labels.

        Returns
        -------
        pd.DataFrame
            Result of the summarize step
        """
        return summarize(s_contrib, var_dict, x_sorted, mask, columns_dict, features_dict)

    def compute_features_import(self, contributions: pd.DataFrame, norm: int | float = 1) -> pd.Series:
        """
        Compute a relative features importance, sum of absolute values
        of the contributions for each
        features importance compute in base 100
        Parameters
        ----------
        contributions : pd.DataFrame
            Matrix containing contributions

        Returns
        -------
        pd.Series
            feature importance, One row by feature,
            index of the serie = contributions.columns
        """
        return compute_features_import(contributions, norm)

    def compute_grouped_contributions(
        self, contributions: pd.DataFrame, features_groups: dict[str, list[str]]
    ) -> pd.DataFrame:
        """
        Regroup contributions according to features_groups parameter.

        Parameters
        ----------
        contributions : pd.DataFrame
            Contributions of each unique feature.
        features_groups : dict[str, list[str]]
            Mapping from group names to the feature names to regroup.

        Returns
        -------
        pd.DataFrame
        """
        return group_contributions(contributions, features_groups)
