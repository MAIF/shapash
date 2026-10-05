"""
Multi Decorator module
"""

from typing import Any

import pandas as pd

from shapash.explainer.smart_state import SmartState


class MultiDecorator:
    """
    Decorator pattern. It simply iterates the method of its member as many times as needed.
    It thus extends any class to apply its methods to a list of arguments.
    """

    def __init__(self, member: SmartState) -> None:
        self.member = member

    def __getattr__(self, item: str) -> Any:
        if item in [x for x in dir(SmartState) if not x.startswith("__")]:

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.delegate(item, *args, **kwargs)

            return wrapper
        else:
            return self.__getattribute__(item)

    def delegate(self, func: str, *args: Any, **kwargs: Any) -> list[Any]:
        """
        Delegate the call to a function with arguments to its member.
        The function is executed as many times as there are elements in the first argument,
        which should be a list.

        Parameters
        ----------
        func : str
            Name of the method to apply.
        *args : Any
            Positional arguments passed to each delegated call. The first
            argument must be a list; if its elements are tuples, each tuple is
            expanded as positional arguments.
        **kwargs : Any
            Keyword arguments passed unchanged to each delegated call.

        Returns
        -------
        list[Any]
            Result of the function applied iteratively to all elements of the first argument.
        """
        self.check_args(args, func)
        method = getattr(self.member, func)
        self.check_method(method, func)
        first_arg, other_args = args[0], args[1:]
        self.check_first_arg(first_arg, func)
        if isinstance(first_arg[0], tuple):
            output_list = [method(*elem, *other_args, **kwargs) for elem in first_arg]
        else:
            output_list = [method(elem, *other_args, **kwargs) for elem in first_arg]
        return output_list

    def check_args(self, args: tuple[Any, ...], name: str) -> None:
        """
        Check if there are arguments in a function call. Raise exception otherwise.

        Parameters
        ----------
        args : tuple[Any, ...]
            Arguments of a function call.
        name : str
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if there is no argument.
        """
        if not args:
            raise ValueError(
                f"{name} is applied without arguments, please check that you have specified contributions."
            )

    def check_method(self, method: Any, name: str) -> None:
        """
        Check if the method is callable. Raise exception otherwise.

        Parameters
        ----------
        method : Any
            Class method or function.
        name : str
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if not callable.
        """
        if not callable(method):
            raise ValueError(f"{name} is not an allowed function, please check for any typo")

    def check_first_arg(self, arg: Any, name: str) -> None:
        """
        Check if the first argument is a list. Raise exception otherwise.

        Parameters
        ----------
        arg : Any
            Any argument, should be a list.
        name : str
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if first argument is not a list.
        """
        if not isinstance(arg, list):
            raise ValueError(
                f"{name} is not applied to a list of contributions,"
                "please check that you are dealing with a multi-class problem."
            )

    def assign_contributions(self, ranked: list[list[pd.DataFrame]]) -> dict[str, list[pd.DataFrame]]:
        """
        Override assign_contributions from SmartState. Turn a nested list into a dict of lists.

        Parameters
        ----------
        ranked : list[list[pd.DataFrame]]
            Nested list coming from multiple applications of rank_contributions.

        Returns
        -------
        dict[str, list[pd.DataFrame]]
            Dictionary containing three keys, and whose values are the successive results.

        Raises
        ------
        ValueError
            The output of a single call to rank_contributions should always be of length three.
        """
        dicts = self.delegate("assign_contributions", ranked)
        keys = list(dicts[0].keys())
        return {key: [d[key] for d in dicts] for key in keys}

    def check_contributions(
        self, contributions: list[pd.DataFrame], x_init: pd.DataFrame, features_names: bool = True
    ) -> bool:
        """
        Override check_contributions from SmartState.
        Return True if all conditions computed are True.

        Parameters
        ----------
        contributions : list[pd.DataFrame]
            List of local contributions to check.
        x_init : pd.DataFrame
            Prediction set.
        features_names : bool, optional
            Whether to check that contribution and input feature names match.

        Returns
        -------
        bool
            True if all inputs share same shape and index with the prediction set.
        """
        bools = self.delegate("check_contributions", contributions, x_init, features_names)
        return all(bools)

    def combine_masks(self, masks: list[list[pd.DataFrame]]) -> list[pd.DataFrame]:
        """
        Override combine_masks. Combine a nested list of masks with the AND operator.

        Parameters
        ----------
        masks : list[list[pd.DataFrame]]
            Nested list of boolean pandas.DataFrames.

        Returns
        -------
        list[pd.DataFrame]
            Combined mask for each contribution set.
        """
        transposed_masks = list(map(list, zip(*masks, strict=False)))
        return self.delegate("combine_masks", transposed_masks)

    def compute_masked_contributions(self, s_contrib: list[pd.DataFrame], masks: list[pd.DataFrame]) -> list[pd.Series]:
        """
        Override compute_masked_contributions. Apply a list of masks to a list of
        contribution matrix and compute for each pair the total masked contributions.

        Parameters
        ----------
        s_contrib : list[pd.DataFrame]
            List of local contributions matrices (pandas.DataFrames).
        masks : list[pd.DataFrame]
            List of masks to apply to contributions matrices0 (pandas.DataFrames, same order).

        Returns
        -------
        list[pd.Series]
            List of masked contributions (pandas.Series).
        """
        arg_tup = list(zip(s_contrib, masks, strict=False))
        return self.delegate("compute_masked_contributions", arg_tup)

    def summarize(
        self,
        s_contribs: list[pd.DataFrame],
        var_dicts: list[pd.DataFrame],
        xs_sorted: list[pd.DataFrame],
        masks: list[pd.DataFrame],
        columns_dict: dict[Any, str],
        features_dict: dict[str, str],
    ) -> list[pd.DataFrame]:
        """
        Compute the summarized contributions of hidden features.

        Parameters
        ----------
        s_contribs : list[pd.DataFrame]
            list of Matrix contributions that will be summarized
        var_dicts : list[pd.DataFrame]
            list of Matrix of features names that will be summarized
        xs_sorted : list[pd.DataFrame]
            list of Matrix containing the value of each feature
        masks : list[pd.DataFrame]
            list of Mask to apply during the summary step
        columns_dict : dict[Any, str]
            Dict of column Names, matches column num with column name
        features_dict : dict[str, str]
            Dict of column Label, matches column name with column label

        Returns
        -------
        list[pd.DataFrame]
            Result of the summarize step
        """
        arg_tup = list(zip(s_contribs, var_dicts, xs_sorted, masks, strict=False))
        return self.delegate("summarize", arg_tup, columns_dict, features_dict)

    def compute_features_import(self, contributions: list[pd.DataFrame], norm: int | float = 1) -> list[pd.Series]:
        """
        Compute a relative features importance, sum of absolute values
        of the contributions for each
        features importance compute in base 100

        Parameters
        ----------
        contributions : list[pd.DataFrame]
            list of pandas.DataFrames containing contributions
        norm : int or float, optional
            Norm used to compute the feature importance. Defaults to 1.

        Returns
        -------
        list[pd.Series]
            list of features importance pandas.series
        """
        return self.delegate("compute_features_import", contributions, norm)

    def compute_grouped_contributions(
        self, contributions: list[pd.DataFrame], features_groups: dict[str, list[str]]
    ) -> list[pd.DataFrame]:
        """
        Regroup contributions according to features_groups parameter.

        Parameters
        ----------
        contributions : list[pd.DataFrame]
            List of contributions of each unique feature.
        features_groups : dict[str, list[str]]
            Python dict that inform which features to regroup.

        Returns
        -------
        list[pd.DataFrame]
            Grouped contributions for each contribution matrix.
        """
        return self.delegate("compute_grouped_contributions", contributions, features_groups)
