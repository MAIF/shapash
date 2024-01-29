"""
Multi Decorator module
"""
from shapash.explainer.smart_state import SmartState


class MultiDecorator:
    """
    Decorator pattern. It simply iterates the method of its member as many times as needed.
    It thus extends any class to apply its methods to a list of arguments.
    """

    def __init__(self, member):
        self.member = member

    def __getattr__(self, item):
        if item in [x for x in dir(SmartState) if not x.startswith("__")]:

            def wrapper(*args, **kwargs):
                return self.delegate(item, *args, **kwargs)

            return wrapper
        else:
            return self.__getattribute__(item)

    def delegate(self, func, *args, **kwargs):
        """
        Delegate the call to a function with arguments to its member.
        The function is executed as many times as there are elements in the first argument,
        which should be a list.

        Parameters
        ----------
        func : string
            Name of the method to apply.
        first_arg : list
            contains arguments to specify to each call of method
            if first_arg is a list of tuple, the delegate function use each element of tuple as
            an argument
        other_args : str, list, pd.DataFrame, array (optional)
            any argument that method needs
            notice: other_args is constant for each call of method

        Returns
        -------
        list
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

    def check_args(self, args, name):
        """
        Check if there are arguments in a function call. Raise exception otherwise.

        Parameters
        ----------
        args : object
            Arguments of a function call.
        name : string
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if there is no argument.
        """
        if not args:
            raise ValueError(
                "{} is applied without arguments," "please check that you have specified contributions.".format(name)
            )

    def check_method(self, method, name):
        """
        Check if the method is callable. Raise exception otherwise.

        Parameters
        ----------
        method : object
            Class method or function.
        name : string
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if not callable.
        """
        if not callable(method):
            raise ValueError(f"{name} is not an allowed function, please check for any typo")

    def check_first_arg(self, arg, name):
        """
        Check if the first argument is a list. Raise exception otherwise.

        Parameters
        ----------
        arg : object
            Any argument, should be a list.
        name : string
            Name of function targeted.

        Raises
        ------
        ValueError
            Raise if first argument is not a list.
        """
        if not isinstance(arg, list):
            raise ValueError(
                "{} is not applied to a list of contributions,"
                "please check that you are dealing with a multi-class problem.".format(name)
            )

    def assign_contributions(self, ranked):
        """
        Override assign_contributions from SmartState. Turn a nested list into a dict of lists.

        Parameters
        ----------
        ranked : list
            Nested list coming from multiple applications of rank_contributions.

        Returns
        -------
        dict
            Dictionary containing three keys, and whose values are the successive results.

        Raises
        ------
        ValueError
            The output of a single call to rank_contributions should always be of length three.
        """
        dicts = self.delegate("assign_contributions", ranked)
        keys = list(dicts[0].keys())
        return {key: [d[key] for d in dicts] for key in keys}

    def check_contributions(self, contributions, x_init, features_names=True):
        """
        Override check_contributions from SmartState.
        Return True if all conditions computed are True.

        Parameters
        ----------
        contributions : list
            List of local contributions to check.
        x_init : pandas.DataFrame
            Prediction set.

        Returns
        -------
        Bool
            True if all inputs share same shape and index with the prediction set.
        """
        bools = self.delegate("check_contributions", contributions, x_init, features_names)
        return all(bools)

    def combine_masks(self, masks):
        """
        Override combine_masks. Combine a nested list of masks with the AND operator.

        Parameters
        ----------
        masks : list
            Nested list of boolean pandas.DataFrames.

        Returns
        -------
        pd.Dataframe
            Combination of all masks.
        """
        transposed_masks = list(map(list, zip(*masks)))
        return self.delegate("combine_masks", transposed_masks)

    def compute_masked_contributions(self, s_contrib, masks):
        """
        Override compute_masked_contributions. Apply a list of masks to a list of
        contribution matrix and compute for each pair the total masked contributions.

        Parameters
        ----------
        s_contrib : list
            List of local contributions matrices (pandas.DataFrames).
        masks : list
            List of masks to apply to contributions matrices0 (pandas.DataFrames, same order).

        Returns
        -------
        list
            List of masked contributions (pandas.Series).
        """
        arg_tup = list(zip(s_contrib, masks))
        return self.delegate("compute_masked_contributions", arg_tup)

    def summarize(self, s_contribs, var_dicts, xs_sorted, masks, columns_dict, features_dict):
        """
        Compute the summarized contributions of hidden features.

        Parameters
        ----------
        s_contribs: list
            list of Matrix contributions that will be summarized
        var_dicts: list
            list of Matrix of features names that will be summarized
        xs_sorted: list
            list of Matrix containing the value of each feature
        masks: list
            list of Mask to apply during the summary step
        columns_dict: dict
            Dict of column Names, matches column num with column name
        features_dict: dict
            Dict of column Label, matches column name with column label

        Returns
        -------
        list of pd.DataFrame
            Result of the summarize step
        """
        arg_tup = list(zip(s_contribs, var_dicts, xs_sorted, masks))
        return self.delegate("summarize", arg_tup, columns_dict, features_dict)

    def compute_features_import(self, contributions):
        """
        Compute a relative features importance, sum of absolute values
         ​​of the contributions for each
         features importance compute in base 100

        Parameters
        ----------
        contributions : list
             list of pandas.DataFrames containing contributions

        Returns
        -------
        list
            list of features importance pandas.series
        """
        return self.delegate("compute_features_import", contributions)

    def compute_grouped_contributions(self, contributions, features_groups):
        """
        Regroup contributions according to features_groups parameter.

        Parameters
        ----------
        contributions : list
            List of contributions of each unique feature.
        features_groups : dict
            Python dict that inform which features to regroup.

        Returns
        -------
        pd.DataFrame
        """
        return self.delegate("compute_grouped_contributions", contributions, features_groups)
