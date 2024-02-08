"""
Smart predictor module
"""
import copy

import pandas as pd

import shapash.explainer.smart_explainer
from shapash.decomposition.contributions import assign_contributions, rank_contributions
from shapash.manipulation.filters import (
    cap_contributions,
    combine_masks,
    cutoff_contributions,
    hide_contributions,
    sign_contributions,
)
from shapash.manipulation.mask import compute_masked_contributions, init_mask
from shapash.manipulation.select_lines import keep_right_contributions
from shapash.manipulation.summarize import create_grouped_features_values, group_contributions, summarize
from shapash.utils.check import (
    check_consistency_model_features,
    check_consistency_model_label,
    check_features_name,
    check_label_dict,
    check_mask_params,
    check_model,
    check_preprocessing,
    check_preprocessing_options,
    check_y,
)
from shapash.utils.columntransformer_backend import columntransformer
from shapash.utils.io import save_pickle
from shapash.utils.model import predict_proba
from shapash.utils.transform import adapt_contributions, apply_postprocessing, apply_preprocessing, preprocessing_tolist


class SmartPredictor:
    """
    The SmartPredictor class is an object lighter than SmartExplainer Object with
    additionnal consistency checks.

    The SmartPredictor object is provided to deploy the summary of local explanation
    for the operational needs.

    Switching from SmartExplainer to SmartPredictor, allows users to reproduce
    the same results automatically on datasets with right structure.

    SmartPredictor is designed to make new results understandable:
        - It checks consistency of all parameters
        - It applies preprocessing and postprocessing
        - It computes models contributions
        - It makes predictions
        - It summarizes local explainability

    This class allows the user to automatically summarize the results of his model
    on new datasets (prediction, preprocessing and postprocessing linking,
    explainability).
    The SmartPredictor has several methods described below.

    The SmartPredictor Attributes :

    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    model: model object
        model used to check the different values of target estimate predict_proba
    backend: str or backend object
        backend (explainer) used to compute contributions
    columns_dict: dict
        Dictionary mapping integer column number (in the same order of the trained dataset) to technical feature names.
    features_types: dict
        Dictionary mapping features with the right types needed.
    label_dict: dict (optional)
        Dictionary mapping integer labels to domain names (classification - target values).
    preprocessing: category_encoders, ColumnTransformer, list or dict (optional)
        The processing apply to the original data.
    postprocessing: dict (optional)
        Dictionary of postprocessing modifications to apply in x_init dataframe.
    _case: string
        String that informs if the model used is for classification or regression problem.
    _classes: list, None
        List of labels if the model used is for classification problem, None otherwise.
    mask_params: dict (optional)
        Dictionary that specify how to summarize the explainability.

    How to declare a new SmartPredictor object?

    Example
    -------
    >>> predictor = SmartPredictor(features_dict=my_features_dict,
    >>>                             model=my_model,
    >>>                             backend=my_backend,
    >>>                             columns_dict=my_columns_dict,
    >>>                             features_types=my_features_type_dict,
    >>>                             label_dict=my_label_dict,
    >>>                             preprocessing=my_preprocess,
    >>>                             postprocessing=my_postprocess)

    or the most common syntax

    >>> predictor = xpl.to_smartpredictor()

    xpl, explainer: object
        SmartExplainer instance to point to.
    """

    def __init__(
        self,
        features_dict,
        model,
        columns_dict,
        backend,
        features_types,
        label_dict=None,
        preprocessing=None,
        postprocessing=None,
        features_groups=None,
        mask_params=None,
    ):
        params_dict = [features_dict, features_types, label_dict, columns_dict, postprocessing]

        for params in params_dict:
            if (params is not None) and (not isinstance(params, dict)):
                raise ValueError(
                    """
                    {} must be a dict.
                    """.format(
                        str(params)
                    )
                )

        self.model = model
        self._case, self._classes = self.check_model()
        self.backend = backend
        self.preprocessing = preprocessing
        self.check_preprocessing()
        self.features_dict = features_dict
        self.features_types = features_types
        self.label_dict = label_dict
        self.check_label_dict()
        self.columns_dict = columns_dict
        self.mask_params = (
            mask_params
            if mask_params is not None
            else {"features_to_hide": None, "threshold": None, "positive": None, "max_contrib": None}
        )
        self.check_mask_params()
        self.postprocessing = postprocessing
        self.features_groups = features_groups
        list_preprocessing = preprocessing_tolist(self.preprocessing)
        check_consistency_model_features(
            self.features_dict,
            self.model,
            self.columns_dict,
            self.features_types,
            self.mask_params,
            self.preprocessing,
            self.postprocessing,
            list_preprocessing,
            self.features_groups,
        )
        check_consistency_model_label(self.columns_dict, self.label_dict)
        self._drop_option = check_preprocessing_options(columns_dict, features_dict, preprocessing, list_preprocessing)

    def check_model(self):
        """
        Check if model has a predict_proba method is a one column dataframe of integer or float
        and if y_pred index matches x_init index

        Returns
        -------
        string:
            'regression' or 'classification' according to the attributes of the model
        """
        _case, _classes = check_model(self.model)
        return _case, _classes

    def check_preprocessing(self):
        """
        Check that all transformation of the preprocessing are supported.
        """
        return check_preprocessing(self.preprocessing)

    def check_label_dict(self):
        """
        Check if label_dict and model _classes match
        """
        if self._case != "regression":
            return check_label_dict(self.label_dict, self._case, self._classes)

    def check_mask_params(self):
        """
        Check if mask_params given respect the expected format.
        """
        return check_mask_params(self.mask_params)

    def add_input(self, x=None, ypred=None, contributions=None):
        """
        The add_input method is the first step to add a dataset for prediction and explainability.

        add_input applies to x parameter :
            - consistencies checks
            - preprocessing and postprocessing specified during the initialisation
            - features reordering with the right order for the model

        If you don't specify ypred or contributions, add_input compute them.
        It's possible to not specified one parameter if it has already been defined before.
        For example, if the user want to specified an ypred without reinitialize the dataset x already defined.
        If the user declare a new input x, all the parameters stored will be cleaned.

        Example
        --------
        >>> predictor.add_input(x=xtest_df)
        >>> predictor.add_input(ypred=ytest_df)

        Parameters
        ----------
        x: dict, pandas.DataFrame (optional)
            Raw dataset used by the model to perform the prediction (not preprocessed).
        ypred: pandas.DataFrame (optional)
            User-specified prediction values.
        contributions: pandas.DataFrame (regression) or list (classification) (optional)
            local contributions aggregated if the preprocessing part requires it (e.g. one-hot encoding).
        """
        if x is not None:
            x = self.check_dataset_features(self.check_dataset_type(x))
            self.data = self.clean_data(x)
            self.data["x_postprocessed"] = self.apply_postprocessing()
            try:
                self.data["x_preprocessed"] = self.apply_preprocessing()
            except BaseException:
                raise ValueError(
                    """
                    Preprocessing has failed. The preprocessing specified or the dataset doesn't match.
                    """
                )
        else:
            if not hasattr(self, "data"):
                raise ValueError("No dataset x specified.")

        if ypred is not None:
            self.data["ypred_init"] = self.check_ypred(ypred)

        if contributions is not None:
            self.data["ypred"], self.data["contributions"] = self.compute_contributions(
                contributions=contributions, use_groups=False
            )
        else:
            self.data["ypred"], self.data["contributions"] = self.compute_contributions(use_groups=False)

        if self.features_groups is not None:
            self._add_groups_input()

    def _add_groups_input(self):
        """
        Compute groups of features values, contributions the same way as add_input method
        and stores it in data_groups attribute
        """
        self.data_groups = dict()
        self.data_groups["x_postprocessed"] = create_grouped_features_values(
            x_init=self.data["x_postprocessed"],
            x_encoded=self.data["x_preprocessed"],
            preprocessing=self.preprocessing,
            features_groups=self.features_groups,
            features_dict=self.features_dict,
            how="dict_of_values",
        )
        self.data_groups["ypred"] = self.data["ypred"]
        self.data_groups["contributions"] = group_contributions(
            contributions=self.data["contributions"], features_groups=self.features_groups
        )

    def check_dataset_type(self, x=None):
        """
        Check if dataset x given respect the expected format.

        Parameters
        ----------
        x: dict, pandas.DataFrame (optional)
            Raw dataset used by the model to perform the prediction (not preprocessed).

        Returns
        -------
        x: pandas.DataFrame
            Raw dataset used by the model to perform the prediction (not preprocessed).

        """
        if not (type(x) in [pd.DataFrame, dict]):
            raise ValueError(
                """
                x must be a dict or a pandas.DataFrame.
                """
            )
        else:
            x = self.convert_dict_dataset(x)
        return x

    def convert_dict_dataset(self, x):
        """
        Convert a dict to a dataframe if the dataset specified is a dict.

        Parameters
        ----------
        x: dict
            Raw dataset used by the model to perform the prediction (not preprocessed).

        Returns
        -------
        x: pandas.DataFrame
            Raw dataset used by the model to perform the prediction (not preprocessed).
        """
        if type(x) == dict:
            if not all([column in self.features_types.keys() for column in x.keys()]):
                raise ValueError(
                    """
                All features from dataset x must be in the features_types dict initialized.
                """
                )
            try:
                x = pd.DataFrame.from_dict(x, orient="index").T
                for feature, type_feature in self.features_types.items():
                    x[feature] = x[feature].astype(type_feature)
            except BaseException:
                raise ValueError(
                    """
                    The structure of the given dict x isn't at the right format.
                    """
                )
        return x

    def check_dataset_features(self, x):
        """
        Check if the features of the dataset x has the expected types before using preprocessing and model.

        Parameters
        ----------
        x: pandas.DataFrame (optional)
            Raw dataset used by the model to perform the prediction (not preprocessed).
        """
        assert all(column in self.columns_dict.values() for column in x.columns)
        if not all([type(key) == int for key in self.columns_dict.keys()]):
            raise ValueError("columns_dict must have only integers keys for features order.")
        features_order = []
        for order in range(min(self.columns_dict.keys()), max(self.columns_dict.keys()) + 1):
            features_order.append(self.columns_dict[order])
        x = x[features_order]

        assert all(column in self.features_types.keys() for column in x.columns)
        if not all([str(x[feature].dtypes) == self.features_types[feature] for feature in x.columns]):
            raise ValueError(
                """
                  Types of features in x doesn't match with the expected one in features_types.
                  x input must be initial dataset without preprocessing applied.
                  """
            )
        return x

    def check_ypred(self, ypred=None):
        """
        Check that ypred given has the right shape and expected value.

        Parameters
        ----------
        ypred: pandas.DataFrame (optional)
            User-specified prediction values.
        """
        return check_y(self.data["x"], ypred)

    def adapt_contributions(self, contributions):
        """
        If _case is "classification" and contributions a np.array or pd.DataFrame
        this function transform contributions matrix in a list of 2 contributions
        matrices: Opposite contributions and contributions matrices.

        Parameters
        ----------
        contributions : pandas.DataFrame, np.ndarray or list

        Returns
        -------
            pandas.DataFrame, np.ndarray or list
            contributions object modified
        """
        return adapt_contributions(self._case, contributions)

    def check_contributions(self, contributions):
        """
        Check if contributions and prediction set match in terms of shape and index.
        """
        if self._drop_option is not None:
            x = self.data["x"][self.data["x"].columns.difference(self._drop_option["features_to_drop"])]
        else:
            x = self.data["x"]

        if not self.backend.state.check_contributions(contributions, x, features_names=False):
            raise ValueError(
                """
                Prediction set and contributions should have exactly the same number of lines
                and number of columns. the order of the columns must be the same
                Please check x, contributions and preprocessing arguments.
                """
            )

    def clean_data(self, x):
        """
        Clean data stored if x is defined and not None.

        Parameters
        ----------
        x: pandas.DataFrame
            Raw dataset used by the model to perform the prediction (not preprocessed).

        Returns
        -------
            dict of data stored
        """
        return {
            "x": x,
            "ypred_init": None,
            "ypred": None,
            "contributions": None,
            "x_preprocessed": None,
            "x_postprocessed": None,
        }

    def predict_proba(self):
        """
        The predict_proba compute the probabilities predicted for each x row defined in add_input.

        Returns
        -------
        pandas.DataFrame
            A dataset with all probabilities of each label if there is no ypred data or a dataset with ypred and the associated probability.

        Example
        --------
        >>> predictor.add_input(x=xtest_df)
        >>> predictor.predict_proba()

        """
        return predict_proba(self.model, self.data["x_preprocessed"], self._classes)

    def compute_contributions(self, contributions=None, use_groups=None):
        """
        The compute_contributions compute the contributions associated to data ypred specified.
        Need a data ypred specified in an add_input to display detail_contributions.

        Parameters
        -------
        contributions : object (optional)
            Local contributions, or list of local contributions.
        use_groups : bool (optional)
            Whether or not to compute groups of features contributions.

        Returns
        -------
        pandas.DataFrame
            Data with contributions associated to the ypred specified.
        pandas.DataFrame
            ypred data with right probabilities associated.

        """
        use_groups = True if (use_groups is not False and self.features_groups is not None) else False

        if not hasattr(self, "data"):
            raise ValueError("add_input method must be called at least once.")
        if self.data["x"] is None:
            raise ValueError(
                """
                x must be specified in an add_input method to apply detail_contributions.
                """
            )
        if self.data["ypred_init"] is None:
            self.predict()

        if contributions is None:
            explain_data = self.backend.run_explainer(x=self.data["x_preprocessed"])
            contributions = self.backend.get_local_contributions(
                explain_data=explain_data, x=self.data["x_preprocessed"]
            )
        else:
            contributions = self.backend.format_and_aggregate_local_contributions(
                x=self.data["x_preprocessed"], contributions=contributions
            )
        self.check_contributions(contributions)
        proba_values = self.predict_proba() if self._case == "classification" else None
        y_pred, match_contrib = keep_right_contributions(
            self.data["ypred_init"], contributions, self._case, self._classes, self.label_dict, proba_values
        )
        if use_groups:
            match_contrib = group_contributions(match_contrib, features_groups=self.features_groups)

        return y_pred, match_contrib

    def detail_contributions(self, contributions=None, use_groups=None):
        """
        The detail_contributions method associates the right contributions with the right data predicted.
        (with ypred specified in add_input or computed automatically)

        Parameters
        -------
        contributions : object (optional)
            Local contributions, or list of local contributions.
        use_groups : bool (optional)
            Whether or not to compute groups of features contributions.

        Returns
        -------
        pandas.DataFrame
            A Dataset with ypred and the right associated contributions.

        Example
        --------

        >>> predictor.add_input(x=xtest_df)
        >>> predictor.detail_contributions()

        """
        y_pred, detail_contrib = self.compute_contributions(contributions=contributions, use_groups=use_groups)
        return pd.concat([y_pred, detail_contrib], axis=1)

    def save(self, path):
        """
        Save method allows users to save SmartPredictor object on disk using a pickle file.
        Save method can be useful: you don't have to recompile to display results later.

        Load_smartpredictor method allow to load your SmartPredictor object saved. (See example below)

        Parameters
        ----------
        path : str
            File path to store the pickle file

        Example
        --------

        >>> predictor.save('path_to_pkl/predictor.pkl')
        >>> from shapash.utils.load_smartpredictor import load_smartpredictor
        >>> predictor_load = load_smartpredictor('path_to_pkl/predictor.pkl')
        """
        save_pickle(self, path)

    def apply_preprocessing(self):
        """
        Apply preprocessing on new dataset input specified.
        """
        return apply_preprocessing(self.data["x"], self.model, self.preprocessing)

    def filter(self):
        """
        The filter method is an important method which allows to summarize the local explainability
        by using the user defined mask_params parameters which correspond to its use case.
        """
        mask = [init_mask(self.summary["contrib_sorted"], True)]
        if self.mask_params["features_to_hide"] is not None:
            mask.append(
                hide_contributions(
                    self.summary["var_dict"],
                    features_list=self.check_features_name(self.mask_params["features_to_hide"]),
                )
            )
        if self.mask_params["threshold"] is not None:
            mask.append(cap_contributions(self.summary["contrib_sorted"], threshold=self.mask_params["threshold"]))
        if self.mask_params["positive"] is not None:
            mask.append(sign_contributions(self.summary["contrib_sorted"], positive=self.mask_params["positive"]))
        self.mask = combine_masks(mask)
        if self.mask_params["max_contrib"] is not None:
            self.mask = cutoff_contributions(mask=self.mask, k=self.mask_params["max_contrib"])
        self.masked_contributions = compute_masked_contributions(self.summary["contrib_sorted"], self.mask)

    def summarize(self, use_groups=None):
        """
        The summarize method allows to display the summary of local explainability.
        This method can be configured with modify_mask method to summarize the explainability to suit needs.

        If the user doesn't use modify_mask, the summarize method uses the mask_params parameters specified during
        the initialisation of the SmartPredictor.

        In classification case, The summarize method summarizes the explainability which corresponds to :
            - the predicted values specified by the user or automatically computed (with add_input method)
            - the right probabilities from predict_proba associated to the right predicted values
            - the right contributions ranked and filtered as specify with modify_mask method

        Parameters
        ----------
        use_groups : bool (optional)
            Whether or not to compute groups of features contributions.

        Returns
        -------
        pandas.DataFrame
            - selected explanation of each row for classification case

        Examples
        --------
        >>> summary_df = predictor.summarize()
        >>> summary_df
                pred	proba	    feature_1	value_1	    contribution_1	feature_2	value_2	    contribution_2
        0	0	    0.756416	Sex	        1.0	        0.322308	    Pclass	    3.0	        0.155069
        1	3	    0.628911	Sex	        2.0	        0.585475	    Pclass	    1.0	        0.370504
        2	0	    0.543308	Sex	        2.0	        -0.486667	    Pclass	    3.0	        0.255072

        >>> predictor.modify_mask(max_contrib=1)
        >>> summary_df = predictor.summarize()
        >>> summary_df
                pred	proba	    feature_1	value_1	    contribution_1
        0	0	    0.756416	Sex	        1.0	        0.322308
        1	3	    0.628911	Sex	        2.0	        0.585475
        2	0	    0.543308	Sex	        2.0	        -0.486667
        """
        # data is needed : add_input() method must be called at least once
        use_groups = True if (use_groups is not False and self.features_groups is not None) else False

        if not hasattr(self, "data"):
            raise ValueError("You have to specify dataset x and y_pred arguments. Please use add_input() method.")

        if use_groups is True:
            data = self.data_groups
        else:
            data = self.data

        if self._drop_option is not None:
            columns_to_keep = [
                x for x in self._drop_option["columns_dict_op"].values() if x in data["x_postprocessed"].columns
            ]
            if use_groups:
                columns_to_keep += list(self.features_groups.keys())
            x_preprocessed = data["x_postprocessed"][columns_to_keep]
        else:
            x_preprocessed = data["x_postprocessed"]

        columns_dict = {i: col for i, col in enumerate(x_preprocessed.columns)}
        features_dict = {k: v for k, v in self.features_dict.items() if k in x_preprocessed.columns}

        self.summary = assign_contributions(rank_contributions(data["contributions"], x_preprocessed))
        # Apply filter method with mask_params attributes parameters
        self.filter()

        # Summarize information
        data["summary"] = summarize(
            self.summary["contrib_sorted"],
            self.summary["var_dict"],
            self.summary["x_sorted"],
            self.mask,
            columns_dict,
            features_dict,
        )

        # Matching with y_pred
        return pd.concat([data["ypred"], data["summary"]], axis=1)

    def modify_mask(self, features_to_hide=None, threshold=None, positive=None, max_contrib=None):
        """
        This method allows the users to modify the mask_params values.
        Each parameter is optional, modify_mask method modifies only the values specified in parameters.

        This method has to be used to configure the summary displayed with summarize method.

        Parameters
        ----------
        features_to_hide : list, optional (default: None)
            List of strings, containing features to hide.
        threshold : float, optional (default: None)
            Absolute threshold below which any contribution is hidden.
        positive: bool, optional (default: None)
            If True, hide negative values. False, hide positive values
            If None, hide nothing.
        max_contrib : int, optional (default: None)
            Maximum number of contributions to show.

        Examples
        --------
        >>> predictor.modify_mask(max_contrib=1)
        >>> summary_df = predictor.summarize()
        >>> summary_df
                pred	proba	    feature_1	value_1	    contribution_1
        0	0	    0.756416	Sex	        1.0	        0.322308
        1	3	    0.628911	Sex	        2.0	        0.585475
        2	0	    0.543308	Sex	        2.0	        -0.486667

        """
        Attributes = {
            "features_to_hide": features_to_hide,
            "threshold": threshold,
            "positive": positive,
            "max_contrib": max_contrib,
        }
        for label, attribute in Attributes.items():
            if attribute is not None:
                self.mask_params[label] = attribute

    def predict(self):
        """
        The predict method compute the predicted values for each x row defined in add_input.

        Returns
        -------
        pandas.DataFrame
            A dataset with predicted values for each x row.

        Example
        --------
        >>> predictor.add_input(x=xtest_df)
        >>> predictor.predict()


        """
        if not hasattr(self, "data"):
            raise ValueError("add_input method must be called at least once.")
        if self.data["x_preprocessed"] is None:
            raise ValueError(
                """
                x must be specified in an add_input method to apply predict.
                """
            )
        if hasattr(self.model, "predict"):
            self.data["ypred_init"] = pd.DataFrame(
                self.model.predict(self.data["x_preprocessed"]),
                columns=["ypred"],
                index=self.data["x_preprocessed"].index,
            )
        else:
            raise ValueError("model has no predict method")

        return self.data["ypred_init"]

    def apply_postprocessing(self):
        """
        Modifies x Dataframe according to postprocessing modifications, if exists.

        Parameters
        ----------
        postprocessing: Dict
            Dictionnary of postprocessing modifications to apply in x.

        Returns
        -------
        pandas.Dataframe
            Returns x_init if postprocessing is empty, modified dataframe otherwise.
        """
        if self.postprocessing:
            return apply_postprocessing(self.data["x"], self.postprocessing)
        else:
            return self.data["x"]

    def check_features_name(self, features):
        """
        Convert a list of feature names (string) or features ids into features ids.
        Features names can be part of columns_dict or features_dict.

        Parameters
        ----------
        features : List
            List of ints (columns ids) or of strings (business names)

        Returns
        -------
        list of ints
            Columns ids compatible with var_dict
        """
        return check_features_name(self.columns_dict, self.features_dict, features)

    def to_smartexplainer(self):
        """
        Create a SmartExplainer object compiled with the data specified in add_input method with
        SmartPredictor attributes
        """
        if not hasattr(self, "data"):
            raise ValueError("add_input method must be called at least once.")

        if self.data["x"] is None:
            raise ValueError(
                """
                x must be specified in an add_input method to apply to_smartexplainer method.
                """
            )

        list_preprocessing = preprocessing_tolist(self.preprocessing)
        for enc in list_preprocessing:
            if str(type(enc)) in columntransformer:
                raise ValueError("SmartPredictor can't switch to SmartExplainer for ColumnTransformer preprocessing.")

        xpl = shapash.explainer.smart_explainer.SmartExplainer(
            model=self.model,
            backend=self.backend,
            preprocessing=self.preprocessing,
            postprocessing=self.postprocessing,
            features_groups=self.features_groups,
            features_dict=copy.deepcopy(self.features_dict),
            label_dict=copy.deepcopy(self.label_dict),
        )
        xpl.compile(x=copy.deepcopy(self.data["x_preprocessed"]), y_pred=copy.deepcopy(self.data["ypred_init"]))
        return xpl
