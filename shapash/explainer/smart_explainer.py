"""
Smart explainer module
"""
import copy
import logging
import shutil
import tempfile

import numpy as np
import pandas as pd

import shapash.explainer.smart_predictor
from shapash.backend import BaseBackend, get_backend_cls_from_name
from shapash.backend.shap_backend import get_shap_interaction_values
from shapash.manipulation.select_lines import keep_right_contributions
from shapash.manipulation.summarize import create_grouped_features_values
from shapash.report import check_report_requirements
from shapash.style.style_utils import colors_loading, select_palette
from shapash.utils.check import (
    check_additional_data,
    check_features_name,
    check_label_dict,
    check_model,
    check_postprocessing,
    check_y,
)
from shapash.utils.explanation_metrics import find_neighbors, get_distance, get_min_nb_features, shap_neighbors
from shapash.utils.io import load_pickle, save_pickle
from shapash.utils.model import predict, predict_error, predict_proba
from shapash.utils.threading import CustomThread
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform
from shapash.utils.utils import get_host_name
from shapash.webapp.smart_app import SmartApp

from .smart_plotter import SmartPlotter

logging.basicConfig(level=logging.INFO)


class SmartExplainer:
    """
    The SmartExplainer class is the main object of the Shapash library.
    It allows the Data Scientists to perform many operations to make the
    results more understandable :
    linking encoders, models, predictions, label dict and datasets.
    SmartExplainer users have several methods which are described below.
    Parameters
    ----------
    model : model object
        model used to consistency check. model object can also be used by some method to compute
        predict and predict_proba values
    backend : str or shapash.backend object (default: 'shap')
        Select which computation method to use in order to compute contributions
        and feature importance. Possible values are 'shap' or 'lime'. Default is 'shap'.
        It is also possible to pass a backend class inherited from shpash.backend.BaseBackend.
    preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
        --> Differents types of preprocessing are available:
        - A single category_encoders (OrdinalEncoder/OnehotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
        - A single ColumnTransformer with scikit-learn encoding or category_encoders transformers
        - A list with multiple category_encoders with optional (dict, list of dict)
        - A list with a single ColumnTransformer with optional (dict, list of dict)
        - A dict
        - A list of dict
    postprocessing : dict, optional (default: None)
        Dictionnary of postprocessing modifications to apply in x_init dataframe.
        Dictionnary with feature names as keys (or number, or well labels referencing to features names),
        which modifies dataset features by features.
        --> Different types of postprocessing are available, but the syntax is this one:
        One key by features, 5 different types of modifications:
            features_groups : dict, optional (default: None)
        Dictionnary containing features that should be grouped together. This option allows
        to compute and display the contributions and importance of this group of features.
        Features that are grouped together will still be displayed in the webapp when clicking
        on a group.
        >>> {
        ‘feature1’ : { ‘type’ : ‘prefix’, ‘rule’ : ‘age: ‘ },
        ‘feature2’ : { ‘type’ : ‘suffix’, ‘rule’ : ‘$/week ‘ },
        ‘feature3’ : { ‘type’ : ‘transcoding’, ‘rule‘: { ‘code1’ : ‘single’, ‘code2’ : ‘married’}},
        ‘feature4’ : { ‘type’ : ‘regex’ , ‘rule‘: { ‘in’ : ‘AND’, ‘out’ : ‘ & ‘ }},
        ‘feature5’ : { ‘type’ : ‘case’ , ‘rule‘: ‘lower’‘ }
        }
        Only one transformation by features is possible.
    features_groups : dict, optional (default: None)
        Dictionnary containing features that should be grouped together. This option allows
        to compute and display the contributions and importance of this group of features.
        Features that are grouped together will still be displayed in the webapp when clicking
        on a group.
        >>> {
        ‘feature_group_1’ : ['feature3', 'feature7', 'feature24'],
        ‘feature_group_2’ : ['feature1', 'feature12'],
        }
    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    label_dict: dict
        Dictionary mapping integer labels to domain names (classification - target values).
    title_story: str (default: None)
        The default title is empty. You can specify a custom title
        which can be used the webapp, or other methods
    palette_name : str
        Name of the palette used for the colors of the report (refer to style folder).
    colors_dic : dict
        dictionnary contaning every palettes of colors. You can use this parameter to change
        any color of the graphs.
    **backend_kwargs : dict
        Keyword parameters to be passed to the backend.

    Attributes
    ----------
    data: dict
        Data dictionary has 3 entries. Each key returns a pd.DataFrame (regression) or a list of pd.DataFrame
        (classification - The length of the lists is equivalent to the number of labels).
        All pd.DataFrame have she same shape (n_samples, n_features).
        For the regression case, data that should be regarded as a single array
        of size (n_samples, n_features, 3).
        data['contrib_sorted']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            Contains local contributions of the prediction set, with common line index.
            Columns are 'contrib_1', 'contrib_2', ... and contains the top contributions
            for each line from left to right. In multi-class problems, this is a list of
            contributions, one for each class.
        data['var_dict']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            Must contain only ints. It gives, for each line, the list of most import features
            regarding the local decomposition. In order to save space, columns are denoted by
            integers, the conversion being done with the columns_dict member. In multi-class
            problems, this is a list of dataframes, one for each class.
        data['x_sorted']: pandas.DataFrame (regression) or list of pandas.DataFrame (classification)
            It gives, for each line, the list of most important features values regarding the local
            decomposition. These values can only be understood with respect to data['var_dict']
    backend_name:
        backend name if backend passed is a string
    x_encoded: pandas.DataFrame
        preprocessed dataset used by the model to perform the prediction.
    x_init: pandas.DataFrame
        x_encoded dataset with inverse transformation with eventual postprocessing modifications.
    x_contrib_plot: pandas.DataFrame
        x_encoded dataset with inverse transformation, without postprocessing used for contribution_plot.
    y_pred: pandas.DataFrame
        User-specified prediction values.
    contributions: pandas.DataFrame (regression) or list (classification)
        local contributions aggregated if the preprocessing part requires it (e.g. one-hot encoding).
    features_dict: dict
        Dictionary mapping technical feature names to domain names.
    inv_features_dict: dict
        Inverse features_dict mapping.
    label_dict: dict
        Dictionary mapping integer labels to domain names (classification - target values).
    inv_label_dict: dict
        Inverse label_dict mapping.
    columns_dict: dict
        Dictionary mapping integer column number to technical feature names.
    plot: object
        Helper object containing all plotting functions (Bridge pattern).
    model: model object
        model used to check the different values of target estimate predict proba
    features_desc: dict
        Dictionary that references the numbers of feature values ​​in the x_init
    features_imp: pandas.Series (regression) or list (classification)
        Features importance values
    local_neighbors: dict
        Dictionary of values to be displayed on the local_neighbors plot.
        The key is "norm_shap (normalized contributions values of instance and neighbors)
    features_stability: dict
        Dictionary of arrays to be displayed on the stability plot.
        The keys are "amplitude" (average contributions values for selected instances) and
        "stability" (stability metric across neighborhood)
    preprocessing : category_encoders, ColumnTransformer, list or dict
        The processing apply to the original data.
    postprocessing : dict
        Dictionnary of postprocessing modifications to apply in x_init dataframe.
    y_target : pandas.Series or pandas.DataFrame, optional (default: None)
        Target values

    Example
    --------
    >>> xpl = SmartExplainer(model, features_dict=featd,label_dict=labeld)
    >>> xpl.compile(x=x_encoded, y_target=y)
    >>> xpl.plot.features_importance()
    """

    def __init__(
        self,
        model,
        backend="shap",
        preprocessing=None,
        postprocessing=None,
        features_groups=None,
        features_dict=None,
        label_dict=None,
        title_story: str = None,
        palette_name=None,
        colors_dict=None,
        **backend_kwargs,
    ):
        if features_dict is not None and not isinstance(features_dict, dict):
            raise ValueError(
                """
                features_dict must be a dict
                """
            )
        if label_dict is not None and isinstance(label_dict, dict) is False:
            raise ValueError(
                """
                label_dict must be a dict
                """
            )
        self.model = model
        self.preprocessing = preprocessing
        self.backend_name = None
        if isinstance(backend, str):
            self.backend_name = backend
        elif isinstance(backend, BaseBackend):
            self.backend = backend
            if backend.preprocessing is None and self.preprocessing is not None:
                self.backend.preprocessing = self.preprocessing
        else:
            raise NotImplementedError(f"Unknown backend : {backend}")

        self.backend_kwargs = backend_kwargs
        self.features_dict = dict() if features_dict is None else copy.deepcopy(features_dict)
        self.label_dict = label_dict
        self.plot = SmartPlotter(self)
        self.title_story = title_story if title_story is not None else ""
        self.palette_name = palette_name if palette_name else "default"
        self.colors_dict = copy.deepcopy(select_palette(colors_loading(), self.palette_name))
        if colors_dict is not None:
            self.colors_dict.update(colors_dict)
        self.plot.define_style_attributes(colors_dict=self.colors_dict)

        self._case, self._classes = check_model(self.model)
        self.postprocessing = postprocessing
        self.check_label_dict()
        if self.label_dict:
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        self.features_groups = features_groups
        self.local_neighbors = None
        self.features_stability = None
        self.features_compacity = None
        self.contributions = None
        self.explain_data = None
        self.features_imp = None

    def compile(
        self, x, contributions=None, y_pred=None, y_target=None, additional_data=None, additional_features_dict=None
    ):
        """
        The compile method is the first step to understand model and
        prediction. It performs the sorting of contributions, the reverse
        preprocessing steps and performs all the calculations necessary for
        a quick display of plots and efficient display of summary of
        explanation. This step can last a few moments with large datasets.
        Parameters
        ----------
        x : pandas.DataFrame
            Prediction set.
            IMPORTANT: this should be the raw prediction set,
            whose values are seen by the end user.
            x is a preprocessed dataset: Shapash can apply the model to it
        contributions : pandas.DataFrame, np.ndarray or list
            single or multiple contributions (multi-class) to handle.
            if pandas.Dataframe, the index and columns should be share with
            the prediction set. if np.ndarray, index and columns will be
            generated according to x dataset
        y_pred : pandas.Series or pandas.DataFrame, optional (default: None)
            Prediction values (1 column only).
            The index must be identical to the index of x_init.
            This is an interesting parameter for more explicit outputs.
            Shapash lets users define their own predict,
            as they may wish to set their own threshold (classification)
        y_target : pandas.Series or pandas.DataFrame, optional (default: None)
            Target values (1 column only).
            The index must be identical to the index of x_init.
            This is an interesting parameter for outputs on prediction
        additional_data : pandas.DataFrame, optional (default: None)
            Additional dataset of features outsite the model.
            The index must be identical to the index of x_init.
            This is an interesting parameter for visualisation and filtering
            in Shapash SmartApp.
        additional_features_dict : dict
            Dictionary mapping technical feature names to domain names for additional data.

        Example
        --------
        >>> xpl.compile(x=x_test)
        """
        if isinstance(self.backend_name, str):
            backend_cls = get_backend_cls_from_name(self.backend_name)
            self.backend = backend_cls(
                model=self.model, preprocessing=self.preprocessing, masker=x, **self.backend_kwargs
            )
        self.x_encoded = handle_categorical_missing(x)
        x_init = inverse_transform(self.x_encoded, self.preprocessing)
        self.x_init = handle_categorical_missing(x_init)
        self.y_pred = check_y(self.x_init, y_pred, y_name="y_pred")
        self.y_target = check_y(self.x_init, y_target, y_name="y_target")
        self.prediction_error = predict_error(self.y_target, self.y_pred, self._case)

        self._get_contributions_from_backend_or_user(x, contributions)
        self.check_contributions()

        self.columns_dict = {i: col for i, col in enumerate(self.x_init.columns)}
        self.check_features_dict()
        self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        self._apply_all_postprocessing_modifications()

        self.data = self.state.assign_contributions(self.state.rank_contributions(self.contributions, self.x_init))
        self.features_desc = dict(self.x_init.nunique())
        if self.features_groups is not None:
            self._compile_features_groups(self.features_groups)
        self.additional_features_dict = (
            dict()
            if additional_features_dict is None
            else self._compile_additional_features_dict(additional_features_dict)
        )
        self.additional_data = self._compile_additional_data(additional_data)

    def _get_contributions_from_backend_or_user(self, x, contributions):
        # Computing contributions using backend
        if contributions is None:
            self.explain_data = self.backend.run_explainer(x=x)
            self.contributions = self.backend.get_local_contributions(x=x, explain_data=self.explain_data)
        else:
            self.explain_data = contributions
            self.contributions = self.backend.format_and_aggregate_local_contributions(
                x=x,
                contributions=contributions,
            )
        self.state = self.backend.state

    def _apply_all_postprocessing_modifications(self):
        postprocessing = self.modify_postprocessing(self.postprocessing)
        check_postprocessing(self.x_init, postprocessing)
        self.postprocessing_modifications = self.check_postprocessing_modif_strings(postprocessing)
        self.postprocessing = postprocessing
        if self.postprocessing_modifications:
            self.x_contrib_plot = copy.deepcopy(self.x_init)
        self.x_init = self.apply_postprocessing(postprocessing)

    def _compile_features_groups(self, features_groups):
        """
        Performs required computations for groups of features.
        """
        if self.backend.support_groups is False:
            raise AssertionError(f"Selected backend ({self.backend.name}) " f"does not support groups of features.")
        # Compute contributions for groups of features
        self.contributions_groups = self.state.compute_grouped_contributions(self.contributions, features_groups)
        self.features_imp_groups = None
        # Update features dict with groups names
        self._update_features_dict_with_groups(features_groups=features_groups)
        # Compute t-sne projections for groups of features
        self.x_init_groups = create_grouped_features_values(
            x_init=self.x_init,
            x_encoded=self.x_encoded,
            preprocessing=self.preprocessing,
            features_groups=self.features_groups,
            features_dict=self.features_dict,
            how="dict_of_values",
        )
        # Compute data attribute for groups of features
        self.data_groups = self.state.assign_contributions(
            self.state.rank_contributions(self.contributions_groups, self.x_init_groups)
        )
        self.columns_dict_groups = {i: col for i, col in enumerate(self.x_init_groups.columns)}

    def _compile_additional_features_dict(self, additional_features_dict):
        """
        Performs required computations for additional features dict.
        """
        if not isinstance(additional_features_dict, dict):
            raise ValueError(
                """
                additional_features_dict must be a dict
                """
            )
        additional_features_dict = {f"_{key}": f"_{value}" for key, value in additional_features_dict.items()}
        return additional_features_dict

    def _compile_additional_data(self, additional_data):
        """
        Performs required computations for additional data.
        """
        if additional_data is not None:
            check_additional_data(self.x_init, additional_data)
            for feature in additional_data.columns:
                if feature in self.features_dict.keys() and feature not in self.columns_dict.values():
                    self.additional_features_dict[f"_{feature}"] = f"_{self.features_dict[feature]}"
                    del self.features_dict[feature]
            additional_data = additional_data.add_prefix("_")
            for feature in set(list(additional_data.columns)) - set(self.additional_features_dict):
                self.additional_features_dict[feature] = feature
        return additional_data

    def define_style(self, palette_name=None, colors_dict=None):
        """
        Set the color set to use in plots.
        """
        if palette_name is None and colors_dict is None:
            raise ValueError("At least one of palette_name or colors_dict parameters must be defined")
        new_palette_name = palette_name or self.palette_name
        new_colors_dict = copy.deepcopy(select_palette(colors_loading(), new_palette_name))
        if colors_dict is not None:
            new_colors_dict.update(colors_dict)
        self.colors_dict.update(new_colors_dict)
        self.plot.define_style_attributes(colors_dict=self.colors_dict)

    def add(
        self,
        y_pred=None,
        y_target=None,
        label_dict=None,
        features_dict=None,
        title_story: str = None,
        additional_data=None,
        additional_features_dict=None,
    ):
        """
        add method allows the user to add a label_dict, features_dict
        or y_pred without compiling again (and it can last a few moments).
        y_pred can be used in the plot to color scatter.
        y_pred is needed in the to_pandas method.
        label_dict and features_dict displays allow to display clearer results.
        Parameters
        ----------
        y_pred : pandas.Series, optional (default: None)
            Prediction values (1 column only).
            The index must be identical to the index of x_init.
        label_dict: dict, optional (default: None)
            Dictionary mapping integer labels to domain names.
        features_dict: dict, optional (default: None)
            Dictionary mapping technical feature names to domain names.
        title_story: str (default: None)
            The default title is empty. You can specify a custom title
            which can be used the webapp, or other methods
        y_target : pandas.Series or pandas.DataFrame, optional (default: None)
            Target values (1 column only).
            The index must be identical to the index of x_init.
            This is an interesting parameter for outputs on prediction
        additional_data : pandas.DataFrame, optional (default: None)
            Additional dataset of features outsite the model.
            The index must be identical to the index of x_init.
            This is an interesting parameter for visualisation and filtering
            in Shapash SmartApp.
        additional_features_dict : dict
            Dictionary mapping technical feature names to domain names for additional data.
        """
        if y_pred is not None:
            self.y_pred = check_y(self.x_init, y_pred, y_name="y_pred")
            if hasattr(self, "y_target"):
                self.prediction_error = predict_error(self.y_target, self.y_pred, self._case)
        if y_target is not None:
            self.y_target = check_y(self.x_init, y_target, y_name="y_target")
            if hasattr(self, "y_pred"):
                self.prediction_error = predict_error(self.y_target, self.y_pred, self._case)
        if label_dict is not None:
            if isinstance(label_dict, dict) is False:
                raise ValueError(
                    """
                    label_dict must be a dict
                    """
                )
            self.label_dict = label_dict
            self.check_label_dict()
            self.inv_label_dict = {v: k for k, v in self.label_dict.items()}
        if features_dict is not None:
            if isinstance(features_dict, dict) is False:
                raise ValueError(
                    """
                    features_dict must be a dict
                    """
                )
            self.features_dict = features_dict
            self.check_features_dict()
            self.inv_features_dict = {v: k for k, v in self.features_dict.items()}
        if title_story is not None:
            self.title_story = title_story
        if additional_features_dict is not None:
            self.additional_features_dict = self._compile_additional_features_dict(additional_features_dict)
        if additional_data is not None:
            self.additional_data = self._compile_additional_data(additional_data)

    def get_interaction_values(self, n_samples_max=None, selection=None):
        """
        Compute shap interaction values for each row of x_encoded.
        This function is only available for explainer of type TreeExplainer (used for tree based models).
        Please refer to the official tree shap paper for more information : https://arxiv.org/pdf/1802.03888.pdf
        Parameters
        ----------
        n_samples_max : int, optional
            Limit the number of points for which we compute the interactions.
        selection : list, optional
            Contains list of index, subset of the input DataFrame that we want to plot
        Returns
        -------
        np.ndarray
            Shap interaction values for each sample as an array of shape (# samples x # features x # features).
        """
        x = copy.deepcopy(self.x_encoded)

        if selection:
            x = x.loc[selection]

        if hasattr(self, "x_interaction"):
            if self.x_interaction.equals(x[:n_samples_max]):
                return self.interaction_values

        self.x_interaction = x[:n_samples_max]
        self.interaction_values = get_shap_interaction_values(self.x_interaction, self.backend.explainer)
        return self.interaction_values

    def check_postprocessing_modif_strings(self, postprocessing=None):
        """
        Check if any modification of postprocessing will convert numeric values into strings values.
        If so, return True, otherwise False.
        Parameters
        ----------
        postprocessing: dict
            Dict of postprocessing modifications to apply.
        Returns
        -------
        modif: bool
            Boolean which is True if any numerical variable will be converted into string.
        """
        modif = False
        if postprocessing is not None:
            for key in postprocessing.keys():
                dict_postprocess = postprocessing[key]
                if dict_postprocess["type"] in {"prefix", "suffix"} and pd.api.types.is_numeric_dtype(self.x_init[key]):
                    modif = True
        return modif

    def modify_postprocessing(self, postprocessing=None):
        """
        Modifies postprocessing parameter, to change only keys, with features name,
        in case of parameters are not real feature names (with columns_dict,
        or inv_features_dict).
        Parameters
        ----------
        postprocessing : Dict
            Dictionnary of postprocessing to modify.
        Returns
        -------
        Dict
            Modified dictionnary, with same values but keys directly referencing to feature names.
        """
        if postprocessing:
            new_dic = dict()
            for key in postprocessing.keys():
                if key in self.features_dict:
                    new_dic[key] = postprocessing[key]

                elif key in self.columns_dict.keys():
                    new_dic[self.columns_dict[key]] = postprocessing[key]

                elif key in self.inv_features_dict:
                    new_dic[self.inv_features_dict[key]] = postprocessing[key]

                else:
                    raise ValueError(f"Feature name '{key}' not found in the dataset.")

            return new_dic

    def apply_postprocessing(self, postprocessing=None):
        """
        Modifies x_init Dataframe according to postprocessing modifications, if exists.
        Parameters
        ----------
        postprocessing: Dict
            Dictionnary of postprocessing modifications to apply in x_init.
        Returns
        -------
        pandas.Dataframe
            Returns x_init if postprocessing is empty, modified dataframe otherwise.
        """
        if postprocessing:
            return apply_postprocessing(self.x_init, postprocessing)
        else:
            return self.x_init

    def check_label_dict(self):
        """
        Check if label_dict and model _classes match
        """
        if self._case != "regression":
            return check_label_dict(self.label_dict, self._case, self._classes)

    def check_features_dict(self):
        """
        Check the features_dict and add the necessary keys if all the
        input X columns are not present
        """
        for feature in set(list(self.columns_dict.values())) - set(list(self.features_dict)):
            self.features_dict[feature] = feature

    def _update_features_dict_with_groups(self, features_groups):
        """
        Add groups into features dict and inv_features_dict if not present.
        """
        for group_name in features_groups.keys():
            self.features_desc[group_name] = 1000
            if group_name not in self.features_dict.keys():
                self.features_dict[group_name] = group_name
                self.inv_features_dict[group_name] = group_name

    def check_contributions(self):
        """
        Check if contributions and prediction set match in terms of shape and index.
        """
        if not self.state.check_contributions(self.contributions, self.x_init):
            raise ValueError(
                """
                Prediction set and contributions should have exactly the same number of lines
                and number of columns. the order of the columns must be the same
                Please check x, contributions and preprocessing arguments.
                """
            )

    def check_label_name(self, label, origin=None):
        """
        Convert a string label in integer. If the label is already
        an integer nothing is done. In all other cases an error is raised.
        Parameters
        ----------
        label: int or string
            Integer (id) or string (business names)
        origin: None, 'num', 'code', 'value' (default: None)
            Kind of the label used in parameter
        Returns
        -------
        tuple
            label num, label code (class of the mode), label value
        """
        if origin is None:
            if label in self._classes:
                origin = "code"
            elif self.label_dict is not None and label in self.label_dict.values():
                origin = "value"
            elif isinstance(label, int) and label in range(-1, len(self._classes)):
                origin = "num"

        try:
            if origin == "num":
                label_num = label
                label_code = self._classes[label]
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == "code":
                label_code = label
                label_num = self._classes.index(label)
                label_value = self.label_dict[label_code] if self.label_dict else label_code
            elif origin == "value":
                label_code = self.inv_label_dict[label]
                label_num = self._classes.index(label_code)
                label_value = label
            else:
                raise ValueError

        except ValueError:
            raise Exception({"message": "Origin must be 'num', 'code' or 'value'."})

        except Exception:
            raise Exception({"message": f"Label ({label}) not found for origin ({origin})"})

        return label_num, label_code, label_value

    def check_features_name(self, features, use_groups=False):
        """
        Convert a list of feature names (string) or features ids into features ids.
        Features names can be part of columns_dict or features_dict.
        Parameters
        ----------
        features : List
            List of ints (columns ids) or of strings (business names)
        use_groups : bool
            Whether or not features parameter includes groups of features
        Returns
        -------
        list of ints
            Columns ids compatible with var_dict
        """
        columns_dict = self.columns_dict if use_groups is False else self.columns_dict_groups
        return check_features_name(columns_dict, self.features_dict, features)

    def check_attributes(self, attribute):
        """
        Check that explainer has the attribute precised
        Parameters
        ----------
        attribute: string
            the label of the attribute to test
        Returns
        -------
        Object content of the attribute specified from SmartExplainer instance
        """
        if not hasattr(self, attribute):
            raise ValueError(
                """
                attribute {} isn't an attribute of the explainer precised.
                """.format(
                    attribute
                )
            )

        return self.__dict__[attribute]

    def filter(self, features_to_hide=None, threshold=None, positive=None, max_contrib=None, display_groups=None):
        """
        The filter method is an important method which allows to summarize the local explainability
        by using the user defined parameters which correspond to its use case.
        Filter method is used with the local_plot method of Smarplotter to see the concrete result of this summary
        with a local contribution barchart
        Please, watch the local_plot tutorial to see how these two methods are combined with a concrete example
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
        display_groups : bool (default: None)
            Whether or not to display groups of features. This option is
            only useful if groups of features are declared when compiling
            SmartExplainer object.
        """
        display_groups = True if (display_groups is not False and self.features_groups is not None) else False
        if display_groups:
            data = self.data_groups
        else:
            data = self.data
        mask = [self.state.init_mask(data["contrib_sorted"], True)]
        if features_to_hide:
            mask.append(
                self.state.hide_contributions(
                    data["var_dict"],
                    features_list=self.check_features_name(features_to_hide, use_groups=display_groups),
                )
            )
        if threshold:
            mask.append(self.state.cap_contributions(data["contrib_sorted"], threshold=threshold))
        if positive is not None:
            mask.append(self.state.sign_contributions(data["contrib_sorted"], positive=positive))
        self.mask = self.state.combine_masks(mask)
        if max_contrib:
            self.mask = self.state.cutoff_contributions(self.mask, max_contrib=max_contrib)
        self.masked_contributions = self.state.compute_masked_contributions(data["contrib_sorted"], self.mask)
        self.mask_params = {
            "features_to_hide": features_to_hide,
            "threshold": threshold,
            "positive": positive,
            "max_contrib": max_contrib,
        }

    def save(self, path):
        """
        Save method allows user to save SmartExplainer object on disk
        using a pickle file.
        Save method can be useful: you don't have to recompile to display
        results later
        Parameters
        ----------
        path : str
            File path to store the pickle file
        Example
        --------
        >>> xpl.save('path_to_pkl/xpl.pkl')
        """
        if hasattr(self, "smartapp"):
            self.smartapp = None
        save_pickle(self, path)

    @classmethod
    def load(cls, path):
        """
        Load method allows Shapash user to use pickled SmartExplainer.
        To use this method you must first declare your SmartExplainer object
        Watch the following example
        Parameters
        ----------
        path : str
            File path of the pickle file.
        Example
        --------
        >>> xpl = SmartExplainer.load('path_to_pkl/xpl.pkl')
        """
        xpl = load_pickle(path)
        if isinstance(xpl, SmartExplainer):
            smart_explainer = cls(model=xpl.model)
            smart_explainer.__dict__.update(xpl.__dict__)
            return smart_explainer
        else:
            raise ValueError("File is not a SmartExplainer object")

    def predict_proba(self):
        """
        The predict_proba compute the proba values for each x_encoded row
        """
        self.proba_values = predict_proba(self.model, self.x_encoded, self._classes)

    def predict(self):
        """
        The predict method computes the model output for each x_encoded row and stores it in y_pred attribute
        """
        self.y_pred = predict(self.model, self.x_encoded)
        if hasattr(self, "y_target"):
            self.prediction_error = predict_error(self.y_target, self.y_pred, self._case)

    def to_pandas(
        self, features_to_hide=None, threshold=None, positive=None, max_contrib=None, proba=False, use_groups=None
    ):
        """
        The to_pandas method allows to export the summary of local explainability.
        This method proposes a set of parameters to summarize the explainability of each point.
        If the user does not specify any, the to_pandas method uses the parameter specified during
        the last execution of the filter method.
        In classification case, The method to_pandas summarizes the explicability which corresponds
        to the predicted values specified by the user (with compile or add method).
        the proba parameter displays the corresponding predict proba value for each point
        In classification case, There are 2 ways to use this to pandas method.
        - Provide a real prediction set to explain
        - Focus on a constant target value and look at the proba and explainability corresponding to each point.
        (in that case, specify a constant pd.Series with add or compile method)
        Examples are presented in the tutorial local_plot (please check tutorial part of this doc)
        Parameters
        ----------
        features_to_hide : list, optional (default: None)
            List of strings, containing features to hide.
        threshold : float, optional (default: None)
            Absolute threshold below which any contribution is hidden.
        positive: bool, optional (default: None)
            If True, hide negative values. Hide positive values otherwise. If None, hide nothing.
        max_contrib : int, optional (default: 5)
            Number of contributions to show in the pandas df
        proba : bool, optional (default: False)
            adding proba in output df
        use_groups : bool (optional)
            Whether or not to use groups of features contributions (only available if features_groups
            parameter was not empty when calling compile method).
        Returns
        -------
        pandas.DataFrame
            - selected explanation of each row for classification case
        Examples
        --------
        >>> summary_df = xpl.to_pandas(max_contrib=2,proba=True)
        >>> summary_df
            pred	proba	    feature_1	value_1	    contribution_1	feature_2	value_2	    contribution_2
        0	0	    0.756416	Sex	        1.0	        0.322308	    Pclass	    3.0	        0.155069
        1	3	    0.628911	Sex	        2.0	        0.585475	    Pclass	    1.0	        0.370504
        2	0	    0.543308	Sex	        2.0	        -0.486667	    Pclass	    3.0	        0.255072
        """
        use_groups = True if (use_groups is not False and self.features_groups is not None) else False
        if use_groups:
            data = self.data_groups
        else:
            data = self.data

        # Classification: y_pred is needed
        if self.y_pred is None:
            raise ValueError("You have to specify y_pred argument. Please use add() or compile() method")

        # Apply filter method if necessary
        if (
            all(var is None for var in [features_to_hide, threshold, positive, max_contrib])
            and hasattr(self, "mask_params")
            and (
                # if the already computed mask does not have the right shape (this can happen when
                # we use groups of features once and then use method without groups)
                (
                    isinstance(data["contrib_sorted"], pd.DataFrame)
                    and len(data["contrib_sorted"].columns) == len(self.mask.columns)
                )
                or (
                    isinstance(data["contrib_sorted"], list)
                    and len(data["contrib_sorted"][0].columns) == len(self.mask[0].columns)
                )
            )
        ):
            print("to_pandas params: " + str(self.mask_params))
        else:
            self.filter(
                features_to_hide=features_to_hide,
                threshold=threshold,
                positive=positive,
                max_contrib=max_contrib,
                display_groups=use_groups,
            )
        if use_groups:
            columns_dict = {i: col for i, col in enumerate(self.x_init_groups.columns)}
        else:
            columns_dict = self.columns_dict
        # Summarize information
        data["summary"] = self.state.summarize(
            data["contrib_sorted"], data["var_dict"], data["x_sorted"], self.mask, columns_dict, self.features_dict
        )
        # Matching with y_pred
        if proba:
            self.predict_proba() if proba else None
            proba_values = self.proba_values
        else:
            proba_values = None

        y_pred, summary = keep_right_contributions(
            self.y_pred, data["summary"], self._case, self._classes, self.label_dict, proba_values
        )

        return pd.concat([y_pred, summary], axis=1)

    def compute_features_import(self, force=False):
        """
        Compute a relative features importance, sum of absolute values
        of the contributions for each.
        Features importance compute in base 100
        Parameters
        ----------
        force: bool (default: False)
            True to force de compute if features importance is
            already calculated
        Returns
        -------
        pd.Serie (Regression)
        or list of pd.Serie (Classification: One Serie for each target modality)
            Each Serie: feature importance, One row by feature,
            index of the serie = contributions.columns
        """
        self.features_imp = self.backend.get_global_features_importance(
            contributions=self.contributions, explain_data=self.explain_data, subset=None
        )

        if self.features_groups is not None and self.features_imp_groups is None:
            self.features_imp_groups = self.state.compute_features_import(self.contributions_groups)

    def compute_features_stability(self, selection):
        """
        For a selection of instances, compute features stability metrics used in
        methods `local_neighbors_plot` and `local_stability_plot`.
        - If selection is a single instance, the method returns the (normalized) contribution values
        of instance and corresponding neighbors.
        - If selection represents multiple instances, the method returns the average (normalized) contribution values
        of instances and neighbors (=amplitude), as well as the variability of those values in the neighborhood (=variability)
        Parameters
        ----------
        selection: list
            Indices of rows to be displayed on the stability plot
        Returns
        -------
        Dictionary
            Values that will be displayed on the graph. Keys are "amplitude", "variability" and "norm_shap"
        """
        if (self._case == "classification") and (len(self._classes) > 2):
            raise AssertionError("Multi-class classification is not supported")

        all_neighbors = find_neighbors(selection, self.x_encoded, self.model, self._case)

        # Check if entry is a single instance or not
        if len(selection) == 1:
            # Compute explanations for instance and neighbors
            norm_shap, _, _ = shap_neighbors(all_neighbors[0], self.x_encoded, self.contributions, self._case)
            self.local_neighbors = {"norm_shap": norm_shap}
        else:
            numb_expl = len(selection)
            amplitude = np.zeros((numb_expl, self.x_init.shape[1]))
            variability = np.zeros((numb_expl, self.x_init.shape[1]))
            # For each instance (+ neighbors), compute explanation
            for i in range(numb_expl):
                (
                    _,
                    variability[i, :],
                    amplitude[i, :],
                ) = shap_neighbors(all_neighbors[i], self.x_encoded, self.contributions, self._case)
            self.features_stability = {"variability": variability, "amplitude": amplitude}

    def compute_features_compacity(self, selection, distance, nb_features):
        """
        For a selection of instances, compute features compacity metrics used in method `compacity_plot`.
        The method returns :
        * the minimum number of features needed for a given approximation level
        * conversely, the approximation reached with a given number of features
        Parameters
        ----------
        selection: list
            Indices of rows to be displayed on the stability plot
        distance : float
            How close we want to be from model with all features
        nb_features : int
            Number of features used
        """
        if (self._case == "classification") and (len(self._classes) > 2):
            raise AssertionError("Multi-class classification is not supported")

        features_needed = get_min_nb_features(selection, self.contributions, self._case, distance)
        distance_reached = get_distance(selection, self.contributions, self._case, nb_features)
        # We clip large approximations to 100%
        distance_reached = np.clip(distance_reached, 0, 1)

        self.features_compacity = {"features_needed": features_needed, "distance_reached": distance_reached}

    def init_app(self, settings: dict = None):
        """
        Simple init of SmartApp in case of host smartapp by another way

        Parameters
        ----------
        settings : dict (default: None)
            A dict describing the default webapp settings values to be used
            Possible settings (dict keys) are 'rows', 'points', 'violin', 'features'
            Values should be positive ints
        """
        if self.y_pred is None:
            self.predict()
        self.smartapp = SmartApp(self, settings)

    def run_app(
        self, port: int = None, host: str = None, title_story: str = None, settings: dict = None
    ) -> CustomThread:
        """
        run_app method launches the interpretability web app associated with the shapash object.
        run_app method can be used directly in a Jupyter notebook
        The link to the webapp is directly mentioned in the Jupyter output
        Use object.kill() method to kill the current instance
        Examples are presented in the web_app tutorial (please check tutorial part of this doc)
        Parameters
        ----------
        port: int (default: None)
            The port is by default on 8050. You can specify a custom port
            for your webapp.
        host: str (default: None)
            The default host is '0.0.0.0'. You can specify a custom
            ip address for your webapp
        title_story: str (default: None)
            The default title is empty. You can specify a custom title
            for your webapp (can be reused in other methods like in a report, ...)
        settings : dict (default: None)
            A dict describing the default webapp settings values to be used
            Possible settings (dict keys) are 'rows', 'points', 'violin', 'features'
            Values should be positive ints
        Returns
        -------
        CustomThread
            Return the thread instance of your server.
        Example
        --------
        >>> app = xpl.run_app()
        >>> app.kill()
        """

        if title_story is not None:
            self.title_story = title_story
        if self.y_pred is None:
            self.predict()
        if hasattr(self, "_case"):
            self.smartapp = SmartApp(self, settings)
            if host is None:
                host = "0.0.0.0"
            if port is None:
                port = 8050
            host_name = get_host_name()
            server_instance = CustomThread(
                target=lambda: self.smartapp.app.run_server(debug=False, host=host, port=port)
            )
            if host_name is None:
                host_name = host
            elif host != "0.0.0.0":
                host_name = host
            server_instance.start()
            logging.info(f"Your Shapash application run on http://{host_name}:{port}/")
            logging.info("Use the method .kill() to down your app.")
            return server_instance

        else:
            raise ValueError("Explainer must be compiled before running app.")

    def to_smartpredictor(self):
        """
        Create a SmartPredictor object designed from the following attributes
        needed from the SmartExplainer Object :
        features_dict: dict
            Dictionary mapping technical feature names to domain names.
        label_dict: dict
            Dictionary mapping integer labels to domain names (classification - target values).
        columns_dict: dict
            Dictionary mapping integer column number to technical feature names.
        features_types: dict
            Dictionnary mapping features with the right types needed.
        model: model object
            model used to check the different values of target estimate predict proba
        backend : backend object
            backend used to compute contributions
        preprocessing: category_encoders, ColumnTransformer, list or dict
            The processing apply to the original data.
        postprocessing: dict
            Dictionnary of postprocessing modifications to apply in x_init dataframe.
        _case: string
            String that informs if the model used is for classification or regression problem.
        _classes: list, None
            List of labels if the model used is for classification problem, None otherwise.
        mask_params: dict (optional)
            Dictionnary allowing the user to define a apply a filter to summarize the local explainability.
        """
        if self.backend is None:
            raise ValueError(
                """
                SmartPredictor needs a backend (explainer).
                Please compile without contributions or specify  the
                explainer used. Make change in compile() step.
                """
            )

        self.features_types = {features: str(self.x_init[features].dtypes) for features in self.x_init.columns}

        listattributes = [
            "features_dict",
            "model",
            "columns_dict",
            "backend",
            "features_types",
            "label_dict",
            "preprocessing",
            "postprocessing",
            "features_groups",
        ]

        params_smartpredictor = [self.check_attributes(attribute) for attribute in listattributes]

        if not hasattr(self, "mask_params"):
            self.mask_params = {"features_to_hide": None, "threshold": None, "positive": None, "max_contrib": None}
        params_smartpredictor.append(self.mask_params)

        return shapash.explainer.smart_predictor.SmartPredictor(*params_smartpredictor)

    def check_x_y_attributes(self, x_str, y_str):
        """
        Check if x_str and y_str are attributes of the SmartExplainer
        Parameters
        ----------
        x_str: string
            label of the attribute x
        y_str: string
            label of the attribute y
        Returns
        -------
        list of object detained by attributes x and y.
        """
        if not (isinstance(x_str, str) and isinstance(y_str, str)):
            raise ValueError(
                """
                x and y must be strings.
                """
            )
        params_checkypred = []
        attributs_explainer = [x_str, y_str]

        for attribut in attributs_explainer:
            if hasattr(self, attribut):
                params_checkypred.append(self.__dict__[attribut])
            else:
                params_checkypred.append(None)
        return params_checkypred

    def generate_report(
        self,
        output_file,
        project_info_file,
        x_train=None,
        y_train=None,
        y_test=None,
        title_story=None,
        title_description=None,
        metrics=None,
        working_dir=None,
        notebook_path=None,
        kernel_name=None,
    ):
        """
        This method will generate an HTML report containing different information about the project.
        It analyzes the data and the model used in order to provide interesting
        insights that can be shared using the HTML format.
        It requires a project info yml file on which can figure different information about the project.
        Parameters
        ----------
        output_file : str
            Path to the HTML file to write.
        project_info_file : str
            Path to the file used to display some information about the project in the report.
        x_train : pd.DataFrame, optional
            DataFrame used for training the model.
        y_train: pd.Series or pd.DataFrame, optional
            Series of labels in the training set.
        y_test : pd.Series or pd.DataFrame, optional
            Series of labels in the test set.
        title_story : str, optional
            Report title.
        title_description : str, optional
            Report title description (as written just below the title).
        metrics : list, optional
            Metrics used in the model performance section. The metrics parameter should be a list
            of dict. Each dict contains they following keys :
            'path' (path to the metric function, ex: 'sklearn.metrics.mean_absolute_error'),
            'name' (optional, name of the metric as displayed in the report),
            and 'use_proba_values' (optional, possible values are False (default) or True
            if the metric uses proba values instead of predicted values).
            For example, metrics=[{'name': 'F1 score', 'path': 'sklearn.metrics.f1_score'}]
        working_dir : str, optional
            Working directory in which will be generated the notebook used to create the report
            and where the objects used to execute it will be saved. This parameter can be usefull
            if one wants to create its own custom report and debug the notebook used to generate
            the html report. If None, a temporary directory will be used.
        notebook_path : str, optional
            Path to the notebook used to generate the report. If None, the Shapash base report
            notebook will be used.
        kernel_name : str, optional
            Name of the kernel used to generate the report. This parameter can be usefull if
            you have multiple jupyter kernels and that the method does not use the right kernel
            by default.
        Examples
        --------
        >>> xpl.generate_report(
                output_file='report.html',
                project_info_file='utils/project_info.yml',
                x_train=x_train,
                y_train=y_train,
                y_test=ytest,
                title_story="House prices project report",
                title_description="This document is a data science report of the kaggle house prices project."
                metrics=[
                    {
                        'path': 'sklearn.metrics.mean_squared_error',
                        'name': 'Mean squared error',  # Optional : name that will be displayed next to the metric
                    },
                    {
                        'path': 'sklearn.metrics.mean_absolute_error',
                        'name': 'Mean absolute error',
                    }
                ]
            )
        """
        check_report_requirements()
        if x_train is not None:
            x_train = handle_categorical_missing(x_train)
        # Avoid Import Errors with requirements specific to the Shapash Report
        from shapash.report.generation import execute_report, export_and_save_report

        rm_working_dir = False
        if not working_dir:
            working_dir = tempfile.mkdtemp()
            rm_working_dir = True

        if not hasattr(self, "model"):
            raise AssertionError(
                "Explainer object was not compiled. Please compile the explainer "
                "object using .compile(...) method before generating the report."
            )

        try:
            execute_report(
                working_dir=working_dir,
                explainer=self,
                project_info_file=project_info_file,
                x_train=x_train,
                y_train=y_train,
                y_test=y_test,
                config=dict(
                    title_story=title_story,
                    title_description=title_description,
                    metrics=metrics,
                ),
                notebook_path=notebook_path,
                kernel_name=kernel_name,
            )
            export_and_save_report(working_dir=working_dir, output_file=output_file)

            if rm_working_dir:
                shutil.rmtree(working_dir)

        except Exception as e:
            if rm_working_dir:
                shutil.rmtree(working_dir)
            raise e
