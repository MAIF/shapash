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
    check_columns_order,
    check_features_name,
    check_label_dict,
    check_model,
    check_postprocessing,
    check_y,
)
from shapash.utils.custom_thread import CustomThread
from shapash.utils.explanation_metrics import find_neighbors, get_distance, get_min_nb_features, shap_neighbors
from shapash.utils.io import load_pickle, save_pickle
from shapash.utils.model import predict, predict_error, predict_proba
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform
from shapash.utils.utils import get_host_name
from shapash.webapp.smart_app import SmartApp

from .smart_plotter import SmartPlotter

logging.basicConfig(level=logging.INFO)


class SmartExplainer:
    """
    The main class of the Shapash library, designed to make machine learning model
    results more interpretable and understandable.

    `SmartExplainer` links together the model, encoders, datasets, predictions,
    and label dictionaries. It provides a variety of methods to visualize and
    analyze model explanations both in notebooks and in the Shapash WebApp.

    Parameters
    ----------
    model : object
        The model to be explained. Used for consistency checks and, in some cases,
        to compute `predict` and `predict_proba` values.
    backend : str or shapash.backend.BaseBackend, default='shap'
        Defines the backend used to compute feature contributions and importances.
        Options:
        - `'shap'`: use SHAP as backend.
        - `'lime'`: use LIME as backend.
        You can also pass a custom backend class that inherits from
        `shapash.backend.BaseBackend`.
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
    features_groups : dict, optional
        Groups of features to be aggregated together in plots and importance
        computations. Each key defines a group name, and its value is a list of
        feature names.

        Example:
        >>> {
        ...   'feature_group_1': ['feature3', 'feature7', 'feature24'],
        ...   'feature_group_2': ['feature1', 'feature12']
        ... }
    features_dict : dict, optional
        Mapping from technical feature names to domain-specific (readable) names.
    label_dict : dict, optional
        Mapping from numeric labels to human-readable class names (for classification tasks).
    title_story : str, optional
        Custom title used in visualizations and reports. Default is empty.
    palette_name : str, optional
        Name of the color palette used for visualizations (see the `style` folder for options).
    colors_dict : dict, optional
        Dictionary containing the full color palette configuration.
        Can be used to override default plot colors.
    **backend_kwargs : dict
        Additional keyword arguments passed to the backend.

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
    backend_name : str
        Name of the backend if specified as a string.
    x_encoded : pandas.DataFrame
        Preprocessed dataset used by the model.
    x_init : pandas.DataFrame
        Inverse-transformed dataset (after preprocessing) with optional postprocessing.
    x_contrib_plot : pandas.DataFrame
        Inverse-transformed dataset without postprocessing (used for plots).
    y_pred : pandas.DataFrame
        Model predictions.
    contributions : pandas.DataFrame or list
        Local feature contributions. Aggregated if preprocessing expands features
        (e.g., one-hot encoding).
    features_dict : dict
        Mapping from technical feature names to domain names.
    inv_features_dict : dict
        Reverse mapping of `features_dict`.
    label_dict : dict
        Mapping from numeric labels to class names.
    inv_label_dict : dict
        Reverse mapping of `label_dict`.
    columns_dict : dict
        Mapping from feature index to technical feature name.
    plot : SmartPlotter
        Object providing access to all plotting functions.
    model : object
        The model being explained.
    features_desc : dict
        Number of unique values per feature in `x_init`.
    features_imp : pandas.Series or list
        Computed feature importance values.
    local_neighbors : dict
        Data displayed in local neighbor plots (normalized SHAP values, etc.).
    features_stability : dict
        Data used for stability plots, including:
        - `'amplitude'`: average contribution values for selected instances.
        - `'stability'`: metric assessing stability across neighborhoods.
    preprocessing : category_encoders object, ColumnTransformer, list, or dict
        Preprocessing transformations applied to raw input data.
    postprocessing : dict
        Postprocessing rules applied after inverse preprocessing.
    y_target : pandas.Series or pandas.DataFrame, optional
        True target values.

    Example
    -------
    >>> xpl = SmartExplainer(model, features_dict=featd, label_dict=labeld)
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
        self.title_story = title_story if title_story is not None else ""
        self.palette_name = palette_name if palette_name else "default"
        self.colors_dict = copy.deepcopy(select_palette(colors_loading(), self.palette_name))
        if colors_dict is not None:
            self.colors_dict.update(colors_dict)
        self.plot = SmartPlotter(self, self.colors_dict)

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
        self,
        x,
        contributions=None,
        y_pred=None,
        proba_values=None,
        y_target=None,
        columns_order=None,
        additional_data=None,
        additional_features_dict=None,
    ):
        """
        Prepare and structure all data needed for interpreting the model and its predictions.

        The `compile` method is the first essential step to make your model explainable
        with Shapash. It organizes the model’s inputs, outputs, and contributions into
        a consistent format, applies inverse preprocessing, and computes all elements
        required for visualization and summaries.

        Depending on dataset size and backend, this step may take some time.

        Parameters
        ----------
        x : pandas.DataFrame
            Prediction dataset — the same data seen by the end user.
            It should correspond to the **raw prediction input** (post-preprocessing).
            Shapash will use this dataset to compute and align explanations.
        contributions : pandas.DataFrame, numpy.ndarray, or list, optional
            Local feature contributions for each sample.
            - If a `DataFrame`, its index and columns must match those of `x`.
            - If a `numpy.ndarray`, Shapash will automatically generate the corresponding
            index and column names based on `x`.
            - In multi-class settings, provide a list of contributions (one per class).
        y_pred : pandas.Series or pandas.DataFrame, optional
            Model predictions.
            Must have the same index as `x_init`.
            Useful for customizing predicted values, for example when applying
            a custom threshold in classification tasks.
        proba_values : pandas.Series or pandas.DataFrame, optional
            Prediction probabilities.
            Must have the same index as `x_init`.
            Useful for visualizations and for comparing probabilities across classes.
        y_target : pandas.Series or pandas.DataFrame, optional
            True target values used for comparison or performance display.
            Must have the same index as `x_init`.
        columns_order : list or str, optional
            Defines the display order of columns in the dataset.
            - If a **list** is provided, it specifies the exact order of columns.
            Any columns not included in the list will be added automatically.
            - If set to `'additional_data_first'`, all additional columns are placed first.
            - If set to `'additional_data_last'`, all additional columns are placed last.
            This option helps control column order in the Shapash WebApp and SmartApp.
        additional_data : pandas.DataFrame, optional
            Additional features not used by the model but relevant for visualization or filtering
            in the WebApp.
            Must have the same index as `x_init`.
        additional_features_dict : dict, optional
            Mapping of additional feature names (technical names) to user-friendly
            domain names, used to improve readability in plots and dashboards.
            Must have the same index as `x_init`.

        Example
        -------
        >>> xpl.compile(x=x_test)
        >>> xpl.plot.features_importance()
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
        if (self.y_pred is None) and (hasattr(self.model, "predict")):
            self.predict()

        self.proba_values = check_y(self.x_init, proba_values, y_name="proba_values")
        if (self._case == "classification") and (self.proba_values is None) and (hasattr(self.model, "predict_proba")):
            self.predict_proba()

        self.y_target = check_y(self.x_init, y_target, y_name="y_target")
        self.prediction_error = predict_error(
            self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
        )

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
        self.columns_order = self._compile_columns_order(columns_order)
        self.plot._tuning_round_digit()

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
            raise AssertionError(f"Selected backend ({self.backend.name}) does not support groups of features.")
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

    def _compile_columns_order(self, columns_order):
        """
        Performs required computations for ordering data.
        """
        if isinstance(columns_order, list):
            check_columns_order(columns_order)
            # Prefix column name with "_" if it's listed in additional_features_dict
            columns_order = [f"_{col}" if f"_{col}" in self.additional_features_dict else col for col in columns_order]

            x_cols = set(self.x_encoded.columns)
            additional_cols = set(self.additional_features_dict)
            columns_order_set = set(columns_order)

            # Check for missing or unexpected columns
            missing_cols = x_cols - columns_order_set
            extra_cols = columns_order_set - x_cols - additional_cols

            if missing_cols:
                raise ValueError(f"The following columns are missing from columns_order: {missing_cols}")
            if extra_cols:
                raise ValueError(
                    f"The following columns in columns_order do not exist in x or additional data: {extra_cols}"
                )

        return columns_order

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
        proba_values=None,
        y_target=None,
        label_dict=None,
        features_dict=None,
        title_story: str = None,
        columns_order=None,
        additional_data=None,
        additional_features_dict=None,
    ):
        """
        Add or update metadata and outputs without recompiling the explainer.

        The `add` method lets users attach or modify supplementary information such as
        predictions, label or feature dictionaries, and display options **without**
        rerunning the full `compile()` process (which can be time-consuming for large datasets).

        It can be used to:
        - Add or update `y_pred` (used to color plots or export results).
        - Add or update `label_dict` and `features_dict` for clearer labels in visualizations.
        - Include additional data or adjust column display order in the WebApp.

        Parameters
        ----------
        y_pred : pandas.Series or pandas.DataFrame, optional
            Model predictions (one column only).
            Must have the same index as `x_init`.
            Used in plots (e.g., to color scatter plots) and in export methods like `to_pandas()`.
        proba_values : pandas.Series or pandas.DataFrame, optional
            Prediction probabilities (one column only).
            Must have the same index as `x_init`.
            Useful for visualizations or probabilistic outputs.
        y_target : pandas.Series or pandas.DataFrame, optional
            True target values (one column only).
            Must have the same index as `x_init`.
            Used for comparison and performance-oriented visualizations.
        label_dict : dict, optional
            Mapping of integer labels to domain names (for classification targets).
            Enables clearer class naming in plots and tables.
        features_dict : dict, optional
            Mapping of technical feature names to human-readable (domain) names.
            Improves interpretability of plots and exported data.
        title_story : str, optional
            Custom title for reports or visualizations.
            Default is empty.
        columns_order : list or str, optional
            Defines the display order of columns in the dataset.
            - If a **list** is provided, it specifies the exact order of columns.
            Columns not included will be appended automatically.
            - If set to `'additional_data_first'`, additional columns appear first.
            - If set to `'additional_data_last'`, additional columns appear last.
            Especially useful for controlling display order in the Shapash SmartApp.
        additional_data : pandas.DataFrame, optional
            Extra dataset containing features outside the model.
            Must have the same index as `x_init`.
            Useful for filtering and enrichment in the Shapash WebApp.
        additional_features_dict : dict, optional
            Dictionary mapping technical feature names to human-readable names
            for columns in `additional_data`.

        Example
        -------
        >>> # Add predictions and friendly feature names after compiling
        >>> xpl.add(y_pred=preds, features_dict=feat_dict)
        >>> xpl.plot.local_plot(index=5)
        """
        if y_pred is not None:
            self.y_pred = check_y(self.x_init, y_pred, y_name="y_pred")
        if proba_values is not None:
            self.proba_values = check_y(self.x_init, proba_values, y_name="proba_values")
        if y_target is not None:
            self.y_target = check_y(self.x_init, y_target, y_name="y_target")
        if hasattr(self, "y_target") and self.y_target is not None:
            self.prediction_error = predict_error(
                self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
            )
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
        if columns_order is not None:
            self.columns_order = self._compile_columns_order(columns_order)

    def get_interaction_values(self, n_samples_max=None, selection=None):
        """
        Compute SHAP interaction values for the encoded dataset.

        This method calculates pairwise SHAP interaction effects between features
        for each sample in `x_encoded`. It is only available when using a backend
        based on `TreeExplainer` (i.e., for tree-based models such as LightGBM,
        XGBoost, or CatBoost).

        For more details, see the official Tree SHAP paper:
        https://arxiv.org/pdf/1802.03888.pdf

        Parameters
        ----------
        n_samples_max : int, optional
            Maximum number of samples to compute interaction values for.
            If provided, the computation will be limited to this number of samples,
            selected randomly or according to the backend implementation.
        selection : list of int, optional
            List of specific sample indices for which to compute interactions.
            Useful to focus on a subset of the dataset rather than the entire `x_encoded`.

        Returns
        -------
        numpy.ndarray
            Array of SHAP interaction values with shape `(n_samples, n_features, n_features)`.
            Each entry `[i, j, k]` represents the interaction strength between features `j`
            and `k` for sample `i`.
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
        Check whether postprocessing transformations will convert numeric values to strings.

        This method inspects the provided `postprocessing` configuration and determines
        if any transformation rule would change a numerical feature into a string representation
        (e.g., by adding prefixes, suffixes, or other text-based modifications).

        Parameters
        ----------
        postprocessing : dict, optional
            Dictionary of postprocessing transformations to apply.
            Keys correspond to feature names, and values define transformation rules.

        Returns
        -------
        bool
            `True` if at least one numeric feature will be converted to string,
            otherwise `False`.
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
        Adjust the postprocessing dictionary so that all keys reference actual feature names.

        This method ensures that postprocessing rules are aligned with the real feature names
        used in the dataset. If the provided dictionary uses alternative identifiers
        (such as column indices or encoded names), they are converted into the corresponding
        feature names using `columns_dict` or `inv_features_dict`.

        Parameters
        ----------
        postprocessing : dict, optional
            Dictionary of postprocessing transformations to adjust.
            Keys may be feature names, indices, or label references.

        Returns
        -------
        dict
            Modified postprocessing dictionary, where all keys correspond directly
            to real feature names while preserving the original transformation rules.
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
        Apply postprocessing transformations to the `x_init` DataFrame, if defined.

        This method updates `x_init` according to the transformation rules specified
        in the `postprocessing` dictionary. If no postprocessing is provided,
        the original `x_init` is returned unchanged.

        Parameters
        ----------
        postprocessing : dict, optional
            Dictionary of postprocessing transformations to apply to `x_init`.
            Keys correspond to feature names, and values define the transformation rules.

        Returns
        -------
        pandas.DataFrame
            The modified `x_init` DataFrame if postprocessing rules are applied,
            otherwise the unmodified `x_init`.
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
        Synchronize features_dict with dataset columns:
        - Remove features not present in dataset
        - Add missing dataset features to features_dict
        """

        dataset_features = set(self.columns_dict.values())
        current_features = set(self.features_dict.keys())

        # Remove features not present in dataset
        for feature in current_features - dataset_features:
            self.features_dict.pop(feature, None)

        # Add features present in dataset but missing in features_dict
        for feature in dataset_features - current_features:
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
        Validate and convert a label name into its corresponding integer identifier.

        If the provided label is already an integer, it is returned unchanged.
        If it is a string corresponding to a class name, the method converts it
        into the appropriate integer label using the label dictionary.
        An error is raised if the label cannot be recognized.

        Parameters
        ----------
        label : int or str
            Label identifier, provided either as an integer (class index)
            or as a string (human-readable class name).
        origin : {'num', 'code', 'value', None}, optional
            Specifies the form of the input label:
            - `'num'`: integer class index
            - `'code'`: internal label code
            - `'value'`: business or display name
            - `None`: automatically inferred (default)

        Returns
        -------
        tuple
            A tuple containing:
            - `label_num` : int — numerical class index
            - `label_code` : object — internal class code used by the model
            - `label_value` : str — human-readable class name
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
        Validate and convert feature names or IDs into their corresponding column indices.

        This method ensures that the provided list of features is aligned with
        the internal column indexing used in Shapash. It supports both
        technical feature names and business (domain) names, as defined in
        `columns_dict` or `features_dict`.

        Parameters
        ----------
        features : list
            List of feature identifiers, where each element can be either:
            - an integer (column ID), or
            - a string (technical or business feature name).
        use_groups : bool, optional
            If True, the method also resolves feature groups defined in
            `features_groups`. Default is False.

        Returns
        -------
        list of int
            List of column indices corresponding to the input features,
            compatible with `var_dict`.
        """
        columns_dict = self.columns_dict if use_groups is False else self.columns_dict_groups
        return check_features_name(columns_dict, self.features_dict, features)

    def check_attributes(self, attribute):
        """
        Verify that the SmartExplainer instance contains the specified attribute.

        This method checks whether the given attribute exists within the
        current `SmartExplainer` instance and returns its content if found.

        Parameters
        ----------
        attribute : str
            Name of the attribute to check.

        Returns
        -------
        object
            The value of the specified attribute from the `SmartExplainer` instance.

        Raises
        ------
        ValueError
            If the specified attribute does not exist in the current explainer.
        """
        if not hasattr(self, attribute):
            raise ValueError(f"The attribute '{attribute}' does not exist in this SmartExplainer instance.")

        return self.__dict__[attribute]

    def filter(self, features_to_hide=None, threshold=None, positive=None, max_contrib=None, display_groups=None):
        """
        Apply filtering rules to summarize local explainability results.

        The `filter` method allows users to control which feature contributions
        are displayed or hidden when visualizing local explanations.
        It is typically used in combination with the `local_plot` method of
        `SmartPlotter` to display a filtered local contribution bar chart.

        For detailed examples, see the **Local Plot** tutorial in the Shapash documentation.

        Parameters
        ----------
        features_to_hide : list of str, optional
            List of feature names to hide from the visualization.
            These can be individual feature names or group names if
            `display_groups=True`.
        threshold : float, optional
            Absolute value threshold below which contributions are hidden.
            For example, setting `threshold=0.01` hides all features with
            contribution magnitudes smaller than 0.01.
        positive : bool, optional
            Defines whether to hide contributions by sign:
            - If `True`, hides negative contributions.
            - If `False`, hides positive contributions.
            - If `None` (default), all contributions are displayed.
        max_contrib : int, optional
            Maximum number of contributions to display.
            Only the top `max_contrib` features (by absolute contribution)
            will be shown.
        display_groups : bool, optional
            If `True`, feature groups defined in `features_groups` are displayed
            and filtered together.
            If `False`, only individual features are considered.
            By default, this is automatically set to `True` if
            feature groups are defined.

        Notes
        -----
        - The filtering configuration is stored in `self.mask_params`.
        - The resulting filtered contributions are available in
        `self.masked_contributions`.

        Example
        -------
        >>> # Hide specific features and small contributions
        >>> xpl.filter(features_to_hide=['Age', 'Gender'], threshold=0.01, max_contrib=10)
        >>> xpl.plot.local_plot(index=5)
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
        Save the SmartExplainer object to disk as a pickle file.

        This method serializes the current `SmartExplainer` instance and saves it
        to a `.pkl` file. It allows users to reload an explainer later without
        recompiling, which is especially useful for large datasets or models.

        Parameters
        ----------
        path : str
            Destination file path where the pickle file will be saved.

        Notes
        -----
        - The `smartapp` attribute is removed before saving to avoid serialization issues.
        - The saved object can be reloaded using the `load` method.

        Example
        -------
        >>> xpl.save("path_to_file/xpl.pkl")
        >>> xpl_loaded = SmartExplainer.load("path_to_file/xpl.pkl")
        """
        if hasattr(self, "smartapp"):
            self.smartapp = None
        save_pickle(self, path)

    @classmethod
    def load(cls, path):
        """
        Load a previously saved SmartExplainer object from a pickle file.

        This class method restores a `SmartExplainer` instance that was saved
        using the `save` method. It allows users to quickly reload a compiled
        explainer without repeating the full preprocessing and explanation steps.

        Parameters
        ----------
        path : str
            File path to the pickle file containing the saved `SmartExplainer` object.

        Returns
        -------
        SmartExplainer
            A reloaded `SmartExplainer` instance identical to the one saved on disk.

        Raises
        ------
        ValueError
            If the provided file does not contain a valid `SmartExplainer` object.

        Example
        -------
        >>> xpl = SmartExplainer.load("path_to_file/xpl.pkl")
        >>> xpl.plot.features_importance()
        """
        xpl = load_pickle(path)
        if isinstance(xpl, SmartExplainer):
            smart_explainer = cls(model=xpl.model)
            smart_explainer.__dict__.update(xpl.__dict__)
            return smart_explainer
        else:
            raise ValueError("The provided file does not contain a SmartExplainer object.")

    def predict_proba(self):
        """
        Compute and store prediction probabilities for each sample in `x_encoded`.

        This method applies the model’s `predict_proba` function to the encoded
        dataset (`x_encoded`) and saves the resulting probability values in
        `self.proba_values`.

        It is typically used for classification models to display or analyze
        predicted probabilities in visualizations or summaries.

        Returns
        -------
        None
            The computed probabilities are stored in the `proba_values` attribute.

        Example
        -------
        >>> xpl.predict_proba()
        >>> xpl.proba_values.head()
        """
        self.proba_values = predict_proba(self.model, self.x_encoded, self._classes)

    def predict(self):
        """
        Compute and store model predictions for each sample in `x_encoded`.

        This method applies the model’s `predict` function to the encoded dataset
        (`x_encoded`) and saves the resulting predictions in the `y_pred` attribute.
        If target values (`y_target`) are available, it also computes and stores
        the prediction error in `prediction_error`.

        Returns
        -------
        None
            The computed predictions are stored in the `y_pred` attribute.
            If available, prediction errors are stored in `prediction_error`.

        Example
        -------
        >>> xpl.predict()
        >>> xpl.y_pred.head()
        >>> xpl.prediction_error
        """
        self.y_pred = predict(self.model, self.x_encoded)
        if hasattr(self, "y_target"):
            self.prediction_error = predict_error(
                self.y_target, self.y_pred, self._case, proba_values=self.proba_values, classes=self._classes
            )

    def to_pandas(
        self,
        features_to_hide=None,
        threshold=None,
        positive=None,
        max_contrib=None,
        proba=False,
        use_groups=None,
    ):
        """
        Export a summarized view of local explainability results as a pandas DataFrame.

        The `to_pandas` method summarizes the local contributions of each feature
        for every sample, returning a DataFrame that combines predictions, probabilities
        (if applicable), and the top feature contributions.

        If no filtering parameters are provided, the method automatically reuses
        the configuration from the most recent call to the `filter` method.

        In classification tasks, this summary corresponds to the predicted values
        specified by the user (using either `compile()` or `add()`).
        You can also choose to include prediction probabilities using the `proba` parameter.

        There are two main usage modes in classification:
        1. Provide a real prediction set to explain.
        2. Focus on a constant target value and analyze its explainability and associated
        probabilities (using a constant `pd.Series` passed during `compile()` or `add()`).

        See the **Local Plot** tutorial for detailed examples.

        Parameters
        ----------
        features_to_hide : list of str, optional
            List of feature names to hide from the output summary.
        threshold : float, optional
            Absolute value threshold below which feature contributions are hidden.
        positive : bool, optional
            Determines which contribution signs to hide:
            - `True`: hide negative values.
            - `False`: hide positive values.
            - `None` (default): show all contributions.
        max_contrib : int, optional
            Maximum number of top feature contributions to include for each sample.
            Default is 5.
        proba : bool, optional
            If `True`, adds predicted probability values to the output DataFrame.
            Default is `False`.
        use_groups : bool, optional
            If `True`, aggregates feature contributions by groups defined in
            `features_groups` (if available).
            Default automatically activates grouping if `features_groups` were defined
            during `compile()`.

        Returns
        -------
        pandas.DataFrame
            A DataFrame summarizing local explanations for each sample.
            Columns typically include:
            - Predicted class or value (`pred`)
            - Probability (`proba`, if `proba=True`)
            - Top N feature names, values, and corresponding contributions

        Raises
        ------
        ValueError
            If predictions (`y_pred`) are missing.
            Use `compile()` or `add()` before calling this method.

        Example
        -------
        >>> # Export a summary of local explanations with probabilities
        >>> summary_df = xpl.to_pandas(max_contrib=2, proba=True)
        >>> summary_df.head()

            pred    proba       feature_1   value_1     contribution_1   feature_2   value_2     contribution_2
        0     0     0.756416    Sex         1.0         0.322308         Pclass      3.0         0.155069
        1     3     0.628911    Sex         2.0         0.585475         Pclass      1.0         0.370504
        2     0     0.543308    Sex         2.0         -0.486667        Pclass      3.0         0.255072
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
            self.predict_proba()
            proba_values = self.proba_values
        else:
            proba_values = None

        y_pred, summary = keep_right_contributions(
            self.y_pred, data["summary"], self._case, self._classes, self.label_dict, proba_values
        )

        return pd.concat([y_pred, summary], axis=1)

    def compute_features_import(self, force=False, local=False):
        """
        Compute the relative feature importance based on contribution magnitudes.

        This method calculates the global feature importance as the sum of the absolute
        values of feature contributions across all samples.
        The importance values are normalized on a base-100 scale.

        For models with defined feature groups, grouped importances are also computed.
        Optionally, local-level importances can be generated to capture finer-grained
        feature effects at multiple neighborhood scales.

        Parameters
        ----------
        force : bool, optional
            If `True`, recomputes feature importance even if it has already been calculated.
            Default is `False`.
        local : bool, optional
            If `True`, computes additional local-level importances at multiple aggregation
            scales (level 1 and level 2).
            Default is `False`.

        Returns
        -------
        pandas.Series or list of pandas.Series
            - **Regression:** a single `Series` with one row per feature.
            - **Classification:** a list of `Series`, one per class label.
            Each `Series` represents the normalized feature importances,
            indexed by feature name.

        Notes
        -----
        - Feature importances are computed using the backend’s `get_global_features_importance` method.
        - Grouped importances are computed if `features_groups` are defined.
        - When `local=True`, additional granular importances are computed with
        alternative normalization factors (norm=3 and norm=7).

        Example
        -------
        >>> # Compute standard global feature importance
        >>> xpl.compute_features_import()

        >>> # Compute both global and local-level importances
        >>> xpl.compute_features_import(local=True)
        >>> xpl.features_imp.head()
        """
        self.features_imp = self.backend.get_global_features_importance(
            contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=1
        )

        if self.features_groups is not None and self.features_imp_groups is None:
            self.features_imp_groups = self.state.compute_features_import(self.contributions_groups, norm=1)

        if local:
            self.features_imp_local_lev1 = self.backend.get_global_features_importance(
                contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=3
            )
            self.features_imp_local_lev2 = self.backend.get_global_features_importance(
                contributions=self.contributions, explain_data=self.explain_data, subset=None, norm=7
            )
            if self.features_groups is not None:
                self.features_imp_groups_local_lev1 = self.state.compute_features_import(
                    self.contributions_groups, norm=3
                )
                self.features_imp_groups_local_lev2 = self.state.compute_features_import(
                    self.contributions_groups, norm=7
                )

    def compute_features_stability(self, selection):
        """
        Compute feature stability metrics for a given selection of instances.

        This method calculates how stable feature contributions are within the
        neighborhood of selected samples.
        The resulting metrics are used in the visualizations
        `local_neighbors_plot` and `local_stability_plot`.

        Behavior depends on the size of the selection:
        - **Single instance:** returns the normalized contribution values of the
          instance and its neighbors (`norm_shap`).
        - **Multiple instances:** returns the average normalized contributions
          (`amplitude`) and their variability across neighborhoods (`variability`).

        Parameters
        ----------
        selection : list of int
            Indices of samples in `x_encoded` for which to compute stability metrics.
            Each index corresponds to a row in the dataset.

        Returns
        -------
        dict
            Dictionary containing arrays to be displayed in stability plots:
            - `"amplitude"` : average normalized contribution values of selected instances and their neighbors
            - `"variability"` : variation in contributions across the neighborhood
            - `"norm_shap"` : normalized SHAP (or contribution) values for the selected instance(s)

        Raises
        ------
        AssertionError
            If the explainer handles a multi-class classification problem (currently unsupported).

        Notes
        -----
        - Only binary classification and regression tasks are supported.
        - For each instance, nearest neighbors are identified using the encoded data (`x_encoded`).
        - Contributions are normalized to enable comparison across samples.

        Example
        -------
        >>> # Compute stability for a single instance
        >>> xpl.compute_features_stability(selection=[5])
        >>> xpl.local_neighbors["norm_shap"]

        >>> # Compute stability for multiple instances
        >>> xpl.compute_features_stability(selection=[2, 8, 12])
        >>> xpl.features_stability["variability"].shape
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
        Compute feature compacity metrics for a given selection of instances.

        This method evaluates how efficiently a model’s predictions can be
        approximated using only a subset of features. It returns:
        - the minimum number of features needed to reach a specified approximation level, and
        - the approximation level reached with a given number of features.

        These metrics are used in the `compacity_plot` visualization to illustrate
        the trade-off between explanation simplicity and fidelity.

        Parameters
        ----------
        selection : list of int
            Indices of samples in `x_encoded` for which to compute compacity metrics.
        distance : float
            Target approximation level (between 0 and 1) indicating how close
            the reduced-feature model should be to the full model.
        nb_features : int
            Number of features to use when computing the achieved approximation.

        Raises
        ------
        AssertionError
            If the explainer handles a multi-class classification problem (currently unsupported).

        Returns
        -------
        dict
            Dictionary containing:
            - `"features_needed"` : number of features required to reach the target approximation level
            - `"distance_reached"` : approximation level achieved using the given number of features

        Notes
        -----
        - Only regression and binary classification tasks are supported.
        - Approximation values are clipped between 0 and 1.
        - Feature compacity measures how well the model’s predictions can be summarized
          with fewer explanatory variables.

        Example
        -------
        >>> xpl.compute_features_compacity(selection=[0, 5, 10], distance=0.9, nb_features=10)
        >>> xpl.features_compacity["features_needed"]
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
        Initialize a SmartApp instance for the current SmartExplainer object.

        This method provides a simple way to create and configure the Shapash
        WebApp (`SmartApp`) when it is hosted or launched through a custom setup,
        rather than via the standard `run_app()` method.

        Parameters
        ----------
        settings : dict, optional
            Dictionary specifying default configuration values for the WebApp.
            Possible keys include:
            - `'rows'` : int — number of rows to display by default
            - `'points'` : int — number of points shown in scatter plots
            - `'violin'` : int — number of points displayed in violin plots
            - `'features'` : int — number of features shown in plots

            All values must be positive integers.

        Returns
        -------
        None
            Initializes the `smartapp` attribute with a configured `SmartApp` instance.

        Example
        -------
        >>> # Initialize SmartApp with custom settings
        >>> xpl.init_app(settings={"rows": 100, "features": 10})
        >>> xpl.smartapp.run()
        """
        self.smartapp = SmartApp(self, settings)

    def run_app(
        self,
        port: int = None,
        host: str = None,
        title_story: str = None,
        settings: dict = None,
    ) -> CustomThread:
        """
        Launch the Shapash interpretability WebApp associated with this SmartExplainer.

        This method starts the interactive Shapash WebApp that enables users to
        explore model predictions, feature importances, and local explanations
        directly in their browser.
        It can be called directly from a Jupyter notebook — the application link
        will appear in the notebook output.

        To stop the running app, use the `.kill()` method on the returned object.

        Examples of usage are provided in the **WebApp tutorial** in the Shapash documentation.

        Parameters
        ----------
        port : int, optional
            Port number for the WebApp server.
            Defaults to `8050` if not specified.
        host : str, optional
            Host address for the WebApp server.
            Defaults to `"0.0.0.0"`, allowing external access.
        title_story : str, optional
            Custom title to display in the WebApp interface.
            This title can also be reused in reports or other visualizations.
        settings : dict, optional
            Dictionary specifying default configuration values for the WebApp.
            Possible keys include:
            - `'rows'` : int — number of rows displayed by default
            - `'points'` : int — number of points in scatter plots
            - `'violin'` : int — number of points in violin plots
            - `'features'` : int — number of features shown in graphs
            All values must be positive integers.

        Returns
        -------
        CustomThread
            A thread instance running the WebApp server.

        Raises
        ------
        ValueError
            If the SmartExplainer has not been compiled before launching the app.

        Example
        -------
        >>> # Launch the WebApp in a Jupyter notebook
        >>> app = xpl.run_app(port=8050)
        >>> # Stop the app
        >>> app.kill()
        """

        if title_story is not None:
            self.title_story = title_story
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
        Create and return a SmartPredictor object derived from the current SmartExplainer instance.

        This method builds a `SmartPredictor` — a lightweight, production-oriented object
        that encapsulates all necessary components from the `SmartExplainer` to generate
        model predictions and interpretability outputs without requiring re-explanation.

        The generated `SmartPredictor` includes the model, preprocessing and postprocessing
        steps, feature and label mappings, and backend configuration used to compute
        contributions.

        Returns
        -------
        SmartPredictor
            A `SmartPredictor` instance initialized with the relevant attributes
            from the current `SmartExplainer`.

        Raises
        ------
        ValueError
            If no backend is defined in the current `SmartExplainer`.

        Attributes Transferred
        ----------------------
        - **features_dict** : dict
          Mapping from technical feature names to human-readable (domain) names.
        - **label_dict** : dict
          Mapping from integer labels to domain names (classification target values).
        - **columns_dict** : dict
          Mapping from integer column indices to technical feature names.
        - **features_types** : dict
          Mapping from feature names to their inferred data types.
        - **model** : object
          The trained model used for prediction.
        - **backend** : BaseBackend
          The backend used to compute feature contributions (e.g., SHAP, LIME).
        - **preprocessing** : category_encoders object, ColumnTransformer, list, or dict
          Preprocessing transformations applied to the original data.
        - **postprocessing** : dict
          Postprocessing transformations applied after inverse preprocessing.
        - **features_groups** : dict, optional
          Feature grouping structure, if defined during compilation.
        - **_case** : str
          Indicates whether the task is `"classification"` or `"regression"`.
        - **_classes** : list or None
          List of class labels for classification models, `None` for regression.
        - **mask_params** : dict, optional
          Parameters defining contribution filters used to summarize local explainability.

        Example
        -------
        >>> # Convert a SmartExplainer into a deployable SmartPredictor
        >>> sp = xpl.to_smartpredictor()
        >>> sp.predict(data_sample)
        >>> sp.explain(data_sample)
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
        Validate and retrieve two attributes from the SmartExplainer instance.

        This method checks whether the given attribute names exist in the current
        `SmartExplainer` object. It returns the corresponding attribute values if found,
        or `None` for any attribute that does not exist.

        Parameters
        ----------
        x_str : str
            Name of the first attribute to check.
        y_str : str
            Name of the second attribute to check.

        Returns
        -------
        list
            A two-element list containing the retrieved attributes in order:
            `[x_attribute, y_attribute]`.
            Each element is the attribute’s value if it exists, otherwise `None`.

        Raises
        ------
        ValueError
            If either `x_str` or `y_str` is not provided as a string.

        Example
        -------
        >>> x_attr, y_attr = xpl.check_x_y_attributes("x_encoded", "y_pred")
        >>> print(x_attr.shape, y_attr.shape)
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
        max_points=200,
        display_interaction_plot=False,
        nb_top_interactions=5,
    ):
        """
        Generate an interactive HTML report summarizing the model and its explainability.

        This method produces a comprehensive HTML report containing visual and textual
        insights about the project, dataset, and model performance.
        It leverages a predefined or custom Jupyter notebook template to analyze
        the model, generate plots, compute metrics, and export the final report.

        A project information YAML file is required to describe key project details
        (e.g., model name, author, date, context).

        Parameters
        ----------
        output_file : str
            Path to the output HTML file where the report will be saved.
        project_info_file : str
            Path to a YAML file containing project metadata to be displayed in the report
            (e.g., project name, author, date, description).
        x_train : pandas.DataFrame, optional
            Training dataset used to fit the model.
            Used for generating feature summaries and training-related analyses.
        y_train : pandas.Series or pandas.DataFrame, optional
            Target values corresponding to `x_train`.
        y_test : pandas.Series or pandas.DataFrame, optional
            Target values for the test dataset.
        title_story : str, optional
            Title displayed at the top of the report.
        title_description : str, optional
            Short descriptive text displayed below the main title.
        metrics : list of dict, optional
            List of metrics to compute and display in the performance section.
            Each dictionary should include:
            - `'path'`: str — import path to the metric function (e.g., `"sklearn.metrics.f1_score"`)
            - `'name'`: str, optional — display name for the metric
            - `'use_proba_values'`: bool, optional — if True, use predicted probabilities instead of labels
            Example:
            `metrics=[{'name': 'F1 score', 'path': 'sklearn.metrics.f1_score'}]`
        working_dir : str, optional
            Directory used to temporarily store generated files (e.g., notebook, outputs).
            If `None`, a temporary directory is automatically created and deleted after report generation.
        notebook_path : str, optional
            Path to a custom notebook used as a template for generating the report.
            If `None`, the default Shapash report notebook is used.
        kernel_name : str, optional
            Name of the Jupyter kernel to use for report execution.
            Useful when multiple kernels are available and the default one is incorrect.
        max_points : int, optional, default=200
            Maximum number of points displayed in contribution plots.
        display_interaction_plot : bool, optional, default=False
            If True, includes interaction plots in the report.
            (Note: this can increase computation time.)
        nb_top_interactions : int, optional, default=5
            Number of top feature interactions to include in the report.

        Returns
        -------
        None
            The report is saved as an HTML file at the specified `output_file` location.

        Raises
        ------
        AssertionError
            If the SmartExplainer instance is not compiled before report generation.
        Exception
            If an unexpected error occurs during report execution or export.

        Notes
        -----
        - The method internally executes a notebook that generates the report content.
        - Temporary files are automatically cleaned up unless a custom `working_dir` is provided.
        - Interaction plots can be disabled to optimize runtime performance.

        Example
        -------
        >>> xpl.generate_report(
        ...     output_file="report.html",
        ...     project_info_file="utils/project_info.yml",
        ...     x_train=x_train,
        ...     y_train=y_train,
        ...     y_test=y_test,
        ...     title_story="House Prices Project Report",
        ...     title_description="Comprehensive interpretability analysis for the Kaggle house prices dataset.",
        ...     metrics=[
        ...         {"path": "sklearn.metrics.mean_squared_error", "name": "Mean Squared Error"},
        ...         {"path": "sklearn.metrics.mean_absolute_error", "name": "Mean Absolute Error"},
        ...     ],
        ...     display_interaction_plot=True,
        ...     nb_top_interactions=5,
        ... )
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
                    max_points=max_points,
                    display_interaction_plot=display_interaction_plot,
                    nb_top_interactions=nb_top_interactions,
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

    def _local_pred(self, index, label=None):
        """
        Compute the model prediction or probability for a single observation.

        This internal method retrieves the prediction or class probability
        corresponding to a specific sample index.

        Parameters
        ----------
        index : int, str, or float
            Index of the sample for which to compute the prediction.
            Must correspond to a valid index in `x_encoded`.
        label : int, optional
            Class label for which to extract the probability in classification tasks.
            If `None`, the method returns the prediction for the main target.

        Returns
        -------
        float
            The predicted value (for regression) or predicted probability
            (for classification).

        Notes
        -----
        - For classification, returns the class probability if `proba_values` are available.
        - For regression, returns the predicted numeric value.
        - This is an internal helper used primarily for visualization.

        Example
        -------
        >>> # Retrieve the predicted value for observation at index 12
        >>> xpl._local_pred(index=12)
        0.7421
        """
        if self._case == "classification":
            if self.proba_values is not None:
                value = self.proba_values.iloc[:, [label]].loc[index].values[0]
            else:
                value = None
        elif self._case == "regression":
            if self.y_pred is not None:
                value = self.y_pred.loc[index]
            else:
                value = self.model.predict(self.x_encoded.loc[[index]])[0]

        if isinstance(value, pd.Series):
            value = value.values[0]

        return value
