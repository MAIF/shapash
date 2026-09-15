"""
Smart explainer module
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from werkzeug.serving import make_server

import shapash.explainer.smart_predictor
from shapash.backend import BaseBackend
from shapash.style.style_utils import colors_loading, select_palette
from shapash.utils.custom_thread import CustomThread
from shapash.utils.io import load_pickle, save_pickle
from shapash.utils.transform import handle_categorical_missing
from shapash.utils.utils import get_host_name
from shapash.webapp.smart_app import SmartApp

from .explainer import Explainer
from .smart_plotter import SmartPlotter

if TYPE_CHECKING:
    from shapash.report.blocks import ReportBlockMixin

REPORT_DEPENDENCIES_AVAILABLE = False
ReportTemplate: Any | None = None
_ReportBlockMixin: type[Any] | None = None


def _generate_smart_report_unavailable(*args: Any, **kwargs: Any) -> None:
    raise ImportError("Report dependencies are not installed. Please install shapash report extras.")


generate_smart_report = _generate_smart_report_unavailable


try:
    from shapash.report import ReportTemplate as _ReportTemplate
    from shapash.report.blocks import ReportBlockMixin as _ReportBlockMixin
    from shapash.report.core import generate_report as _generate_smart_report
except ImportError:
    # [report] optional dependencies may not be installed
    ...
else:
    REPORT_DEPENDENCIES_AVAILABLE = True
    ReportTemplate = _ReportTemplate
    generate_smart_report = _generate_smart_report

logging.basicConfig(level=logging.INFO)

DEFAULT_HOST = "127.0.0.1"


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
        model: Any,
        backend: str | BaseBackend = "shap",
        preprocessing: Any | None = None,
        postprocessing: dict[str, Any] | None = None,
        features_groups: dict[str, list[str]] | None = None,
        features_dict: dict[str, str] | None = None,
        label_dict: dict[Any, Any] | None = None,
        title_story: str | None = None,
        palette_name: str | None = None,
        colors_dict: dict[str, Any] | None = None,
        **backend_kwargs: Any,
    ) -> None:
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
        self.smartapp: Any = None
        self.model = model
        title_story = title_story if title_story is not None else ""
        self.title_story = title_story
        self.palette_name = palette_name if palette_name else "default"
        self.colors_dict = copy.deepcopy(select_palette(colors_loading(), self.palette_name))
        if colors_dict is not None:
            self.colors_dict.update(colors_dict)

        self.explainer = Explainer(
            model=model,
            backend=backend,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            features_groups=features_groups,
            features_dict=features_dict,
            label_dict=label_dict,
            **backend_kwargs,
        )
        self.plot = SmartPlotter(self.explainer, self.colors_dict)
        self.explainer.plot = self.plot

    def compile(
        self,
        x: pd.DataFrame,
        contributions: pd.DataFrame | np.ndarray | list[pd.DataFrame] | list[np.ndarray] | None = None,
        y_pred: pd.Series | pd.DataFrame | None = None,
        proba_values: pd.Series | pd.DataFrame | None = None,
        y_target: pd.Series | pd.DataFrame | None = None,
        columns_order: list[str] | None = None,
        additional_data: pd.DataFrame | None = None,
        additional_features_dict: dict[str, str] | None = None,
    ) -> None:
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
        self.explainer.compile(
            x=x,
            contributions=contributions,
            y_pred=y_pred,
            proba_values=proba_values,
            y_target=y_target,
            columns_order=columns_order,
            additional_data=additional_data,
            additional_features_dict=additional_features_dict,
        )

    def define_style(self, palette_name: str | None = None, colors_dict: dict[str, Any] | None = None) -> None:
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
        y_pred: pd.Series | pd.DataFrame | None = None,
        proba_values: pd.Series | pd.DataFrame | None = None,
        y_target: pd.Series | pd.DataFrame | None = None,
        label_dict: dict[Any, Any] | None = None,
        features_dict: dict[str, str] | None = None,
        title_story: str | None = None,
        columns_order: list[str] | None = None,
        additional_data: pd.DataFrame | None = None,
        additional_features_dict: dict[str, str] | None = None,
    ) -> None:
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
        if title_story is not None:
            self.title_story = title_story

        self.explainer.add(
            y_pred=y_pred,
            proba_values=proba_values,
            y_target=y_target,
            label_dict=label_dict,
            features_dict=features_dict,
            columns_order=columns_order,
            additional_data=additional_data,
            additional_features_dict=additional_features_dict,
        )

    def check_attributes(self, attribute: str) -> Any:
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
        if hasattr(self, attribute):
            return getattr(self, attribute)

        if not hasattr(self.explainer, attribute):
            raise ValueError(f"The attribute '{attribute}' does not exist in this SmartExplainer instance.")

        return getattr(self.explainer, attribute)

    def filter(
        self,
        features_to_hide: list[str] | None = None,
        threshold: float | None = None,
        positive: bool | None = None,
        max_contrib: int | None = None,
        display_groups: bool | None = None,
    ) -> None:
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
        features_to_hide_values: list[Any] | None = features_to_hide
        if features_to_hide is not None:
            use_groups = True if (display_groups is not False and self.explainer.features_groups is not None) else False
            features_to_hide_values = self.explainer.check_features_name(features_to_hide, use_groups=use_groups)

        self.explainer.filter(
            features_to_hide=features_to_hide_values,
            threshold=threshold,
            positive=positive,
            max_contrib=max_contrib,
            display_groups=display_groups,
        )

    def save(self, path: str) -> None:
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
    def load(cls, path: str) -> SmartExplainer:
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
            # Rebind plot<->explainer links after unpickling.
            smart_explainer.plot = SmartPlotter(smart_explainer.explainer, smart_explainer.colors_dict)
            smart_explainer.explainer.plot = smart_explainer.plot
            return smart_explainer
        else:
            raise ValueError("The provided file does not contain a SmartExplainer object.")

    def to_pandas(
        self,
        features_to_hide: list[str] | None = None,
        threshold: float | None = None,
        positive: bool | None = None,
        max_contrib: int | None = None,
        proba: bool = False,
        use_groups: bool | None = None,
    ) -> pd.DataFrame:
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
        return self.explainer.to_pandas(
            features_to_hide=features_to_hide,
            threshold=threshold,
            positive=positive,
            max_contrib=max_contrib,
            proba=proba,
            use_groups=use_groups,
        )

    def init_app(self, settings: dict[str, Any] | None = None):
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
            - `'toggle_group'` : bool — default state of the group toggle in the UI

            All integer values must be positive.

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
        self.smartapp = SmartApp(self.explainer, settings, title_story=self.title_story)

    def run_app(
        self,
        port: int | None = None,
        host: str | None = None,
        title_story: str | None = None,
        settings: dict[str, Any] | None = None,
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
            Defaults to local host `"127.0.0.1"`.
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
        if hasattr(self.explainer, "_case"):
            self.smartapp = SmartApp(self.explainer, settings, title_story=self.title_story)
            if host is None:
                host = DEFAULT_HOST
            if port is None:
                port = 8050
            host_name = get_host_name()
            wsgi_server = make_server(host, port, self.smartapp.server)
            server_instance = CustomThread(target=wsgi_server.serve_forever)

            def _kill():
                wsgi_server.shutdown()
                server_instance.killed = True

            cast(Any, server_instance).kill = _kill
            if host_name is None:
                host_name = host
            elif host != DEFAULT_HOST:
                host_name = host
            server_instance.start()
            logging.info(f"Your Shapash application run on http://{host_name}:{port}/")
            logging.info("Use the method .kill() to down your app.")
            return server_instance

        else:
            raise ValueError("Explainer must be compiled before running app.")

    def to_smartpredictor(self) -> Any:
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
        if self.explainer.backend is None:
            raise ValueError(
                """
                SmartPredictor needs a backend (explainer).
                Please compile without contributions or specify  the
                explainer used. Make change in compile() step.
                """
            )

        features_types = {
            features: str(self.explainer.x_init[features].dtypes) for features in self.explainer.x_init.columns
        }

        listattributes = [
            "features_dict",
            "model",
            "columns_dict",
            "backend",
            "label_dict",
            "preprocessing",
            "postprocessing",
            "features_groups",
        ]

        params_smartpredictor = [self.check_attributes(attribute) for attribute in listattributes]
        params_smartpredictor.insert(4, features_types)

        if hasattr(self.explainer, "mask_params"):
            mask_params = self.explainer.mask_params
        else:
            mask_params = {"features_to_hide": None, "threshold": None, "positive": None, "max_contrib": None}
        params_smartpredictor.append(mask_params)

        return shapash.explainer.smart_predictor.SmartPredictor(*params_smartpredictor)

    def check_x_y_attributes(self, x_str: str, y_str: str) -> list[Any]:
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
            if hasattr(self.explainer, attribut):
                params_checkypred.append(getattr(self.explainer, attribut))
            else:
                params_checkypred.append(None)
        return params_checkypred

    def generate_report(
        self,
        output_file: str,
        x_train: pd.DataFrame | None = None,
        y_train: pd.Series | pd.DataFrame | list | None = None,
        y_test: pd.Series | pd.DataFrame | list | None = None,
        yaml_path: str | Path | None = None,
        max_points: int = 200,
        block_instance: ReportBlockMixin | None = None,
    ) -> None:
        """
        Generate an interactive HTML report summarizing the model and its explainability.

        This method produces a comprehensive HTML report containing visual and textual
        insights about the project, dataset, and model performance using the
        smart_report block-based HTML renderer.

        A report configuration is provided through a YAML file. If no YAML file is
        specified, a default configuration is generated automatically.

        Parameters
        ----------
        output_file : str
            Path to the output HTML file where the report will be saved.
        x_train : pandas.DataFrame, optional
            Training dataset used to fit the model.
            Used for generating feature summaries and training-related analyses.
        y_train : pandas.Series or pandas.DataFrame, optional
            Target values corresponding to `x_train`.
        y_test : pandas.Series or pandas.DataFrame, optional
            Target values for the test dataset.
        yaml_path : str, optional
            Path to a custom YAML configuration file used to generate the report.
            If `None`, a default YAML configuration is generated.
        max_points : int, optional, default=200
            Maximum number of points displayed in contribution plots.
        block_instance : object, optional
            Optional custom block runtime used to resolve block methods during report generation.
            The instance must already be fully initialized by the user and should implement
            methods named `block_<type>` for YAML block entries.

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
        - The method renders the report from block definitions in a YAML configuration.
        - Temporary files are automatically cleaned up unless a custom `working_dir` is provided.
        - Interaction plots can be disabled to optimize runtime performance.

        Example
        -------
        >>> xpl.generate_report(
        ...     output_file="report.html",
        ...     x_train=x_train,
        ...     y_train=y_train,
        ...     y_test=y_test,
        ...     display_interaction_plot=True,
        ...     nb_top_interactions=5,
        ... )
        """

        # input checks
        if not hasattr(self, "model"):
            raise AssertionError(
                "Explainer object was not compiled. Please compile the explainer "
                "object using .compile(...) method before generating the report."
            )

        report_template = ReportTemplate
        report_block_cls = _ReportBlockMixin
        if (not REPORT_DEPENDENCIES_AVAILABLE) or report_template is None or report_block_cls is None:
            raise ImportError("Report dependencies are not installed. Please install shapash report extras.")

        if block_instance is not None:
            if (x_train is not None) and (block_instance.x_train_init is not x_train):
                logging.warning("block_instance's x_train is different from provided x_train. Latter is ignored.")
            if (y_train is not None) and (block_instance.y_train is not y_train):
                logging.warning("block_instance's y_train is different from provided y_train. Latter is ignored.")
            if (y_test is not None) and (block_instance.y_test is not y_test):
                logging.warning("block_instance's y_test is different from provided y_test. Latter is ignored.")
            if max_points != block_instance.max_points:
                logging.warning("block_instance's max_points is different from provided max_points. Latter is ignored.")

            report_runtime = block_instance

        else:
            if x_train is not None:
                x_train = handle_categorical_missing(x_train)

            report_runtime = report_block_cls(
                explainer=self.explainer,
                x_train=x_train,
                y_train=y_train,
                y_test=y_test,
                max_points=max_points,
            )
        if self.explainer._case == "classification":
            default_report = report_template.DEFAULT_CLASSIFICATION
        else:
            default_report = report_template.DEFAULT_REGRESSION

        config_file = (
            Path(yaml_path)
            if yaml_path is not None
            else Path(__file__).resolve().parent.parent / "report" / "assets" / str(default_report)
        )
        generate_smart_report(
            runtime=report_runtime,
            config_file=config_file,
            output_file=output_file,
        )

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
        if self.explainer._case == "classification":
            if self.explainer.proba_values is not None:
                value = self.explainer.proba_values.iloc[:, [label]].loc[index].values[0]
            else:
                value = None
        elif self.explainer._case == "regression":
            if self.explainer.y_pred is not None:
                value = self.explainer.y_pred.loc[index]
            else:
                value = self.explainer.model.predict(self.explainer.x_encoded.loc[[index]])[0]

        if isinstance(value, pd.Series):
            value = value.values[0]

        return value
