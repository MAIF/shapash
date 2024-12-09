"""
Smart plotter module
"""

import math
import random
from typing import Optional

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.manipulation.select_lines import select_lines
from shapash.manipulation.summarize import project_feature_values_1d
from shapash.plots import plot_compacity
from shapash.plots.plot_bar_chart import plot_bar_chart
from shapash.plots.plot_contribution import plot_scatter, plot_violin
from shapash.plots.plot_correlations import plot_correlations
from shapash.plots.plot_evaluation_metrics import plot_confusion_matrix, plot_scatter_prediction
from shapash.plots.plot_feature_importance import plot_feature_importance
from shapash.plots.plot_interactions import plot_interactions_scatter, plot_interactions_violin, update_interactions_fig
from shapash.plots.plot_line_comparison import plot_line_comparison
from shapash.plots.plot_stability import plot_amplitude_vs_stability, plot_stability_distribution
from shapash.plots.plot_univariate import plot_distribution
from shapash.style.style_utils import colors_loading, define_style, select_palette
from shapash.utils.sampling import subset_sampling
from shapash.utils.utils import (
    add_line_break,
    add_text,
    adjust_title_height,
    compute_digit_number,
    compute_sorted_variables_interactions_list_indices,
    maximum_difference_sort_value,
    truncate_str,
    tuning_colorscale,
)


class SmartPlotter:
    """
    SmartPlotter is a Bridge pattern decoupling plotting functions from SmartExplainer.
    The smartplotter class includes all the methods used to display graphics
    Each SmartPlotter method is easy to use from a Smart explainer object,
    just use the following syntax
    Attributes :
    explainer: object
        SmartExplainer instance to point to.
    Example
    --------
    >>> xpl.plot.my_plot_method(param=value)
    """

    def __init__(self, explainer, colors_dict=None):
        self._explainer = explainer
        if colors_dict:
            self._style_dict = define_style(colors_dict)
        else:
            palette_name = list(colors_loading().keys())[0]
            self._style_dict = define_style(select_palette(colors_loading(), palette_name))
        self._last_stability_selection = False
        self._last_compacity_selection = False
        self._tuning_round_digit()

    def define_style_attributes(self, colors_dict):
        """
        define_style_attributes allows shapash user to change the color of plot
        Parameters
        ----------
        colors_dict: dict
            Dict of the colors used in the different plots
        """
        self._style_dict = define_style(colors_dict)

    def _tuning_round_digit(self):
        """
        adapts the display of the number of digit to the distribution of points
        """
        quantile = [0.25, 0.75]
        if hasattr(self._explainer, "y_pred") and self._explainer.y_pred is not None:
            desc_df = self._explainer.y_pred.describe(percentiles=quantile)
            perc1, perc2 = list(desc_df.loc[[str(int(p * 100)) + "%" for p in quantile]].values)
            p_diff = perc2 - perc1
            self._round_digit = compute_digit_number(p_diff)
        else:
            self._round_digit = 0

    def _get_selection(self, line, var_dict, x_val, contrib):
        """
        An auxiliary function to select the row of interest.
        Parameters
        ----------
        line: list
            A one element list containing the index of the observation of interest.
        var_dict: pandas.DataFrame
            A dataframe that indicates for each observation (each row)
            the index of the sorted contribution
            (sorted by descending order, in absolute values).
        x_val: pandas.DataFrame
            A dataframe with sorted features for each observation.
        contrib: pandas.DataFrame
            A dataframe with sorted contributions for each observation.
        Returns
        -------
        numpy arrays
            Unidimensional numpy arrays containing the values for one observation.
        """
        contrib = contrib.loc[line[0], :].values
        x_val = x_val.loc[line[0], :].values
        var_dict = var_dict.loc[line[0], :].values

        return var_dict, x_val, contrib

    def _apply_mask_one_line(self, line, var_dict, x_val, contrib, label=None):
        """
        An auxiliary function to select the mask to apply before plotting local
        explanation.
        Parameters
        ----------
        line: list
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        var_dict: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        x_val: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        contrib: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        label: integer (default None)
            specify the pd.DataFrame of the mask list (classification case) to apply
        Returns
        -------
        lists
            Masked input lists.
        """
        mask = np.array([True] * len(contrib))
        if hasattr(self._explainer, "mask"):
            if isinstance(self._explainer.mask, list):
                mask = self._explainer.mask[label].loc[line[0], :].values
            else:
                mask = self._explainer.mask.loc[line[0], :].values

        contrib = contrib[mask]
        x_val = x_val[mask]
        var_dict = var_dict[mask]

        return var_dict.tolist(), x_val.tolist(), contrib.tolist()

    def _check_masked_contributions(self, line, var_dict, x_val, contrib, label=None):
        """
        Check for masked contributions and update features_values and contrib
        to take the sum of masked contributions into account.
        Parameters
        ----------
        line: list
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        var_dict: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        x_val: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        contrib: numpy array
            Unidimensional numpy array containing the values for the observation of interest.
        Returns
        -------
        numpy arrays
            Input arrays updated with masked contributions.
        """
        if hasattr(self._explainer, "masked_contributions"):
            if isinstance(self._explainer.masked_contributions, list):
                ext_contrib = self._explainer.masked_contributions[label].loc[line[0], :].values
            else:
                ext_contrib = self._explainer.masked_contributions.loc[line[0], :].values

            ext_var_dict = ["Hidden Negative Contributions", "Hidden Positive Contributions"]
            ext_x = ["", ""]
            ext_contrib = ext_contrib.tolist()

            exclusion = np.flatnonzero(np.array(ext_contrib) == 0).tolist()
            exclusion.sort(reverse=True)
            for ind in exclusion:
                del ext_var_dict[ind]
                del ext_x[ind]
                del ext_contrib[ind]

            var_dict.extend(ext_var_dict)
            x_val.extend(ext_x)
            contrib.extend(ext_contrib)

        return var_dict, x_val, contrib

    def local_plot(
        self,
        index=None,
        row_num=None,
        query=None,
        label=None,
        show_masked=True,
        show_predict=True,
        display_groups=None,
        yaxis_max_label=12,
        width=900,
        height=550,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        The local_plot method is used to display the local contributions of
        an individual in the dataset.
        The plot returned is a summary of local explainability.
        you could use the method filter beforehand to modify the parameters of this summary.
        preprocessing is used here to make this graph more intelligible
        index, row_num or query parameter can be used to select the local explanations to display
        local_plot tutorial offers a lot of examples (please check tutorial part of this doc)
        Parameters
        ----------
        index: string, int, float, ... type of index in x_val input matrix (default None)
            1rst option, to select a row whose local contribution will be displayed.
            Use this parameter to select a row by index
        row_num: int (default None)
            2nd option, specify the row number to select the row whose local
            contribution will be displayed.
        query: string
            3rd option: Boolean condition that must filter only one line of the prediction
            set before plotting.
        label: integer or string (default None)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        show_masked: bool (default: False)
            show the sum of the contributions of the hidden variable
        show_predict: bool (default: True)
            show predict or predict proba value
        yaxis_max_label: int
            Maximum number of variables to display labels on the y axis
        display_groups : bool (default: None)
            Whether or not to display groups of features. This option is
            only useful if groups of features are declared when compiling
            SmartExplainer object.
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 550)
            Plotly figure - layout height
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the bar plot or not.
        zoom: bool (default=False)
            graph is currently zoomed
        Returns
        -------
        Plotly Figure Object
            Input arrays updated with masked contributions.
        Example
        --------
        >>> xpl.plot.local_plot(row_num=0)
        """
        display_groups = (
            True if (display_groups is not False and self._explainer.features_groups is not None) else False
        )
        if display_groups:
            data = self._explainer.data_groups
        else:
            data = self._explainer.data

        if index is not None:
            if index in self._explainer.x_init.index:
                line = [index]
            else:
                line = []
        elif row_num is not None:
            line = [self._explainer.x_init.index[row_num]]
        elif query is not None:
            line = select_lines(self._explainer.x_init, query)
        else:
            line = []

        subtitle = ""

        if len(line) != 1:
            if len(line) > 1:
                raise ValueError("Only one line/observation must match the condition")
            contrib = []
            x_val = []
            var_dict = []

        else:
            # apply filter if the method have not yet been asked in order to limit the number of feature to display
            if (
                not hasattr(self._explainer, "mask_params")  # If the filter method has not been called yet
                # Or if the already computed mask was not updated with current display_groups parameter
                or (
                    isinstance(data["contrib_sorted"], pd.DataFrame)
                    and len(data["contrib_sorted"].columns) != len(self._explainer.mask.columns)
                )
                or (
                    isinstance(data["contrib_sorted"], list)
                    and len(data["contrib_sorted"][0].columns) != len(self._explainer.mask[0].columns)
                )
            ):
                self._explainer.filter(max_contrib=20, display_groups=display_groups)

            if self._explainer._case == "classification":
                if label is None:
                    label = -1

                label_num, _, label_value = self._explainer.check_label_name(label)

                contrib = data["contrib_sorted"][label_num]
                x_val = data["x_sorted"][label_num]
                var_dict = data["var_dict"][label_num]

                if show_predict is True:
                    pred = self._explainer._local_pred(line[0], label_num)
                    if pred is None:
                        subtitle = f"Response: <b>{label_value}</b> - No proba available"
                    else:
                        subtitle = f"Response: <b>{label_value}</b> - Proba: <b>{pred:.4f}</b>"

            elif self._explainer._case == "regression":
                contrib = data["contrib_sorted"]
                x_val = data["x_sorted"]
                var_dict = data["var_dict"]
                label_num = None
                if show_predict is True:
                    pred_value = self._explainer._local_pred(line[0])
                    if self._round_digit:
                        digit = self._round_digit
                    else:
                        digit = compute_digit_number(pred_value)
                    subtitle = f"Predict: <b>{round(pred_value, digit)}</b>"

            var_dict, x_val, contrib = self._get_selection(line, var_dict, x_val, contrib)
            var_dict, x_val, contrib = self._apply_mask_one_line(line, var_dict, x_val, contrib, label=label_num)
            # use label of each column
            if display_groups:
                var_dict = [self._explainer.features_dict[self._explainer.x_init_groups.columns[x]] for x in var_dict]
            else:
                var_dict = [self._explainer.features_dict[self._explainer.columns_dict[x]] for x in var_dict]
            if show_masked:
                var_dict, x_val, contrib = self._check_masked_contributions(
                    line, var_dict, x_val, contrib, label=label_num
                )
            # Filtering all negative or positive contrib if specify in mask
            exclusion = []
            if hasattr(self._explainer, "mask_params"):
                positive = self._explainer.mask_params.get("positive")
                if positive is not None:
                    exclusion = np.flatnonzero(np.array(contrib) < 0 if positive else np.array(contrib) > 0).tolist()

            exclusion.sort(reverse=True)
            for expl in exclusion:
                del var_dict[expl]
                del x_val[expl]
                del contrib[expl]

        fig = plot_bar_chart(
            line,
            var_dict,
            x_val,
            contrib,
            self._style_dict,
            self._explainer.features_groups,
            self._explainer.x_init,
            self._explainer.features_dict,
            self._explainer.inv_features_dict,
            yaxis_max_label,
            subtitle,
            width,
            height,
            file_name,
            auto_open,
            zoom,
        )
        return fig

    def contribution_plot(
        self,
        col,
        selection=None,
        label=-1,
        violin_maxf=10,
        max_points=2000,
        proba=True,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        contribution_plot method diplays a Plotly scatter or violin plot of a selected feature.
        It represents the contribution of the selected feature to the predicted value.
        This plot allows the user to understand how the value of a feature affects a prediction
        Type of plot (Violin/scatter) is automatically selected. It depends on the feature
        to be analyzed, the type of use case (regression / classification) and the presence of
        predicted values attribute.
        A sample is taken if the number of points to be displayed is too large
        Using col parameter, shapash user can specify the column num, name or column label of
        the feature
        contribution_plot tutorial offers many examples (please check tutorial part of this doc)
        Parameters
        ----------
        col: String or Int
            Name, label name or column number of the column whose contributions we want to plot
        selection: list (optional)
            Contains list of index, subset of the input DataFrame that we want to plot
        label: integer or string (default -1)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        violin_maxf: int (optional, default: 10)
            maximum number modality to plot violin. If the feature specified with col argument
            has more modalities than violin_maxf, a scatter plot will be choose
        max_points: int (optional, default: 2000)
            maximum number to plot in contribution plot. if input dataset is bigger than max_points,
            a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        proba: bool (optional, default: True)
            use predict_proba to color plot (classification case)
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the bar plot or not.
        zoom: bool (default=False)
            graph is currently zoomed
        Returns
        -------
        Plotly Figure Object
        Example
        --------
        >>> xpl.plot.contribution_plot(0)
        """
        if self._explainer._case == "classification":
            label_num, _, label_value = self._explainer.check_label_name(label)

        if not isinstance(col, (str, int)):
            raise ValueError("parameter col must be string or int.")
        if hasattr(self._explainer, "inv_features_dict"):
            col = self._explainer.inv_features_dict.get(col, col)
        col_is_group = self._explainer.features_groups and col in self._explainer.features_groups.keys()

        # Case where col is a group of features
        if col_is_group:
            contributions = self._explainer.contributions_groups
            col_label = self._explainer.features_dict[col]
            col_name = self._explainer.features_groups[col]  # Here col_name is actually a list of features
            col_value_count = self._explainer.features_desc[col]
        else:
            contributions = self._explainer.contributions
            col_id = self._explainer.check_features_name([col])[0]
            col_name = self._explainer.columns_dict[col_id]
            col_value_count = self._explainer.features_desc[col_name]

            if self._explainer.features_dict:
                col_label = self._explainer.features_dict[col_name]
            else:
                col_label = col_name

        list_ind, addnote = subset_sampling(
            self._explainer.x_init, selection, max_points, None if col_is_group else col, col_value_count
        )

        col_value = None
        proba_values = None
        subtitle = None
        col_scale = None
        cmin = None
        cmax = None

        # Classification Case
        if self._explainer._case == "classification":
            subcontrib = contributions[label_num]
            if self._explainer.y_pred is not None:
                col_value = self._explainer._classes[label_num]
            subtitle = f"Response: <b>{label_value}</b>"
            # predict proba Color scale
            if proba and self._explainer.proba_values is not None:
                proba_values = self._explainer.proba_values.iloc[:, [label_num]]
                # Proba subset:
                proba_values = proba_values.loc[list_ind, :]
                col_scale, cmin, cmax = tuning_colorscale(
                    self._style_dict["init_contrib_colorscale"], proba_values, keep_90_pct=True
                )
            elif self._explainer.y_pred is not None:
                pred_values = self._explainer.y_pred.iloc[:, [label_num]]
                # Prediction subset:
                pred_values = pred_values.loc[list_ind, :]
                col_scale, cmin, cmax = tuning_colorscale(
                    self._style_dict["init_contrib_colorscale"], pred_values, keep_90_pct=True
                )

        # Regression Case - color scale
        elif self._explainer._case == "regression":
            subcontrib = contributions
            if self._explainer.y_pred is not None:
                col_scale, cmin, cmax = tuning_colorscale(
                    self._style_dict["init_contrib_colorscale"], self._explainer.y_pred.loc[list_ind], keep_90_pct=True
                )

        # Subset
        if self._explainer.postprocessing_modifications:
            feature_values = self._explainer.x_contrib_plot.loc[list_ind, col_name]
        else:
            feature_values = self._explainer.x_init.loc[list_ind, col_name]

        if isinstance(col_name, list):
            for el in col_name:
                if feature_values[el].dtype == "bool":
                    feature_values[el] = feature_values[el].astype(int)
        else:
            if feature_values.dtype == "bool":
                feature_values = feature_values.astype(int)

        if col_is_group:
            feature_values = project_feature_values_1d(
                feature_values,
                col,
                self._explainer.x_init,
                self._explainer.x_encoded,
                self._explainer.preprocessing,
                features_dict=self._explainer.features_dict,
            )
            contrib = subcontrib.loc[list_ind, col].to_frame()
            if self._explainer.features_imp is None:
                self._explainer.compute_features_import()
            features_imp = (
                self._explainer.features_imp
                if isinstance(self._explainer.features_imp, pd.Series)
                else self._explainer.features_imp[0]
            )
            top_features_of_group = (
                features_imp.loc[self._explainer.features_groups[col]].sort_values(ascending=False)[:4].index
            )  # Displaying top 4 features
            metadata = {
                self._explainer.features_dict[f_name]: self._explainer.x_init[f_name]
                for f_name in top_features_of_group
            }
            text_group = "Features values were projected on the x axis using t-SNE"
            # if group don't show addnote, if not, it's too long
            # if addnote is not None:
            #    addnote = add_text([addnote, text_group], sep=' - ')
            # else:
            addnote = text_group
        else:
            contrib = subcontrib.loc[list_ind, col_name].to_frame()
            metadata = None
        feature_values = feature_values.to_frame()

        if self._explainer.y_pred is not None:
            y_pred = self._explainer.y_pred.loc[list_ind]
            # Add labels if exist
            if self._explainer._case == "classification" and self._explainer.label_dict is not None:
                y_pred = y_pred.map(lambda x: self._explainer.label_dict[x])
                col_value = self._explainer.label_dict[col_value]
            # round predict
            elif self._explainer._case == "regression":
                y_pred = y_pred.map(lambda x: round(x, self._round_digit))
        else:
            y_pred = None

        max_len_by_row = max([round(50 / self._explainer.features_desc[feature_values.columns.values[0]]), 8])

        # selecting the best plot : Scatter, Violin?
        if col_value_count > violin_maxf:
            fig = plot_scatter(
                feature_values,
                contrib,
                col_label,
                self._explainer._case,
                self._style_dict,
                y_pred,
                proba_values,
                col_value,
                col_scale,
                cmin,
                cmax,
                metadata,
                addnote,
                subtitle,
                max_len_by_row,
                width,
                height,
                file_name,
                auto_open,
                zoom,
            )
        else:
            fig = plot_violin(
                feature_values,
                contrib,
                col_label,
                self._explainer._case,
                self._style_dict,
                y_pred,
                proba_values,
                col_value,
                col_scale,
                cmin,
                cmax,
                addnote,
                subtitle,
                max_len_by_row,
                width,
                height,
                file_name,
                auto_open,
                zoom,
            )

        return fig

    def features_importance(
        self,
        mode="global",
        max_features=20,
        page="top",
        selection=None,
        label=-1,
        group_name=None,
        display_groups=True,
        force=False,
        width=900,
        height=500,
        file_name=None,
        auto_open=False,
        zoom=False,
        normalize_by_nb_samples=False,
        degree="slider",
    ):
        """
        Display a Plotly feature importance plot.

        This method generates a feature importance plot for both classification and regression models.
        For multiclass classification, the plot will focus on the specified `label`.

        Parameters
        ----------
        mode : str, optional, default: 'global'
            Defines the type of plot to display.
            - 'global': Displays the feature importance plot from a global perspective.
            - 'global-local': Shows the global feature importance plot with local importance indicators.
            - 'cumulative': Shows the cumulative sum of feature contributions, ordered by descending importance.
        max_features : int, optional, default: 20
            Limits the number of features to display in the plot.
            For example, `max_features=20` will display the 20 most important features.
        page : int or str, optional, default: 'top'
            Allows the user to select which set of features to display.
            - 'top': Shows the most important features.
            - 'worst': Shows the least important features.
            - Page number (integer) allows navigation between different sets of features.
        selection : list, optional, default: None
            Specifies a subset of features to compare to the global feature importance.
            This is only applicable when `mode` is set to 'global'. If provided, the list must contain
            indices corresponding to the subset of features to be displayed.
        label : int or str, optional, default: -1
            Specifies the label for which to display feature importance in multiclass classification.
            If a string label is provided, it will be converted to an integer if applicable.
        group_name : str, optional, default: None
            Displays feature importance for a specific group of features.
            This is only available if the `SmartExplainer` object has been compiled with feature groups.
            The group name must correspond to a key in the `features_groups` dictionary.
        display_groups : bool, optional, default: True
            If feature groups are declared in the `SmartExplainer` object, this parameter specifies
            whether or not to display them in the plot.
        force : bool, optional, default: False
            If `True`, forces recomputation of feature importance, even if it has already been computed.
        width : int, optional, default: 900
            The width of the Plotly figure layout.
        height : int, optional, default: 500
            The height of the Plotly figure layout.
        file_name : str, optional
            The name of the file to save the Plotly bar chart.
            If `None`, the chart will not be saved.
        auto_open : bool, optional
            If `True`, automatically opens the generated plot.
        zoom : bool, optional, default: False
            Indicates whether the graph is currently zoomed in.
        normalize_by_nb_samples : bool, optional, default: False
            Normalizes feature importance by the number of samples.
            This is only applicable when `mode` is set to 'cumulative'.
        degree : int, optional, default: 0
            Degree of adjustment to apply to the cumulative feature contributions curve.
            This is only applicable when `mode` is set to 'cumulative'.

        Returns
        -------
        plotly.graph_objs._figure.Figure
            The generated Plotly figure object containing the feature importance plot.

        Examples
        --------
        >>> xpl.plot.features_importance()
        """

        def get_feature_importance_page(features_importance, page, max_features):
            if isinstance(page, int):
                nb_features = len(features_importance)
                nb_page_max = nb_features // max_features + 1
                page = (page - 1) % nb_page_max + 1

            if (page == "top") or (page == 1):
                return features_importance.tail(max_features)
            elif page == "worst":
                return features_importance.head(max_features)
            elif isinstance(page, int):
                start_index = (page - 1) * max_features
                end_index = start_index + max_features
                return features_importance.iloc[-end_index:-start_index]
            else:
                raise ValueError("Invalid value for page. It must be 'top', 'worst', or an integer.")

        # Compute the feature importance based on mode
        self._explainer.compute_features_import(force=force, local=(mode == "global-local"))

        # Determine title based on the mode
        titles = {
            "global": "Feature Importance",
            "global-local": "Global and Local Feature Importance",
            "cumulative": "Cumulative Feature Contribution Curve",
        }
        title = titles.get(mode, "Feature Importance")

        # Check if feature groups should be displayed
        display_groups = self._explainer.features_groups is not None and display_groups

        # Handle feature groups and group-specific cases
        local_imp_lev1, local_imp_lev2 = None, None
        if display_groups:
            if group_name:  # Case where we have groups of features and we want to display only features inside a group
                if group_name not in self._explainer.features_groups.keys():
                    raise ValueError(
                        f"group_name parameter : {group_name} is not in features_groups keys. "
                        f"Possible values are : {list(self._explainer.features_groups.keys())}"
                    )
                title += f" - {truncate_str(self._explainer.features_dict.get(group_name), 20)}"
                if isinstance(self._explainer.features_imp, list):
                    features_importance = [
                        label_feat_imp.loc[label_feat_imp.index.isin(self._explainer.features_groups[group_name])]
                        for label_feat_imp in self._explainer.features_imp
                    ]
                    if mode == "global-local":
                        local_imp_lev1 = [
                            label_feat_imp.loc[label_feat_imp.index.isin(self._explainer.features_groups[group_name])]
                            for label_feat_imp in self._explainer.features_imp_local_lev1
                        ]
                        local_imp_lev2 = [
                            label_feat_imp.loc[label_feat_imp.index.isin(self._explainer.features_groups[group_name])]
                            for label_feat_imp in self._explainer.features_imp_local_lev2
                        ]
                else:
                    index = self._explainer.features_imp.index.isin(self._explainer.features_groups[group_name])
                    features_importance = self._explainer.features_imp.loc[index]
                    if mode == "global-local":
                        local_imp_lev1 = self._explainer.features_imp_local_lev1.loc[index]
                        local_imp_lev2 = self._explainer.features_imp_local_lev2.loc[index]
                contributions = self._explainer.contributions
            else:
                features_importance = self._explainer.features_imp_groups
                if mode == "global-local":
                    local_imp_lev1 = self._explainer.features_imp_groups_local_lev1
                    local_imp_lev2 = self._explainer.features_imp_groups_local_lev2
                contributions = self._explainer.contributions_groups
        else:
            features_importance = self._explainer.features_imp
            if mode == "global-local":
                local_imp_lev1 = self._explainer.features_imp_local_lev1
                local_imp_lev2 = self._explainer.features_imp_local_lev2
            contributions = self._explainer.contributions

        subtitle = ""

        # Classification case
        if self._explainer._case == "classification":
            label_num, _, label_value = self._explainer.check_label_name(label)
            features_importance_case = features_importance[label_num]
            contributions_case = contributions[label_num]
            subtitle = f"Response: <b>{label_value}</b>"

        # Regression case
        elif self._explainer._case == "regression":
            label_num = None
            features_importance_case = features_importance
            contributions_case = contributions
        else:
            raise ValueError("Invalid case. Case must be either 'classification' or 'regression'.")

        global_feat_imp = get_feature_importance_page(features_importance_case, page, max_features)

        if mode == "global-local":
            local_imp_lev1, local_imp_lev2 = self._get_local_feature_importance(
                global_feat_imp.index, local_imp_lev1, local_imp_lev2, label_num
            )
        subset_feat_imp = self._get_subset_importance(contributions_case, selection)
        if subset_feat_imp is not None:
            subset_feat_imp = subset_feat_imp.reindex(global_feat_imp.index)
            subset_feat_imp.index = subset_feat_imp.index.map(self._explainer.features_dict)
            if subset_feat_imp.dropna().shape[0] == 0:
                raise ValueError("selection argument doesn't return any row")

        addnote = self._build_additional_notes(subset_feat_imp, selection, max_features)

        features_groups_keys = None
        if self._explainer.features_groups is not None:
            features_groups_keys = self._explainer.features_groups.keys()

        # Generate and return the plot
        return plot_feature_importance(
            mode,
            global_feat_imp,
            contributions_case,
            self._style_dict,
            features_groups_keys,
            self._explainer.features_dict,
            self._explainer.inv_features_dict,
            local_imp_lev1,
            local_imp_lev2,
            subset_feat_imp,
            display_groups,
            title,
            addnote,
            subtitle,
            width,
            height,
            file_name,
            auto_open,
            zoom,
            normalize_by_nb_samples,
            degree,
        )

    def _get_group_feature_importance(self, group_name):
        """Retrieve the feature importance for a specific group of features."""
        if isinstance(self._explainer.features_imp, list):
            return [
                label_feat_imp.loc[label_feat_imp.index.isin(self._explainer.features_groups[group_name])]
                for label_feat_imp in self._explainer.features_imp
            ]
        return self._explainer.features_imp.loc[
            self._explainer.features_imp.index.isin(self._explainer.features_groups[group_name])
        ]

    def _get_local_feature_importance(self, indices, local_imp_lev1, local_imp_lev2, label_num=None):
        """Retrieve local feature importance for global-local mode."""
        if label_num is not None:
            local_imp_lev1 = local_imp_lev1[label_num]
            local_imp_lev2 = local_imp_lev2[label_num]

        local_imp_lev1 = local_imp_lev1.loc[indices]
        local_imp_lev2 = local_imp_lev2.loc[indices]

        return local_imp_lev1, local_imp_lev2

    def _get_subset_importance(self, contributions, selection):
        """Retrieve feature importance for a subset of features, if specified."""
        if selection is not None:
            return self._explainer.backend.get_global_features_importance(
                contributions=contributions, explain_data=self._explainer.explain_data, subset=selection
            )
        return None

    def _build_additional_notes(self, subset_feat_imp, selection, max_features):
        """Generate additional notes to display in the plot."""
        addnote = ""
        if subset_feat_imp is not None:
            subset_len = len(selection)
            total_len = self._explainer.x_init.shape[0]
            addnote = add_text(
                [addnote, f"Subset length: {subset_len} ({int(np.round(100 * subset_len / total_len))}%)"], sep=" - "
            )
        if self._explainer.x_init.shape[1] >= max_features:
            addnote = add_text(
                [addnote, f"Total number of features: {int(self._explainer.x_init.shape[1])}"], sep=" - "
            )
        return addnote

    def compare_plot(
        self,
        index=None,
        row_num=None,
        label=None,
        max_features=20,
        width=900,
        height=550,
        show_predict=True,
        file_name=None,
        auto_open=True,
    ):
        """
        Plotly comparison plot of several individuals' contributions. Plots contributions feature by feature.
        Allows to see the differences of contributions between two or more individuals,
        with each individual represented by a unique line.
        Parameters
        ----------
        index: list
            1st option to select individual rows.
            Int list of index referencing rows.
        row_num: list
            2nd option to select individual rows.
            int list corresponding to the row numbers of individuals (starting at 0).
        label: int or string (default: None)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        max_features: int (optional, default: 20)
            Number of contributions to show.
            If greater than the total of features, shows all.
        width: int (default: 900)
            Plotly figure - layout width.
        height: int (default: 550)
            Plotly figure - layout height.
        show_predict: boolean (default: True)
            Shows predict or predict_proba value.
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: boolean (optional)
            Indicates whether to open the bar plot or not.
        Returns
        -------
        Plotly Figure Object
            Comparison plot of the contributions of the different individuals.
        Example
        -------
        >>> xpl.plot.compare_plot(row_num=[0, 1, 2])
        """
        # Checking input is okay
        if sum(arg is not None for arg in [row_num, index]) != 1:
            raise ValueError("You have to specify just one of these arguments: index, row_num")
        # Getting indexes in a list
        line_reference = []
        if index is not None:
            for ident in index:
                if ident in self._explainer.x_init.index:
                    line_reference.append(ident)

        elif row_num is not None:
            line_reference = [
                self._explainer.x_init.index.values[row_nb_reference]
                for row_nb_reference in row_num
                if self._explainer.x_init.index.values[row_nb_reference] in self._explainer.x_init.index
            ]

        subtitle = ""
        if len(line_reference) < 1:
            raise ValueError("No matching entry for index")

        # Classification case
        if self._explainer._case == "classification":
            if label is None:
                label = -1

            label_num, _, label_value = self._explainer.check_label_name(label)
            contrib = self._explainer.contributions[label_num]

            if show_predict:
                preds = [self._explainer._local_pred(line, label_num) for line in line_reference]
                subtitle = (
                    f"Response: <b>{label_value}</b> - "
                    + "Probas: "
                    + " ; ".join(
                        [str(id) + ": <b>" + str(round(proba, 2)) + "</b>" for proba, id in zip(preds, line_reference)]
                    )
                )

        # Regression case
        elif self._explainer._case == "regression":
            contrib = self._explainer.contributions

            if show_predict:
                preds = [self._explainer._local_pred(line) for line in line_reference]
                subtitle = "Predictions: " + " ; ".join(
                    [str(id) + ": <b>" + str(round(pred, 2)) + "</b>" for id, pred in zip(line_reference, preds)]
                )

        new_contrib = list()
        for ident in line_reference:
            new_contrib.append(contrib.loc[ident])
        new_contrib = np.array(new_contrib).T

        # Well labels if available
        feature_values = [0] * len(contrib.columns)
        if hasattr(self._explainer, "columns_dict"):
            for i, name in enumerate(contrib.columns):
                feature_name = self._explainer.features_dict[name]
                feature_values[i] = feature_name

        preds = [self._explainer.x_init.loc[id] for id in line_reference]
        dict_features = self._explainer.inv_features_dict

        iteration_list = list(zip(new_contrib, feature_values))
        iteration_list.sort(key=lambda x: maximum_difference_sort_value(x), reverse=True)
        iteration_list = iteration_list[:max_features]
        iteration_list = iteration_list[::-1]
        new_contrib, feature_values = list(zip(*iteration_list))

        fig = plot_line_comparison(
            line_reference,
            feature_values,
            new_contrib,
            self._style_dict,
            predictions=preds,
            dict_features=dict_features,
            width=width,
            height=height,
            subtitle=subtitle,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def _select_indices_interactions_plot(self, selection, max_points):
        """
        Method used for sampling indices.
        Parameters
        ----------
        selection : list
            Contains list of index, subset of the input DataFrame that we want to plot
        max_points : int
            Maximum number to plot in contribution plot. if input dataset is bigger than max_points,
            a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        Returns
        -------
        list_ind : list
            List of indices to select
        addnote : str
            Text to inform the user the selection that has been done.
        """
        # Sampling
        addnote = None
        if selection is None:
            # interaction_selection attribute is used to store already computed indices of interaction_values
            if hasattr(self, "interaction_selection"):
                list_ind = self.interaction_selection
            elif self._explainer.x_init.shape[0] <= max_points:
                list_ind = self._explainer.x_init.index.tolist()
            else:
                list_ind = random.sample(self._explainer.x_init.index.tolist(), max_points)
                addnote = "Length of random Subset : "
        elif isinstance(selection, list):
            if len(selection) <= max_points:
                list_ind = selection
                addnote = "Length of user-defined Subset : "
            elif hasattr(self, "interaction_selection"):
                if set(selection).issubset(set(self.interaction_selection)):
                    list_ind = self.interaction_selection
            else:
                list_ind = random.sample(selection, max_points)
                addnote = "Length of random Subset : "
        else:
            raise ValueError("parameter selection must be a list")
        self.interaction_selection = list_ind

        return list_ind, addnote

    def interactions_plot(
        self,
        col1,
        col2,
        selection=None,
        violin_maxf=10,
        max_points=500,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
    ):
        """
        Diplays a Plotly scatter plot or violin plot of two selected features and their combined
        contributions for each of their values.
        This plot allows the user to understand how the different combinations of values of the
        two selected features influence the importance of the two features in the model output.
        A sample is taken if the number of points to be displayed is too large
        Parameters
        ----------
        col1: String or Int
            Name, label name or column number of the first column whose contributions we want to plot
        col2: String or Int
            Name, label name or column number of the second column whose contributions we want to plot
        selection: list (optional)
            Contains list of index, subset of the input DataFrame that we want to plot
        violin_maxf: int (optional, default: 10)
            maximum number modality to plot violin. If the feature specified with col argument
            has more modalities than violin_maxf, a scatter plot will be choose
        max_points: int (optional, default: 2000)
            maximum number of points to plot in contribution plot. if input dataset is bigger than
            max_points, a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the bar plot or not.
        Returns
        -------
        Plotly Figure Object
        Example
        --------
        >>> xpl.plot.interactions_plot(0, 1)
        """

        if not (isinstance(col1, (str, int)) or isinstance(col2, (str, int))):
            raise ValueError("parameters col1 and col2 must be string or int.")

        col_id1 = self._explainer.check_features_name([col1])[0]
        col_name1 = self._explainer.columns_dict[col_id1]

        col_id2 = self._explainer.check_features_name([col2])[0]
        col_name2 = self._explainer.columns_dict[col_id2]

        col_value_count1 = self._explainer.features_desc[col_name1]

        list_ind, addnote = self._select_indices_interactions_plot(selection=selection, max_points=max_points)

        if addnote is not None:
            addnote = add_text(
                [addnote, f"{len(list_ind)} ({int(np.round(100 * len(list_ind) / self._explainer.x_init.shape[0]))}%)"],
                sep="",
            )

        # Subset
        if self._explainer.postprocessing_modifications:
            feature_values1 = self._explainer.x_contrib_plot.loc[list_ind, col_name1].to_frame()
            feature_values2 = self._explainer.x_contrib_plot.loc[list_ind, col_name2].to_frame()
        else:
            feature_values1 = self._explainer.x_init.loc[list_ind, col_name1].to_frame()
            feature_values2 = self._explainer.x_init.loc[list_ind, col_name2].to_frame()

        interaction_values = self._explainer.get_interaction_values(selection=list_ind)[:, col_id1, col_id2]
        if col_id1 != col_id2:
            interaction_values = interaction_values * 2

        # add break line to X label if necessary
        max_len_by_row = max([round(50 / self._explainer.features_desc[feature_values1.columns.values[0]]), 8])
        args = (max_len_by_row, 120)
        feature_values_str = feature_values1.iloc[:, 0].apply(add_line_break, args=args)
        feature_values1 = pd.DataFrame({feature_values1.columns[0]: feature_values_str})

        # selecting the best plot : Scatter, Violin?
        if col_value_count1 > violin_maxf:
            fig = plot_interactions_scatter(
                x_name=col_name1,
                y_name="Shap interaction value",
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self._style_dict["interactions_col_scale"],
                style_dict=self._style_dict,
            )
        else:
            fig = plot_interactions_violin(
                x_name=col_name1,
                y_name="Shap interaction value",
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self._style_dict["interactions_col_scale"],
                style_dict=self._style_dict,
            )

        update_interactions_fig(
            fig=fig,
            col_name1=col_name1,
            col_name2=col_name2,
            addnote=addnote,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
            style_dict=self._style_dict,
        )

        return fig

    def top_interactions_plot(
        self,
        nb_top_interactions=5,
        selection=None,
        violin_maxf=10,
        max_points=500,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
    ):
        """
        Displays a dynamic plot with the `nb_top_interactions` most important interactions existing
        between two variables.
        The most important interactions are determined computing the sum of all absolute shap interactions
        values between all existing pairs of variables.
        A button allows to select and display the corresponding features values and their shap contribution values.
        Parameters
        ----------
        nb_top_interactions : int
            Number of top interactions to display.
        selection : list (optional)
            Contains list of index, subset of the input DataFrame that we want to plot
        violin_maxf : int (optional, default: 10)
            maximum number modality to plot violin. If the feature specified with col argument
            has more modalities than violin_maxf, a scatter plot will be choose
        max_points : int (optional, default: 500)
            maximum number to plot in contribution plot. if input dataset is bigger than max_points,
            a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the bar plot or not.
        Returns
        -------
        go.Figure
        Example
        --------
        >>> xpl.plot.top_interactions_plot()
        """

        list_ind, addnote = self._select_indices_interactions_plot(selection=selection, max_points=max_points)

        interaction_values = self._explainer.get_interaction_values(selection=list_ind)

        sorted_top_features_indices = compute_sorted_variables_interactions_list_indices(interaction_values)

        indices_to_plot = sorted_top_features_indices[:nb_top_interactions]
        interactions_indices_traces_mapping = []
        fig = go.Figure()
        for i, ids in enumerate(indices_to_plot):
            id0, id1 = ids

            fig_one_interaction = self.interactions_plot(
                col1=self._explainer.columns_dict[id0],
                col2=self._explainer.columns_dict[id1],
                selection=selection,
                violin_maxf=violin_maxf,
                max_points=max_points,
                width=width,
                height=height,
                file_name=None,
                auto_open=False,
            )

            # The number of traces of each figure is stored
            interactions_indices_traces_mapping.append(len(fig_one_interaction.data))

            for trace in fig_one_interaction.data:
                trace.visible = True if i == 0 else False
                fig.add_trace(trace=trace)

        def generate_title_dict(col_name1, col_name2, addnote):
            title = f"<b>{truncate_str(col_name1)} and {truncate_str(col_name2)}</b> shap interaction values"
            if addnote:
                title += f"<span style='font-size: 12px;'><br />{add_text([addnote], sep=' - ')}</span>"
            dict_t = self._style_dict["dict_title"] | {
                "text": title,
                "y": 0.88,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
            return dict_t

        fig.layout.coloraxis.colorscale = self._style_dict["interactions_col_scale"]
        updatemenus = [
            dict(
                active=0,
                buttons=list(
                    [
                        dict(
                            label=f"{self._explainer.columns_dict[i]} - {self._explainer.columns_dict[j]}",
                            method="update",
                            args=[
                                {
                                    "visible": [
                                        True if i == id_trace else False
                                        for i, x in enumerate(interactions_indices_traces_mapping)
                                        for _ in range(x)
                                    ]
                                },
                                {
                                    "xaxis": {
                                        "title": {
                                            **{"text": self._explainer.columns_dict[i]},
                                            **self._style_dict["dict_xaxis"],
                                        }
                                    },
                                    "legend": {"title": {"text": self._explainer.columns_dict[j]}},
                                    "coloraxis": {
                                        "colorbar": {"title": {"text": self._explainer.columns_dict[j]}},
                                        "colorscale": fig.layout.coloraxis.colorscale,
                                    },
                                    "title": generate_title_dict(
                                        self._explainer.columns_dict[i], self._explainer.columns_dict[j], addnote
                                    ),
                                },
                            ],
                        )
                        for id_trace, (i, j) in enumerate(indices_to_plot)
                    ]
                ),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=1.25,
                yanchor="top",
            )
        ]
        fig.update_layout(
            xaxis_title=self._explainer.columns_dict[sorted_top_features_indices[0][0]],
            yaxis_title="Shap interaction value",
            updatemenus=updatemenus,
            annotations=[
                dict(
                    text=f"Sorted top {len(indices_to_plot)} SHAP interaction Variables :",
                    x=0,
                    xref="paper",
                    y=1.2,
                    yref="paper",
                    align="left",
                    showarrow=False,
                )
            ],
        )

        update_interactions_fig(
            fig=fig,
            col_name1=self._explainer.columns_dict[sorted_top_features_indices[0][0]],
            col_name2=self._explainer.columns_dict[sorted_top_features_indices[0][1]],
            addnote=addnote,
            width=width,
            height=height,
            file_name=None,
            auto_open=False,
            style_dict=self._style_dict,
        )

        fig.update_layout(title={"y": 0.88, "x": 0.5, "xanchor": "center", "yanchor": "top"})

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def correlations_plot(
        self,
        df=None,
        optimized=False,
        max_features=20,
        features_to_hide=None,
        facet_col=None,
        how="phik",
        width=900,
        height=500,
        degree=2.5,
        decimals=2,
        file_name=None,
        auto_open=False,
    ):
        """
        Correlations matrix heatmap plot.
        The method can use phik or pearson correlations.
        The correlations computed can be changed using the parameter 'how'.
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame for which we want to compute correlations. Will use x_init by default.
        optimized : boolean, optional
            True if we want to potentially accelerate the computation of the correlation matrix by reducing the
            lenght of the data and the number of modalties per columns.
        max_features : int (default: 10)
            Max number of features to show on the matrix.
        features_to_hide : list (optional)
            List of features that will not appear on the graph
        facet_col : str (optional)
            Name of the column used to split the graph in two (or more) plots. One correlation
            subplot will be computed for each value of this column.
        how : str (default: 'phik')
            Correlation method used. 'phik' or 'pearson' are possible values. 'phik' is used by default.
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        degree  : int, optional, (default 2.5)
            degree applied on the correlation matrix in order to focus more or less the clustering
            on strong correlated variables
        decimals : int, optional, (default 2)
            number of decimals to plot for correlation values
        file_name: string (optional)
            File name to use to save the plotly bar chart. If None the bar chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the bar plot or not.
        Returns
        -------
        go.Figure
        Example
        --------
        >>> xpl.plot.correlations()
        """
        if df is None:
            df = self._explainer.x_init.copy()

        fig = plot_correlations(
            df=df,
            style_dict=self._style_dict,
            features_dict=self._explainer.features_dict,
            optimized=optimized,
            max_features=max_features,
            features_to_hide=features_to_hide,
            facet_col=facet_col,
            how=how,
            width=width,
            height=height,
            degree=degree,
            decimals=decimals,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def local_neighbors_plot(self, index, max_features=10, file_name=None, auto_open=False, height="auto", width=900):
        """
        The Local_neighbors_plot has the main objective of increasing confidence \
        in interpreting the contribution values of a selected instance.
        This plot analyzes the local neighborhood of the instance, \
        and compares its contribution values with those of its neighbors.
        Intuitively, for similar instances, we would expect similar contributions.
        Those neighbors are selected as follows :
        * We select top N neighbors for each instance (using L1 norm + variance normalization)
        * We discard neighbors whose model output is too **different** (see equations below)
        from the instance output
        * We discard additional neighbors if their distance to the instance \
        is bigger than a predefined value (to remove outliers)
        In this neighborhood, we would expect instances to have similar SHAP values. \
        If not, one might need to be cautious when interpreting SHAP values.
        The **difference** between outputs is measured with the following distance definition :
        * For regression:
        .. math::
            distance = \\frac{|output_{allFeatures} - output_{currentFeatures}|}{|output_{allFeatures}|}
        * For classification:
        .. math::
            distance = |output_{allFeatures} - output_{currentFeatures}|

        Parameters
        ----------
        index: int
            Contains index row of the input DataFrame that we use to display contribution values in the neighborhood
        max_features: int, optional
            Maximum number of displayed features, by default 10
        file_name: string, optional
            Specify the save path of html files. If it is not provided, no file will be saved, by default None
        auto_open: bool, optional
            open automatically the plot, by default False
        height : str or int, optional
            Height of the figure. Default is 'auto'.
        width : int, optional
            Width of the figure. Default is 900.

        Returns
        -------
        fig
            The figure that will be displayed
        """
        assert index in self._explainer.x_init.index, "index must exist in pandas dataframe"

        self._explainer.compute_features_stability([index])

        column_names = np.array([self._explainer.features_dict.get(x) for x in self._explainer.x_init.columns])

        def ordinal(n):
            return "%d%s" % (n, "tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10 :: 4])

        # Compute explanations for instance and neighbors
        g = self._explainer.local_neighbors["norm_shap"]

        # Reorder indices based on absolute values of the 1st row (i.e. the instance) in descending order
        inds = np.flip(np.abs(g[0, :]).argsort())
        g = g[:, inds]
        columns = [column_names[i] for i in inds]

        # Plot
        g_df = pd.DataFrame(g, columns=columns).T.rename(
            columns={
                **{0: "instance", 1: "closest neighbor"},
                **{i: ordinal(i) + " closest neighbor" for i in range(2, len(g))},
            }
        )

        # Keep only max_features
        if max_features is not None:
            g_df = g_df[:max_features]

        fig = go.Figure(
            data=[
                go.Bar(
                    name=g_df.iloc[::-1, ::-1].columns[i],
                    y=g_df.iloc[::-1, ::-1].index.tolist(),
                    x=g_df.iloc[::-1, ::-1].iloc[:, i],
                    marker_color=(
                        self._style_dict["dict_stability_bar_colors"][1]
                        if i == g_df.shape[1] - 1
                        else self._style_dict["dict_stability_bar_colors"][0]
                    ),
                    orientation="h",
                    opacity=np.clip(0.2 + i * (1 - 0.2) / (g_df.shape[1] - 1), 0.2, 1) if g_df.shape[1] > 1 else 1,
                )
                for i in range(g_df.shape[1])
            ]
        )

        if height == "auto":
            height = max(500, 11 * g_df.shape[0] * g_df.shape[1])
        title = f"<br>Comparing local explanations in a neighborhood - Id: <b>{index}</b>"
        title += "<br><sup>How similar are explanations for closeby neighbours?</sup>"
        dict_t = self._style_dict["dict_title_stability"] | {"text": title, "y": adjust_title_height(height)}
        dict_xaxis = self._style_dict["dict_xaxis"] | {"text": "Normalized contribution values"}
        dict_yaxis = self._style_dict["dict_yaxis"] | {"text": ""}

        fig.update_layout(
            template="none",
            autosize=False,
            width=width,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode="closest",
            barmode="group",
            height=height,
            legend={"traceorder": "reversed"},
            xaxis={"side": "bottom"},
            margin={"l": 150, "r": 20, "t": 95, "b": 70},
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name is not None:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def stability_plot(
        self,
        selection=None,
        max_points=500,
        force=False,
        max_features=10,
        distribution="none",
        file_name=None,
        auto_open=False,
        height="auto",
        width=900,
    ):
        """
        The Stability_plot has the main objective of increasing confidence in contribution values, \
        and helping determine if we can trust an explanation.
        The idea behind local stability is the following : if instances are very similar, \
        then one would expect the explanations to be similar as well.
        Therefore, locally stable explanations are an important factor that help \
        build trust around a particular explanation method.
        The generated graphs can take multiple forms, but they all analyze \
        the same two aspects: for each feature we look at Amplitude vs. Variability. \
        in order terms, how important the feature is on average vs. how the feature impact \
        changes in the instance neighborhood.
        The average importance of the feature is the average SHAP value of the feature acros all considered instances
        The neighborhood is defined as follows :
        * We select top N neighbors for each instance (using L1 norm + variance normalization)
        * We discard neighbors whose model output is too **different** (see equations below) from the instance output
        * We discard additional neighbors if their distance to the instance \
        is bigger than a predefined value (to remove outliers)
        The **difference** between outputs is measured with the following distance definition:
        * For regression:
        .. math::
            distance = \\frac{|output_{allFeatures} - output_{currentFeatures}|}{|output_{allFeatures}|}
        * For classification:
        .. math::
            distance = |output_{allFeatures} - output_{currentFeatures}|
        Parameters
        ----------
        selection: list
            Contains list of index, subset of the input DataFrame that we use for the compute of stability statistics
        max_points: int, optional
            Maximum number to plot in compacity plot, by default 500
        force: bool, optional
            force == True, force the compute of stability values, by default False
        distribution: str, optional
            Add distribution of variability for each feature, by default 'none'.
            The other values are 'boxplot' or 'violin' that specify the type of plot
        file_name: string, optional
            Specify the save path of html files. If it is not provided, no file will be saved, by default None
        auto_open: bool, optional
            open automatically the plot, by default False
        height: int or 'auto'
            Plotly figure - layout height
        width: int
            Plotly figure - layout width

        Returns
        -------
        If single instance:
            * plot -- Normalized contribution values of instance and neighbors
        If multiple instances:
            * if distribution == "none": Mean amplitude of each feature contribution vs. mean variability across neighbors
            * if distribution == "boxplot": Distribution of contributions of each feature in instances neighborhoods.
            Graph type is box plot
            * if distribution == "violin": Distribution of contributions of each feature in instances neighborhoods.
            Graph type is violin plot
        """
        # Sampling
        if selection is None:
            # By default, don't compute calculation if it has already been done
            if (self._explainer.features_stability is None) or self._last_stability_selection or force:
                list_ind = self._explainer.x_init.index.tolist()
                if self._explainer.x_init.shape[0] > max_points:
                    list_ind = random.sample(list_ind, max_points)
                self._explainer.compute_features_stability(list_ind)
            else:
                print("Computed values from previous call are used")
            self._last_stability_selection = False
        elif isinstance(selection, list):
            if len(selection) == 1:
                raise ValueError("Selection must include multiple points")
            if len(selection) > max_points:
                print(
                    f"Size of selection is bigger than max_points (default: {max_points}). \
                    Computation time might be affected"
                )
            self._explainer.compute_features_stability(selection)
            self._last_stability_selection = True
        else:
            raise ValueError("Parameter selection must be a list")

        column_names = np.array([self._explainer.features_dict.get(x) for x in self._explainer.x_init.columns])

        variability = self._explainer.features_stability["variability"]
        amplitude = self._explainer.features_stability["amplitude"]

        mean_variability = variability.mean(axis=0)
        mean_amplitude = amplitude.mean(axis=0)

        # Plot 1 : only show average variability on y-axis
        if distribution not in ["boxplot", "violin"]:
            fig = plot_amplitude_vs_stability(
                mean_variability,
                mean_amplitude,
                column_names,
                file_name,
                auto_open,
                self._style_dict["init_contrib_colorscale"],
                self._style_dict,
                height=height,
                width=width,
            )

        # Plot 2 : Show distribution of variability
        else:
            # If set, only keep features with the highest mean amplitude
            if max_features is not None:
                keep = mean_amplitude.argsort()[::-1][:max_features]
                keep = np.sort(keep)

                variability = variability[:, keep]
                mean_amplitude = mean_amplitude[keep]
                dataset = self._explainer.x_init.iloc[:, keep]
                column_names = column_names[keep]

            fig = plot_stability_distribution(
                variability,
                distribution,
                mean_amplitude,
                dataset,
                column_names,
                file_name,
                auto_open,
                self._style_dict["init_contrib_colorscale"],
                self._style_dict,
                height=height,
                width=width,
            )

        return fig

    def compacity_plot(
        self,
        selection=None,
        max_points=2000,
        force=False,
        approx=0.9,
        nb_features=5,
        file_name=None,
        auto_open=False,
        height=600,
        width=900,
    ):
        """
        The Compacity_plot has the main objective of determining if a small subset of features
        can be extracted to provide a simpler explanation of the model.
        indeed, having too many features might negatively affect the model explainability and make it harder to undersand.
        The following two plots are proposed:
        * We identify the minimum number of required features (based on the top contribution values)
        that well approximate the model, and thus, provide accurate explanations.
        In particular, the prediction with the chosen subset needs to be close enough (*see distance definition below*)
        to the one obtained with all features.
        * Conversely, we determine how close we get to the output with all features by using only a subset of them.
        *Distance definition*
        * For regression:
        .. math::
            distance = \\frac{|output_{allFeatures} - output_{currentFeatures}|}{|output_{allFeatures}|}
        * For classification:
        .. math::
            distance = |output_{allFeatures} - output_{currentFeatures}|
        Parameters
        ----------
        selection: list
            Contains list of index, subset of the input DataFrame that we use for the compute of stability statistics
        max_points: int, optional
            Maximum number to plot in compacity plot, by default 2000
        force: bool, optional
            force == True, force the compute of stability values, by default False
        approx: float, optional
            How close we want to be from model with all features, by default 0.9 (=90%)
        nb_features: int, optional
            Number of features used, by default 5
        file_name: string, optional
            Specify the save path of html files. If it is not provided, no file will be saved, by default None
        auto_open: bool, optional
            open automatically the plot, by default False
        height: int, optional
            height of the plot, by default 600
        width:  int, optional
            width of the plot, by default 900
        """
        # Sampling
        if selection is None:
            if self._explainer.x_init.shape[0] <= max_points:
                list_ind = self._explainer.x_init.index.tolist()
            else:
                list_ind = random.sample(self._explainer.x_init.index.tolist(), max_points)
            # By default, don't compute calculation if it has already been done
            if (self._explainer.features_compacity is None) or self.last_compacity_selection or force:
                self._explainer.compute_features_compacity(list_ind, 1 - approx, nb_features)
            else:
                print("Computed values from previous call are used")
            self.last_compacity_selection = False
        elif isinstance(selection, list):
            if len(selection) > max_points:
                print(
                    f"Size of selection is bigger than max_points (default: {max_points}). \
                    Computation time might be affected"
                )
            self._explainer.compute_features_compacity(selection, 1 - approx, nb_features)
            self._last_compacity_selection = True
        else:
            raise ValueError("Parameter selection must be a list")

        # Data Processing
        features_needed = self._explainer.features_compacity["features_needed"]
        distance_reached = self._explainer.features_compacity["distance_reached"]

        # Plot generation
        fig = plot_compacity(
            features_needed,
            distance_reached,
            self._style_dict,
            approx,
            nb_features,
            file_name,
            auto_open,
            height,
            width,
        )

        return fig

    def scatter_plot_prediction(
        self,
        selection=None,
        label=-1,
        max_points=2000,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
    ):
        """
        scatter_plot_prediction displays a Plotly scatter or violin plot of predictions in comparison to the target variable.
        This plot represents Trues Values versus Predicted Values.

        This plot allows the user to understand the distribution of predictions in comparison to the target variable.
        With the web app, it is possible to select the wrong or correct predictions or a subset of predictions.

        Parameters
        ----------
        selection: list (optional)
            Contains list of index, subset of the input DataFrame that we want to plot
        label: integer or string (default -1)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        max_points: int (optional, default: 2000)
            maximum number to plot in contribution plot. if input dataset is bigger than max_points,
            a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        """

        # Classification Case
        if self._explainer._case == "classification":
            label_num, _, label_value = self._explainer.check_label_name(label)
            y_pred = self._explainer.y_pred
            y_proba_values = self._explainer.proba_values.copy()
        # Regression Case
        elif self._explainer._case == "regression":
            label_num, label_value = None, None
            y_pred = self._explainer.y_pred
            y_proba_values = None

        fig = plot_scatter_prediction(
            x_data=self._explainer.x_init,
            y_pred=y_pred,
            y_proba_values=y_proba_values,
            y_target=self._explainer.y_target,
            prediction_error=self._explainer.prediction_error,
            case=self._explainer._case,
            style_dict=self._style_dict,
            round_digit=self._round_digit,
            label_dict=self._explainer.label_dict,
            selection=selection,
            label_num=label_num,
            label_value=label_value,
            max_points=max_points,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def confusion_matrix_plot(
        self,
        width: int = 700,
        height: int = 500,
        file_name=None,
        auto_open=False,
    ):
        """
        Returns a matplotlib figure containing a confusion matrix that is computed using y_true and
        y_pred parameters.

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated targets as returned by a classifier.
        colors_dict : dict
            dict of colors used
        width : int, optional, default=7
            The width of the generated figure, in inches.
        height : int, optional, default=4
            The height of the generated figure, in inches.

        Returns
        -------
        matplotlib.pyplot.Figure
        """

        # Classification Case
        if self._explainer._case == "classification":
            y_true = self._explainer.y_target.iloc[:, 0]
            y_pred = self._explainer.y_pred.iloc[:, 0]
            if self._explainer.label_dict is not None:
                y_true = y_true.map(self._explainer.label_dict)
                y_pred = y_pred.map(self._explainer.label_dict)
        # Regression Case
        elif self._explainer._case == "regression":
            raise (ValueError("Confusion matrix is only available for classification case"))

        return plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            colors_dict=self._style_dict,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
        )

    def distribution_plot(
        self,
        col: str,
        hue: Optional[str] = None,
        width: int = 700,
        height: int = 500,
        nb_cat_max: int = 7,
        nb_hue_max: int = 7,
        file_name=None,
        auto_open=False,
    ) -> go.Figure:
        """
        Generate a Plotly figure displaying the univariate distribution of a feature
        (continuous or categorical) in the dataset.

        For categorical features with too many unique categories, the least frequent
        categories are grouped into a new 'Other' category to ensure the plot remains
        readable. Continuous features are visualized using KDE plots.

        The input DataFrame must contain the column of interest (`col`) and a second column
        (`hue`) used to distinguish between two groups (e.g., 'train' and 'test').

        Parameters
        ----------
        col : str
            The name of the column of interest whose distribution is to be visualized.
        hue : Optional[str], optional
            The name of the column used to differentiate between groups.
        width : int, optional, default=700
            The width of the generated figure, in pixels.
        height : int, optional, default=500
            The height of the generated figure, in pixels.
        nb_cat_max : int, optional, default=7
            Maximum number of categories to display. Categories beyond this limit
            are grouped into a new 'Other' category (only for categorical features).
        nb_hue_max : int, optional, default=7
            Maximum number of hue categories to display. Categories beyond this limit
            are grouped into a new 'Other' category.
        file_name : str, optional
            Path to save the plot as an HTML file. If None, the plot will not be saved, by default None.
        auto_open : bool, optional
            If True, the plot will automatically open in a web browser after being generated, by default False.

        Returns
        -------
        go.Figure
            A Plotly figure object representing the distribution of the feature.
        """
        if self._explainer.y_target is not None:
            data = pd.concat([self._explainer.x_init, self._explainer.y_target], axis=1)
        else:
            data = self._explainer.x_init

        return plot_distribution(
            data,
            col,
            hue=hue,
            colors_dict=self._style_dict,
            width=width,
            height=height,
            nb_cat_max=nb_cat_max,
            nb_hue_max=nb_hue_max,
            file_name=file_name,
            auto_open=auto_open,
        )
