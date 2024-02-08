"""
Smart plotter module
"""
import copy
import math
import random
import warnings
from numbers import Number

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as sch
from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from shapash.manipulation.select_lines import select_lines
from shapash.manipulation.summarize import compute_corr, project_feature_values_1d
from shapash.style.style_utils import colors_loading, define_style, select_palette
from shapash.utils.utils import (
    add_line_break,
    add_text,
    compute_digit_number,
    compute_sorted_variables_interactions_list_indices,
    compute_top_correlations_features,
    maximum_difference_sort_value,
    truncate_str,
)
from shapash.webapp.utils.utils import round_to_k


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

    def __init__(self, explainer):
        self.explainer = explainer
        self._palette_name = list(colors_loading().keys())[0]
        self._style_dict = define_style(select_palette(colors_loading(), self._palette_name))
        self.round_digit = None
        self.last_stability_selection = False
        self.last_compacity_selection = False

    def define_style_attributes(self, colors_dict):
        """
        define_style_attributes allows shapash user to change the color of plot
        Parameters
        ----------
        colors_dict: dict
            Dict of the colors used in the different plots
        """
        self._style_dict = define_style(colors_dict)

        if hasattr(self, "pred_colorscale"):
            delattr(self, "pred_colorscale")

    def tuning_colorscale(self, values):
        """
        adapts the color scale to the distribution of points
        Parameters
        ----------
        values: 1 column pd.DataFrame
            values ​​whose quantiles must be calculated
        """
        desc_df = values.describe(percentiles=np.arange(0.1, 1, 0.1).tolist())
        min_pred, max_init = list(desc_df.loc[["min", "max"]].values)
        desc_pct_df = (desc_df.loc[~desc_df.index.isin(["count", "mean", "std"])] - min_pred) / (max_init - min_pred)
        color_scale = list(map(list, (zip(desc_pct_df.values.flatten(), self._style_dict["init_contrib_colorscale"]))))
        return color_scale

    def tuning_round_digit(self):
        """
        adapts the display of the number of digit to the distribution of points
        """
        quantile = [0.25, 0.75]
        desc_df = self.explainer.y_pred.describe(percentiles=quantile)
        perc1, perc2 = list(desc_df.loc[[str(int(p * 100)) + "%" for p in quantile]].values)
        p_diff = perc2 - perc1
        self.round_digit = compute_digit_number(p_diff)

    def _update_contributions_fig(
        self,
        fig,
        feature_name,
        pred,
        proba_values,
        col_modality,
        col_scale,
        addnote,
        subtitle,
        width,
        height,
        file_name,
        auto_open,
    ):
        """
        Function used by both violin and scatter methods for contributions plots in order to
        update the layout of the (already) created plotly figure.
        Parameters
        ----------
        fig : go.Figure
            Plotly figure to be modified.
        feature_name : String
            Name of the feature, used in title
        pred: 1 column pd.DataFrame (optional)
            predicted values used to color plot - One Vs All in multiclass case
        proba_values: 1 column pd.DataFrame (optional)
            predicted proba used to color points - One Vs All in multiclass case
        col_modality: Int, Float or String (optional)
            parameter used in classification case,
            specify the modality to color in scatter plot (One Vs All)
        col_scale: list (optional)
            specify the color of points in scatter data
        addnote : String (default: None)
            Specify a note to display
        subtitle : String (default: None)
            Subtitle to display
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        """
        title = f"<b>{truncate_str(feature_name)}</b> - Feature Contribution"
        # Add subtitle and / or addnote
        if subtitle or addnote:
            # title += f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
            if subtitle and addnote:
                title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
            elif subtitle:
                title += "<br><sup>" + subtitle + "</sup>"
            else:
                title += "<br><sup>" + addnote + "</sup>"
        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_t["text"] = title
        dict_xaxis["text"] = truncate_str(feature_name, 110)
        dict_yaxis["text"] = "Contribution"

        if self.explainer._case == "regression":
            colorpoints = pred
            colorbar_title = "Predicted"
        elif self.explainer._case == "classification":
            colorpoints = proba_values
            colorbar_title = "Predicted Proba"

        if colorpoints is not None:
            fig.data[-1].marker.color = colorpoints.values.flatten()
            fig.data[-1].marker.coloraxis = "coloraxis"
            fig.layout.coloraxis.colorscale = col_scale
            fig.layout.coloraxis.colorbar = {"title": {"text": colorbar_title}}

        elif fig.data[0].type != "violin":
            if self.explainer._case == "classification" and pred is not None:
                fig.data[-1].marker.color = pred.iloc[:, 0].apply(
                    lambda x: self._style_dict["violin_area_classif"][1]
                    if x == col_modality
                    else self._style_dict["violin_area_classif"][0]
                )
            else:
                fig.data[-1].marker.color = self._style_dict["violin_default"]

        fig.update_traces(marker={"size": 10, "opacity": 0.8, "line": {"width": 0.8, "color": "white"}})

        fig.update_layout(
            template="none",
            title=dict_t,
            width=width,
            height=height,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode="closest",
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

    def plot_scatter(
        self,
        feature_values,
        contributions,
        feature_name,
        pred=None,
        proba_values=None,
        col_modality=None,
        col_scale=None,
        metadata=None,
        addnote=None,
        subtitle=None,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        Scatter plot of one feature contribution across the prediction set.
        Parameters
        ----------
        feature_values : 1 column pd.Dataframe
            The values of one feature
        contributions : 1 column pd.Dataframe
            The contributions associate
        feature_name : String
            Name of the feature, used in title
        pred: 1 column pd.DataFrame (optional)
            predicted values used to color plot - One Vs All in multiclass case
        proba_values: 1 column pd.DataFrame (optional)
            predicted proba used to color points - One Vs All in multiclass case
        col_modality: Int, Float or String (optional)
            parameter used in classification case,
            specify the modality to color in scatter plot (One Vs All)
        col_scale: list (optional)
            specify the color of points in scatter data
        addnote : String (default: None)
            Specify a note to display
        subtitle : String (default: None)
            Subtitle to display
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        zoom: bool (default=False)
            graph is currently zoomed
        """
        fig = go.Figure()

        # add break line to X label if necessary
        max_len_by_row = max([round(50 / self.explainer.features_desc[feature_values.columns.values[0]]), 8])
        feature_values.iloc[:, 0] = feature_values.iloc[:, 0].apply(
            add_line_break,
            args=(
                max_len_by_row,
                120,
            ),
        )

        if pred is not None:
            hv_text = [f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten())]
        else:
            hv_text = [f"Id: {x}" for x in feature_values.index]

        if metadata:
            metadata = {k: [round_to_k(x, 3) if isinstance(x, Number) else x for x in v] for k, v in metadata.items()}
            text_groups_features = np.swap = np.array([col_values for col_values in metadata.values()])
            text_groups_features = np.swapaxes(text_groups_features, 0, 1)
            text_groups_features_keys = list(metadata.keys())
            hovertemplate = (
                "<b>%{hovertext}</b><br />"
                + "Contribution: %{y:.4f} <br />"
                + "<br />".join(
                    [
                        "{}: %{{text[{}]}}".format(text_groups_features_keys[i], i)
                        for i in range(len(text_groups_features_keys))
                    ]
                )
                + "<extra></extra>"
            )
        else:
            hovertemplate = (
                "<b>%{hovertext}</b><br />"
                + f"{feature_name}: "
                + "%{customdata[0]}<br />Contribution: %{y:.4f}<extra></extra>"
            )
            text_groups_features = None

        fig.add_scatter(
            x=feature_values.values.flatten(),
            y=contributions.values.flatten(),
            mode="markers",
            hovertext=hv_text,
            hovertemplate=hovertemplate,
            text=text_groups_features,
        )
        # To change ticktext when the x label size is upper than 10 and zoom is False
        if (type(feature_values.values.flatten()[0]) == str) & (not zoom):
            feature_val = [x.replace("<br />", "") for x in feature_values.values.flatten()]
            feature_val = [x.replace(x[3 : len(x) - 3], "...") if len(x) > 10 else x for x in feature_val]

            fig.update_xaxes(
                tickangle=45, ticktext=feature_val, tickvals=feature_values.values.flatten(), tickmode="array", dtick=1
            )
        # Customdata contains the values and index of feature_values.
        # The values are used in the hovertext and the indexes are used for
        # the interactions between the graphics.
        customdata = np.stack((feature_values.values.flatten(), feature_values.index.values), axis=-1)

        fig.update_traces(customdata=customdata, hovertemplate=hovertemplate)

        self._update_contributions_fig(
            fig=fig,
            feature_name=feature_name,
            pred=pred,
            proba_values=proba_values,
            col_modality=col_modality,
            col_scale=col_scale,
            addnote=addnote,
            subtitle=subtitle,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def plot_violin(
        self,
        feature_values,
        contributions,
        feature_name,
        pred=None,
        proba_values=None,
        col_modality=None,
        col_scale=None,
        addnote=None,
        subtitle=None,
        width=900,
        height=600,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        Violin plot of one feature contribution across the prediction set.
        Parameters
        ----------
        feature_values : 1 column pd.Dataframe
            The values of one feature
        contributions : 1 column pd.Dataframe
            The contributions associate
        feature_name : String
            Name of the feature, used in title
        pred: 1 column pd.DataFrame (optional)
            predicted values used to color plot - One Vs All in multiclass case
        proba_values: 1 column pd.DataFrame (optional)
            predicted proba used to color points - One Vs All in multiclass case
        col_modality: Int, Float or String (optional)
            parameter used in classification case,
            specify the modality to color in scatter plot (One Vs All)
        col_scale: list (optional)
            specify the color of points in scatter data
        addnote : String (default: None)
            Specify a note to display
        subtitle : String (default: None)
            Subtitle to display
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 600)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        zoom: bool (default=False)
            graph is currently zoomed
        """
        fig = go.Figure()

        points_param = False if proba_values is not None else "all"
        jitter_param = 0.075

        if pred is not None:
            hv_text = [f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten())]
        else:
            hv_text = [f"Id: {x}" for x in feature_values.index]
        hv_text_df = pd.DataFrame(hv_text, columns=["text"], index=feature_values.index)
        hv_temp = f"{feature_name} :<br />" + "%{customdata[0]}<br />Contribution: %{y:.4f}<extra></extra>"
        # add break line to X label
        max_len_by_row = max([round(50 / self.explainer.features_desc[feature_values.columns.values[0]]), 8])
        feature_values.iloc[:, 0] = feature_values.iloc[:, 0].apply(
            add_line_break,
            args=(
                max_len_by_row,
                120,
            ),
        )

        uniq_l = list(pd.unique(feature_values.values.flatten()))
        uniq_l.sort()

        for i in uniq_l:
            if pred is not None and self.explainer._case == "classification":
                contribution_neg = contributions.loc[
                    (pred.iloc[:, 0] != col_modality) & (feature_values.iloc[:, 0] == i)
                ].values.flatten()
                # Check if contribution is not empty
                if len(contribution_neg) != 0:
                    fig.add_trace(
                        go.Violin(
                            x=feature_values.loc[
                                (pred.iloc[:, 0] != col_modality) & (feature_values.iloc[:, 0] == i)
                            ].values.flatten(),
                            y=contribution_neg,
                            points=points_param,
                            pointpos=-0.1,
                            side="negative",
                            line_color=self._style_dict["violin_area_classif"][0],
                            showlegend=False,
                            jitter=jitter_param,
                            meanline_visible=True,
                            hovertext=hv_text_df.loc[
                                (pred.iloc[:, 0] != col_modality) & (feature_values.iloc[:, 0] == i)
                            ].values.flatten(),
                        )
                    )

                contribution_pos = contributions.loc[
                    (pred.iloc[:, 0] == col_modality) & (feature_values.iloc[:, 0] == i)
                ].values.flatten()
                if len(contribution_pos) != 0:
                    fig.add_trace(
                        go.Violin(
                            x=feature_values.loc[
                                (pred.iloc[:, 0] == col_modality) & (feature_values.iloc[:, 0] == i)
                            ].values.flatten(),
                            y=contribution_pos,
                            points=points_param,
                            pointpos=0.1,
                            side="positive",
                            line_color=self._style_dict["violin_area_classif"][1],
                            showlegend=False,
                            jitter=jitter_param,
                            meanline_visible=True,
                            scalemode="count",
                            hovertext=hv_text_df.loc[
                                (pred.iloc[:, 0] == col_modality) & (feature_values.iloc[:, 0] == i)
                            ].values.flatten(),
                        )
                    )

            else:
                feature = feature_values.loc[feature_values.iloc[:, 0] == i].values.flatten()
                fig.add_trace(
                    go.Violin(
                        x=feature,
                        y=contributions.loc[feature_values.iloc[:, 0] == i].values.flatten(),
                        line_color=self._style_dict["violin_default"],
                        showlegend=False,
                        meanline_visible=True,
                        scalemode="count",
                        hovertext=hv_text_df.loc[feature_values.iloc[:, 0] == i].values.flatten(),
                    )
                )
                if pred is None:
                    fig.data[-1].points = points_param
                    fig.data[-1].pointpos = 0
                    fig.data[-1].jitter = jitter_param

        colorpoints = (
            pred
            if self.explainer._case == "regression"
            else proba_values
            if self.explainer._case == "classification"
            else None
        )

        hovertemplate = "<b>%{hovertext}</b><br />" + hv_temp
        feature = feature_values.values.flatten()
        customdata = np.stack((feature_values.values.flatten(), contributions.index.values), axis=-1)

        if colorpoints is not None:
            fig.add_trace(
                go.Scatter(
                    x=feature_values.values.flatten(),
                    y=contributions.values.flatten(),
                    mode="markers",
                    showlegend=False,
                    hovertext=hv_text,
                    hovertemplate=hovertemplate,
                )
            )

        fig.update_layout(violingap=0.05, violingroupgap=0, violinmode="overlay", xaxis_type="category")

        # To change ticktext when the x label size is upper than 10 and zoom is False
        if (type(feature[0]) == str) & (not zoom):
            feature_val = [x.replace("<br />", "") for x in np.unique(feature_values.values.flatten())]
            feature_val = [x.replace(x[3 : len(x) - 3], "...") if len(x) > 10 else x for x in feature_val]
            fig.update_xaxes(
                tickangle=45,
                ticktext=feature_val,
                tickvals=np.unique(feature_values.values.flatten()),
                tickmode="array",
                dtick=1,
                range=[-0.6, len(uniq_l) - 0.4],
            )
        else:
            fig.update_xaxes(range=[-0.6, len(uniq_l) - 0.4])

        # Update customdata and hovertemplate
        fig.update_traces(customdata=customdata, hovertemplate=hovertemplate)

        self._update_contributions_fig(
            fig=fig,
            feature_name=feature_name,
            pred=pred,
            proba_values=proba_values,
            col_modality=col_modality,
            col_scale=col_scale,
            addnote=addnote,
            subtitle=subtitle,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def plot_features_import(
        self,
        feature_imp1,
        feature_imp2=None,
        title="Features Importance",
        addnote=None,
        subtitle=None,
        width=900,
        height=500,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        Plot features importance computed with the prediction set.
        Parameters
        ----------
        feature_imp1 : pd.Series
            Feature importance computed with every rows
        feature_imp2 : pd.Series, optional (default: None)
            The contributions associate
        title : str
            Title of the plot, default set to 'Features Importance'
        addnote : String (default: None)
            Specify a note to display
        subtitle : String (default: None)
            Subtitle to display
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 500)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        zoom: bool (default=False)
            graph is currently zoomed
        """
        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        topmargin = 80
        # Add subtitle and / or addnote
        if subtitle or addnote:
            # title += f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
            if subtitle and addnote:
                title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
            elif subtitle:
                title += "<br><sup>" + subtitle + "</sup>"
            else:
                title += "<br><sup>" + addnote + "</sup>"
            topmargin = topmargin + 15
        dict_t.update(text=title)
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_xaxis.update(text="Mean absolute Contribution")
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_yaxis.update(text=None)
        dict_style_bar1 = self._style_dict["dict_featimp_colors"][1]
        dict_style_bar2 = self._style_dict["dict_featimp_colors"][2]
        dict_yaxis["text"] = None

        # Change bar color for groups of features
        marker_color = [
            self._style_dict["featureimp_groups"][0]
            if (
                self.explainer.features_groups is not None
                and self.explainer.inv_features_dict.get(f.replace("<b>", "").replace("</b>", ""))
                in self.explainer.features_groups.keys()
            )
            else dict_style_bar1["color"]
            for f in feature_imp1.index
        ]

        layout = go.Layout(
            barmode="group",
            template="none",
            autosize=False,
            width=width,
            height=height,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode="closest",
            margin={"l": 160, "r": 0, "t": topmargin, "b": 50},
        )
        # To change ticktext when the x label size is upper than 30 and zoom is False
        if (type(feature_imp1.index[0]) == str) & (not zoom):
            # change index to abc...abc if its length is upper than 30
            index_val = [y.replace(y[24 : len(y) - 3], "...") if len(y) > 30 else y for y in feature_imp1.index]
        else:
            index_val = feature_imp1.index
        bar1 = go.Bar(
            x=feature_imp1.round(4),
            y=feature_imp1.index,
            orientation="h",
            name="Global",
            marker=dict_style_bar1,
            marker_color=marker_color,
            hovertemplate="Feature: %{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
            customdata=feature_imp1.index,
        )
        if feature_imp2 is not None:
            bar2 = go.Bar(
                x=feature_imp2.round(4),
                y=feature_imp2.index,
                orientation="h",
                name="Subset",
                marker=dict_style_bar2,
                hovertemplate="Feature: %{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
                customdata=feature_imp2.index,
            )
            data = [bar2, bar1]
        else:
            data = bar1

        fig = go.Figure(data=data, layout=layout)
        # Update ticktext
        fig.update_yaxes(ticktext=index_val, tickvals=feature_imp1.index, tickmode="array", dtick=1)
        fig.update_yaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
        return fig

    def plot_bar_chart(
        self,
        index_value,
        var_dict,
        x_val,
        contrib,
        yaxis_max_label=12,
        subtitle=None,
        width=900,
        height=550,
        file_name=None,
        auto_open=False,
        zoom=False,
    ):
        """
        Plotly bar plot of local explainers
        Parameters
        ----------
        index_value:
            the index of row, used in title of local contribution plot
        var_dict: numpy array
            Unidimensional numpy array containing the features names for the observation of interest.
        x_val: numpy array
            Unidimensional numpy array containing the features values for the observation of interest.
        contrib: numpy array
            Unidimensional numpy array containing the contribution value for the observation of interest.
        yaxis_max_label: int (default: 12)
            Maximum number of variables to display labels on the y axis
        subtitle: string (default: None)
            subtitle to display
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 550)
            Plotly figure - layout height
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        zoom: bool (default=False)
            graph is currently zoomed
        Returns
        -------
        plotly bar plot
            A bar plot with selected contributions and
            associated feature values for one observation.
        """
        if len(index_value) != 0:
            dict_t = copy.deepcopy(self._style_dict["dict_title"])
            topmargin = 80
            dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
            dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
            dict_local_plot_colors = copy.deepcopy(self._style_dict["dict_local_plot_colors"])
            title = f"Local Explanation - Id: <b>{index_value[0]}</b>"
            # Add subtitle
            if subtitle:
                title += "<br><sup>" + subtitle + "</sup>"
                topmargin += 15
            dict_t["text"] = title
            dict_xaxis["text"] = "Contribution"
            dict_yaxis["text"] = None

            layout = go.Layout(
                barmode="group",
                template="none",
                width=width,
                height=height,
                title=dict_t,
                xaxis_title=dict_xaxis,
                yaxis_title=dict_yaxis,
                yaxis_type="category",
                hovermode="closest",
                margin={"l": 150, "r": 20, "t": topmargin, "b": 70},
            )
            bars = []
            for num, expl in enumerate(list(zip(var_dict, x_val, contrib))):
                group_name = None
                if expl[1] == "":
                    ylabel = "<i>{}</i>".format(expl[0])
                    hoverlabel = "<b>{}</b>".format(expl[0])
                else:
                    # If bar is a group of features, hovertext includes the values of the features of the group
                    # And color changes
                    if (
                        self.explainer.features_groups is not None
                        and self.explainer.inv_features_dict.get(expl[0]) in self.explainer.features_groups.keys()
                        and len(index_value) > 0
                    ):
                        group_name = self.explainer.inv_features_dict.get(expl[0])
                        feat_groups_values = self.explainer.x_init[self.explainer.features_groups[group_name]].loc[
                            index_value[0]
                        ]
                        hoverlabel = "<br />".join(
                            [
                                "<b>{} :</b>{}".format(
                                    add_line_break(self.explainer.features_dict.get(f_name, f_name), 40, maxlen=120),
                                    add_line_break(f_value, 40, maxlen=160),
                                )
                                for f_name, f_value in feat_groups_values.to_dict().items()
                            ]
                        )
                    else:
                        hoverlabel = "<b>{} :</b><br />{}".format(
                            add_line_break(expl[0], 40, maxlen=120), add_line_break(expl[1], 40, maxlen=160)
                        )
                    trunc_value = truncate_str(expl[0], 45)
                    if not zoom:
                        # Truncate value if length is upper than 30
                        trunc_new_value = (
                            trunc_value.replace(trunc_value[24 : len(trunc_value) - 3], "...")
                            if len(trunc_value) > 30
                            else trunc_value
                        )
                    else:
                        trunc_new_value = trunc_value
                    if len(contrib) <= yaxis_max_label and (
                        self.explainer.features_groups is None
                        # We don't want to display label values for t-sne projected values of groups of features.
                        or (
                            self.explainer.features_groups is not None
                            and self.explainer.inv_features_dict.get(expl[0])
                            not in self.explainer.features_groups.keys()
                        )
                    ):
                        # ylabel is based on trunc_new_value
                        ylabel = "<b>{} :</b><br />{}".format(trunc_new_value, truncate_str(expl[1], 45))
                    else:
                        ylabel = f"<b>{trunc_new_value}</b>"
                contrib_value = expl[2]
                # colors
                if contrib_value >= 0:
                    color = 1 if expl[1] != "" else 0
                else:
                    color = -1 if expl[1] != "" else -2

                # If the bar is a group of features we modify the color
                if group_name is not None:
                    bar_color = (
                        self._style_dict["featureimp_groups"][0]
                        if color == 1
                        else self._style_dict["featureimp_groups"][1]
                    )
                else:
                    bar_color = dict_local_plot_colors[color]["color"]

                barobj = go.Bar(
                    x=[contrib_value],
                    y=[ylabel],
                    customdata=[hoverlabel],
                    orientation="h",
                    marker=dict_local_plot_colors[color],
                    marker_color=bar_color,
                    showlegend=False,
                    hovertemplate="%{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
                )

                bars.append([color, contrib_value, num, barobj])

            bars.sort()
            fig = go.Figure(data=[x[-1] for x in bars], layout=layout)
            fig.update_yaxes(dtick=1)
            fig.update_yaxes(automargin=True)
            # fig.update_xaxes(automargin=True)

            if file_name:
                plot(fig, filename=file_name, auto_open=auto_open)
        else:
            fig = go.Figure()
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[
                    {
                        "text": "Select a valid single sample to display<br />Local Explanation plot.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14},
                    }
                ],
            )
        return fig

    def get_selection(self, line, var_dict, x_val, contrib):
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

    def apply_mask_one_line(self, line, var_dict, x_val, contrib, label=None):
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
        if hasattr(self.explainer, "mask"):
            if isinstance(self.explainer.mask, list):
                mask = self.explainer.mask[label].loc[line[0], :].values
            else:
                mask = self.explainer.mask.loc[line[0], :].values

        contrib = contrib[mask]
        x_val = x_val[mask]
        var_dict = var_dict[mask]

        return var_dict.tolist(), x_val.tolist(), contrib.tolist()

    def check_masked_contributions(self, line, var_dict, x_val, contrib, label=None):
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
        if hasattr(self.explainer, "masked_contributions"):
            if isinstance(self.explainer.masked_contributions, list):
                ext_contrib = self.explainer.masked_contributions[label].loc[line[0], :].values
            else:
                ext_contrib = self.explainer.masked_contributions.loc[line[0], :].values

            ext_var_dict = ["Hidden Negative Contributions", "Hidden Positive Contributions"]
            ext_x = ["", ""]
            ext_contrib = ext_contrib.tolist()

            exclusion = np.where(np.array(ext_contrib) == 0)[0].tolist()
            exclusion.sort(reverse=True)
            for ind in exclusion:
                del ext_var_dict[ind]
                del ext_x[ind]
                del ext_contrib[ind]

            var_dict.extend(ext_var_dict)
            x_val.extend(ext_x)
            contrib.extend(ext_contrib)

        return var_dict, x_val, contrib

    def local_pred(self, index, label=None):
        """
        compute a local pred to display in local_plot
        Parameters
        ----------
        index: string, int, float, ...
            specify the row we want to pred
        label: int (default: None)
        Returns
        -------
        float: Predict or predict_proba value
        """
        if self.explainer._case == "classification":
            if hasattr(self.explainer.model, "predict_proba"):
                if not hasattr(self.explainer, "proba_values"):
                    self.explainer.predict_proba()
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
        display_groups = True if (display_groups is not False and self.explainer.features_groups is not None) else False
        if display_groups:
            data = self.explainer.data_groups
        else:
            data = self.explainer.data

        if index is not None:
            if index in self.explainer.x_init.index:
                line = [index]
            else:
                line = []
        elif row_num is not None:
            line = [self.explainer.x_init.index[row_num]]
        elif query is not None:
            line = select_lines(self.explainer.x_init, query)
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
                not hasattr(self.explainer, "mask_params")  # If the filter method has not been called yet
                # Or if the already computed mask was not updated with current display_groups parameter
                or (
                    isinstance(data["contrib_sorted"], pd.DataFrame)
                    and len(data["contrib_sorted"].columns) != len(self.explainer.mask.columns)
                )
                or (
                    isinstance(data["contrib_sorted"], list)
                    and len(data["contrib_sorted"][0].columns) != len(self.explainer.mask[0].columns)
                )
            ):
                self.explainer.filter(max_contrib=20, display_groups=display_groups)

            if self.explainer._case == "classification":
                if label is None:
                    label = -1

                label_num, _, label_value = self.explainer.check_label_name(label)

                contrib = data["contrib_sorted"][label_num]
                x_val = data["x_sorted"][label_num]
                var_dict = data["var_dict"][label_num]

                if show_predict is True:
                    pred = self.local_pred(line[0], label_num)
                    if pred is None:
                        subtitle = f"Response: <b>{label_value}</b> - No proba available"
                    else:
                        subtitle = f"Response: <b>{label_value}</b> - Proba: <b>{pred:.4f}</b>"

            elif self.explainer._case == "regression":
                contrib = data["contrib_sorted"]
                x_val = data["x_sorted"]
                var_dict = data["var_dict"]
                label_num = None
                if show_predict is True:
                    pred_value = self.local_pred(line[0])
                    if self.explainer.y_pred is not None:
                        if self.round_digit is None:
                            self.tuning_round_digit()
                        digit = self.round_digit
                    else:
                        digit = compute_digit_number(pred_value)
                    subtitle = f"Predict: <b>{round(pred_value, digit)}</b>"

            var_dict, x_val, contrib = self.get_selection(line, var_dict, x_val, contrib)
            var_dict, x_val, contrib = self.apply_mask_one_line(line, var_dict, x_val, contrib, label=label_num)
            # use label of each column
            if display_groups:
                var_dict = [self.explainer.features_dict[self.explainer.x_init_groups.columns[x]] for x in var_dict]
            else:
                var_dict = [self.explainer.features_dict[self.explainer.columns_dict[x]] for x in var_dict]
            if show_masked:
                var_dict, x_val, contrib = self.check_masked_contributions(
                    line, var_dict, x_val, contrib, label=label_num
                )
            # Filtering all negative or positive contrib if specify in mask
            exclusion = []
            if hasattr(self.explainer, "mask_params"):
                if self.explainer.mask_params["positive"] is True:
                    exclusion = np.where(np.array(contrib) < 0)[0].tolist()
                elif self.explainer.mask_params["positive"] is False:
                    exclusion = np.where(np.array(contrib) > 0)[0].tolist()
            exclusion.sort(reverse=True)
            for expl in exclusion:
                del var_dict[expl]
                del x_val[expl]
                del contrib[expl]

        fig = self.plot_bar_chart(
            line, var_dict, x_val, contrib, yaxis_max_label, subtitle, width, height, file_name, auto_open, zoom
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

        if self.explainer._case == "classification":
            label_num, _, label_value = self.explainer.check_label_name(label)

        if not isinstance(col, (str, int)):
            raise ValueError("parameter col must be string or int.")
        if hasattr(self.explainer, "inv_features_dict"):
            col = self.explainer.inv_features_dict.get(col, col)
        col_is_group = self.explainer.features_groups and col in self.explainer.features_groups.keys()

        # Case where col is a group of features
        if col_is_group:
            contributions = self.explainer.contributions_groups
            col_label = self.explainer.features_dict[col]
            col_name = self.explainer.features_groups[col]  # Here col_name is actually a list of features
            col_value_count = self.explainer.features_desc[col]
        else:
            contributions = self.explainer.contributions
            col_id = self.explainer.check_features_name([col])[0]
            col_name = self.explainer.columns_dict[col_id]
            col_value_count = self.explainer.features_desc[col_name]

            if self.explainer.features_dict:
                col_label = self.explainer.features_dict[col_name]
            else:
                col_label = col_name

        list_ind, addnote = self.explainer.plot._subset_sampling(selection, max_points)

        col_value = None
        proba_values = None
        subtitle = None
        col_scale = None

        # Classification Case
        if self.explainer._case == "classification":
            subcontrib = contributions[label_num]
            if self.explainer.y_pred is not None:
                col_value = self.explainer._classes[label_num]
            subtitle = f"Response: <b>{label_value}</b>"
            # predict proba Color scale
            if proba and hasattr(self.explainer.model, "predict_proba"):
                if not hasattr(self.explainer, "proba_values"):
                    self.explainer.predict_proba()
                proba_values = self.explainer.proba_values.iloc[:, [label_num]]
                if not hasattr(self, "pred_colorscale"):
                    self.pred_colorscale = {}
                if label_num not in self.pred_colorscale:
                    self.pred_colorscale[label_num] = self.tuning_colorscale(proba_values)
                col_scale = self.pred_colorscale[label_num]
                # Proba subset:
                proba_values = proba_values.loc[list_ind, :]

        # Regression Case - color scale
        elif self.explainer._case == "regression":
            subcontrib = contributions
            if self.explainer.y_pred is not None:
                if not hasattr(self, "pred_colorscale"):
                    self.pred_colorscale = self.tuning_colorscale(self.explainer.y_pred)
                col_scale = self.pred_colorscale

        # Subset
        if self.explainer.postprocessing_modifications:
            feature_values = self.explainer.x_contrib_plot.loc[list_ind, col_name]
        else:
            feature_values = self.explainer.x_init.loc[list_ind, col_name]

        if col_is_group:
            feature_values = project_feature_values_1d(
                feature_values,
                col,
                self.explainer.x_init,
                self.explainer.x_encoded,
                self.explainer.preprocessing,
                features_dict=self.explainer.features_dict,
            )
            contrib = subcontrib.loc[list_ind, col].to_frame()
            if self.explainer.features_imp is None:
                self.explainer.compute_features_import()
            features_imp = (
                self.explainer.features_imp
                if isinstance(self.explainer.features_imp, pd.Series)
                else self.explainer.features_imp[0]
            )
            top_features_of_group = (
                features_imp.loc[self.explainer.features_groups[col]].sort_values(ascending=False)[:4].index
            )  # Displaying top 4 features
            metadata = {
                self.explainer.features_dict[f_name]: self.explainer.x_init[f_name] for f_name in top_features_of_group
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

        if self.explainer.y_pred is not None:
            y_pred = self.explainer.y_pred.loc[list_ind]
            # Add labels if exist
            if self.explainer._case == "classification" and self.explainer.label_dict is not None:
                y_pred = y_pred.applymap(lambda x: self.explainer.label_dict[x])
                col_value = self.explainer.label_dict[col_value]
            # round predict
            elif self.explainer._case == "regression":
                if self.round_digit is None:
                    self.tuning_round_digit()
                y_pred = y_pred.applymap(lambda x: round(x, self.round_digit))
        else:
            y_pred = None

        # selecting the best plot : Scatter, Violin?
        if col_value_count > violin_maxf:
            fig = self.plot_scatter(
                feature_values,
                contrib,
                col_label,
                y_pred,
                proba_values,
                col_value,
                col_scale,
                metadata,
                addnote,
                subtitle,
                width,
                height,
                file_name,
                auto_open,
                zoom,
            )
        else:
            fig = self.plot_violin(
                feature_values,
                contrib,
                col_label,
                y_pred,
                proba_values,
                col_value,
                col_scale,
                addnote,
                subtitle,
                width,
                height,
                file_name,
                auto_open,
                zoom,
            )

        return fig

    def features_importance(
        self,
        max_features=20,
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
    ):
        """
        features_importance display a plotly features importance plot.
        in Multiclass Case, this features_importance focus on a label value.
        User specifies the label value using label parameter.
        the selection parameter allows the user to compare a subset to the global features
        importance
        features_importance tutorial offers several examples
         (please check tutorial part of this doc)
        Parameters
        ----------
        max_features: int (optional, default 20)
            this argument limit the number of hbar in features importance plot
            if max_features is 20, plot selects the 20 most important features
        selection: list (optional, default None)
            This argument allows to represent the importance calculated with a subset.
            Subset features importance is compared to global in the plot
            Argument must contains list of index, subset of the input DataFrame that we want to plot
        label: integer or string (default -1)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        group_name : str (optional, default None)
            Allows to display the features importance of the variables that are grouped together
            inside a group of features.
            This parameter is only available if the SmartExplainer object has been compiled using
            the features_groups optional parameter and should correspond to a key of
            features_groups dictionary.
        display_groups : bool (default True)
            If groups of features are declared in SmartExplainer object, this parameter allows to
            specify whether or not to display them.
        force: bool (optional, default False)
            force == True, force the compute features importance if it's already done
        width : Int (default: 900)
            Plotly figure - layout width
        height : Int (default: 500)
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
        >>> xpl.plot.features_importance()
        """
        self.explainer.compute_features_import(force=force)
        subtitle = None
        title = "Features Importance"
        display_groups = self.explainer.features_groups is not None and display_groups
        if display_groups:
            if group_name:  # Case where we have groups of features and we want to display only features inside a group
                if group_name not in self.explainer.features_groups.keys():
                    raise ValueError(
                        f"group_name parameter : {group_name} is not in features_groups keys. "
                        f"Possible values are : {list(self.explainer.features_groups.keys())}"
                    )
                title += f" - {truncate_str(self.explainer.features_dict.get(group_name), 20)}"
                if isinstance(self.explainer.features_imp, list):
                    features_importance = [
                        label_feat_imp.loc[label_feat_imp.index.isin(self.explainer.features_groups[group_name])]
                        for label_feat_imp in self.explainer.features_imp
                    ]
                else:
                    features_importance = self.explainer.features_imp.loc[
                        self.explainer.features_imp.index.isin(self.explainer.features_groups[group_name])
                    ]
                contributions = self.explainer.contributions
            else:
                features_importance = self.explainer.features_imp_groups
                contributions = self.explainer.contributions_groups
        else:
            features_importance = self.explainer.features_imp
            contributions = self.explainer.contributions

        # classification
        if self.explainer._case == "classification":
            label_num, _, label_value = self.explainer.check_label_name(label)
            global_feat_imp = features_importance[label_num].tail(max_features)
            if selection is not None:
                subset_feat_imp = self.explainer.backend.get_global_features_importance(
                    contributions=contributions[label_num], explain_data=self.explainer.explain_data, subset=selection
                )
            else:
                subset_feat_imp = None
            subtitle = f"Response: <b>{label_value}</b>"
        # regression
        elif self.explainer._case == "regression":
            global_feat_imp = features_importance.tail(max_features)
            if selection is not None:
                subset_feat_imp = self.explainer.backend.get_global_features_importance(
                    contributions=contributions, explain_data=self.explainer.explain_data, subset=selection
                )
            else:
                subset_feat_imp = None
        addnote = ""
        if subset_feat_imp is not None:
            subset_feat_imp = subset_feat_imp.reindex(global_feat_imp.index)
            subset_feat_imp.index = subset_feat_imp.index.map(self.explainer.features_dict)
            if subset_feat_imp.dropna().shape[0] == 0:
                raise ValueError("selection argument doesn't return any row")
            subset_len = len(selection)
            total_len = self.explainer.x_init.shape[0]
            addnote = add_text(
                [addnote, f"Subset length: {subset_len} ({int(np.round(100 * subset_len / total_len))}%)"], sep=" - "
            )
        if self.explainer.x_init.shape[1] >= max_features:
            addnote = add_text([addnote, f"Total number of features: {int(self.explainer.x_init.shape[1])}"], sep=" - ")

        global_feat_imp.index = global_feat_imp.index.map(self.explainer.features_dict)
        if display_groups:
            # Bold font for groups of features
            global_feat_imp.index = [
                "<b>" + str(f)
                if self.explainer.inv_features_dict.get(f) in self.explainer.features_groups.keys()
                else str(f)
                for f in global_feat_imp.index
            ]
            if subset_feat_imp is not None:
                subset_feat_imp.index = [
                    "<b>" + str(f)
                    if self.explainer.inv_features_dict.get(f) in self.explainer.features_groups.keys()
                    else str(f)
                    for f in subset_feat_imp.index
                ]

        fig = self.plot_features_import(
            global_feat_imp, subset_feat_imp, title, addnote, subtitle, width, height, file_name, auto_open, zoom
        )
        return fig

    def plot_line_comparison(
        self,
        index,
        feature_values,
        contributions,
        predictions=None,
        dict_features=None,
        subtitle=None,
        width=900,
        height=550,
        file_name=None,
        auto_open=False,
    ):
        """
        Plotly plot for comparisons. Displays
        the contributions of several individuals. One line represents
        the different contributions of a unique individual.
        Parameters
        ----------
        index: list
            List of index corresponding to the individuals we want to compare.
        feature_values: list
            String list corresponding to the name of the features.
        contributions: numpy.ndarray
            Matrix of contributions.
            Each row corresponds to an individual.
        predictions: list
            List of pandas.Series containing values of individuals.
        dict_features: dict
            Dictionnary of feature names.
        subtitle: string (default : None)
            Subtitle to display.
        width: int (default: 900)
            Plotly figure - layout width
        height: int (default: 550)
            Plotly figure - layout height.
        file_name: string (optional)
            File name to use to save the plotly scatter chart. If None the scatter chart will not be saved.
        auto_open: Boolean (optional)
            Indicate whether to open the scatter plot or not.
        Returns
        -------
        Plotly Figure Object
            Plot of the contributions of individuals, feature by feature.
        """

        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        topmargin = 80
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])

        if len(index) == 0:
            warnings.warn("No individuals matched", UserWarning)
            dict_t["text"] = "Compare plot - <b>No Matching Reference Entry</b>"
        elif len(index) < 2:
            warnings.warn("Comparison needs at least 2 individuals", UserWarning)
            dict_t["text"] = "Compare plot - index : " + " ; ".join(["<b>" + str(id) + "</b>" for id in index])
        else:
            dict_t["text"] = "Compare plot - index : " + " ; ".join(["<b>" + str(id) + "</b>" for id in index])

            dict_xaxis["text"] = "Contributions"

        dict_yaxis["text"] = None

        if subtitle is not None:
            topmargin += 15 * height / 275
            dict_t["text"] = (
                truncate_str(dict_t["text"], 120)
                + f"<span style='font-size: 12px;'><br />{truncate_str(subtitle, 200)}</span>"
            )

        layout = go.Layout(
            template="none",
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            yaxis_type="category",
            width=width,
            height=height,
            hovermode="closest",
            legend=dict(x=1, y=1),
            margin={"l": 150, "r": 20, "t": topmargin, "b": 70},
        )

        iteration_list = list(zip(contributions, feature_values))

        dic_color = copy.deepcopy(self._style_dict["dict_compare_colors"])
        lines = list()

        for i, id_i in enumerate(index):
            x_i = list()
            features = list()
            x_val = predictions[i]
            x_hover = list()

            for contrib, feat in iteration_list:
                x_i.append(contrib[i])
                features.append("<b>" + str(feat) + "</b>")
                pred_x_val = x_val[dict_features[feat]]
                x_hover.append(
                    f"Id: <b>{add_line_break(id_i, 40, 160)}</b>"
                    + f"<br /><b>{add_line_break(feat, 40, 160)}</b> <br />"
                    + f"Contribution: {contrib[i]:.4f} <br />Value: "
                    + str(add_line_break(pred_x_val, 40, 160))
                )

            lines.append(
                go.Scatter(
                    x=x_i,
                    y=features,
                    mode="lines+markers",
                    showlegend=True,
                    name=f"Id: <b>{index[i]}</b>",
                    hoverinfo="text",
                    hovertext=x_hover,
                    marker={"color": dic_color[i % len(dic_color)]},
                )
            )

        fig = go.Figure(data=lines, layout=layout)
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name is not None:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

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
                if ident in self.explainer.x_init.index:
                    line_reference.append(ident)

        elif row_num is not None:
            line_reference = [
                self.explainer.x_init.index.values[row_nb_reference]
                for row_nb_reference in row_num
                if self.explainer.x_init.index.values[row_nb_reference] in self.explainer.x_init.index
            ]

        subtitle = ""
        if len(line_reference) < 1:
            raise ValueError("No matching entry for index")

        # Classification case
        if self.explainer._case == "classification":
            if label is None:
                label = -1

            label_num, _, label_value = self.explainer.check_label_name(label)
            contrib = self.explainer.contributions[label_num]

            if show_predict:
                preds = [self.local_pred(line, label_num) for line in line_reference]
                subtitle = (
                    f"Response: <b>{label_value}</b> - "
                    + "Probas: "
                    + " ; ".join(
                        [str(id) + ": <b>" + str(round(proba, 2)) + "</b>" for proba, id in zip(preds, line_reference)]
                    )
                )

        # Regression case
        elif self.explainer._case == "regression":
            contrib = self.explainer.contributions

            if show_predict:
                preds = [self.local_pred(line) for line in line_reference]
                subtitle = "Predictions: " + " ; ".join(
                    [str(id) + ": <b>" + str(round(pred, 2)) + "</b>" for id, pred in zip(line_reference, preds)]
                )

        new_contrib = list()
        for ident in line_reference:
            new_contrib.append(contrib.loc[ident])
        new_contrib = np.array(new_contrib).T

        # Well labels if available
        feature_values = [0] * len(contrib.columns)
        if hasattr(self.explainer, "columns_dict"):
            for i, name in enumerate(contrib.columns):
                feature_name = self.explainer.features_dict[name]
                feature_values[i] = feature_name

        preds = [self.explainer.x_init.loc[id] for id in line_reference]
        dict_features = self.explainer.inv_features_dict

        iteration_list = list(zip(new_contrib, feature_values))
        iteration_list.sort(key=lambda x: maximum_difference_sort_value(x), reverse=True)
        iteration_list = iteration_list[:max_features]
        iteration_list = iteration_list[::-1]
        new_contrib, feature_values = list(zip(*iteration_list))

        fig = self.plot_line_comparison(
            line_reference,
            feature_values,
            new_contrib,
            predictions=preds,
            dict_features=dict_features,
            width=width,
            height=height,
            subtitle=subtitle,
            file_name=file_name,
            auto_open=auto_open,
        )

        return fig

    def _plot_interactions_scatter(self, x_name, y_name, col_name, x_values, y_values, col_values, col_scale):
        """
        Function used to generate a scatter plot figure for the interactions plots.
        Parameters
        ----------
        x_name : str
            Name of the variable used as the x axis
        y_name : str
            Name of the variable used as the y axis
        col_name : str
            Name of the variable used as the color attribute
        x_values : pd.DataFrame
            Values of the points on the x axis as a 1 column DataFrame
        y_values : pd.DataFrame
            Values of the points on the y axis as a 1 column DataFrame
        col_values : pd.DataFrame
            Values of the color of the points as a 1 column DataFrame
        col_scale : list
            color scale
        Returns
        -------
        go.Figure
        """
        # add break line to X label if necessary
        max_len_by_row = max([round(50 / self.explainer.features_desc[x_values.columns.values[0]]), 8])
        x_values.iloc[:, 0] = x_values.iloc[:, 0].apply(
            add_line_break,
            args=(
                max_len_by_row,
                120,
            ),
        )

        data_df = pd.DataFrame(
            {
                x_name: x_values.values.flatten(),
                y_name: y_values.values.flatten(),
                col_name: col_values.values.flatten(),
            }
        )

        if isinstance(col_values.values.flatten()[0], str):
            fig = px.scatter(
                data_df,
                x=x_name,
                y=y_name,
                color=col_name,
                color_discrete_sequence=self._style_dict["interactions_discrete_colors"],
            )
        else:
            fig = px.scatter(data_df, x=x_name, y=y_name, color=col_name, color_continuous_scale=col_scale)

        fig.update_traces(mode="markers")

        return fig

    def _plot_interactions_violin(self, x_name, y_name, col_name, x_values, y_values, col_values, col_scale):
        """
        Function used to generate a violin plot figure for the interactions plots.
        Parameters
        ----------
        x_name : str
            Name of the variable used as the x axis
        y_name : str
            Name of the variable used as the y axis
        col_name : str
            Name of the variable used as the color attribute
        x_values : pd.DataFrame
            Values of the points on the x axis as a 1 column DataFrame
        y_values : pd.DataFrame
            Values of the points on the y axis as a 1 column DataFrame
        col_values : pd.DataFrame
            Values of the color of the points as a 1 column DataFrame
        col_scale : list
            color scale
        Returns
        -------
        go.Figure
        """

        fig = go.Figure()

        # add break line to X label
        max_len_by_row = max([round(50 / self.explainer.features_desc[x_values.columns.values[0]]), 8])
        x_values.iloc[:, 0] = x_values.iloc[:, 0].apply(
            add_line_break,
            args=(
                max_len_by_row,
                120,
            ),
        )

        uniq_l = list(pd.unique(x_values.values.flatten()))
        uniq_l.sort()

        for i in uniq_l:
            fig.add_trace(
                go.Violin(
                    x=x_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                    y=y_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                    line_color=self._style_dict["violin_default"],
                    showlegend=False,
                    meanline_visible=True,
                    scalemode="count",
                )
            )
        scatter_fig = self._plot_interactions_scatter(
            x_name=x_name,
            y_name=y_name,
            col_name=col_name,
            x_values=x_values,
            y_values=y_values,
            col_values=col_values,
            col_scale=col_scale,
        )
        for trace in scatter_fig.data:
            fig.add_trace(trace)

        fig.update_layout(
            autosize=False,
            hovermode="closest",
            violingap=0.05,
            violingroupgap=0,
            violinmode="overlay",
            xaxis_type="category",
        )

        fig.update_xaxes(range=[-0.6, len(uniq_l) - 0.4])

        return fig

    def _update_interactions_fig(self, fig, col_name1, col_name2, addnote, width, height, file_name, auto_open):
        """
        Function used for the interactions plot to update the layout of the plotly figure.
        Parameters
        ----------
        col_name1 : str
            Name of the first column whose contributions we want to plot
        col_name2 : str
            Name of the second column whose contributions we want to plot
        addnote : str
            Text to be added to the figure title
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
        """

        if fig.data[-1]["showlegend"] is False:  # Case where col2 is not categorical
            fig.layout.coloraxis.colorscale = self._style_dict["interactions_col_scale"]
        else:
            fig.update_layout(legend=dict(title=dict(text=col_name2)))

        title = f"<b>{truncate_str(col_name1)} and {truncate_str(col_name2)}</b> shap interaction values"
        if addnote:
            title += f"<span style='font-size: 12px;'><br />{add_text([addnote], sep=' - ')}</span>"
        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        dict_t["text"] = title

        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_xaxis["text"] = truncate_str(col_name1, 110)
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_yaxis["text"] = "Shap interaction value"

        fig.update_traces(marker={"size": 8, "opacity": 0.8, "line": {"width": 0.8, "color": "white"}})

        fig.update_layout(
            coloraxis=dict(colorbar={"title": {"text": col_name2}}),
            yaxis_title=dict_yaxis,
            title=dict_t,
            template="none",
            width=width,
            height=height,
            xaxis_title=dict_xaxis,
            hovermode="closest",
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

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
            elif self.explainer.x_init.shape[0] <= max_points:
                list_ind = self.explainer.x_init.index.tolist()
            else:
                list_ind = random.sample(self.explainer.x_init.index.tolist(), max_points)
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
            ValueError("parameter selection must be a list")
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

        col_id1 = self.explainer.check_features_name([col1])[0]
        col_name1 = self.explainer.columns_dict[col_id1]

        col_id2 = self.explainer.check_features_name([col2])[0]
        col_name2 = self.explainer.columns_dict[col_id2]

        col_value_count1 = self.explainer.features_desc[col_name1]

        list_ind, addnote = self._select_indices_interactions_plot(selection=selection, max_points=max_points)

        if addnote is not None:
            addnote = add_text(
                [addnote, f"{len(list_ind)} ({int(np.round(100 * len(list_ind) / self.explainer.x_init.shape[0]))}%)"],
                sep="",
            )

        # Subset
        if self.explainer.postprocessing_modifications:
            feature_values1 = self.explainer.x_contrib_plot.loc[list_ind, col_name1].to_frame()
            feature_values2 = self.explainer.x_contrib_plot.loc[list_ind, col_name2].to_frame()
        else:
            feature_values1 = self.explainer.x_init.loc[list_ind, col_name1].to_frame()
            feature_values2 = self.explainer.x_init.loc[list_ind, col_name2].to_frame()

        interaction_values = self.explainer.get_interaction_values(selection=list_ind)[:, col_id1, col_id2]
        if col_id1 != col_id2:
            interaction_values = interaction_values * 2

        # selecting the best plot : Scatter, Violin?
        if col_value_count1 > violin_maxf:
            fig = self._plot_interactions_scatter(
                x_name=col_name1,
                y_name="Shap interaction value",
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self._style_dict["interactions_col_scale"],
            )
        else:
            fig = self._plot_interactions_violin(
                x_name=col_name1,
                y_name="Shap interaction value",
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self._style_dict["interactions_col_scale"],
            )

        self._update_interactions_fig(
            fig=fig,
            col_name1=col_name1,
            col_name2=col_name2,
            addnote=addnote,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open,
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

        interaction_values = self.explainer.get_interaction_values(selection=list_ind)

        sorted_top_features_indices = compute_sorted_variables_interactions_list_indices(interaction_values)

        indices_to_plot = sorted_top_features_indices[:nb_top_interactions]
        interactions_indices_traces_mapping = []
        fig = go.Figure()
        for i, ids in enumerate(indices_to_plot):
            id0, id1 = ids

            fig_one_interaction = self.interactions_plot(
                col1=self.explainer.columns_dict[id0],
                col2=self.explainer.columns_dict[id1],
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
            dict_t = copy.deepcopy(self._style_dict["dict_title"])
            dict_t.update({"text": title, "y": 0.88, "x": 0.5, "xanchor": "center", "yanchor": "top"})
            return dict_t

        fig.layout.coloraxis.colorscale = self._style_dict["interactions_col_scale"]
        fig.update_layout(
            xaxis_title=self.explainer.columns_dict[sorted_top_features_indices[0][0]],
            yaxis_title="Shap interaction value",
            updatemenus=[
                dict(
                    active=0,
                    buttons=list(
                        [
                            dict(
                                label=f"{self.explainer.columns_dict[i]} - {self.explainer.columns_dict[j]}",
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
                                                **{"text": self.explainer.columns_dict[i]},
                                                **self._style_dict["dict_xaxis"],
                                            }
                                        },
                                        "legend": {"title": {"text": self.explainer.columns_dict[j]}},
                                        "coloraxis": {
                                            "colorbar": {"title": {"text": self.explainer.columns_dict[j]}},
                                            "colorscale": fig.layout.coloraxis.colorscale,
                                        },
                                        "title": generate_title_dict(
                                            self.explainer.columns_dict[i], self.explainer.columns_dict[j], addnote
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
            ],
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

        self._update_interactions_fig(
            fig=fig,
            col_name1=self.explainer.columns_dict[sorted_top_features_indices[0][0]],
            col_name2=self.explainer.columns_dict[sorted_top_features_indices[0][1]],
            addnote=addnote,
            width=width,
            height=height,
            file_name=None,
            auto_open=False,
        )

        fig.update_layout(title={"y": 0.88, "x": 0.5, "xanchor": "center", "yanchor": "top"})

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def correlations(
        self,
        df=None,
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

        def cluster_corr(corr, degree, inplace=False):
            """
            Rearranges the correlation matrix, corr, so that groups of highly
            correlated variables are next to eachother

            Parameters
            ----------
            corr : pandas.DataFrame or numpy.ndarray
                a NxN correlation matrix
            degree  : int
                degree applied on the correlation matrix in order to focus more or less the clustering
                on strong correlated variables
            inplace : bool, optional
                to replace the original correlation matrix by the new one, by default False

            Returns
            -------
            pandas.DataFrame or numpy.ndarray
                a NxN correlation matrix with the columns and rows rearranged
            """

            if corr.shape[0] < 2:
                return corr

            pairwise_distances = sch.distance.pdist(corr ** degree)
            linkage = sch.linkage(pairwise_distances, method="complete")
            cluster_distance_threshold = pairwise_distances.max() / 2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion="distance")
            idx = np.argsort(idx_to_cluster_array)

            if not inplace:
                corr = corr.copy()

            if isinstance(corr, pd.DataFrame):
                return corr.iloc[idx, :].T.iloc[idx, :]

            return corr[idx, :][:, idx]

        if features_to_hide is None:
            features_to_hide = []

        if df is None:
            # Use x_init by default
            df = self.explainer.x_init

        if facet_col:
            features_to_hide += [facet_col]

        compute_method = how

        hovertemplate = "<b>%{text}<br />Correlation: %{z}</b><extra></extra>"

        list_features = []
        if facet_col:
            facet_col_values = sorted(df[facet_col].unique(), reverse=True)
            fig = make_subplots(
                rows=1,
                cols=df[facet_col].nunique(),
                subplot_titles=[t + " correlation" for t in facet_col_values],
                horizontal_spacing=0.15,
            )
            # Used for the Shapash report to get train then test set
            for i, col_v in enumerate(facet_col_values):
                corr = compute_corr(df.loc[df[facet_col] == col_v].drop(features_to_hide, axis=1), compute_method)

                # Keep the same list of features for each subplot
                if len(list_features) == 0:
                    top_features = compute_top_correlations_features(corr=corr, max_features=max_features)
                    corr = cluster_corr(corr.loc[top_features, top_features], degree=degree)
                    list_features = list(corr.columns)

                fig.add_trace(
                    go.Heatmap(
                        z=corr.loc[list_features, list_features].round(decimals).values,
                        x=list_features,
                        y=list_features,
                        coloraxis="coloraxis",
                        text=[
                            [
                                f"Feature 1: {self.explainer.features_dict.get(y, y)} <br />"
                                f"Feature 2: {self.explainer.features_dict.get(x, x)}"
                                for x in list_features
                            ]
                            for y in list_features
                        ],
                        hovertemplate=hovertemplate,
                    ),
                    row=1,
                    col=i + 1,
                )

        else:
            corr = compute_corr(df.drop(features_to_hide, axis=1), compute_method)
            top_features = compute_top_correlations_features(corr=corr, max_features=max_features)
            corr = cluster_corr(corr.loc[top_features, top_features], degree=degree)
            list_features = [col for col in corr.columns if col in top_features]

            fig = go.Figure(
                go.Heatmap(
                    z=corr.loc[list_features, list_features].round(decimals).values,
                    x=list_features,
                    y=list_features,
                    coloraxis="coloraxis",
                    text=[
                        [
                            f"Feature 1: {self.explainer.features_dict.get(y, y)} <br />"
                            f"Feature 2: {self.explainer.features_dict.get(x, x)}"
                            for x in list_features
                        ]
                        for y in list_features
                    ],
                    hovertemplate=hovertemplate,
                )
            )

        title = f"Correlation ({compute_method})"
        if len(list_features) < len(df.drop(features_to_hide, axis=1).columns):
            subtitle = f"Top {len(list_features)} correlations"
            title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        dict_t["text"] = title

        fig.update_layout(
            coloraxis=dict(colorscale=["rgb(255, 255, 255)"] + self._style_dict["init_contrib_colorscale"][5:-1]),
            showlegend=True,
            title=dict_t,
            width=width,
            height=height,
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def plot_amplitude_vs_stability(self, mean_variability, mean_amplitude, column_names, file_name, auto_open):
        """
        Intermediate function used to display the stability plot when plot_type is "none"
        Parameters
        ----------
        mean_variability: array
            Local stability expressed as a mean value for all instances (one value per feature).
            Displayed on the X-axis on the plot.
        mean_amplitude: array
            Average of the normalized SHAP values in the neighborhood. Displayed on the Y-axis on the plot.
        column_names: list
            Columns names that are displayed on the plot
        file_name: string
            Specify the save path of html files. If it is not provided, no file will be saved
        auto_open: bool
            open automatically the plot
        Returns
        -------
        go.Figure
        """
        xaxis_title = (
            "Variability of the Normalized Local Contribution Values"
            + "<span style='font-size: 12px;'><br />(standard deviation / mean)</span>"
        )
        yaxis_title = "Importance<span style='font-size: 12px;'><br />(Average contributions)</span>"
        col_scale = self.tuning_colorscale(pd.DataFrame(mean_amplitude))
        hv_text = [
            f"<b>Feature: {col}</b><br />Importance: {y}<br />Variability: {x}"
            for col, x, y in zip(column_names, mean_variability, mean_amplitude)
        ]
        hovertemplate = "%{hovertext}" + "<extra></extra>"

        fig = go.Figure()
        fig.add_scatter(
            x=mean_variability,
            y=mean_amplitude,
            showlegend=False,
            mode="markers",
            marker={
                "color": mean_amplitude,
                "size": 10,
                "opacity": 0.8,
                "line": {"width": 0.8, "color": "white"},
                "colorscale": col_scale,
            },
            hovertext=hv_text,
            hovertemplate=hovertemplate,
        )

        fig.update_xaxes(range=[np.min(np.append(mean_variability, [0.15])) - 0.03, np.max(mean_variability) + 0.03])

        self._update_stability_fig(
            fig=fig,
            x_barlen=len(mean_amplitude),
            y_bar=[0, mean_amplitude.max()],
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            file_name=file_name,
            auto_open=auto_open,
        )
        return fig

    def plot_stability_distribution(
        self, variability, plot_type, mean_amplitude, dataset, column_names, file_name, auto_open
    ):
        """
        Intermediate function used to display the stability plot when plot_type is "boxplot" or
        "violin"
        Parameters
        ----------
        variability: array
            Local stability expressed as a distribution across all instances
            (one distribution per feature). Displayed on the X-axis on the plot
        plot_type: string
            Defines the type of plot that will be displayed.
            Possible values are "boxplot" or "violin"
        mean_amplitude: array
            Average of the normalized SHAP values in the neighborhood.
            Displayed as a colorscale in the plot.
        dataset: DataFrame
            x_init dataset
        column_names: list
            Columns names that are displayed on the plot
        file_name: string
            Specify the save path of html files. If it is not provided, no file will be saved
        auto_open: bool
            open automatically the plot
        Returns
        -------
        go.Figure
        """
        # Store distribution of variability in a DataFrame
        var_df = pd.DataFrame(variability, columns=column_names)
        mean_amplitude_normalized = pd.Series(mean_amplitude, index=column_names) / mean_amplitude.max()

        # And sort columns by mean amplitude
        var_df = var_df[column_names[mean_amplitude.argsort()]]

        # Add colorscale
        col_scale = self.tuning_colorscale(pd.DataFrame(mean_amplitude))
        color_list = mean_amplitude_normalized.tolist()
        color_list.sort()
        color_list = [next(pair[1] for pair in col_scale if x <= pair[0]) for x in color_list]
        height_value = max(500, 40 * dataset.shape[1] if dataset.shape[1] < 100 else 13 * dataset.shape[1])

        xaxis_title = "Normalized local contribution value variability"
        yaxis_title = ""

        # Plot the distribution
        if dataset.shape[1] < 500:
            fig = go.Figure()
            for i, c in enumerate(var_df):
                if plot_type == "boxplot":
                    fig.add_trace(
                        go.Box(
                            x=var_df[c],
                            marker_color=color_list[i],
                            name=c,
                            showlegend=False,
                        )
                    )
                elif plot_type == "violin":
                    fig.add_trace(
                        go.Violin(
                            x=var_df[c],
                            line_color=color_list[i],
                            name=c,
                            showlegend=False,
                        )
                    )

            # Dummy invisible plot to add the color scale
            colorbar_trace = go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=1,
                    color=[mean_amplitude.min(), mean_amplitude.max()],
                    colorscale=col_scale,
                    colorbar=dict(
                        thickness=20,
                        lenmode="pixels",
                        len=300,
                        yanchor="top",
                        y=1,
                        ypad=60,
                        title="Importance<br />(Average contributions)",
                    ),
                    showscale=True,
                ),
                hoverinfo="none",
                showlegend=False,
            )

            fig.add_trace(colorbar_trace)

            fig.update_layout(
                height=height_value,
            )

            self._update_stability_fig(
                fig=fig,
                x_barlen=len(mean_amplitude),
                y_bar=column_names,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                file_name=file_name,
                auto_open=auto_open,
            )

            return fig

    def _update_stability_fig(self, fig, x_barlen, y_bar, xaxis_title, yaxis_title, file_name, auto_open):
        """
        Function used for the `plot_stability_distribution` and `plot_amplitude_vs_stability`
        to update the layout of the plotly figure.
        Parameters
        ----------
        fig: plotly.graph_objs._figure.Figure
            Plotly figure to update
        x_barlen: int
            draw a line --> len of x array
        y_bar: list
            draw a line --> y values
        xaxis_title: str
            Title of xaxis
        yaxis_title: str
            Title of yaxis
        file_name: string (optional)
            Specify the save path of html files. If it is not provided, no file will be saved.
        auto_open: bool (default=False)
            open automatically the plot
        Returns
        -------
        go.Figure
        """
        title = "Importance & Local Stability of explanations:"
        title += "<span style='font-size: 16px;'><br />How similar are explanations for closeby neighbours?</span>"
        dict_t = copy.deepcopy(self._style_dict["dict_title_stability"])
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_xaxis["text"] = xaxis_title
        dict_yaxis["text"] = yaxis_title
        dict_stability_bar_colors = copy.deepcopy(self._style_dict["dict_stability_bar_colors"])
        dict_t["text"] = title

        fig.add_trace(
            go.Scatter(
                x=[0.15] * x_barlen,
                y=y_bar,
                mode="lines",
                hoverinfo="none",
                line=dict(color=dict_stability_bar_colors[0], dash="dot"),
                name="<-- Stable",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0.3] * x_barlen,
                y=y_bar,
                mode="lines",
                hoverinfo="none",
                line=dict(color=dict_stability_bar_colors[1], dash="dot"),
                name="--> Unstable",
            )
        )

        fig.update_layout(
            template="none",
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            coloraxis_showscale=False,
            hovermode="closest",
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

    def local_neighbors_plot(self, index, max_features=10, file_name=None, auto_open=False):
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
            distance = \\frac{|output_{allFeatures} -
                              output_{currentFeatures}|}{|output_{allFeatures}|}
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
        Returns
        -------
        fig
            The figure that will be displayed
        """
        assert index in self.explainer.x_init.index, "index must exist in pandas dataframe"

        self.explainer.compute_features_stability([index])

        column_names = np.array([self.explainer.features_dict.get(x) for x in self.explainer.x_init.columns])

        def ordinal(n):
            return "%d%s" % (n, "tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10 :: 4])

        # Compute explanations for instance and neighbors
        g = self.explainer.local_neighbors["norm_shap"]

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
                    marker_color=self._style_dict["dict_stability_bar_colors"][1]
                    if i == g_df.shape[1] - 1
                    else self._style_dict["dict_stability_bar_colors"][0],
                    orientation="h",
                    opacity=np.clip(0.2 + i * (1 - 0.2) / (g_df.shape[1] - 1), 0.2, 1) if g_df.shape[1] > 1 else 1,
                )
                for i in range(g_df.shape[1])
            ]
        )

        title = f"Comparing local explanations in a neighborhood - Id: <b>{index}</b>"
        title += "<span style='font-size: 16px;'><br />How similar are explanations for closeby neighbours?</span>"
        dict_t = copy.deepcopy(self._style_dict["dict_title_stability"])
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_xaxis["text"] = "Normalized contribution values"
        dict_yaxis["text"] = ""
        dict_t["text"] = title
        fig.update_layout(
            template="none",
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode="closest",
            barmode="group",
            height=max(500, 11 * g_df.shape[0] * g_df.shape[1]),
            legend={"traceorder": "reversed"},
            xaxis={"side": "bottom"},
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
            if self.explainer.x_init.shape[0] <= max_points:
                list_ind = self.explainer.x_init.index.tolist()
            else:
                list_ind = random.sample(self.explainer.x_init.index.tolist(), max_points)
            # By default, don't compute calculation if it has already been done
            if (self.explainer.features_stability is None) or self.last_stability_selection or force:
                self.explainer.compute_features_stability(list_ind)
            else:
                print("Computed values from previous call are used")
            self.last_stability_selection = False
        elif isinstance(selection, list):
            if len(selection) == 1:
                raise ValueError("Selection must include multiple points")
            if len(selection) > max_points:
                print(
                    f"Size of selection is bigger than max_points (default: {max_points}).\
                      Computation time might be affected"
                )
            self.explainer.compute_features_stability(selection)
            self.last_stability_selection = True
        else:
            raise ValueError("Parameter selection must be a list")

        column_names = np.array([self.explainer.features_dict.get(x) for x in self.explainer.x_init.columns])

        variability = self.explainer.features_stability["variability"]
        amplitude = self.explainer.features_stability["amplitude"]

        mean_variability = variability.mean(axis=0)
        mean_amplitude = amplitude.mean(axis=0)

        # Plot 1 : only show average variability on y-axis
        if distribution not in ["boxplot", "violin"]:
            fig = self.plot_amplitude_vs_stability(mean_variability, mean_amplitude, column_names, file_name, auto_open)

        # Plot 2 : Show distribution of variability
        else:

            # If set, only keep features with the highest mean amplitude
            if max_features is not None:
                keep = mean_amplitude.argsort()[::-1][:max_features]
                keep = np.sort(keep)

                variability = variability[:, keep]
                mean_amplitude = mean_amplitude[keep]
                dataset = self.explainer.x_init.iloc[:, keep]
                column_names = column_names[keep]

            fig = self.plot_stability_distribution(
                variability, distribution, mean_amplitude, dataset, column_names, file_name, auto_open
            )

        return fig

    def compacity_plot(
        self, selection=None, max_points=2000, force=False, approx=0.9, nb_features=5, file_name=None, auto_open=False
    ):
        """
        The Compacity_plot has the main objective of determining if a small subset of features \
        can be extracted to provide a simpler explanation of the model. \
        indeed, having too many features might negatively affect the model explainability and make it harder to undersand.
        The following two plots are proposed:
        * We identify the minimum number of required features (based on the top contribution values) \
        that well approximate the model, and thus, provide accurate explanations.
        In particular, the prediction with the chosen subset needs to be close enough (*see distance definition below*) \
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
        """
        # Sampling
        if selection is None:
            if self.explainer.x_init.shape[0] <= max_points:
                list_ind = self.explainer.x_init.index.tolist()
            else:
                list_ind = random.sample(self.explainer.x_init.index.tolist(), max_points)
            # By default, don't compute calculation if it has already been done
            if (self.explainer.features_compacity is None) or self.last_compacity_selection or force:
                self.explainer.compute_features_compacity(list_ind, 1 - approx, nb_features)
            else:
                print("Computed values from previous call are used")
            self.last_compacity_selection = False
        elif isinstance(selection, list):
            if len(selection) > max_points:
                print(
                    f"Size of selection is bigger than max_points (default: {max_points}).\
                      Computation time might be affected"
                )
            self.explainer.compute_features_compacity(selection, 1 - approx, nb_features)
            self.last_compacity_selection = True
        else:
            raise ValueError("Parameter selection must be a list")

        features_needed = self.explainer.features_compacity["features_needed"]
        distance_reached = self.explainer.features_compacity["distance_reached"]

        # Make plots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Number of features required<br>to explain " + str(round(100 * approx)) + "% of the model's output",
                "Percentage of the model output<br>explained by the "
                + str(nb_features)
                + " most important<br>features per instance",
            ],
            horizontal_spacing=0.2,
        )

        # Used as titles in make_subplots are considered annotations
        fig.update_annotations(font=self._style_dict["dict_title_compacity"]["font"])

        # First plot: number of features required for a given approximation
        fig.add_trace(
            go.Histogram(
                x=features_needed,
                histnorm="percent",
                cumulative={"enabled": True},
                name="",
                hovertemplate="Top %{x:.0f} features explain at least "
                + str(round(100 * approx))
                + "%<br>of the model for %{y:.1f}% of the instances",
                hovertext="none",
                marker_color=self._style_dict["dict_compacity_bar_colors"][1],
            ),
            row=1,
            col=1,
        )

        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_xaxis["text"] = "Number of selected features"
        dict_yaxis["text"] = "Cumulative distribution over<br>dataset's instances (%)"

        fig.update_xaxes(title=dict_xaxis, row=1, col=1)
        fig.update_yaxes(title=dict_yaxis, row=1, col=1)

        # Second plot: approximation reached for a given number of features
        fig.add_trace(
            go.Histogram(
                x=100 * (1 - distance_reached),
                histnorm="percent",
                cumulative={"enabled": True, "direction": "decreasing"},
                name="",
                hovertemplate="Top "
                + str(nb_features)
                + " features explain at least "
                + "%{x:.0f}"
                + "%<br>of the model for %{y:.1f}% of the instances",
                marker_color=self._style_dict["dict_compacity_bar_colors"][0],
            ),
            row=1,
            col=2,
        )

        dict_xaxis2 = copy.deepcopy(self._style_dict["dict_xaxis"])
        dict_yaxis2 = copy.deepcopy(self._style_dict["dict_yaxis"])
        dict_xaxis2["text"] = "Percentage of model output<br>explained (%)"
        dict_yaxis2["text"] = "Cumulative distribution over<br>dataset's instances (%)"

        fig.update_xaxes(title=dict_xaxis2, row=1, col=2)
        fig.update_yaxes(title=dict_yaxis2, row=1, col=2)

        title = "Compacity of explanations:"
        title += "<span style='font-size: 16px;'><br />How many variables are enough to produce accurate explanations?</span>"
        dict_t = copy.deepcopy(self._style_dict["dict_title_stability"])
        dict_t["text"] = title

        fig.update_layout(
            template="none",
            title=dict_t,
            title_y=0.8,
            hovermode="closest",
            margin={"t": 150},
            showlegend=False,
        )

        if file_name is not None:
            plot(fig, filename=file_name, auto_open=auto_open)

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
        if self.explainer.y_target is not None:
            # Sampling
            list_ind, addnote = self.explainer.plot._subset_sampling(selection, max_points)

            # Classification Case
            if self.explainer._case == "classification":
                fig, subtitle = self.explainer.plot._prediction_classification_plot(list_ind, label)

            # Regression Case
            elif self.explainer._case == "regression":
                fig, subtitle = self.explainer.plot._prediction_regression_plot(list_ind)

            # Add traces, title and template
            title = "True Values Vs Predicted Values"
            if subtitle and addnote:
                title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
            elif subtitle:
                title += "<br><sup>" + subtitle + "</sup>"
            else:
                title += "<br><sup>" + addnote + "</sup>"
            dict_t = copy.deepcopy(self._style_dict["dict_title"])
            dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis"])
            dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis"])
            dict_t["text"] = title
            dict_xaxis["text"] = truncate_str("True Values", 110)
            dict_yaxis["text"] = "Predicted Values"

            fig.update_traces(marker={"size": 10, "opacity": 0.8, "line": {"width": 0.8, "color": "white"}})

            fig.update_layout(
                template="none",
                title=dict_t,
                width=width,
                height=height,
                xaxis_title=dict_xaxis,
                yaxis_title=dict_yaxis,
                hovermode="closest",
            )

            fig.update_yaxes(automargin=True)
            fig.update_xaxes(automargin=True)
            if file_name:
                plot(fig, filename=file_name, auto_open=auto_open)

        else:
            fig = go.Figure()
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                annotations=[
                    {
                        "text": "Provide the y_target argument in the compile() method to display this plot.",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 14},
                    }
                ],
            )

        return fig

    def _prediction_classification_plot(
        self,
        list_ind,
        label=-1,
    ):
        """
        _prediction_classification_plot displays a Plotly violin plot of predictions in comparison to the target variable.
        This plot represents Trues Values versus Predicted Values.

        This plot allows the user to understand the distribution of predictions in comparison to the target variable.
        With the web app, it is possible to select the wrong or correct predictions or a subset of predictions.

        Parameters
        ----------
        label: integer or string (default -1)
            If the label is of string type, check if it can be changed to integer to select the
            good dataframe object.
        list_ind: list
            Contains list of index that we want to plot
        """
        fig = go.Figure()

        label_num, _, label_value = self.explainer.check_label_name(label)
        # predict proba Color scale
        if hasattr(self.explainer.model, "predict_proba"):
            if not hasattr(self.explainer, "proba_values"):
                self.explainer.predict_proba()
            if hasattr(self.explainer.model, "predict"):
                if not hasattr(self.explainer, "y_pred") or self.explainer.y_pred is None:
                    self.explainer.predict()
            # Assign proba values of the target
            df_proba_target = self.explainer.proba_values.copy()
            df_proba_target["proba_target"] = df_proba_target.iloc[:, label_num]
            proba_values = df_proba_target[["proba_target"]]
            # Proba subset:
            proba_values = proba_values.loc[list_ind, :]
            target = self.explainer.y_target.loc[list_ind, :]
            y_pred = self.explainer.y_pred.loc[list_ind, :]
            df_pred = pd.concat(
                [proba_values.reset_index(), y_pred.reset_index(drop=True), target.reset_index(drop=True)], axis=1
            )
            df_pred.set_index(df_pred.columns[0], inplace=True)
            df_pred.columns = ["proba_values", "predict_class", "target"]
            df_pred["wrong_predict"] = 1
            df_pred.loc[(df_pred["predict_class"] == df_pred["target"]), "wrong_predict"] = 0
            subtitle = f"Response: <b>{label_value}</b>"

        # Plot distribution
        fig.add_trace(
            go.Violin(
                x=df_pred["target"].values.flatten(),
                y=df_pred["proba_values"].values.flatten(),
                points=False,
                legendgroup="M",
                scalegroup="M",
                name="Correct Prediction",
                line_color=self._style_dict["violin_area_classif"][1],
                pointpos=-0.1,
                showlegend=False,
                jitter=0.075,
                meanline_visible=True,
                spanmode="hard",
                customdata=df_pred["proba_values"].index.values,
                scalemode="count",
            )
        )

        # Plot points depending if wrong or correct prediction
        df_correct_predict = df_pred[(df_pred["wrong_predict"] == 0)]
        df_wrong_predict = df_pred[(df_pred["wrong_predict"] == 1)]
        hv_text_correct_predict = [
            f"Id: {x}<br />Predicted Values: {y:.3f}<br />Predicted class: {w}<br />True Values: {z}<br />"
            for x, y, w, z in zip(
                df_correct_predict.index,
                df_correct_predict.proba_values.values.round(3).flatten(),
                df_correct_predict.predict_class.values.flatten(),
                df_correct_predict.target.values.flatten(),
            )
        ]
        hv_text_wrong_predict = [
            f"Id: {x}<br />Predicted Values: {y:.3f}<br />Predicted class: {w}<br />True Values: {z}<br />"
            for x, y, w, z in zip(
                df_wrong_predict.index,
                df_wrong_predict.proba_values.values.round(3).flatten(),
                df_wrong_predict.predict_class.values.flatten(),
                df_wrong_predict.target.values.flatten(),
            )
        ]

        fig.add_trace(
            go.Scatter(
                x=df_correct_predict["target"].values.flatten() + np.random.normal(0, 0.02, len(df_correct_predict)),
                y=df_correct_predict["proba_values"].values.flatten(),
                mode="markers",
                marker_color=self._style_dict["prediction_plot"][0],
                showlegend=True,
                name="Correct Prediction",
                hovertext=hv_text_correct_predict,
                hovertemplate="<b>%{hovertext}</b><br />",
                customdata=df_correct_predict["proba_values"].index.values,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_wrong_predict["target"].values.flatten() + np.random.normal(0, 0.02, len(df_wrong_predict)),
                y=df_wrong_predict["proba_values"].values.flatten(),
                mode="markers",
                marker_color=self._style_dict["prediction_plot"][1],
                showlegend=True,
                name="Wrong Prediction",
                hovertext=hv_text_wrong_predict,
                hovertemplate="<b>%{hovertext}</b><br />",
                customdata=df_wrong_predict["proba_values"].index.values,
            )
        )

        fig.update_layout(violingap=0, violinmode="overlay")
        if self.explainer.label_dict is not None:
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(df_pred["target"].unique()),
                ticktext=list(df_pred["target"].apply(lambda x: self.explainer.label_dict[x]).unique()),
            )
        if self.explainer.label_dict is None:
            fig.update_xaxes(tickvals=sorted(list(df_pred["target"].unique())))

        return fig, subtitle

    def _prediction_regression_plot(
        self,
        list_ind,
    ):
        """
        _prediction_regression_plot displays a Plotly scatter plot of predictions in comparison to the target variable.
        This plot represents Trues Values versus Predicted Values.

        This plot allows the user to understand the distribution of predictions in comparison to the target variable.
        With the web app, it is possible to select the wrong or correct predictions or a subset of predictions.

        Parameters
        ----------
        list_ind: list
            Contains list of index that we want to plot
        """
        fig = go.Figure()

        subtitle = None
        if self.explainer.y_pred is None:
            if hasattr(self.explainer.model, "predict"):
                self.explainer.predict()
        prediction_error = self.explainer.prediction_error
        if prediction_error is not None:
            if (self.explainer.y_target == 0).any()[0]:
                subtitle = "Prediction Error = abs(True Values - Predicted Values)"
            else:
                subtitle = "Prediction Error = abs(True Values - Predicted Values) / True Values"
            df_equal_bins = prediction_error.describe(percentiles=np.arange(0.1, 1, 0.1).tolist())
            equal_bins = df_equal_bins.loc[~df_equal_bins.index.isin(["count", "mean", "std"])].values
            equal_bins = np.unique(equal_bins)
            self.pred_colorscale = self.tuning_colorscale(
                pd.DataFrame(
                    pd.cut([val[0] for val in prediction_error.values], bins=[i for i in equal_bins], labels=False)
                )
            )
            col_scale = self.pred_colorscale

            y_pred = self.explainer.y_pred.loc[list_ind]
            y_target = self.explainer.y_target.loc[list_ind]
            prediction_error = np.array(prediction_error.loc[list_ind])
            # round predict
            if self.round_digit is None:
                self.tuning_round_digit()
            y_pred = y_pred.applymap(lambda x: round(x, self.round_digit))

            hv_text = [
                f"Id: {x}<br />True Values: {y:,.2f}<br />Predicted Values: {z:,.2f}<br />Prediction Error: {w:,.2f}"
                for x, y, z, w in zip(
                    y_target.index, y_target.values.flatten(), y_pred.values.flatten(), prediction_error.flatten()
                )
            ]

            fig.add_scatter(
                x=y_target.values.flatten(),
                y=y_pred.values.flatten(),
                mode="markers",
                hovertext=hv_text,
                hovertemplate="<b>%{hovertext}</b><br />",
                customdata=y_pred.index.values,
            )

            colorpoints = pd.cut([val[0] for val in prediction_error], bins=[i for i in equal_bins], labels=False) / 10
            colorbar_title = "Prediction Error"
            fig.data[-1].marker.color = colorpoints.flatten()
            fig.data[-1].marker.coloraxis = "coloraxis"
            fig.layout.coloraxis.colorscale = col_scale
            fig.layout.coloraxis.colorbar = {
                "title": {"text": colorbar_title},
                "tickvals": [col_scale[0][0], col_scale[-1][0] - 0.15],
                "ticktext": [float("{:0.3f}".format(equal_bins[0])), float("{:0.3f}".format(equal_bins[-2]))],
                "tickformat": ".2s",
                "yanchor": "top",
                "y": 1.1,
            }
            range_axis = [
                min(min(y_target.values.flatten()), min(y_pred.values.flatten())),
                max(max(y_target.values.flatten()), max(y_pred.values.flatten())),
            ]
            fig.update_xaxes(range=range_axis)
            fig.update_yaxes(range=range_axis)
            fig.update_layout(
                shapes=[
                    {
                        "type": "line",
                        "yref": "y domain",
                        "xref": "x domain",
                        "y0": 0,
                        "y1": 1,
                        "x0": 0,
                        "x1": 1,
                        "line": dict(color="grey", width=1, dash="dot"),
                    }
                ]
            )

        return fig, subtitle

    def _subset_sampling(self, selection=None, max_points=2000):
        """
        Subset sampling for plots and create addnote for subtitle

        Parameters
        ----------
        selection: list (optional)
            Contains list of index, subset of the input DataFrame that we want to plot
        max_points: int (optional, default: 2000)
            maximum number to plot in contribution plot. if input dataset is bigger than max_points,
            a sample limits the number of points to plot.
            nb: you can also limit the number using 'selection' parameter.
        """

        # Sampling
        if selection is None:
            if self.explainer.x_init.shape[0] <= max_points:
                list_ind = self.explainer.x_init.index.tolist()
                addnote = None
            else:
                if self.explainer.x_init.shape[0] <= max_points:
                    list_ind = self.explainer.x_init.index.tolist()
                    addnote = None
                else:
                    random.seed(79)
                    list_ind = random.sample(self.explainer.x_init.index.tolist(), max_points)
                    addnote = "Length of random Subset: "
        elif isinstance(selection, list):
            if len(selection) <= max_points:
                list_ind = selection
                addnote = "Length of user-defined Subset: "
            else:
                random.seed(79)
                list_ind = random.sample(selection, max_points)
                addnote = "Length of random Subset: "
        else:
            raise ValueError("parameter selection must be a list")
        if addnote is not None:
            addnote = add_text(
                [addnote, f"{len(list_ind)} ({int(np.round(100 * len(list_ind) / self.explainer.x_init.shape[0]))}%)"],
                sep="",
            )

        return list_ind, addnote
