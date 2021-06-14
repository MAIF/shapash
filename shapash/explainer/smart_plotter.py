"""
Smart plotter module
"""
import warnings
from numbers import Number
import random
import copy
import numpy as np
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot
from shapash.manipulation.select_lines import select_lines
from shapash.manipulation.summarize import compute_features_import, project_feature_values_1d, compute_corr
from shapash.utils.utils import add_line_break, truncate_str, compute_digit_number, add_text, \
    maximum_difference_sort_value, compute_sorted_variables_interactions_list_indices, \
    compute_top_correlations_features
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
        self.dict_title = {
            'xanchor': "center",
            'yanchor': "middle",
            'x': 0.5,
            'y': 0.9,
            'font': {
                'size': 24,
                'family': "Arial",
                'color': "rgb(50, 50, 50)"
            }
        }
        self.dict_xaxis = {
            'font': {
                'size': 16,
                'family': "Arial Black",
                'color': "rgb(50, 50, 50)"
            }
        }
        self.dict_yaxis = {
            'font': {
                'size': 16,
                'family': "Arial Black",
                'color': "rgb(50, 50, 50)"
            }
        }
        self.dict_ycolors = {
            1: "rgba(255, 166, 17, 0.9)",
            0: "rgba(117, 152, 189, 0.9)"
        }
        self.init_colorscale = [
            "rgb(52, 55, 54)",
            "rgb(74, 99, 138)",
            "rgb(116, 153, 214)",
            "rgb(162, 188, 213)",
            "rgb(212, 234, 242)",
            "rgb(235, 216, 134)",
            "rgb(255, 204, 83)",
            "rgb(244 ,192, 0)",
            "rgb(255, 166, 17)",
            "rgb(255, 123, 38)",
            "rgb(255, 77, 7)"
        ]
        self.default_color = 'rgba(117, 152, 189, 0.9)'
        self.dict_featimp_colors = {
            1: {
                'color': 'rgba(244, 192, 0, 1.0)',
                'line': {
                    'color': 'rgba(52, 55, 54, 0.8)',
                    'width': 0.5
                }
            },
            2: {
                'color': 'rgba(52, 55, 54, 0.7)'
            }
        }
        self.dict_local_plot_colors = {
            1: {
                'color': 'rgba(244, 192, 0, 1.0)',
                'line': {
                    'color': 'rgba(52, 55, 54, 0.8)',
                    'width': 0.5
                }
            },
            -1: {
                'color': 'rgba(74, 99, 138, 0.7)',
                'line': {
                    'color': 'rgba(27, 28, 28, 1.0)',
                    'width': 0.5
                }
            },
            0: {
                'color': 'rgba(113, 101, 59, 1.0)',
                'line': {
                    'color': 'rgba(52, 55, 54, 0.8)',
                    'width': 0.5
                }
            },
            -2: {
                'color': 'rgba(52, 55, 54, 0.7)',
                'line': {
                    'color': 'rgba(27, 28, 28, 1.0)',
                    'width': 0.5
                }
            }
        }

        self.dict_compare_colors = [
            'rgba(244, 192, 0, 1.0)',
            'rgba(74, 99, 138, 0.7)',
            'rgba(113, 101, 59, 1.0)',
            "rgba(183, 58, 56, 0.9)",
            "rgba(255, 123, 38, 1.0)",
            'rgba(0, 21, 179, 0.97)',
            'rgba(116, 1, 179, 0.9)',
        ]

        self.groups_colors = [
            px.colors.qualitative.T10[1],
            px.colors.qualitative.G10[9]
        ]

        self.round_digit = None

        self.interactions_col_scale = ["rgb(175, 169, 157)", "rgb(255, 255, 255)", "rgb(255, 77, 7)"]

        self.interactions_discrete_colors = px.colors.qualitative.Antique

    def tuning_colorscale(self, values):
        """
        adapts the color scale to the distribution of points

        Parameters
        ----------
        values: 1 column pd.DataFrame
            values ​​whose quantiles must be calculated
        """
        desc_df = values.describe(percentiles=np.arange(0.1, 1, 0.1).tolist())
        min_pred, max_pred = list(desc_df.loc[['min', 'max']].values)
        desc_pct_df = (desc_df.loc[~desc_df.index.isin(['count', 'mean', 'std'])] - min_pred) / \
                      (max_pred - min_pred)
        color_scale = list(map(list, (zip(desc_pct_df.values.flatten(), self.init_colorscale))))
        return color_scale

    def tuning_round_digit(self):
        """
        adapts the display of the number of digit to the distribution of points
        """
        quantile = [0.25, 0.75]
        desc_df = self.explainer.y_pred.describe(percentiles=quantile)
        perc1, perc2 = list(desc_df.loc[[str(int(p * 100)) + '%' for p in quantile]].values)
        p_diff = perc2 - perc1
        self.round_digit = compute_digit_number(p_diff)

    def _update_contributions_fig(self,
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
                                  auto_open):
        """
        Function used by both violin and scatter methods for contributions plots in order to update the layout
        of the (already) created plotly figure.

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
        if subtitle or addnote:
            title += f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
        dict_t = copy.deepcopy(self.dict_title)
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_t['text'] = title
        dict_xaxis['text'] = truncate_str(feature_name, 110)
        dict_yaxis['text'] = 'Contribution'

        if self.explainer._case == "regression":
            colorpoints = pred
            colorbar_title = 'Predicted'
        elif self.explainer._case == "classification":
            colorpoints = proba_values
            colorbar_title = 'Predicted Proba'

        if colorpoints is not None:
            fig.data[-1].marker.color = colorpoints.values.flatten()
            fig.data[-1].marker.coloraxis = 'coloraxis'
            fig.layout.coloraxis.colorscale = col_scale
            fig.layout.coloraxis.colorbar = {'title': {'text': colorbar_title}}

        elif fig.data[0].type != 'violin':
            if self.explainer._case == 'classification' and pred is not None:
                fig.data[-1].marker.color = pred.iloc[:, 0].apply(lambda
                                                                  x: self.dict_ycolors[1] if x == col_modality else
                                                                  self.dict_ycolors[0])
            else:
                fig.data[-1].marker.color = self.default_color

        fig.update_traces(
            marker={
                'size': 10,
                'opacity': 0.8,
                'line': {'width': 0.8, 'color': 'white'}
            }
        )

        fig.update_layout(
            template='none',
            title=dict_t,
            width=width,
            height=height,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode='closest'
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

    def plot_scatter(self,
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
                     auto_open=False):
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
        """
        fig = go.Figure()

        # add break line to X label if necessary
        max_len_by_row = max([round(50 / self.explainer.features_desc[feature_values.columns.values[0]]), 8])
        feature_values.iloc[:, 0] = feature_values.iloc[:, 0].apply(add_line_break, args=(max_len_by_row, 120,))

        if pred is not None:
            hv_text = [f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten())]
        else:
            hv_text = [f"Id: {x}" for x in feature_values.index]

        if metadata:
            metadata = {k: [round_to_k(x, 3) if isinstance(x, Number) else x for x in v]
                        for k, v in metadata.items()}
            text_groups_features = np.swap = np.array([col_values for col_values in metadata.values()])
            text_groups_features = np.swapaxes(text_groups_features, 0, 1)
            text_groups_features_keys = list(metadata.keys())
            hovertemplate = '<b>%{hovertext}</b><br />' + \
                            'Contribution: %{y:.4f} <br />' + \
                            '<br />'.join([
                                '{}: %{{text[{}]}}'.format(text_groups_features_keys[i], i)
                                for i in range(len(text_groups_features_keys))
                            ]) + '<extra></extra>'
        else:
            hovertemplate = '<b>%{hovertext}</b><br />' +\
                            f'{feature_name} : ' +\
                            '%{x}<br />Contribution: %{y:.4f}<extra></extra>'
            text_groups_features = None

        fig.add_scatter(
            x=feature_values.values.flatten(),
            y=contributions.values.flatten(),
            mode='markers',
            hovertext=hv_text,
            hovertemplate=hovertemplate,
            customdata=feature_values.index.values,
            text=text_groups_features
        )

        self._update_contributions_fig(fig=fig,
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
                                       auto_open=auto_open)

        return fig

    def plot_violin(self,
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
                    auto_open=False):
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
        """
        fig = go.Figure()

        points_param = False if proba_values is not None else "all"
        jitter_param = 0.075

        if pred is not None:
            hv_text = [f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten())]
        else:
            hv_text = [f"Id: {x}" for x in feature_values.index]
        hv_text_df = pd.DataFrame(hv_text, columns=['text'], index=feature_values.index)
        hv_temp = f'{feature_name} :<br />' + '%{x}<br />Contribution: %{y:.4f}<extra></extra>'

        # add break line to X label
        max_len_by_row = max([round(50 / self.explainer.features_desc[feature_values.columns.values[0]]), 8])
        feature_values.iloc[:, 0] = feature_values.iloc[:, 0].apply(add_line_break, args=(max_len_by_row, 120,))

        uniq_l = list(pd.unique(feature_values.values.flatten()))
        uniq_l.sort()

        for i in uniq_l:
            if pred is not None and self.explainer._case == 'classification':
                fig.add_trace(go.Violin(x=feature_values.loc[(pred.iloc[:, 0] != col_modality) &
                                                             (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        y=contributions.loc[(pred.iloc[:, 0] != col_modality) &
                                                            (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        points=points_param,
                                        pointpos=-0.1,
                                        side='negative',
                                        line_color=self.dict_ycolors[0],
                                        showlegend=False,
                                        jitter=jitter_param,
                                        meanline_visible=True,
                                        hovertext=hv_text_df.loc[(pred.iloc[:, 0] != col_modality) &
                                                                 (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        hovertemplate='<b>%{hovertext}</b><br />' + hv_temp,
                                        customdata=contributions.loc[(pred.iloc[:, 0] != col_modality) &
                                                                     (feature_values.iloc[:, 0] == i)].index.values
                                        ))
                fig.add_trace(go.Violin(x=feature_values.loc[(pred.iloc[:, 0] == col_modality) &
                                                             (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        y=contributions.loc[(pred.iloc[:, 0] == col_modality) &
                                                            (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        points=points_param,
                                        pointpos=0.1,
                                        side='positive',
                                        line_color=self.dict_ycolors[1],
                                        showlegend=False,
                                        jitter=jitter_param,
                                        meanline_visible=True,
                                        scalemode='count',
                                        hovertext=hv_text_df.loc[(pred.iloc[:, 0] == col_modality) &
                                                                 (feature_values.iloc[:, 0] == i)].values.flatten(),
                                        hovertemplate='<b>%{hovertext}</b><br />' + hv_temp,
                                        customdata=contributions.loc[(pred.iloc[:, 0] == col_modality) &
                                                                     (feature_values.iloc[:, 0] == i)].index.values
                                        ))

            else:
                fig.add_trace(go.Violin(x=feature_values.loc[feature_values.iloc[:, 0] == i].values.flatten(),
                                        y=contributions.loc[feature_values.iloc[:, 0] == i].values.flatten(),
                                        line_color=self.default_color,
                                        showlegend=False,
                                        meanline_visible=True,
                                        scalemode='count',
                                        hovertext=hv_text_df.loc[feature_values.iloc[:, 0] == i].values.flatten(),
                                        hovertemplate='<b>%{hovertext}</b><br />' + hv_temp,
                                        customdata=contributions.index.values
                                        ))
                if pred is None:
                    fig.data[-1].points = points_param
                    fig.data[-1].pointpos = 0
                    fig.data[-1].jitter = jitter_param

        colorpoints = pred if self.explainer._case == "regression" else proba_values if \
            self.explainer._case == 'classification' else None

        if colorpoints is not None:
            fig.add_trace(go.Scatter(
                x=feature_values.values.flatten(),
                y=contributions.values.flatten(),
                mode='markers',
                showlegend=False,
                hovertext=hv_text,
                hovertemplate='<b>%{hovertext}</b><br />' + hv_temp,
                customdata=contributions.index.values,
            ))

        fig.update_layout(
            violingap=0.05,
            violingroupgap=0,
            violinmode='overlay',
            xaxis_type='category'
        )

        fig.update_xaxes(range=[-0.6, len(uniq_l) - 0.4])

        self._update_contributions_fig(fig=fig,
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
                                       auto_open=auto_open)

        return fig

    def plot_features_import(self,
                             feature_imp1,
                             feature_imp2=None,
                             title='Features Importance',
                             addnote=None,
                             subtitle=None,
                             width=900,
                             height=500,
                             file_name=None,
                             auto_open=False):
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
        """
        dict_t = copy.deepcopy(self.dict_title)
        topmargin = 80
        if subtitle or addnote:
            title += f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
            topmargin = topmargin + 15
        dict_t.update(text=title)
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_xaxis.update(text='Contribution')
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_yaxis.update(text=None)
        dict_style_bar1 = self.dict_featimp_colors[1]
        dict_style_bar2 = self.dict_featimp_colors[2]
        dict_yaxis['text'] = None

        # Change bar color for groups of features
        marker_color = [
            self.groups_colors[0]
            if (
                    self.explainer.features_groups is not None
                    and self.explainer.inv_features_dict.get(f.replace("<b>", "").replace("</b>", ""))
                    in self.explainer.features_groups.keys()
            )
            else dict_style_bar1["color"]
            for f in feature_imp1.index
        ]

        layout = go.Layout(
            barmode='group',
            template='none',
            autosize=False,
            width=width,
            height=height,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode='closest',
            margin={
                'l': 160,
                'r': 0,
                't': topmargin,
                'b': 50
            }
        )
        bar1 = go.Bar(
            x=feature_imp1.round(4),
            y=feature_imp1.index,
            orientation='h',
            name='Global',
            marker=dict_style_bar1,
            marker_color=marker_color
        )
        if feature_imp2 is not None:
            bar2 = go.Bar(
                x=feature_imp2.round(4),
                y=feature_imp2.index,
                orientation='h',
                name='Subset',
                marker=dict_style_bar2
            )
            data = [bar2, bar1]
        else:
            data = bar1
        fig = go.Figure(data=data, layout=layout)
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
        return fig

    def plot_bar_chart(self,
                       index_value,
                       var_dict,
                       x_val,
                       contrib,
                       yaxis_max_label=12,
                       subtitle=None,
                       width=900,
                       height=550,
                       file_name=None,
                       auto_open=False):
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

        Returns
        -------
        plotly bar plot
            A bar plot with selected contributions and
            associated feature values for one observation.
        """
        dict_t = copy.deepcopy(self.dict_title)
        topmargin = 80
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_local_plot_colors = copy.deepcopy(self.dict_local_plot_colors)
        if len(index_value) == 0:
            warnings.warn('Only one line/observation must match the condition', UserWarning)
            dict_t['text'] = "Local Explanation - <b>No Matching Entry</b>"
        else:
            title = f"Local Explanation - Id: <b>{index_value[0]}</b>"
            if subtitle:
                title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
                topmargin += 15
            dict_t['text'] = title
        dict_xaxis['text'] = 'Contribution'
        dict_yaxis['text'] = None

        layout = go.Layout(
            barmode='group',
            template='none',
            width=width,
            height=height,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            yaxis_type='category',
            hovermode='closest',
            margin={
                'l': 150,
                'r': 20,
                't': topmargin,
                'b': 70
            }
        )
        bars = []
        for num, expl in enumerate(list(zip(var_dict, x_val, contrib))):
            group_name = None
            if expl[1] == '':
                ylabel = '<i>{}</i>'.format(expl[0])
                hoverlabel = '<b>{}</b>'.format(expl[0])
            else:
                # If bar is a group of features, hovertext includes the values of the features of the group
                # And color changes
                if (self.explainer.features_groups is not None
                        and self.explainer.inv_features_dict.get(expl[0]) in self.explainer.features_groups.keys()
                        and len(index_value) > 0):
                    group_name = self.explainer.inv_features_dict.get(expl[0])
                    feat_groups_values = self.explainer.x_pred[self.explainer.features_groups[group_name]]\
                                                       .loc[index_value[0]]
                    hoverlabel = '<br />'.join([
                        '<b>{} :</b>{}'.format(add_line_break(self.explainer.features_dict.get(f_name, f_name),
                                                              40, maxlen=120),
                                               add_line_break(f_value, 40, maxlen=160))
                        for f_name, f_value in feat_groups_values.to_dict().items()
                    ])
                else:
                    hoverlabel = '<b>{} :</b><br />{}'.format(add_line_break(expl[0], 40, maxlen=120),
                                                              add_line_break(expl[1], 40, maxlen=160))
                if len(contrib) <= yaxis_max_label and (
                        self.explainer.features_groups is None
                        # We don't want to display label values for t-sne projected values of groups of features.
                        or (
                                self.explainer.features_groups is not None
                                and self.explainer.inv_features_dict.get(expl[0])
                                not in self.explainer.features_groups.keys()
                        )
                ):
                        ylabel = '<b>{} :</b><br />{}'.format(truncate_str(expl[0], 45), truncate_str(expl[1], 45))

                else:
                    ylabel = ('<b>{}</b>'.format(truncate_str(expl[0], maxlen=45)))
            contrib_value = expl[2]
            # colors
            if contrib_value >= 0:
                color = 1 if expl[1] != '' else 0
            else:
                color = -1 if expl[1] != '' else -2

            # If the bar is a group of features we modify the color
            if group_name is not None:
                bar_color = self.groups_colors[0] if color == 1 else self.groups_colors[1]
            else:
                bar_color = dict_local_plot_colors[color]['color']

            barobj = go.Bar(
                x=[contrib_value],
                y=[ylabel],
                customdata=[hoverlabel],
                orientation='h',
                marker=dict_local_plot_colors[color],
                marker_color=bar_color,
                showlegend=False,
                hovertemplate='%{customdata}<br />Contribution: %{x:.4f}<extra></extra>'
            )
            bars.append([color, contrib_value, num, barobj])

        bars.sort()
        fig = go.Figure(data=[x[-1] for x in bars], layout=layout)
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
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
        if hasattr(self.explainer, 'mask'):
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
        if hasattr(self.explainer, 'masked_contributions'):
            if isinstance(self.explainer.masked_contributions, list):
                ext_contrib = self.explainer.masked_contributions[label].loc[line[0], :].values
            else:
                ext_contrib = self.explainer.masked_contributions.loc[line[0], :].values

            ext_var_dict = ['Hidden Negative Contributions', 'Hidden Positive Contributions']
            ext_x = ['', '']
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
            if hasattr(self.explainer.model, 'predict_proba'):
                if not hasattr(self.explainer, "proba_values"):
                    self.explainer.predict_proba()
                value = self.explainer.proba_values.iloc[:, [label]].loc[index].values[0]
            else:
                value = None
        elif self.explainer._case == "regression":
            if self.explainer.y_pred is not None:
                value = self.explainer.y_pred.loc[index]
            else:
                value = self.explainer.model.predict(self.explainer.x_init.loc[[index]])[0]

        if isinstance(value, pd.Series):
            value = value.values[0]

        return value

    def local_plot(self,
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
                   auto_open=False):
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
        # checking args
        if sum(arg is not None for arg in [query, row_num, index]) != 1:
            raise ValueError(
                "You have to specify just one of these arguments: query, nrow, index"
            )

        if index is not None:
            if index in self.explainer.x_pred.index:
                line = [index]
            else:
                line = []
        elif row_num is not None:
            line = [self.explainer.x_pred.index[row_num]]
        elif query is not None:
            line = select_lines(self.explainer.x_pred, query)

        subtitle = ""

        if len(line) != 1:
            if len(line) > 1:
                raise ValueError('Only one line/observation must match the condition')
            contrib = []
            x_val = []
            var_dict = []

        else:
            # apply filter if the method have not yet been asked in order to limit the number of feature to display
            if (
                not hasattr(self.explainer, "mask_params")  # If the filter method has not been called yet
                # Or if the already computed mask was not updated with current display_groups parameter
                or (isinstance(data["contrib_sorted"], pd.DataFrame)
                    and len(data["contrib_sorted"].columns) != len(self.explainer.mask.columns))
                or (isinstance(data["contrib_sorted"], list)
                    and len(data["contrib_sorted"][0].columns) != len(self.explainer.mask[0].columns))
            ):
                self.explainer.filter(max_contrib=20, display_groups=display_groups)

            if self.explainer._case == "classification":
                if label is None:
                    label = -1

                label_num, _, label_value = self.explainer.check_label_name(label)

                contrib = data['contrib_sorted'][label_num]
                x_val = data['x_sorted'][label_num]
                var_dict = data['var_dict'][label_num]

                if show_predict is True:
                    pred = self.local_pred(line[0], label_num)
                    if pred is None:
                        subtitle = f"Response: <b>{label_value}</b> - No proba available"
                    else:
                        subtitle = f"Response: <b>{label_value}</b> - Proba: <b>{pred:.4f}</b>"

            elif self.explainer._case == "regression":
                contrib = data['contrib_sorted']
                x_val = data['x_sorted']
                var_dict = data['var_dict']
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
                var_dict = [self.explainer.features_dict[self.explainer.x_pred_groups.columns[x]] for x in var_dict]
            else:
                var_dict = [self.explainer.features_dict[self.explainer.columns_dict[x]] for x in var_dict]
            if show_masked:
                var_dict, x_val, contrib = self.check_masked_contributions(line, var_dict, x_val, contrib, label=label_num)
            # Filtering all negative or positive contrib if specify in mask
            exclusion = []
            if hasattr(self.explainer, 'mask_params'):
                if self.explainer.mask_params['positive'] is True:
                    exclusion = np.where(np.array(contrib) < 0)[0].tolist()
                elif self.explainer.mask_params['positive'] is False:
                    exclusion = np.where(np.array(contrib) > 0)[0].tolist()
            exclusion.sort(reverse=True)
            for expl in exclusion:
                del var_dict[expl]
                del x_val[expl]
                del contrib[expl]

        fig = self.plot_bar_chart(line, var_dict, x_val, contrib, yaxis_max_label, subtitle, width, height, file_name,
                                  auto_open)
        return fig

    def contribution_plot(self,
                          col,
                          selection=None,
                          label=-1,
                          violin_maxf=10,
                          max_points=2000,
                          proba=True,
                          width=900,
                          height=600,
                          file_name=None,
                          auto_open=False):
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
            raise ValueError('parameter col must be string or int.')
        if hasattr(self.explainer, 'inv_features_dict'):
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

        # Sampling
        if selection is None:
            if self.explainer.x_pred.shape[0] <= max_points:
                list_ind = self.explainer.x_pred.index.tolist()
                addnote = None
            else:
                list_ind = random.sample(self.explainer.x_pred.index.tolist(), max_points)
                addnote = "Length of random Subset : "
        elif isinstance(selection, list):
            if len(selection) <= max_points:
                list_ind = selection
                addnote = "Length of user-defined Subset : "
            else:
                list_ind = random.sample(selection, max_points)
                addnote = "Length of random Subset : "
        else:
            raise ValueError('parameter selection must be a list')
        if addnote is not None:
            addnote = add_text([addnote,
                                f"{len(list_ind)} ({int(np.round(100 * len(list_ind) / self.explainer.x_pred.shape[0]))}%)"],
                               sep='')

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
            feature_values = self.explainer.x_pred.loc[list_ind, col_name]

        if col_is_group:
            feature_values = project_feature_values_1d(feature_values, col, self.explainer.x_pred,
                                                       self.explainer.x_init, self.explainer.preprocessing,
                                                       features_dict=self.explainer.features_dict)
            contrib = subcontrib.loc[list_ind, col].to_frame()
            if self.explainer.features_imp is None:
                self.explainer.compute_features_import()
            features_imp = self.explainer.features_imp if isinstance(self.explainer.features_imp, pd.Series) \
                else self.explainer.features_imp[0]
            top_features_of_group = features_imp.loc[self.explainer.features_groups[col]] \
                                                .sort_values(ascending=False)[:4].index  # Displaying top 4 features
            metadata = {
                self.explainer.features_dict[f_name]: self.explainer.x_pred[f_name]
                for f_name in top_features_of_group
            }
            text_group = "Features values were projected on the x axis using t-SNE"
            if addnote is not None:
                addnote = add_text([addnote, text_group], sep=' - ')
            else:
                addnote = text_group
        else:
            contrib = subcontrib.loc[list_ind, col_name].to_frame()
            metadata = None
        feature_values = feature_values.to_frame()

        if self.explainer.y_pred is not None:
            y_pred = self.explainer.y_pred.loc[list_ind]
            # Add labels if exist
            if self.explainer._case == 'classification' and self.explainer.label_dict is not None:
                y_pred = y_pred.applymap(lambda x: self.explainer.label_dict[x])
                col_value = self.explainer.label_dict[col_value]
            # round predict
            elif self.explainer._case == 'regression':
                if self.round_digit is None:
                    self.tuning_round_digit()
                y_pred = y_pred.applymap(lambda x: round(x, self.round_digit))
        else:
            y_pred = None

        # selecting the best plot : Scatter, Violin?
        if col_value_count > violin_maxf:
            fig = self.plot_scatter(feature_values, contrib, col_label, y_pred, proba_values, col_value, col_scale,
                                    metadata, addnote,
                                    subtitle, width, height, file_name, auto_open)
        else:
            fig = self.plot_violin(feature_values, contrib, col_label, y_pred, proba_values, col_value, col_scale,
                                   addnote,
                                   subtitle, width, height, file_name, auto_open)

        return fig

    def features_importance(self,
                            max_features=20,
                            selection=None,
                            label=-1,
                            group_name=None,
                            display_groups=True,
                            force=False,
                            width=900,
                            height=500,
                            file_name=None,
                            auto_open=False):
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
            This  argument allows to represent the importance calculated with a subset.
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

        Returns
        -------
        Plotly Figure Object

        Example
        --------
        >>> xpl.plot.features_importance()
        """

        self.explainer.compute_features_import(force=force)
        subtitle = None
        title = 'Features Importance'
        display_groups = self.explainer.features_groups is not None and display_groups

        if display_groups:
            if group_name:  # Case where we have groups of features and we want to display only features inside a group
                if group_name not in self.explainer.features_groups.keys():
                    raise ValueError(f"group_name parameter : {group_name} is not in features_groups keys. "
                                     f"Possible values are : {list(self.explainer.features_groups.keys())}")
                title += f' - {truncate_str(self.explainer.features_dict.get(group_name), 20)}'
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
                subset = contributions[label_num].loc[selection]
                subset_feat_imp = compute_features_import(subset)
                subset_feat_imp = subset_feat_imp.reindex(global_feat_imp.index)
            else:
                subset_feat_imp = None
            subtitle = f"Response: <b>{label_value}</b>"
        # regression
        elif self.explainer._case == "regression":
            global_feat_imp = features_importance.tail(max_features)
            if selection is not None:
                subset = contributions.loc[selection]
                subset_feat_imp = compute_features_import(subset)
                subset_feat_imp = subset_feat_imp.reindex(global_feat_imp.index)
            else:
                subset_feat_imp = None
        addnote = ''
        if subset_feat_imp is not None:
            subset_feat_imp.index = subset_feat_imp.index.map(self.explainer.features_dict)
            if subset_feat_imp.dropna().shape[0] == 0:
                raise ValueError("selection argument doesn't return any row")
            subset_len = subset.shape[0]
            total_len = self.explainer.x_pred.shape[0]
            addnote = add_text([addnote,
                                f"Subset length: {subset_len} ({int(np.round(100 * subset_len / total_len))}%)"],
                               sep=" - ")
        if self.explainer.x_pred.shape[1] >= max_features:
            addnote = add_text([addnote,
                                f"Total number of features: {int(self.explainer.x_pred.shape[1])}"],
                               sep=" - ")

        global_feat_imp.index = global_feat_imp.index.map(self.explainer.features_dict)
        if display_groups:
            # Bold font for groups of features
            global_feat_imp.index = [
                '<b>' + str(f) + '</b>'
                if self.explainer.inv_features_dict.get(f) in self.explainer.features_groups.keys()
                else str(f) for f in global_feat_imp.index
            ]

            if subset_feat_imp is not None:
                subset_feat_imp.index = [
                    '<b>' + str(f) + '</b>'
                    if self.explainer.inv_features_dict.get(f) in self.explainer.features_groups.keys()
                    else str(f) for f in subset_feat_imp.index
                ]

        fig = self.plot_features_import(global_feat_imp, subset_feat_imp, title, addnote,
                                        subtitle, width, height, file_name, auto_open)
        return fig

    def plot_line_comparison(self,
                             index,
                             feature_values,
                             contributions,
                             predictions=None,
                             dict_features=None,
                             subtitle=None,
                             width=900,
                             height=550,
                             file_name=None,
                             auto_open=False):
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

        dict_t = copy.deepcopy(self.dict_title)
        topmargin = 80
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)

        if len(index) == 0:
            warnings.warn('No individuals matched', UserWarning)
            dict_t['text'] = "Compare plot - <b>No Matching Reference Entry</b>"
        elif len(index) < 2:
            warnings.warn('Comparison needs at least 2 individuals', UserWarning)
            dict_t['text'] = "Compare plot - index : " + ' ; '.join(['<b>' + str(id) + '</b>' for id in index])
        else:
            dict_t['text'] = "Compare plot - index : " + ' ; '.join(['<b>' + str(id) + '</b>' for id in index])

            dict_xaxis['text'] = "Contributions"

        dict_yaxis['text'] = None

        if subtitle is not None:
            topmargin += 15 * height / 275
            dict_t['text'] = truncate_str(dict_t['text'], 120) \
                + f"<span style='font-size: 12px;'><br />{truncate_str(subtitle, 200)}</span>"

        layout = go.Layout(
            template='none',
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            yaxis_type='category',
            width=width,
            height=height,
            hovermode='closest',
            legend=dict(x=1, y=1),
            margin={
                'l': 150,
                'r': 20,
                't': topmargin,
                'b': 70
            }
        )

        iteration_list = list(zip(contributions, feature_values))

        dic_color = copy.deepcopy(self.dict_compare_colors)
        lines = list()

        for i, id_i in enumerate(index):
            x_i = list()
            features = list()
            x_val = predictions[i]
            x_hover = list()

            for contrib, feat in iteration_list:
                x_i.append(contrib[i])
                features.append('<b>' + str(feat) + '</b>')
                pred_x_val = x_val[dict_features[feat]]
                x_hover.append(f"Id: <b>{add_line_break(id_i, 40, 160)}</b>"
                               + f"<br /><b>{add_line_break(feat, 40, 160)}</b> <br />"
                               + f"Contribution: {contrib[i]:.4f} <br />Value: "
                               + str(add_line_break(pred_x_val, 40, 160)))

            lines.append(go.Scatter(
                x=x_i,
                y=features,
                mode='lines+markers',
                showlegend=True,
                name=f"Id: <b>{index[i]}</b>",
                hoverinfo="text",
                hovertext=x_hover,
                marker={'color': dic_color[i % len(dic_color)]}
            )
            )

        fig = go.Figure(data=lines, layout=layout)
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name is not None:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def compare_plot(self,
                     index=None,
                     row_num=None,
                     label=None,
                     max_features=20,
                     width=900,
                     height=550,
                     show_predict=True,
                     file_name=None,
                     auto_open=True):
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
            raise ValueError(
                "You have to specify just one of these arguments: index, row_num"
            )
        # Getting indexes in a list
        line_reference = []
        if index is not None:
            for ident in index:
                if ident in self.explainer.x_pred.index:
                    line_reference.append(ident)

        elif row_num is not None:
            line_reference = [self.explainer.x_pred.index.values[row_nb_reference]
                              for row_nb_reference in row_num
                              if self.explainer.x_pred.index.values[row_nb_reference] in self.explainer.x_pred.index]

        subtitle = ""
        if len(line_reference) < 1:
            raise ValueError('No matching entry for index')

        # Classification case
        if self.explainer._case == 'classification':
            if label is None:
                label = -1

            label_num, _, label_value = self.explainer.check_label_name(label)
            contrib = self.explainer.contributions[label_num]

            if show_predict:
                preds = [self.local_pred(line, label_num) for line in line_reference]
                subtitle = f"Response: <b>{label_value}</b> - " \
                           + "Probas: " \
                           + ' ; '.join([str(id) + ': <b>' + str(round(proba, 2)) + '</b>'
                                         for proba, id in zip(preds, line_reference)])

        # Regression case
        elif self.explainer._case == 'regression':
            contrib = self.explainer.contributions

            if show_predict:
                preds = [self.local_pred(line) for line in line_reference]
                subtitle = "Predictions: " + ' ; '.join([str(id) + ': <b>' + str(round(pred, 2)) + '</b>'
                                                         for id, pred in zip(line_reference, preds)])

        new_contrib = list()
        for ident in line_reference:
            new_contrib.append(contrib.loc[ident])
        new_contrib = np.array(new_contrib).T

        # Well labels if available
        feature_values = [0] * len(contrib.columns)
        if hasattr(self.explainer, 'columns_dict'):
            for i, name in enumerate(contrib.columns):
                feature_name = self.explainer.features_dict[name]
                feature_values[i] = feature_name

        preds = [self.explainer.x_pred.loc[id] for id in line_reference]
        dict_features = self.explainer.inv_features_dict

        iteration_list = list(zip(new_contrib, feature_values))
        iteration_list.sort(key=lambda x: maximum_difference_sort_value(x), reverse=True)
        iteration_list = iteration_list[:max_features]
        iteration_list = iteration_list[::-1]
        new_contrib, feature_values = list(zip(*iteration_list))

        fig = self.plot_line_comparison(line_reference, feature_values, new_contrib,
                                        predictions=preds, dict_features=dict_features,
                                        width=width, height=height, subtitle=subtitle,
                                        file_name=file_name, auto_open=auto_open)

        return fig

    def _plot_interactions_scatter(self,
                                   x_name,
                                   y_name,
                                   col_name,
                                   x_values,
                                   y_values,
                                   col_values,
                                   col_scale):
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
        x_values.iloc[:, 0] = x_values.iloc[:, 0].apply(add_line_break, args=(max_len_by_row, 120,))

        data_df = pd.DataFrame({
            x_name: x_values.values.flatten(),
            y_name: y_values.values.flatten(),
            col_name: col_values.values.flatten()
        })

        if isinstance(col_values.values.flatten()[0], str):
            fig = px.scatter(data_df, x=x_name, y=y_name, color=col_name,
                             color_discrete_sequence=self.interactions_discrete_colors)
        else:
            fig = px.scatter(data_df, x=x_name, y=y_name, color=col_name, color_continuous_scale=col_scale)

        fig.update_traces(mode='markers')

        return fig

    def _plot_interactions_violin(self,
                                  x_name,
                                  y_name,
                                  col_name,
                                  x_values,
                                  y_values,
                                  col_values,
                                  col_scale):
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
        x_values.iloc[:, 0] = x_values.iloc[:, 0].apply(add_line_break, args=(max_len_by_row, 120,))

        uniq_l = list(pd.unique(x_values.values.flatten()))
        uniq_l.sort()

        for i in uniq_l:
            fig.add_trace(go.Violin(x=x_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                                    y=y_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                                    line_color=self.default_color,
                                    showlegend=False,
                                    meanline_visible=True,
                                    scalemode='count',
                                    ))
        scatter_fig = self._plot_interactions_scatter(x_name=x_name, y_name=y_name, col_name=col_name,
                                                      x_values=x_values, y_values=y_values, col_values=col_values,
                                                      col_scale=col_scale)
        for trace in scatter_fig.data:
            fig.add_trace(trace)

        fig.update_layout(
            autosize=False,
            hovermode='closest',
            violingap=0.05,
            violingroupgap=0,
            violinmode='overlay',
            xaxis_type='category'
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

        if fig.data[-1]['showlegend'] is False:  # Case where col2 is not categorical
            fig.layout.coloraxis.colorscale = self.interactions_col_scale
        else:
            fig.update_layout(legend=dict(title=dict(text=col_name2)))

        title = f"<b>{truncate_str(col_name1)} and {truncate_str(col_name2)}</b> shap interaction values"
        if addnote:
            title += f"<span style='font-size: 12px;'><br />{add_text([addnote], sep=' - ')}</span>"
        dict_t = copy.deepcopy(self.dict_title)
        dict_t['text'] = title

        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_xaxis['text'] = truncate_str(col_name1, 110)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_yaxis['text'] = 'Shap interaction value'

        fig.update_traces(
            marker={
                'size': 8,
                'opacity': 0.8,
                'line': {'width': 0.8, 'color': 'white'}
            }
        )

        fig.update_layout(
            coloraxis=dict(colorbar={'title': {'text': col_name2}}),
            yaxis_title=dict_yaxis,
            title=dict_t,
            template='none',
            width=width,
            height=height,
            xaxis_title=dict_xaxis,
            hovermode='closest'
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
            if hasattr(self, 'interaction_selection'):
                list_ind = self.interaction_selection
            elif self.explainer.x_pred.shape[0] <= max_points:
                list_ind = self.explainer.x_pred.index.tolist()
            else:
                list_ind = random.sample(self.explainer.x_pred.index.tolist(), max_points)
                addnote = "Length of random Subset : "
        elif isinstance(selection, list):
            if hasattr(self, 'interaction_selection'):
                if set(self.interaction_selection).issubset(set(selection)):
                    list_ind = self.interaction_selection
            elif len(selection) <= max_points:
                list_ind = selection
                addnote = "Length of user-defined Subset : "
            else:
                list_ind = random.sample(selection, max_points)
                addnote = "Length of random Subset : "
        else:
            ValueError('parameter selection must be a list')
        self.interaction_selection = list_ind

        return list_ind, addnote

    def interactions_plot(self,
                          col1,
                          col2,
                          selection=None,
                          violin_maxf=10,
                          max_points=500,
                          width=900,
                          height=600,
                          file_name=None,
                          auto_open=False):
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
            raise ValueError('parameters col1 and col2 must be string or int.')

        col_id1 = self.explainer.check_features_name([col1])[0]
        col_name1 = self.explainer.columns_dict[col_id1]

        col_id2 = self.explainer.check_features_name([col2])[0]
        col_name2 = self.explainer.columns_dict[col_id2]

        col_value_count1 = self.explainer.features_desc[col_name1]

        list_ind, addnote = self._select_indices_interactions_plot(selection=selection, max_points=max_points)

        if addnote is not None:
            addnote = add_text([addnote,
                                f"{len(list_ind)} ({int(np.round(100 * len(list_ind) / self.explainer.x_pred.shape[0]))}%)"],
                               sep='')

        # Subset
        if self.explainer.postprocessing_modifications:
            feature_values1 = self.explainer.x_contrib_plot.loc[list_ind, col_name1].to_frame()
            feature_values2 = self.explainer.x_contrib_plot.loc[list_ind, col_name2].to_frame()
        else:
            feature_values1 = self.explainer.x_pred.loc[list_ind, col_name1].to_frame()
            feature_values2 = self.explainer.x_pred.loc[list_ind, col_name2].to_frame()

        interaction_values = self.explainer.get_interaction_values(selection=list_ind)[:, col_id1, col_id2]

        # selecting the best plot : Scatter, Violin?
        if col_value_count1 > violin_maxf:
            fig = self._plot_interactions_scatter(
                x_name=col_name1,
                y_name='Shap interaction value',
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self.interactions_col_scale
            )
        else:
            fig = self._plot_interactions_violin(
                x_name=col_name1,
                y_name='Shap interaction value',
                col_name=col_name2,
                x_values=feature_values1,
                y_values=pd.DataFrame(interaction_values, index=feature_values1.index),
                col_values=feature_values2,
                col_scale=self.interactions_col_scale
            )

        self._update_interactions_fig(
            fig=fig,
            col_name1=col_name1,
            col_name2=col_name2,
            addnote=addnote,
            width=width,
            height=height,
            file_name=file_name,
            auto_open=auto_open
        )

        return fig

    def top_interactions_plot(self,
                              nb_top_interactions=5,
                              selection=None,
                              violin_maxf=10,
                              max_points=500,
                              width=900,
                              height=600,
                              file_name=None,
                              auto_open=False):
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
                auto_open=False
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
            dict_t = copy.deepcopy(self.dict_title)
            dict_t.update({'text': title, 'y': 0.88, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
            return dict_t

        fig.layout.coloraxis.colorscale = self.interactions_col_scale
        fig.update_layout(
            xaxis_title=self.explainer.columns_dict[sorted_top_features_indices[0][0]],
            yaxis_title="Shap interaction value",
            updatemenus=[
                dict(
                    active=0,
                    buttons=list([
                        dict(label=f"{self.explainer.columns_dict[i]} - {self.explainer.columns_dict[j]}",
                             method="update",
                             args=[{"visible": [True if i == id_trace else False
                                                for i, x in enumerate(interactions_indices_traces_mapping)
                                                for _ in range(x)]},
                                   {'xaxis': {'title': {**{'text': self.explainer.columns_dict[i]}, **self.dict_xaxis}},
                                    'legend': {'title': {'text': self.explainer.columns_dict[j]}},
                                    'coloraxis': {'colorbar': {'title': {'text': self.explainer.columns_dict[j]}},
                                                  'colorscale': fig.layout.coloraxis.colorscale},
                                    'title': generate_title_dict(self.explainer.columns_dict[i],
                                                                 self.explainer.columns_dict[j], addnote)},
                                   ])
                        for id_trace, (i, j) in enumerate(indices_to_plot)
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.37,
                    xanchor="left",
                    y=1.25,
                    yanchor="top"
                )],
            annotations=[
                dict(text=f"Sorted top {len(indices_to_plot)} SHAP interaction Variables :",
                     x=0, xref="paper", y=1.2, yref="paper", align="left", showarrow=False)
            ]
        )

        self._update_interactions_fig(
            fig=fig,
            col_name1=self.explainer.columns_dict[sorted_top_features_indices[0][0]],
            col_name2=self.explainer.columns_dict[sorted_top_features_indices[0][1]],
            addnote=addnote,
            width=width,
            height=height,
            file_name=None,
            auto_open=False
        )

        fig.update_layout(
            title={
                'y': 0.88,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'}
        )

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig

    def correlations(
            self,
            df=None,
            max_features=20,
            features_to_hide=None,
            facet_col=None,
            how='phik',
            width=900,
            height=500,
            file_name=None,
            auto_open=False
    ):
        """
        Correlations matrix heatmap plot.
        The method can use phik or pearson correlations.
        The correlations computed can be changed using the parameter 'how'.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame for which we want to compute correlations. Will use x_pred by default.
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

        if features_to_hide is None:
            features_to_hide = []

        if df is None:
            # Use x_pred by default
            df = self.explainer.x_pred

        if facet_col:
            features_to_hide += [facet_col]

        # We use phik by default as it is a convenient method for numeric and categorical data
        if how == 'phik':
            try:
                from phik import phik_matrix
                compute_method = 'phik'
            except (ImportError, ModuleNotFoundError):
                warnings.warn('Cannot compute phik correlations. Install phik using "pip install phik".', UserWarning)
                compute_method = "pearson"
        else:
            compute_method = how

        hovertemplate = '<b>%{text}<br />Correlation: %{z}</b><extra></extra>'

        list_features = []
        if facet_col:
            facet_col_values = sorted(df[facet_col].unique(), reverse=True)
            fig = make_subplots(
                rows=1,
                cols=df[facet_col].nunique(),
                subplot_titles=[t + " correlation" for t in facet_col_values],
                horizontal_spacing=0.15
            )
            # Used for the Shapash report to get train then test set
            for i, col_v in enumerate(facet_col_values):
                corr = compute_corr(df.loc[df[facet_col] == col_v].drop(features_to_hide, axis=1), compute_method)

                # Keep the same list of features for each subplot
                if len(list_features) == 0:
                    list_features = compute_top_correlations_features(corr=corr, max_features=max_features)

                fig.add_trace(
                    go.Heatmap(
                        z=corr.loc[list_features, list_features].round(2).values,
                        x=list_features,
                        y=list_features,
                        coloraxis='coloraxis',
                        text=[[f'Feature 1: {self.explainer.features_dict.get(y, y)} <br />'
                               f'Feature 2: {self.explainer.features_dict.get(x, x)}' for x in list_features]
                              for y in list_features],
                        hovertemplate=hovertemplate,
                    ), row=1, col=i+1)

        else:
            corr = compute_corr(df.drop(features_to_hide, axis=1), compute_method)
            list_features = compute_top_correlations_features(corr=corr, max_features=max_features)

            fig = go.Figure(go.Heatmap(
                        z=corr.loc[list_features, list_features].round(2).values,
                        x=list_features,
                        y=list_features,
                        coloraxis='coloraxis',
                        text=[[f'Feature 1: {self.explainer.features_dict.get(y, y)} <br />'
                               f'Feature 2: {self.explainer.features_dict.get(x, x)}' for x in list_features]
                              for y in list_features],
                        hovertemplate=hovertemplate,
                    ))

        title = f'Correlation ({compute_method})'
        if len(list_features) < len(df.drop(features_to_hide, axis=1).columns):
            subtitle = f"Top {len(list_features)} correlations"
            title += f"<span style='font-size: 12px;'><br />{subtitle}</span>"
        dict_t = copy.deepcopy(self.dict_title)
        dict_t['text'] = title

        fig.update_layout(
            coloraxis=dict(colorscale=['rgb(255, 255, 255)'] + self.init_colorscale[5:-1]),
            showlegend=True,
            title=dict_t,
            width=width,
            height=height
        )

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)

        return fig
