"""
Smart plotter module
"""
import warnings
import random
import copy
import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot
from shapash.manipulation.select_lines import select_lines
from shapash.manipulation.summarize import compute_features_import
from shapash.utils.utils import add_line_break, truncate_str, compute_digit_number, add_text, \
    maximum_difference_sort_value

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

        self.round_digit = None

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

    def plot_scatter(self,
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
        Scatter plot of one feature contribution across the prediction set.

        Parameters
        ----------
        feature_values : List of values or pd.Series or 1d Array
            The values of one feature
        contributions : List of values or pd.Serie or 1d Array
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
        title = f"<b>{truncate_str(feature_name)}</b> - Feature Contribution"
        if subtitle or addnote:
            title = title + f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
        dict_t = copy.deepcopy(self.dict_title)
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        default_color = self.default_color
        dict_colors = copy.deepcopy(self.dict_ycolors)
        dict_t['text'] = title
        dict_xaxis['text'] = truncate_str(feature_name, 110)
        dict_yaxis['text'] = 'Contribution'
        if self.explainer._case == "regression":
            colorpoints = pred
            colorbar_title = 'Predicted'
        elif self.explainer._case == "classification":
            colorpoints = proba_values
            colorbar_title = 'Predicted Proba'

        # add break line to X label if necessary
        max_len_by_row = max([round(50 / self.explainer.features_desc[feature_values.columns.values[0]]), 8])
        feature_values.iloc[:, 0] = feature_values.iloc[:, 0].apply(add_line_break, args=(max_len_by_row, 120,))

        if pred is not None:
            hv_text = [f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten())]
        else:
            hv_text = [f"Id: {x}<br />" for x in feature_values.index]
        fig.update_layout(
            template='none',
            title=dict_t,
            width=width,
            height=height,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            hovermode='closest'
        )
        fig.add_scatter(
            x=feature_values.values.flatten(),
            y=contributions.values.flatten(),
            mode='markers',
            hovertext=hv_text,
            hovertemplate='<b>%{hovertext}</b><br />' +
                          f'{feature_name} : ' +
                          '%{x}<br />Contribution: %{y:.4f}<extra></extra>',
            marker={
                'size': 10,
                'opacity': 0.8,
                'line': {'width': 0.8, 'color': 'white'}
            },
            customdata=contributions.index.values
        )
        if colorpoints is not None:
            fig.data[0].marker.color = colorpoints.values.flatten()
            fig.data[0].marker.coloraxis = 'coloraxis'
            fig.layout.coloraxis.colorscale = col_scale
            fig.layout.coloraxis.colorbar = {'title': {'text': colorbar_title}}

        elif self.explainer._case == 'classification' and pred is not None:
            fig.data[0].marker.color = pred.iloc[:, 0].apply(lambda \
                                                                     x: dict_colors[1] if x == col_modality else
            dict_colors[0])
        else:
            fig.data[0].marker.color = default_color

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
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
        dict_t = copy.deepcopy(self.dict_title)
        title = f"<b>{truncate_str(feature_name)}</b> - Feature Contribution"
        if subtitle or addnote:
            title = title + f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_colors = copy.deepcopy(self.dict_ycolors)
        default_color = self.default_color
        dict_t['text'] = title
        dict_xaxis['text'] = truncate_str(feature_name, 110)
        dict_yaxis['text'] = 'Contribution'
        points_param = False if proba_values is not None else "all"
        jitter_param = 0.075
        if self.explainer._case == "regression":
            colorpoints = pred
            colorbar_title = 'Predicted'
        elif self.explainer._case == 'classification':
            colorpoints = proba_values
            colorbar_title = 'Predicted Proba'

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
                                        line_color=dict_colors[0],
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
                                        line_color=dict_colors[1],
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
                                        line_color=default_color,
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

        if colorpoints is not None:
            fig.add_trace(go.Scatter(
                x=feature_values.values.flatten(),
                y=contributions.values.flatten(),
                mode='markers',
                showlegend=False,
                hovertext=hv_text,
                hovertemplate='<b>%{hovertext}</b><br />' + hv_temp,
                customdata=contributions.index.values,
                marker={
                    'color': colorpoints.values.flatten(),
                    'showscale': True,
                    'coloraxis': 'coloraxis'
                }
            ))
            fig.layout.coloraxis.colorscale = col_scale
            fig.layout.coloraxis.colorbar = {'title': {'text': colorbar_title}}

        fig.update_traces(
            marker={
                'size': 10,
                'opacity': 0.8,
                'line': {'width': 0.8, 'color': 'white'}
            }
        )
        fig.update_layout(
            template='none',
            autosize=False,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            width=width,
            height=height,
            hovermode='closest',
            violingap=0.05,
            violingroupgap=0,
            violinmode='overlay',
            xaxis_type='category'
        )

        fig.update_xaxes(range=[-0.6, len(uniq_l) - 0.4])

        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
        return fig

    def plot_features_import(self,
                             feature_imp1,
                             feature_imp2=None,
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
        title = "Features Importance"
        topmargin = 80
        if subtitle or addnote:
            title = title + f"<span style='font-size: 12px;'><br />{add_text([subtitle, addnote], sep=' - ')}</span>"
            topmargin = topmargin + 15
        dict_t.update(text=title)
        dict_xaxis = copy.deepcopy(self.dict_xaxis)
        dict_xaxis.update(text='Contribution')
        dict_yaxis = copy.deepcopy(self.dict_yaxis)
        dict_yaxis.update(text=None)
        dict_style_bar1 = self.dict_featimp_colors[1]
        dict_style_bar2 = self.dict_featimp_colors[2]
        dict_yaxis['text'] = None

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
            marker=dict_style_bar1
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
                title = title + f"<span style='font-size: 12px;'><br />{subtitle}</span>"
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
            if expl[1] == '':
                ylabel = '<i>{}</i>'.format(expl[0])
                hoverlabel = '<b>{}</b>'.format(expl[0])
            else:
                hoverlabel = '<b>{} :</b><br />{}'.format(add_line_break(expl[0], 40, maxlen=120), \
                                                          add_line_break(expl[1], 40, maxlen=160))
                if len(contrib) <= yaxis_max_label:
                    ylabel = '<b>{} :</b><br />{}'.format(truncate_str(expl[0], 45), truncate_str(expl[1], 45))
                else:
                    ylabel = ('<b>{}</b>'.format(truncate_str(expl[0], maxlen=45)))

            contrib_value = expl[2]
            # colors
            if contrib_value >= 0:
                color = 1 if expl[1] != '' else 0
            else:
                color = -1 if expl[1] != '' else -2

            barobj = go.Bar(
                x=[contrib_value],
                y=[ylabel],
                customdata=[hoverlabel],
                orientation='h',
                marker=dict_local_plot_colors[color],
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
            value = self.explainer.model.predict(self.explainer.x_init.loc[index].to_frame().T)[0]

        return value

    def local_plot(self,
                   index=None,
                   row_num=None,
                   query=None,
                   label=None,
                   show_masked=True,
                   show_predict=True,
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
            if not hasattr(self.explainer, 'mask_params'):
                self.explainer.filter(max_contrib=20)

            if self.explainer._case == "classification":
                if label is None:
                    label = -1

                label_num, label_code, label_value = self.explainer.check_label_name(label)

                contrib = self.explainer.data['contrib_sorted'][label_num]
                x_val = self.explainer.data['x_sorted'][label_num]
                var_dict = self.explainer.data['var_dict'][label_num]

                if show_predict is True:
                    pred = self.local_pred(line[0], label_num)
                    if pred is None:
                        subtitle = f"Response: <b>{label_value}</b> - No proba available"
                    else:
                        subtitle = f"Response: <b>{label_value}</b> - Proba: <b>{pred:.4f}</b>"

            elif self.explainer._case == "regression":
                contrib = self.explainer.data['contrib_sorted']
                x_val = self.explainer.data['x_sorted']
                var_dict = self.explainer.data['var_dict']
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
            var_dict = [self.explainer.features_dict[self.explainer.columns_dict[x]] for x in var_dict]
            if show_masked:
                var_dict, x_val, contrib = self.check_masked_contributions(line, var_dict, x_val, contrib,
                                                                           label=label_num)

            # Filtering all negative or positive contrib if specify in mask
            exclusion = []
            if hasattr(self.explainer, 'mask_params'):
                if self.explainer.mask_params['positive'] == True:
                    exclusion = np.where(np.array(contrib) < 0)[0].tolist()
                elif self.explainer.mask_params['positive'] == False:
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
            label_num, label_code, label_value = self.explainer.check_label_name(label)

        if not isinstance(col, (str, int)):
            raise ValueError('parameter col must be string or int.')

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
            ValueError('parameter selection must be a list')

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
            subcontrib = self.explainer.contributions[label_num]
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
            subcontrib = self.explainer.contributions
            if self.explainer.y_pred is not None:
                if not hasattr(self, "pred_colorscale"):
                    self.pred_colorscale = self.tuning_colorscale(self.explainer.y_pred)
                col_scale = self.pred_colorscale

        # Subset
        if self.explainer.postprocessing_modifications:
            feature_values = self.explainer.x_contrib_plot.loc[list_ind, col_name].to_frame()
        else:
            feature_values = self.explainer.x_pred.loc[list_ind, col_name].to_frame()
        contrib = subcontrib.loc[list_ind, col_name].to_frame()

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
                                    addnote,
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

        # classification
        if self.explainer._case == "classification":
            label_num, label_code, label_value = self.explainer.check_label_name(label)
            global_feat_imp = self.explainer.features_imp[label_num].tail(max_features)
            if selection is not None:
                subset = self.explainer.contributions[label_num].loc[selection]
                subset_feat_imp = compute_features_import(subset)
                subset_feat_imp = subset_feat_imp.reindex(global_feat_imp.index)
            else:
                subset_feat_imp = None
            subtitle = f"Response: <b>{label_value}</b>"
        # regression
        elif self.explainer._case == "regression":
            global_feat_imp = self.explainer.features_imp.tail(max_features)
            if selection is not None:
                subset = self.explainer.contributions.loc[selection]
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
        fig = self.plot_features_import(global_feat_imp, subset_feat_imp, addnote,
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

            label_num, label_code, label_value = self.explainer.check_label_name(label)
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
