from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.plots.plot_contribution import _calculate_percentage_intervals, _create_jittered_points
from shapash.utils.utils import add_text, adjust_title_height, truncate_str


def plot_interactions_scatter(
    x_name: str,
    y_name: str,
    col_name: str,
    x_values: pd.DataFrame,
    y_values: pd.DataFrame,
    col_values: pd.DataFrame,
    col_scale: list,
    style_dict: dict,
    cmin: float | None = None,
    cmax: float | None = None,
    x_values_hover: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Generate a scatter-plot figure for interactions.

    Supports both categorical and continuous color encoding. When continuous
    color is used, optional `cmin`/`cmax` bounds can be applied.
    The x-values used in hover can differ from plotted x-values via
    `x_values_hover` (useful when plotted x is jittered).

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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    cmin : float, optional
        Lower bound of the continuous color range.
    cmax : float, optional
        Upper bound of the continuous color range.
    x_values_hover : pd.DataFrame, optional
        Values displayed in hover for x-axis. If None, `x_values` are used.

    Returns
    -------
    go.Figure
    """

    if x_values_hover is None:
        x_values_hover = x_values

    data_df = pd.DataFrame(
        {
            x_name: x_values.values.flatten(),
            y_name: y_values.values.flatten(),
            col_name: col_values.values.flatten(),
            "__x_hover__": x_values_hover.values.flatten(),
        }
    )

    if isinstance(col_values.values.flatten()[0], str):
        fig = px.scatter(
            data_df,
            x=x_name,
            y=y_name,
            color=col_name,
            color_discrete_sequence=style_dict["interactions_discrete_colors"],
            hover_data={x_name: False, "__x_hover__": True},
            labels={"__x_hover__": x_name},
        )
    else:
        scatter_args = {
            "data_frame": data_df,
            "x": x_name,
            "y": y_name,
            "color": col_name,
            "color_continuous_scale": col_scale,
            "hover_data": {x_name: False, "__x_hover__": True},
            "labels": {"__x_hover__": x_name},
        }
        if cmin is not None and cmax is not None:
            scatter_args["range_color"] = [cmin, cmax]
        fig = px.scatter(**scatter_args)

    fig.update_traces(mode="markers")

    return fig


def plot_interactions_violin(
    x_name: str,
    y_name: str,
    col_name: str,
    x_values: pd.DataFrame,
    y_values: pd.DataFrame,
    col_values: pd.DataFrame,
    col_scale: list,
    style_dict: dict,
    cmin: float | None = None,
    cmax: float | None = None,
) -> go.Figure:
    """
    Generate a violin-plot figure for interactions with point dispersion.

    For each x-modality, a violin is drawn and a jittered scatter layer is
    overlaid to spread points. Hover displays the original x value (non-jittered).

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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    cmin : float, optional
        Lower bound of the continuous color range used by the scatter overlay.
    cmax : float, optional
        Upper bound of the continuous color range used by the scatter overlay.

    Returns
    -------
    go.Figure
    """

    fig = go.Figure()

    uniq_l = list(pd.unique(x_values.values.flatten()))
    uniq_l.sort()

    x_numeric = pd.Series(index=x_values.index, dtype=float)
    x_jittered = pd.Series(index=x_values.index, dtype=float)

    for idx, modality in enumerate(uniq_l):
        if pd.isna(modality):
            x_cond = x_values.iloc[:, 0].isna()
        else:
            x_cond = x_values.iloc[:, 0] == modality

        x_numeric.loc[x_cond] = idx

        percentage_series = _calculate_percentage_intervals(y_values.loc[x_cond].iloc[:, 0], bins=20)
        x_jittered.loc[x_cond] = _create_jittered_points(
            x_numeric.loc[x_cond].to_numpy(), percentage_series, side="both"
        )

        fig.add_trace(
            go.Violin(
                x=x_numeric.loc[x_cond].to_numpy(),
                y=y_values.loc[x_cond].values.flatten(),
                name="missing" if pd.isna(modality) else modality,
                line_color=style_dict["violin_default"],
                showlegend=False,
                meanline_visible=True,
                scalemode="count",
            )
        )

    x_values_dispersion = pd.DataFrame({x_name: x_jittered}, index=x_values.index)
    scatter_fig = plot_interactions_scatter(
        x_name=x_name,
        y_name=y_name,
        col_name=col_name,
        x_values=x_values_dispersion,
        y_values=y_values,
        col_values=col_values,
        col_scale=col_scale,
        style_dict=style_dict,
        cmin=cmin,
        cmax=cmax,
        x_values_hover=x_values,
    )
    for trace in scatter_fig.data:
        fig.add_trace(trace)

    fig.update_layout(
        autosize=False,
        hovermode="closest",
        violingap=0.05,
        violingroupgap=0,
        violinmode="overlay",
        xaxis_type="linear",
    )

    xs_labels = ["missing" if pd.isna(x) else x for x in uniq_l]
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(uniq_l))), ticktext=xs_labels)
    fig.update_xaxes(range=[-0.6, len(uniq_l) - 0.4])

    return fig


def update_interactions_fig(
    fig: go.Figure,
    col_name1: str,
    col_name2: str,
    addnote: str | None,
    width: int,
    height: int,
    file_name: str | None,
    auto_open: bool,
    style_dict: dict,
    col_scale: list | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
) -> go.Figure:
    """
    Update the final layout for interactions figures.

    Handles title, axes, marker style, and color axis formatting for
    continuous color encodings.

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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    col_scale : list, optional
        Colorscale to apply for continuous color encoding.
    cmin : float, optional
        Lower bound of the continuous color range.
    cmax : float, optional
        Upper bound of the continuous color range.

    Returns
    -------
    go.Figure
    """

    if fig.data[-1]["showlegend"] is False:  # Case where col2 is not categorical
        fig.layout.coloraxis.colorscale = col_scale if col_scale is not None else style_dict["interactions_col_scale"]
        if cmin is not None and cmax is not None:
            fig.layout.coloraxis.cmin = cmin
            fig.layout.coloraxis.cmax = cmax
    else:
        fig.update_layout(legend=dict(title=dict(text=col_name2)))

    title = f"<b>{truncate_str(col_name1)} and {truncate_str(col_name2)}</b> shap interaction values"
    if addnote:
        title += f"<span style='font-size: 12px;'><br />{add_text([addnote], sep=' - ')}</span>"
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
    dict_xaxis = style_dict["dict_xaxis"] | {"text": truncate_str(col_name1, 110)}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": "Shap interaction value"}

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
