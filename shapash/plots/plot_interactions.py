import pandas as pd
import plotly.express as px
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_text, adjust_title_height, truncate_str


def plot_interactions_scatter(x_name, y_name, col_name, x_values, y_values, col_values, col_scale, style_dict):
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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    Returns
    -------
    go.Figure
    """

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
            color_discrete_sequence=style_dict["interactions_discrete_colors"],
        )
    else:
        fig = px.scatter(data_df, x=x_name, y=y_name, color=col_name, color_continuous_scale=col_scale)

    fig.update_traces(mode="markers")

    return fig


def plot_interactions_violin(x_name, y_name, col_name, x_values, y_values, col_values, col_scale, style_dict):
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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    Returns
    -------
    go.Figure
    """

    fig = go.Figure()

    uniq_l = list(pd.unique(x_values.values.flatten()))
    uniq_l.sort()

    for i in uniq_l:
        fig.add_trace(
            go.Violin(
                x=x_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                y=y_values.loc[x_values.iloc[:, 0] == i].values.flatten(),
                line_color=style_dict["violin_default"],
                showlegend=False,
                meanline_visible=True,
                scalemode="count",
            )
        )
    scatter_fig = plot_interactions_scatter(
        x_name=x_name,
        y_name=y_name,
        col_name=col_name,
        x_values=x_values,
        y_values=y_values,
        col_values=col_values,
        col_scale=col_scale,
        style_dict=style_dict,
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


def update_interactions_fig(fig, col_name1, col_name2, addnote, width, height, file_name, auto_open, style_dict):
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
    style_dict: dict
        the different styles used in the different outputs of Shapash

    Returns
    -------
    go.Figure
    """

    if fig.data[-1]["showlegend"] is False:  # Case where col2 is not categorical
        fig.layout.coloraxis.colorscale = style_dict["interactions_col_scale"]
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
