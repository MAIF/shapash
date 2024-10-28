import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import adjust_title_height, tuning_colorscale


def plot_stability_distribution(
    variability,
    plot_type,
    mean_amplitude,
    dataset,
    column_names,
    file_name,
    auto_open,
    init_colorscale,
    style_dict,
    height="auto",
    width=900,
):
    """
    Generates and displays a stability distribution plot for feature variability using either a boxplot or violin plot.

    This function is useful for visualizing the stability (or variability) of local feature contributions across all
    instances in the dataset. The function supports both boxplot and violin plot visualizations, and colors the plot
    based on the mean amplitude of normalized SHAP values.

    Parameters
    ----------
    variability : numpy.ndarray
        A 2D array where each row represents the local stability (variability) for each feature across instances.
        The X-axis of the plot represents the feature's normalized contribution value variability.
    plot_type : str
        The type of plot to be displayed. Can be either:
        - "boxplot": Displays a boxplot for each feature's variability distribution.
        - "violin": Displays a violin plot for each feature's variability distribution.
    mean_amplitude : numpy.ndarray
        1D array containing the average of the normalized SHAP values in the neighborhood for each feature.
        Used to generate a colorscale representing feature importance in the plot.
    dataset : pandas.DataFrame
        The original dataset (`x_init`) on which the SHAP values were computed. Used to adjust plot dimensions
        and ensure compatibility with `variability`.
    column_names : list
        A list of strings representing the feature names. These names are used as labels for the Y-axis in the plot.
    file_name : str, optional
        The file path to save the generated plot as an HTML file. If not provided, the plot will not be saved.
    auto_open : bool
        If True, the saved plot will automatically open in the browser. Only applicable if `file_name` is provided.
    init_colorscale : str or list
        The initial colorscale used for displaying the mean amplitude of the SHAP values. This will determine the
        gradient of colors applied to the plot.
    style_dict : dict
        A dictionary specifying the various style options such as font size, color, and other aesthetic parameters
        for the plot.
    height: int or 'auto'
        Plotly figure - layout height
    width: int
        Plotly figure - layout width

    Returns
    -------
    go.Figure
        A Plotly `Figure` object representing the generated stability distribution plot (either boxplot or violin plot),
        including a color scale to indicate feature importance.

    Notes
    -----
    - This function generates stability distribution plots for each feature based on their variability across instances.
    It also includes a color scale based on the mean SHAP value amplitude to provide insights into feature importance.
    - The function adjusts plot height dynamically based on the number of features in the dataset.
    - Supports saving the generated plot as an interactive HTML file if `file_name` is specified.
    - If the number of features exceeds 500, the function might not display all features due to space constraints.
    """
    # Store distribution of variability in a DataFrame
    var_df = pd.DataFrame(variability, columns=column_names)
    mean_amplitude_normalized = pd.Series(mean_amplitude, index=column_names) / mean_amplitude.max()

    # And sort columns by mean amplitude
    var_df = var_df[column_names[mean_amplitude.argsort()]]

    # Add colorscale
    col_scale, _, _ = tuning_colorscale(init_colorscale, pd.DataFrame(mean_amplitude))
    color_list = mean_amplitude_normalized.tolist()
    color_list.sort()
    color_list = [next(pair[1] for pair in col_scale if x <= pair[0]) for x in color_list]
    if height == "auto":
        height_value = max(500, 40 * dataset.shape[1] if dataset.shape[1] < 100 else 13 * dataset.shape[1])
    else:
        height_value = height

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

        _update_stability_fig(
            fig=fig,
            x_barlen=len(mean_amplitude),
            y_bar=column_names,
            style_dict=style_dict,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            file_name=file_name,
            auto_open=auto_open,
            height=height_value,
            width=width,
        )

        return fig


def _update_stability_fig(
    fig, x_barlen, y_bar, style_dict, xaxis_title, yaxis_title, file_name, auto_open, height=500, width=900
):
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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    xaxis_title: str
        Title of xaxis
    yaxis_title: str
        Title of yaxis
    file_name: string (optional)
        Specify the save path of html files. If it is not provided, no file will be saved.
    auto_open: bool (default=False)
        open automatically the plot
    height: int
        Plotly figure - layout height

    Returns
    -------
    go.Figure
    """
    title = "<br>Importance & Local Stability of explanations:"
    title += "<br><sup>How similar are explanations for closeby neighbours?</sup>"
    dict_t = style_dict["dict_title_stability"] | {"text": title, "y": adjust_title_height(height)}

    dict_xaxis = style_dict["dict_xaxis"] | {"text": xaxis_title}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": yaxis_title}

    fig.add_trace(
        go.Scatter(
            x=[0.15] * x_barlen,
            y=y_bar,
            mode="lines",
            hoverinfo="none",
            line=dict(color=style_dict["dict_stability_bar_colors"][0], dash="dot"),
            name="<-- Stable",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0.3] * x_barlen,
            y=y_bar,
            mode="lines",
            hoverinfo="none",
            line=dict(color=style_dict["dict_stability_bar_colors"][1], dash="dot"),
            name="--> Unstable",
        )
    )

    fig.update_layout(
        template="none",
        autosize=False,
        title=dict_t,
        xaxis_title=dict_xaxis,
        yaxis_title=dict_yaxis,
        coloraxis_showscale=False,
        hovermode="closest",
        height=height,
        width=width,
        margin={"l": 150, "r": 20, "t": 95, "b": 70},
    )

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)


def plot_amplitude_vs_stability(
    mean_variability,
    mean_amplitude,
    column_names,
    file_name,
    auto_open,
    col_scale,
    style_dict,
    height="auto",
    width=900,
):
    """
    Generates and displays a scatter plot showing the relationship between feature variability and importance.

    This plot helps visualize the trade-off between the stability (variability) of feature contributions and their
    average importance, providing insights into the reliability of feature attributions. Each feature is represented
    as a point, where the X-axis shows the variability of the feature's contribution, and the Y-axis shows its average
    contribution (importance). The color of the points represents the feature importance, based on a provided color scale.

    Parameters
    ----------
    mean_variability : numpy.ndarray
        A 1D array representing the local stability of each feature, averaged across all instances. The values are
        typically computed as the standard deviation of SHAP values divided by the mean. These values are displayed
        on the X-axis.
    mean_amplitude : numpy.ndarray
        A 1D array representing the average normalized SHAP values (importance) for each feature. These values are
        displayed on the Y-axis of the plot.
    column_names : list of str
        The names of the features being plotted. These are used for hover text to identify individual points.
    file_name : str, optional
        The file path to save the generated plot as an HTML file. If not provided, the plot will not be saved.
    auto_open : bool
        If True, the saved plot will automatically open in the browser. Only applicable if `file_name` is provided.
    col_scale : list or str
        The color scale used for visualizing feature importance in the scatter plot. Can be either a named color scale
        (e.g., "Viridis") or a custom list of colors.
    style_dict : dict
        A dictionary specifying various style options such as font size, axis formatting, and other aesthetic
        properties for the plot.
    height: int
        Plotly figure - layout height
    width: int
        Plotly figure - layout width

    Returns
    -------
    go.Figure
        A Plotly `Figure` object representing the generated scatter plot of feature variability vs importance.

    Notes
    -----
    - The plot provides a way to understand the relationship between feature stability and its contribution to the
    model’s predictions.
    - Hover text is included for each feature, showing its name, importance, and variability.
    - The plot automatically adjusts its X-axis range to ensure clear visualization of feature variability.
    - The function can optionally save the plot as an HTML file for further exploration.

    """
    if height == "auto":
        height = 500
    xaxis_title = (
        "Variability of the Normalized Local Contribution Values"
        + "<span style='font-size: 12px;'><br />(standard deviation / mean)</span>"
    )
    yaxis_title = "Importance<span style='font-size: 12px;'><br />(Average contributions)</span>"
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

    _update_stability_fig(
        fig=fig,
        x_barlen=len(mean_amplitude),
        y_bar=[0, mean_amplitude.max()],
        style_dict=style_dict,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        file_name=file_name,
        auto_open=auto_open,
        height=height,
        width=width,
    )
    return fig
