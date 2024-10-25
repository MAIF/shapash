import warnings

from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_line_break, adjust_title_height, truncate_str


def plot_line_comparison(
    index,
    feature_values,
    contributions,
    style_dict,
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
    style_dict: dict
        the different styles used in the different outputs of Shapash
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

    topmargin = 80
    dict_xaxis = style_dict["dict_xaxis"] | {"text": None}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": None}

    if len(index) == 0:
        warnings.warn("No individuals matched", UserWarning)
        title = "Compare plot - <b>No Matching Reference Entry</b>"
    elif len(index) < 2:
        warnings.warn("Comparison needs at least 2 individuals", UserWarning)
        title = "Compare plot - index : " + " ; ".join(["<b>" + str(id) + "</b>" for id in index])
    else:
        title = "Compare plot - index : " + " ; ".join(["<b>" + str(id) + "</b>" for id in index])
        dict_xaxis["text"] = "Contributions"
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}

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
    len_dic_color = len(style_dict["dict_compare_colors"])
    lines = list()

    for i, id_i in enumerate(index):
        x_i = list()
        features = list()
        x_val = predictions[i]
        x_hover = list()
        color = style_dict["dict_compare_colors"][i % len_dic_color]

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
                marker={"color": color},
            )
        )

    fig = go.Figure(data=lines, layout=layout)
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)

    if file_name is not None:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig
