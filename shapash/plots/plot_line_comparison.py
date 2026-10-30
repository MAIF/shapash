import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_line_break, adjust_title_height, truncate_str


def plot_line_comparison(
    index: Sequence[Any],
    feature_values: Sequence[Any],
    contributions: np.ndarray | Sequence[Sequence[float]],
    style_dict: dict[str, Any],
    predictions: Sequence[pd.Series] | None = None,
    dict_features: Mapping[Any, Any] | None = None,
    subtitle: str | None = None,
    width: int = 900,
    height: int = 550,
    file_name: str | None = None,
    auto_open: bool = False,
) -> go.Figure:
    """
    Plotly plot for comparisons. Displays
    the contributions of several individuals. One line represents
    the different contributions of a unique individual.

    Parameters
    ----------
    index : sequence[Any]
        Identifiers of the individuals to compare.
    feature_values : sequence[Any]
        Feature labels or identifiers corresponding to the contribution rows.
    contributions : numpy.ndarray or sequence of sequence[float]
        Matrix of contributions, with one row per feature and one value per individual.
    style_dict : dict[str, Any]
        Styles used in the different outputs of Shapash.
    predictions : sequence[pandas.Series] or None, optional
        Feature values for each individual. Required when ``index`` is not empty.
    dict_features : Mapping[Any, Any] or None, optional
        Mapping from feature labels to the corresponding prediction-series keys. Required when ``index`` is not empty.
    subtitle : str or None, optional
        Subtitle to display.
    width : int, optional
        Plotly figure layout width, by default 900.
    height : int, optional
        Plotly figure layout height, by default 550.
    file_name : str or None, optional
        Path for saving the Plotly chart. If None, the chart will not be saved.
    auto_open : bool, optional
        Whether to open the saved plot, by default False.

    Returns
    -------
    go.Figure
        Plot of the contributions of individuals, feature by feature.
    """

    topmargin = 80.0
    dict_xaxis = style_dict["dict_xaxis"] | {"text": None}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": None}

    if len(index) == 0:
        warnings.warn("No individuals matched", UserWarning, stacklevel=2)
        title = "Compare plot - <b>No Matching Reference Entry</b>"
    elif len(index) < 2:
        warnings.warn("Comparison needs at least 2 individuals", UserWarning, stacklevel=2)
        title = "Compare plot - index : " + " ; ".join(["<b>" + str(index_id) + "</b>" for index_id in index])
    else:
        title = "Compare plot - index : " + " ; ".join(["<b>" + str(index_id) + "</b>" for index_id in index])
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

    iteration_list = list(zip(contributions, feature_values, strict=False))
    len_dic_color = len(style_dict["dict_compare_colors"])
    lines = list()

    for i, id_i in enumerate(index):
        if predictions is None or dict_features is None:
            raise ValueError("predictions and dict_features are required when index is not empty.")
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
                f"Id: <b>{add_line_break(str(id_i), 40, 160)}</b>"
                + f"<br /><b>{add_line_break(str(feat), 40, 160)}</b> <br />"
                + f"Contribution: {contrib[i]:.4f} <br />Value: "
                + add_line_break(str(pred_x_val), 40, 160)
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
