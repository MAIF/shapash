from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from shapash.utils.utils import adjust_title_height


def plot_compacity(
    features_needed,
    distance_reached,
    style_dict,
    approx=0.9,
    nb_features=5,
    file_name=None,
    auto_open=False,
    height=600,
    width=900,
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
        features_needed:
        distance_reached:
        style_dict: dict
            the different styles used in the different outputs of Shapash
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
    fig.update_annotations(font=style_dict["dict_title_compacity"]["font"])

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
            marker_color=style_dict["dict_compacity_bar_colors"][1],
        ),
        row=1,
        col=1,
    )

    title = style_dict["dict_xaxis"] | {"text": "Number of selected features"}
    fig.update_xaxes(title=title, row=1, col=1)
    title = style_dict["dict_yaxis"] | {"text": "Cumulative distribution over<br>dataset's instances (%)"}
    fig.update_yaxes(title=title, row=1, col=1)

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
            marker_color=style_dict["dict_compacity_bar_colors"][0],
        ),
        row=1,
        col=2,
    )

    title = style_dict["dict_xaxis"] | {"text": "Percentage of model output<br>explained (%)"}
    fig.update_xaxes(title=title, row=1, col=2)
    title = style_dict["dict_yaxis"] | {"text": "Cumulative distribution over<br>dataset's instances (%)"}
    fig.update_yaxes(title=title, row=1, col=2)

    title = "<br>Compacity of explanations:"
    title += "<br><sup>How many variables are enough to produce accurate explanations?</sup>"
    dict_t = style_dict["dict_title_stability"] | {"text": title, "y": adjust_title_height()}

    fig.update_layout(
        template="none",
        autosize=False,
        height=height,
        width=width,
        title=dict_t,
        hovermode="closest",
        showlegend=False,
        margin={"l": 150, "r": 20, "t": 150, "b": 70},
    )

    if file_name is not None:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig
