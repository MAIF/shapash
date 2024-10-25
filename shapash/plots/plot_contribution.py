from numbers import Number

import numpy as np
import pandas as pd
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_line_break, adjust_title_height, truncate_str
from shapash.webapp.utils.utils import round_to_k


def plot_scatter(
    feature_values,
    contributions,
    feature_name,
    case,
    style_dict,
    pred=None,
    proba_values=None,
    col_modality=None,
    col_scale=None,
    cmin=None,
    cmax=None,
    metadata=None,
    addnote=None,
    subtitle=None,
    max_len_by_row=20,
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
    case: str
        Type of the model, either 'classification' or 'regression'
    style_dict: dict
        the different styles used in the different outputs of Shapash
    proba_values: 1 column pd.DataFrame (optional)
        predicted proba used to color points - One Vs All in multiclass case
    col_modality: Int, Float or String (optional)
        parameter used in classification case,
        specify the modality to color in scatter plot (One Vs All)
    col_scale: list (optional)
        specify the color of points in scatter data
    cmin : float, optional
        The minimum value for the color scale, providing the lower bound for color normalization.
    cmax : float, optional
        The maximum value for the color scale, providing the upper bound for color normalization.
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

    column_name = feature_values.columns[0]
    feature_values = feature_values.sort_values(by=column_name)
    contributions = contributions.loc[feature_values.index]
    if pred is not None:
        pred = pred.loc[feature_values.index]
    if proba_values is not None:
        proba_values = proba_values.loc[feature_values.index]

    # add break line to X label if necessary
    args = (max_len_by_row, 120)
    feature_values_str = feature_values.iloc[:, 0].apply(add_line_break, args=args)
    feature_values = pd.DataFrame({column_name: feature_values_str})

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
                [f"{text_groups_features_keys[i]}: %{{text[{i}]}}" for i in range(len(text_groups_features_keys))]
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

    feature_values_array = feature_values.values.flatten()

    if len(feature_values_array) > 2:
        contributions_min = contributions.values.flatten().min()
        h = contributions.values.flatten().max() - contributions_min

        if feature_values.iloc[:, 0].dtype.kind in "biufc":
            feature_values_min, feature_values_max = min(feature_values_array), max(feature_values_array)
            val_inter = feature_values_max - feature_values_min
            from sklearn.neighbors import KernelDensity

            feature_np = np.array(feature_values_array)
            feature_np = feature_np[~np.isnan(feature_np)][:, None]
            kde = KernelDensity(bandwidth=val_inter / 100, kernel="epanechnikov").fit(feature_np)
            xs = np.linspace(feature_values_min, feature_values_max, 1000)
            log_dens = kde.score_samples(xs[:, None])
            y_upper = np.exp(log_dens) * h / (np.max(np.exp(log_dens)) * 3) + contributions_min
            y_lower = np.full_like(y_upper, contributions_min)
        else:
            feature_values_counts = feature_values.value_counts()
            xs = feature_values_counts.index.get_level_values(0).sort_values()
            y_upper = (
                feature_values_counts.loc[xs] / feature_values_counts.sum()
            ).values.flatten() / 3 + contributions_min
            y_lower = np.full_like(y_upper, contributions_min)

        # Create the density plot
        density_plot = go.Scatter(
            x=np.concatenate([pd.Series(xs), pd.Series(xs)[::-1]]),
            y=pd.concat([pd.Series(y_upper), pd.Series(y_lower)[::-1]]),
            fill="toself",
            hoverinfo="none",
            showlegend=False,
            line={"color": style_dict["contrib_distribution"]},
        )
        # Add density plot
        fig.add_trace(density_plot)

    fig.add_scatter(
        x=feature_values_array,
        y=contributions.values.flatten(),
        mode="markers",
        hovertext=hv_text,
        hovertemplate=hovertemplate,
        text=text_groups_features,
        showlegend=False,
    )
    # To change ticktext when the x label size is upper than 10 and zoom is False
    if (isinstance(feature_values_array[0], str)) & (not zoom):
        feature_val = [x.replace("<br />", "") for x in feature_values_array]
        feature_val = [x.replace(x[3 : len(x) - 3], "...") if len(x) > 10 else x for x in feature_val]

        fig.update_xaxes(tickangle=45, ticktext=feature_val, tickvals=feature_values_array, tickmode="array", dtick=1)
    # Customdata contains the values and index of feature_values.
    # The values are used in the hovertext and the indexes are used for
    # the interactions between the graphics.
    customdata = np.stack((feature_values_array, feature_values.index.values), axis=-1)

    fig.update_traces(customdata=customdata, hovertemplate=hovertemplate)

    _update_contributions_fig(
        fig=fig,
        feature_name=feature_name,
        pred=pred,
        proba_values=proba_values,
        col_modality=col_modality,
        col_scale=col_scale,
        cmin=cmin,
        cmax=cmax,
        addnote=addnote,
        subtitle=subtitle,
        width=width,
        height=height,
        file_name=file_name,
        auto_open=auto_open,
        case=case,
        style_dict=style_dict,
    )

    return fig


def plot_violin(
    feature_values,
    contributions,
    feature_name,
    case,
    style_dict,
    pred=None,
    proba_values=None,
    col_modality=None,
    col_scale=None,
    cmin=None,
    cmax=None,
    addnote=None,
    subtitle=None,
    max_len_by_row=20,
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
    case: str
        Type of the model, either 'classification' or 'regression'
    style_dict: dict
        the different styles used in the different outputs of Shapash
    pred: 1 column pd.DataFrame (optional)
        predicted values used to color plot - One Vs All in multiclass case
    proba_values: 1 column pd.DataFrame (optional)
        predicted proba used to color points - One Vs All in multiclass case
    col_modality: Int, Float or String (optional)
        parameter used in classification case,
        specify the modality to color in scatter plot (One Vs All)
    col_scale: list (optional)
        specify the color of points in scatter data
    cmin : float, optional
        The minimum value for the color scale, providing the lower bound for color normalization.
    cmax : float, optional
        The maximum value for the color scale, providing the upper bound for color normalization.
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
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    column_name = feature_values.columns[0]
    feature_values = feature_values.sort_values(by=column_name)

    # add break line to X label if necessary
    args = (max_len_by_row, 120)
    feature_values_str = feature_values.iloc[:, 0].apply(add_line_break, args=args)
    feature_values = pd.DataFrame({column_name: feature_values_str})

    contributions = contributions.loc[feature_values.index]
    if pred is not None:
        pred = pred.loc[feature_values.index]
    if proba_values is not None:
        proba_values = proba_values.loc[feature_values.index]

    hv_text_df, hovertemplate = _prepare_hover_text(feature_values, pred, feature_name)

    feature_values_counts = feature_values.value_counts()
    xs = feature_values_counts.index.get_level_values(0).sort_values()

    y_upper = (feature_values_counts.loc[xs] / feature_values_counts.sum()).values.flatten()
    y_upper_max = y_upper.max()

    if case == "classification":
        colorpoints = proba_values
    elif case == "regression":
        colorpoints = pred
    else:
        colorpoints = None

    for i, c in enumerate(xs):
        # Add Density Plot
        fig.add_trace(
            go.Bar(
                x=[i],
                y=[y_upper[i]],
                hoverinfo="none",
                showlegend=False,
                marker=dict(
                    pattern_shape="+",
                    pattern_size=6,
                    pattern_fillmode="replace",
                    pattern_bgcolor=style_dict["contrib_distribution"],
                    color="white",
                ),
            )
        )

        if pred is not None and case == "classification":
            # Negative case
            feature_cond_neg = (pred.iloc[:, 0] != col_modality) & (feature_values.iloc[:, 0] == c)
            _add_violin_and_scatter(
                fig,
                feature_cond_neg,
                contributions,
                feature_values,
                hv_text_df,
                colorpoints,
                col_scale,
                cmin,
                cmax,
                hovertemplate,
                i,
                c,
                line_color=style_dict["violin_area_classif"][0],
                secondary_y=True,
                side="negative",
            )

            # Positive case
            feature_cond_pos = (pred.iloc[:, 0] == col_modality) & (feature_values.iloc[:, 0] == c)
            _add_violin_and_scatter(
                fig,
                feature_cond_pos,
                contributions,
                feature_values,
                hv_text_df,
                colorpoints,
                col_scale,
                cmin,
                cmax,
                hovertemplate,
                i,
                c,
                line_color=style_dict["violin_area_classif"][1],
                secondary_y=True,
                side="positive",
            )
        else:
            # General case
            feature_cond_other = feature_values.iloc[:, 0] == c
            _add_violin_and_scatter(
                fig,
                feature_cond_other,
                contributions,
                feature_values,
                hv_text_df,
                colorpoints,
                col_scale,
                cmin,
                cmax,
                hovertemplate,
                i,
                c,
                line_color=style_dict["violin_default"],
                secondary_y=True,
                side="both",
            )

    if colorpoints is not None:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                showlegend=False,
                hoverinfo="none",
            ),
            secondary_y=True,
        )

    fig.update_layout(
        violingap=0.05,
        violingroupgap=0,
        violinmode="overlay",
        xaxis_type="linear",
        barmode="overlay",
        yaxis=dict(
            side="right",
            range=[0, y_upper_max * 3],
            showticklabels=False,  # Hide tick labels
            showgrid=False,  # Hide grid lines (optional)
            visible=False,  # Make the entire axis invisible
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
        ),
    )

    # To change ticktext
    _update_xaxis_labels(fig, xs, zoom)

    _update_contributions_fig(
        fig=fig,
        feature_name=feature_name,
        pred=pred,
        proba_values=proba_values,
        col_modality=col_modality,
        col_scale=col_scale,
        cmin=cmin,
        cmax=cmax,
        addnote=addnote,
        subtitle=subtitle,
        width=width,
        height=height,
        file_name=file_name,
        auto_open=auto_open,
        case=case,
        style_dict=style_dict,
    )

    return fig


def _update_contributions_fig(
    fig,
    feature_name,
    pred,
    proba_values,
    col_modality,
    col_scale,
    cmin,
    cmax,
    addnote,
    subtitle,
    width,
    height,
    file_name,
    auto_open,
    case,
    style_dict,
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
    cmin : float, optional
        The minimum value for the color scale, providing the lower bound for color normalization.
    cmax : float, optional
        The maximum value for the color scale, providing the upper bound for color normalization.
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
    case: str
        Type of the model, either 'classification' or 'regression'
    style_dict: dict
        the different styles used in the different outputs of Shapash
    """
    title = f"<b>{truncate_str(feature_name)}</b> - Feature Contribution"
    # Add subtitle and / or addnote
    if subtitle or addnote:
        if subtitle and addnote:
            title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
        elif subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
        else:
            title += "<br><sup>" + addnote + "</sup>"
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
    dict_xaxis = style_dict["dict_xaxis"] | {"text": truncate_str(feature_name, 110)}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": "Contribution"}

    if case == "regression":
        colorpoints = pred
        colorbar_title = "Predicted"
    elif case == "classification":
        colorpoints = proba_values
        colorbar_title = "Predicted Proba"

    if colorpoints is not None:
        if fig.data[-1].type == "scatter":
            fig.data[-1].marker.color = colorpoints.values.flatten()
            fig.data[-1].marker.coloraxis = "coloraxis"
        fig.layout.coloraxis.colorscale = col_scale
        fig.layout.coloraxis.colorbar = {"title": {"text": colorbar_title}}
        if (cmin is not None) and (cmax is not None):
            fig.layout.coloraxis.cmin = cmin
            fig.layout.coloraxis.cmax = cmax

    elif fig.data[0].type != "violin":
        if case == "classification" and pred is not None:
            fig.data[-1].marker.color = pred.iloc[:, 0].apply(
                lambda x: (
                    style_dict["violin_area_classif"][1] if x == col_modality else style_dict["violin_area_classif"][0]
                )
            )
        else:
            fig.data[-1].marker.color = style_dict["violin_default"]

    fig.update_traces(marker={"line": {"width": 0.8, "color": "white"}})
    for trace in fig.data:
        if trace.type != "bar":
            trace.marker["size"] = 10

    fig.update_layout(
        boxmode="group",
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


def _update_xaxis_labels(fig, xs, zoom=False):
    """
    Updates the x-axis labels of a Plotly figure based on label length and zoom status.
    Shortens labels if they are longer than a specified threshold.

    Parameters:
    - fig: The Plotly figure object to update.
    - xs: A list of x-axis label strings.
    - zoom: Boolean indicating whether zoom is enabled.
    """

    # Define common x-axis parameters
    params = {"tickvals": list(range(len(xs))), "tickmode": "array", "dtick": 1, "range": [-0.6, len(xs) - 0.4]}

    nb_feature = len(xs)
    # Determine label shortening strategy based on label count and zoom status
    if isinstance(xs[0], str):
        if not zoom:
            feature_val = [x.replace("<br />", "") for x in xs]
            if nb_feature < 6:
                k = 10
            else:
                k = 6

            # Shorten labels that exceed the threshold
            feature_val = [
                x.replace(x[k + k // 2 : -k + k // 2], "...") if len(x) > 2 * k + 3 else x for x in feature_val
            ]
        else:
            k = 10
            feature_val = []
            for feature_name in xs:
                feature_name_splited = [
                    x.replace(x[k + k // 2 : -k + k // 2], "...") if len(x) > 2 * k + 3 else x
                    for x in feature_name.split("<br />")
                ]
                feature_val_name = "<br />".join(feature_name_splited)
                feature_val.append(feature_val_name)

        params["ticktext"] = feature_val

        # Adjust tick angle for longer lists of labels
        if nb_feature > 5 * (zoom + 1):
            params["tickangle"] = 45
    else:
        params["ticktext"] = xs

    # Update the figure with the new x-axis parameters
    fig.update_xaxes(**params)


def _calculate_percentage_intervals(data, bins=20):
    """
    Calculates the percentage of data points within each interval of a binned distribution.

    Parameters:
    - data: DataFrame containing the data to bin and calculate percentages for.
    - bins: Number of bins to use for the distribution.

    Returns:
    - A numpy array of the percentage of points in the interval corresponding to each original data point.
    """
    # Binning data into intervals and calculating the percentage of points in each interval
    intervals = pd.cut(data, bins, duplicates="drop")
    points_per_interval = intervals.value_counts()
    total_points = len(data)
    percentage_per_interval = (points_per_interval / total_points).sort_index().to_dict()

    # Mapping those percentages to the original data points
    percentage_series = intervals.map(percentage_per_interval).to_numpy()

    return percentage_series


def _create_jittered_points(numerical_features, percentages, mean=0, std=0.6, clip_min=-1, clip_max=1, side="both"):
    """
    Creates jittered points by applying a random normal perturbation scaled by calculated percentages.

    Parameters:
    - numerical_features: The numerical features to which jitter will be added.
    - percentages: The percentages to scale the jitter by.
    - mean: Mean of the normal distribution to generate jitter.
    - std: Standard deviation of the normal distribution to generate jitter.
    - clip_min: Minimum value to clip the jitter values to.
    - clip_max: Maximum value to clip the jitter values to.

    Returns:
    - A numpy array of jittered points.
    """
    # Creating jittered points
    rng = np.random.default_rng(seed=79)
    jitter = rng.normal(mean, std, len(percentages))
    if np.isnan(percentages).any():
        percentages.fill(1)

    if side in ["negative", "positive"]:
        jitter = np.abs(jitter)

    jitter = np.clip(jitter, clip_min, clip_max)

    if side == "negative":
        jitter *= -1

    jittered_points = numerical_features + np.clip(jitter * percentages, -0.5, 0.5)

    return jittered_points


def _prepare_hover_text(feature_values, pred, feature_name):
    """
    Prepares the hover text for a Plotly plot based on feature values and predictions.

    Parameters:
    - feature_values: A pandas DataFrame of feature values.
    - pred: A pandas Series of predictions, can be None.
    - feature_name: The name of the feature for which the hover text is being prepared.

    Returns:
    - A pandas DataFrame containing the hover text.
    - The hover template to be used in Plotly.
    """
    # Building the base text for hover
    hv_text = [
        f"Id: {id_val}{f'<br />Predict: {pred_val}' if pred is not None else ''}"
        for id_val, pred_val in zip(
            feature_values.index, pred.values.flatten() if pred is not None else [""] * len(feature_values)
        )
    ]

    # Creating a DataFrame for hover text
    hv_text_df = pd.DataFrame(hv_text, columns=["text"], index=feature_values.index)

    # Hover template with contribution and custom data
    hv_temp = f"{feature_name} :<br />%{{customdata[0]}}<br />Contribution: %{{y:.4f}}<extra></extra>"
    hovertemplate = f"<b>%{{hovertext}}</b><br />{hv_temp}"

    return hv_text_df, hovertemplate


def _add_violin_and_scatter(
    fig,
    feature_cond,
    contributions,
    feature_values,
    hovertext_df,
    colorpoints,
    col_scale,
    cmin,
    cmax,
    hovertemplate,
    i,
    c,
    line_color,
    secondary_y=True,
    side="both",
):
    """Adds a Violin trace and a Scatter trace based on specified conditions."""
    y = contributions.loc[feature_cond].iloc[:, 0].values
    if len(y) > 0:
        x = [i] * len(y)
        hovertext = hovertext_df.loc[feature_cond].values.flatten()

        _add_violin_trace(fig, c, x, y, side, line_color, hovertext, secondary_y)

        percentage_series = _calculate_percentage_intervals(contributions.loc[feature_cond].iloc[:, 0], bins=20)
        x = _create_jittered_points(x, percentage_series, side=side)
        if colorpoints is not None:
            colorpoints_selected = colorpoints.loc[feature_cond].values.flatten()
        customdata = np.stack(
            (feature_values.loc[feature_cond].values.flatten(), contributions.loc[feature_cond].index.values),
            axis=-1,
        )
        marker = None
        if colorpoints is not None:
            marker = {
                "color": colorpoints_selected,
                "colorscale": col_scale,
                "opacity": 0.7,
                "cmin": cmin,
                "cmax": cmax,
            }

        _add_scatter_trace(fig, x, y, c, marker, hovertext, hovertemplate, customdata, secondary_y)


def _add_scatter_trace(fig, x, y, name, marker, hovertext, hovertemplate, customdata, secondary_y=True):
    """Adds a Scatter trace to the figure."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name=name,
            mode="markers",
            marker=marker,
            showlegend=False,
            hovertext=hovertext,
            hovertemplate=hovertemplate,
            customdata=customdata,
        ),
        secondary_y=secondary_y,
    )


def _add_violin_trace(fig, name, x, y, side, line_color, hovertext, secondary_y=True):
    """Adds a Violin trace to the figure."""
    # Violin plot has a problem if for one violin all the points have the same contribution value
    rng = np.random.default_rng(seed=79)
    y = y + rng.normal(size=y.shape) * (max(y.max(), 0) - min(y.min(), 0)) / 10**8
    violin_trace = go.Violin(
        name=name,
        x=x,
        y=y,
        side=side,
        line_color=line_color,
        points=False,
        showlegend=False,
        meanline_visible=True,
        hovertext=hovertext,
    )

    if side:
        violin_trace.update(side=side)

    fig.add_trace(violin_trace, secondary_y=secondary_y)
