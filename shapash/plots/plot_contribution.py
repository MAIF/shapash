from numbers import Number

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn.neighbors import KernelDensity

from shapash.utils.utils import add_line_break, adjust_title_height, truncate_str
from shapash.webapp.utils.utils import round_to_k

NAN_PLACEHOLDER_K = 0.2


def _compute_density_polygon(
    x_values: pd.Series,
    y_values: pd.Series,
    include_na_for_categories: bool = False,
) -> tuple[pd.Series | np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute the polygon coordinates for a density/volumetry background layer.

    Returns ``None`` when there is not enough information to compute a
    meaningful density shape.
    """

    if len(x_values) <= 2:
        return None

    y_min = y_values.min()
    y_span = y_values.max() - y_min

    if pd.api.types.is_numeric_dtype(x_values):
        x_non_null = x_values.dropna().astype(float)
        if x_non_null.nunique() <= 1:
            return None

        x_min, x_max = x_non_null.min(), x_non_null.max()
        val_inter = x_max - x_min
        if val_inter <= 0:
            return None

        kde = KernelDensity(bandwidth=val_inter / 100, kernel="epanechnikov").fit(x_non_null.to_numpy()[:, None])
        xs = np.linspace(x_min, x_max, 1000)
        log_dens = kde.score_samples(xs[:, None])

        if y_span == 0:
            y_upper = np.full_like(xs, y_min, dtype=float)
        else:
            dens = np.exp(log_dens)
            y_upper = dens * y_span / (dens.max() * 3) + y_min
        y_lower = np.full_like(y_upper, y_min)
    else:
        x_counts = x_values.value_counts(dropna=not include_na_for_categories)
        if x_counts.shape[0] <= 1:
            return None

        xs = x_counts.index.to_series().sort_values()
        y_upper = (x_counts.loc[xs] / x_counts.sum()).to_numpy() / 3 + y_min
        y_lower = np.full_like(y_upper, y_min)

    return xs, y_upper, y_lower


def _add_density_trace(
    fig: go.Figure,
    x_values: pd.Series,
    y_values: pd.Series,
    style_dict: dict,
    include_na_for_categories: bool = False,
) -> None:
    """
    Add a density/volumetry background trace to an existing figure.
    """

    density_polygon = _compute_density_polygon(
        x_values=x_values,
        y_values=y_values,
        include_na_for_categories=include_na_for_categories,
    )
    if density_polygon is None:
        return

    xs, y_upper, y_lower = density_polygon
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([pd.Series(xs), pd.Series(xs)[::-1]]),
            y=pd.concat([pd.Series(y_upper), pd.Series(y_lower)[::-1]]),
            fill="toself",
            hoverinfo="none",
            showlegend=False,
            line={"color": style_dict["contrib_distribution"]},
        )
    )


def _build_secondary_y_scatter_figure(scatter_fig: go.Figure, y_values: pd.Series) -> go.Figure:
    """
    Rebuild a scatter figure on a two-y-axis subplot layout.

    Filled density traces are attached to the primary axis, while marker traces
    are attached to the overlaid secondary axis.
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for trace in scatter_fig.data:
        if trace.fill == "toself":
            fig.add_trace(trace, secondary_y=False)
        else:
            trace.mode = "markers"
            fig.add_trace(trace, secondary_y=True)

    fig.update_layout(
        autosize=False,
        hovermode="closest",
        barmode="overlay",
        yaxis=dict(
            side="right",
            range=[float(y_values.min()), float(y_values.max())] if len(y_values) > 0 else None,
            showticklabels=False,
            showgrid=False,
            visible=False,
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
        ),
    )

    return fig


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
    """

    if x_values_hover is None:
        x_values_hover = x_values

    x_series = pd.Series(x_values.values.flatten())
    y_series = pd.Series(y_values.values.flatten())

    data_df = pd.DataFrame(
        {
            x_name: x_series.values,
            y_name: y_series.values,
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

    if x_values_hover.equals(x_values):
        _add_density_trace(fig, x_series, y_series, style_dict, include_na_for_categories=True)

    return _build_secondary_y_scatter_figure(fig, y_series)


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
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    uniq_l = list(pd.unique(x_values.values.flatten()))
    uniq_l.sort()

    x_numeric = pd.Series(index=x_values.index, dtype=float)
    x_jittered = pd.Series(index=x_values.index, dtype=float)

    proportions = (x_values.iloc[:, 0].value_counts(dropna=False) / len(x_values)).to_dict()

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
            go.Bar(
                x=[idx],
                y=[proportions.get(modality, 0.0)],
                hoverinfo="none",
                showlegend=False,
                marker=dict(
                    pattern_shape="+",
                    pattern_size=6,
                    pattern_fillmode="replace",
                    pattern_bgcolor=style_dict["contrib_distribution"],
                    color="white",
                ),
            ),
            secondary_y=False,
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
            ),
            secondary_y=True,
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
        fig.add_trace(trace, secondary_y=True)

    y_upper_max = max(proportions.values()) if proportions else 1.0

    fig.update_layout(
        autosize=False,
        hovermode="closest",
        violingap=0.05,
        violingroupgap=0,
        violinmode="overlay",
        xaxis_type="linear",
        barmode="overlay",
        yaxis=dict(
            side="right",
            range=[0, y_upper_max * 3],
            showticklabels=False,
            showgrid=False,
            visible=False,
        ),
        yaxis2=dict(
            overlaying="y",
            side="left",
        ),
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
    subtitle: str | None,
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
    """

    if fig.data[-1]["showlegend"] is False:
        fig.layout.coloraxis.colorscale = col_scale if col_scale is not None else style_dict["interactions_col_scale"]
        if cmin is not None and cmax is not None:
            fig.layout.coloraxis.cmin = cmin
            fig.layout.coloraxis.cmax = cmax
    else:
        fig.update_layout(legend=dict(title=dict(text=col_name2)))

    title = f"<b>{truncate_str(col_name1)} and {truncate_str(col_name2)}</b> shap interaction values"
    if subtitle or addnote:
        if subtitle and addnote:
            title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
        elif subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
        else:
            title += "<br><sup>" + addnote + "</sup>"
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
    dict_xaxis = style_dict["dict_xaxis"] | {"text": truncate_str(col_name1, 110)}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": "Shap interaction value"}

    fig.update_traces(marker={"line": {"width": 0.8, "color": "white"}})
    for trace in fig.data:
        if trace.type != "bar":
            trace.marker["size"] = 8
            trace.marker["opacity"] = 0.8

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
        hv_text = [
            f"Id: {x}<br />Predict: {y}" for x, y in zip(feature_values.index, pred.values.flatten(), strict=False)
        ]
    else:
        hv_text = [f"Id: {x}" for x in feature_values.index]

    if metadata:
        metadata = {
            k: [
                round_to_k(x, 3) if isinstance(x, Number) else x
                for x in pd.Series(v, index=feature_values.index).reindex(feature_values.index)
            ]
            for k, v in metadata.items()
        }

        text_groups_features = np.array([col_values for col_values in metadata.values()])
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

    _add_density_trace(
        fig=fig,
        x_values=pd.Series(feature_values_array),
        y_values=pd.Series(contributions.values.flatten()),
        style_dict=style_dict,
        include_na_for_categories=False,
    )

    nan_mask_arr = pd.isna(feature_values.iloc[:, 0]).to_numpy()
    has_nan_numeric = bool(nan_mask_arr.any()) and feature_values.iloc[:, 0].dtype.kind in "biufc"
    marker = None
    if has_nan_numeric:
        non_nan_arr = feature_values_array[~nan_mask_arr].astype(float)
        if non_nan_arr.size > 0:
            vmax = float(non_nan_arr.max())
            spread = vmax - float(non_nan_arr.min())
            nan_x = vmax + spread * NAN_PLACEHOLDER_K if spread > 0 else vmax + 1.0
        else:
            nan_x = 0.0
        feature_values_array = np.where(nan_mask_arr, nan_x, feature_values_array)
        marker = {"symbol": np.where(nan_mask_arr, "x", "circle").tolist()}

    fig.add_scatter(
        x=feature_values_array,
        y=contributions.values.flatten(),
        mode="markers",
        hovertext=hv_text,
        hovertemplate=hovertemplate,
        text=text_groups_features,
        marker=marker,
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
    customdata_values = feature_values_array
    if has_nan_numeric:
        customdata_values = feature_values_array.astype(object).copy()
        customdata_values[nan_mask_arr] = "missing"
    customdata = np.stack((customdata_values, feature_values.index.values), axis=-1)

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

    feature_values_counts = feature_values.value_counts(dropna=False)
    xs = feature_values_counts.index.get_level_values(0).sort_values()

    y_upper = (feature_values_counts.sort_index() / feature_values_counts.sum()).values.flatten()
    y_upper_max = y_upper.max()

    if case == "classification":
        colorpoints = proba_values
    elif case == "regression":
        colorpoints = pred
    else:
        colorpoints = None

    for i, c in enumerate(xs):
        if pd.isna(c):
            is_c = feature_values.iloc[:, 0].isna()
            c_label = "missing"
        else:
            is_c = feature_values.iloc[:, 0] == c
            c_label = c

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
            feature_cond_neg = (pred.iloc[:, 0] != col_modality) & is_c
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
                c_label,
                line_color=style_dict["violin_area_classif"][0],
                secondary_y=True,
                side="negative",
            )

            # Positive case
            feature_cond_pos = (pred.iloc[:, 0] == col_modality) & is_c
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
                c_label,
                line_color=style_dict["violin_area_classif"][1],
                secondary_y=True,
                side="positive",
            )
        else:
            # General case
            feature_cond_other = is_c
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
                c_label,
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
    xs_labels = ["missing" if pd.isna(x) else x for x in xs]
    _update_xaxis_labels(fig, xs_labels, zoom)

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
            feature_values.index,
            pred.values.flatten() if pred is not None else [""] * len(feature_values),
            strict=False,
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
