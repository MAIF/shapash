import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from plotly import graph_objs as go
from plotly.offline import plot

from shapash.style.style_utils import get_pyplot_color
from shapash.utils.utils import adjust_title_height


def plot_feature_importance(
    mode,
    global_feat_imp,
    contributions_case,
    style_dict,
    features_groups_keys=None,
    features_dict=None,
    inv_features_dict=None,
    local_imp_lev1=None,
    local_imp_lev2=None,
    subset_feat_imp=None,
    display_groups=False,
    title="",
    addnote="",
    subtitle="",
    width=900,
    height=500,
    file_name=None,
    auto_open=False,
    zoom=False,
    normalize_by_nb_samples=False,
    degree="slider",
):
    """
    Generate feature importance plots using different modes such as global, global-local, or cumulative contributions.

    Parameters
    ----------
    mode : str
        Defines the type of plot to generate. Accepts 'global', 'global-local', or 'cumulative'.
    global_feat_imp : pandas.DataFrame
        Dataframe containing global feature importance values.
    contributions_case : pandas.DataFrame
        Dataframe containing contribution values for individual cases (used for cumulative mode).
    style_dict : dict
        Dictionary containing styles for the plot (e.g., color, font sizes, etc.).
    features_groups_keys : dict, optional
        Dictionary containing groups of features. The keys represent group names, and the values are lists of feature names.
        Used to group features in the display, by default None.
    features_dict : dict, optional
        Dictionary mapping technical feature names to domain names, by default None.
    inv_features_dict : dict, optional
        Inverse dictionary of `features_dict`, used for displaying original feature names, by default None.
    local_imp_lev1 : pandas.DataFrame, optional
        Dataframe containing semi-local feature importance values (level 1), used in 'global-local' mode, by default None.
    local_imp_lev2 : pandas.DataFrame, optional
        Dataframe containing local feature importance values (level 2), used in 'global-local' mode, by default None.
    subset_feat_imp : pandas.DataFrame, optional
        Dataframe containing a subset of feature importance values for selective display, by default None.
    display_groups : bool, optional
        Whether to display feature groups in the plot. If True, features will be grouped accordingly, by default False.
    title : str, optional
        Title of the plot, by default "".
    addnote : str, optional
        Additional notes to be added to the plot, by default "".
    subtitle : str, optional
        Subtitle of the plot, by default "".
    width : int, optional
        Width of the plot, by default 900.
    height : int, optional
        Height of the plot, by default 500.
    file_name : str, optional
        Path to save the plot as an HTML file. If None, the plot will not be saved, by default None.
    auto_open : bool, optional
        If True, the plot will automatically open in a web browser after being generated, by default False.
    zoom : bool, optional
        If True, the plot will be zoomed in, by default False.
    normalize_by_nb_samples : bool, optional
        Whether to normalize the cumulative contributions by the number of samples (used in cumulative mode), by default False.
    degree : str, optional
        Degree of the cumulative plot, often represented as a slider, by default "slider".

    Returns
    -------
    plotly.graph_objects.Figure
        A plotly figure object containing the feature importance plot.

    Raises
    ------
    ValueError
        If an invalid `mode` is passed. The mode must be one of 'global', 'global-local', or 'cumulative'.

    Examples
    --------
    >>> plot_feature_importance(
            mode="global",
            global_feat_imp=global_feat_imp_df,
            contributions_case=contributions_df,
            style_dict=style_dict,
            title="Feature Importance",
            width=1000,
            height=600
        )
    """
    # Map feature names
    if features_dict is not None:
        global_feat_imp.index = global_feat_imp.index.map(features_dict)
        if mode == "global-local":
            local_imp_lev1.index = local_imp_lev1.index.map(features_dict)
            local_imp_lev2.index = local_imp_lev2.index.map(features_dict)

    if inv_features_dict is None:
        inv_features_dict = {}

    # Format indices if display_groups is enabled
    if display_groups:
        global_feat_imp, local_imp_lev1, local_imp_lev2, subset_feat_imp = _apply_bold_formatting(
            global_feat_imp,
            local_imp_lev1,
            local_imp_lev2,
            subset_feat_imp,
            mode,
            features_groups_keys,
            inv_features_dict,
        )

    """Generate the feature importance plot based on the mode."""
    if mode == "global":
        return _plot_features_import(
            global_feat_imp,
            style_dict,
            inv_features_dict,
            features_groups_keys,
            subset_feat_imp,
            title,
            addnote,
            subtitle,
            width,
            height,
            file_name,
            auto_open,
            zoom,
        )
    elif mode == "global-local":
        feat_imp = {"global": global_feat_imp, "semi-local": local_imp_lev1, "local": local_imp_lev2}
        return _plot_local_features_import(
            feat_imp,
            style_dict,
            inv_features_dict,
            title,
            features_groups_keys,
            addnote,
            subtitle,
            width,
            height,
            file_name,
            auto_open,
            zoom,
        )
    elif mode == "cumulative":
        return _plot_feature_contributions_cumulative(
            global_feat_imp,
            contributions_case,
            style_dict,
            inv_features_dict,
            title,
            addnote,
            subtitle,
            width,
            height,
            normalize_by_nb_samples,
            degree,
            file_name,
            auto_open,
            zoom,
        )
    else:
        raise ValueError("Invalid value for mode. It must be 'global', 'global-local', or 'cumulative'.")


def _plot_features_import(
    feature_imp1,
    style_dict,
    inv_features_dict,
    features_groups_keys=None,
    feature_imp2=None,
    title="Features Importance",
    addnote=None,
    subtitle=None,
    width=900,
    height=500,
    file_name=None,
    auto_open=False,
    zoom=False,
):
    """
    Plot features importance computed with the prediction set.
    Parameters
    ----------
    feature_imp1 : pd.Series
        Feature importance computed with every rows
    style_dict: dict
        the different styles used in the different outputs of Shapash
    inv_features_dict: dict
        Inverse features_dict mapping.
    features_groups : dict, optional (default: None)
        Keys of the dictionnary containing features that should be grouped together.
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
    zoom: bool (default=False)
        graph is currently zoomed
    """

    topmargin = 80
    # Add subtitle and / or addnote
    if subtitle or addnote:
        if subtitle and addnote:
            title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
        elif subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
        else:
            title += "<br><sup>" + addnote + "</sup>"
        topmargin = topmargin + 15
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
    dict_xaxis = style_dict["dict_xaxis"] | {"text": "Mean absolute Contribution"}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": None}
    dict_style_bar1 = style_dict["dict_featimp_colors"][1]
    dict_style_bar2 = style_dict["dict_featimp_colors"][2]

    # Change bar color for groups of features
    marker_color = [
        (
            style_dict["featureimp_groups"][0]
            if (
                features_groups_keys is not None
                and inv_features_dict.get(f.replace("<b>", "").replace("</b>", "")) in features_groups_keys
            )
            else dict_style_bar1["color"]
        )
        for f in feature_imp1.index
    ]

    layout = go.Layout(
        barmode="group",
        template="none",
        autosize=False,
        width=width,
        height=height,
        title=dict_t,
        xaxis_title=dict_xaxis,
        yaxis_title=dict_yaxis,
        hovermode="closest",
        margin={"l": 160, "r": 0, "t": topmargin, "b": 50},
    )
    # To change ticktext when the x label size is upper than 30 and zoom is False
    if (isinstance(feature_imp1.index[0], str)) & (not zoom):
        # change index to abc...abc if its length is upper than 30
        index_val = [y.replace(y[24 : len(y) - 3], "...") if len(y) > 30 else y for y in feature_imp1.index]
    else:
        index_val = feature_imp1.index
    bar1 = go.Bar(
        x=feature_imp1.round(4),
        y=feature_imp1.index,
        orientation="h",
        name="Global",
        marker=dict_style_bar1,
        marker_color=marker_color,
        hovertemplate="Feature: %{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
        customdata=feature_imp1.index,
    )
    if feature_imp2 is not None:
        bar2 = go.Bar(
            x=feature_imp2.round(4),
            y=feature_imp2.index,
            orientation="h",
            name="Subset",
            marker=dict_style_bar2,
            hovertemplate="Feature: %{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
            customdata=feature_imp2.index,
        )
        data = [bar2, bar1]
    else:
        data = bar1

    fig = go.Figure(data=data, layout=layout)
    # Update ticktext
    fig.update_yaxes(ticktext=index_val, tickvals=feature_imp1.index, tickmode="array", dtick=1)
    fig.update_yaxes(automargin=True)
    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)
    return fig


def _plot_local_features_import(
    feat_imp,
    style_dict,
    inv_features_dict,
    title="Features Importance Global-Local",
    features_groups_keys=None,
    addnote=None,
    subtitle=None,
    width=900,
    height=500,
    file_name=None,
    auto_open=False,
    zoom=False,
):
    """
    Plot features importance computed with the prediction set.
    Parameters
    ----------
    feat_imp : dict of pd.Series
        Feature importance computed with every rows :global, semi-local and local
    style_dict: dict
        the different styles used in the different outputs of Shapash
    inv_features_dict: dict
        Inverse features_dict mapping.
    title : str
        Title of the plot, default set to 'Features Importance'
    features_groups : dict, optional (default: None)
        Keys of the dictionnary containing features that should be grouped together.
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
    zoom: bool (default=False)
        graph is currently zoomed
    """
    topmargin = 80
    # Add subtitle and / or addnote
    if subtitle or addnote:
        if subtitle and addnote:
            title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
        elif subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
        else:
            title += "<br><sup>" + addnote + "</sup>"
        topmargin = topmargin + 15
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
    dict_xaxis = style_dict["dict_xaxis"] | {"text": "Mean absolute Contribution"}
    dict_yaxis = style_dict["dict_yaxis"] | {"text": None}
    dict_style_bar = {}
    for type_feat, i in zip(["global", "semi-local", "local"], [1, 3, 4]):
        dict_style_bar[type_feat] = style_dict["dict_featimp_colors"][i]

    # Change bar color for groups of features
    marker_color = [
        (
            style_dict["featureimp_groups"][0]
            if (
                features_groups_keys is not None
                and inv_features_dict.get(f.replace("<b>", "").replace("</b>", "")) in features_groups_keys
            )
            else dict_style_bar["global"]["color"]
        )
        for f in feat_imp["global"].index
    ]

    layout = go.Layout(
        barmode="group",
        template="none",
        autosize=False,
        width=width,
        height=height,
        title=dict_t,
        xaxis_title=dict_xaxis,
        yaxis_title=dict_yaxis,
        hovermode="closest",
        margin={"l": 160, "r": 0, "t": topmargin, "b": 50},
    )

    data = []
    for type_feat in ["local", "semi-local", "global"]:
        feature_imp = feat_imp[type_feat]
        style_bar = dict_style_bar[type_feat]

        data.append(
            go.Bar(
                x=feature_imp.round(4),
                y=feature_imp.index,
                orientation="h",
                name=type_feat.capitalize(),
                marker=style_bar,
                marker_color=marker_color if type_feat == "global" else style_bar["color"],
                hovertemplate="Feature: %{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
                customdata=feature_imp.index,
            )
        )

    fig = go.Figure(data=data, layout=layout)

    # Update ticktext
    # To change ticktext when the x label size is upper than 30 and zoom is False
    if (isinstance(feat_imp["global"].index[0], str)) & (not zoom):
        # change index to abc...abc if its length is upper than 30
        index_val = [y.replace(y[24 : len(y) - 3], "...") if len(y) > 30 else y for y in feat_imp["global"].index]
    else:
        index_val = feat_imp["global"].index
    fig.update_yaxes(ticktext=index_val, tickvals=feat_imp["global"].index, tickmode="array", dtick=1)
    fig.update_yaxes(automargin=True)
    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)
    return fig


def _plot_feature_contributions_cumulative(
    feature_imp1,
    contributions_case,
    style_dict,
    inv_features_dict,
    title="Feature Contributions Cumulative Plot",
    addnote=None,
    subtitle=None,
    width=900,
    height=500,
    normalize_by_nb_samples=False,
    degree="slider",
    file_name=None,
    auto_open=False,
    zoom=False,
):
    """
    Generates a cumulative plot of feature contributions with a slider to adjust the degree.

    Parameters
    ----------
    feature_imp1 : pandas.DataFrame
        DataFrame containing the importance values of each feature. The index represents feature names.
    contributions_case : pandas.DataFrame
        DataFrame containing the individual feature contributions for each case (row).
    style_dict : dict
        Dictionary specifying style options such as color schemes and line styles.
    inv_features_dict : dict
        Dictionary mapping feature names to their corresponding indices in `contributions_case`.
    title : str, optional
        Title of the plot. Default is "Feature Contributions Cumulative Plot".
    addnote : str, optional
        Additional notes to be displayed as a subtitle in the plot.
    subtitle : str, optional
        A subtitle for the plot, displayed under the title.
    width : int, optional
        The width of the plot in pixels. Default is 900.
    height : int, optional
        The height of the plot in pixels. Default is 500.
    normalize_by_nb_samples : bool, optional
        Whether to normalize each feature's cumulative contribution by the number of samples. Default is False.
    degree : str or float, optional
        Degree of normalization to apply. If 'slider', an interactive slider will be added to control the normalization
        degree in the range [0, 1]. Default is "slider".
    file_name : str, optional
        The file path to save the generated plot as an HTML file. If not provided, the plot will not be saved. Default is None.
    auto_open : bool, optional
        Whether to automatically open the saved plot in a browser. Default is False.
    zoom : bool, optional
        Whether to allow zooming on long feature names. Default is False.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A Plotly figure containing the cumulative feature contribution plot.

    Notes
    -----
    - The function creates a cumulative plot for each feature, where the contributions are accumulated and plotted as a line graph.
    - If `normalize_by_nb_samples` is True, contributions are normalized by the number of samples (i.e., the number of cases).
    - The `degree` parameter controls how strongly the feature contributions are normalized. If set to "slider", an interactive
    slider is added to dynamically adjust the degree of normalization.
    - Hover information displays the feature name, and the cumulative contribution is shown on hover.

    Raises
    ------
    ValueError
        If `feature_imp1` or `contributions_case` is not a DataFrame, or if there are issues with indexing features.
    """
    # Number of features
    num_features = len(feature_imp1)

    # Generate color scale
    col_scale = get_pyplot_color(colors=style_dict["feature_contributions_cumulative"])
    cmap = LinearSegmentedColormap.from_list("feature_contributions_cumulative", col_scale, N=256)
    colors = [cmap(i / num_features) for i in range(num_features)]
    colors_hex = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _ in colors]

    # Initialize data for storing the series
    data = []
    serie_tot = None

    lst_feat = list(feature_imp1.index)[::-1]
    lst_feat = [f.replace("<b>", "").replace("</b>", "") for f in lst_feat]

    # Process each feature's contributions and compute cumulative sums
    for name in lst_feat:
        serie = (
            contributions_case[inv_features_dict.get(name)]
            .abs()
            .sort_values(ascending=False)
            .cumsum()
            .reset_index(drop=True)
        )
        data.append(serie)

        # Accumulate the total series for normalization
        if serie_tot is None:
            serie_tot = serie.copy()
        else:
            serie_tot += serie

    # Create the Plotly traces for each series
    topmargin = 80
    # Add subtitle and / or addnote
    if subtitle or addnote:
        if subtitle and addnote:
            title += "<br><sup>" + subtitle + " - " + addnote + "</sup>"
        elif subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
        else:
            title += "<br><sup>" + addnote + "</sup>"
        topmargin = topmargin + 15
    dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}

    if (isinstance(lst_feat[0], str)) & (not zoom):
        # change index to abc...abc if its length is upper than 30
        index_val = [y.replace(y[24 : len(y) - 3], "...") if len(y) > 30 else y for y in lst_feat]
    else:
        index_val = lst_feat

    figs = []
    for i, serie in enumerate(data):
        serie_values = serie.copy()

        # Optionally normalize by the number of samples
        if normalize_by_nb_samples:
            serie_values /= pd.Series(range(1, len(serie_values) + 1))

        # Apply initial degree-based normalization
        if degree not in [0, "slider"]:
            serie_values /= serie_tot**degree

        # Append the trace for the current series
        figs.append(
            go.Scatter(
                x=serie.index,
                y=serie_values,
                mode="lines",
                name=index_val[i],
                hoverinfo="text",
                text=lst_feat[i],
                line=dict(color=colors_hex[i], width=3),
                hoverlabel=dict(
                    font_size=12,
                ),
            )
        )

    # Define layout with a clean white background and title
    layout = go.Layout(
        title=dict_t,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        margin={"l": 10, "r": 0, "t": topmargin, "b": 10},
    )

    # Create the initial figure with the data and layout
    fig = go.Figure(data=figs, layout=layout)

    # Create a list of frames with updated data for each degree value
    if degree == "slider":
        frames = []
        degree_range = np.round(np.arange(0, 1.1, 0.1), 1)

        for deg in degree_range:
            new_figs = []
            max_y = 0
            for i, serie in enumerate(data):
                serie_values = serie.copy()

                if normalize_by_nb_samples:
                    serie_values /= pd.Series(range(1, len(serie_values) + 1))

                # Apply degree-based normalization
                if deg != 0:
                    serie_values /= serie_tot ** (-deg)

                max_y = max(max_y, serie_values.max())

                new_figs.append(
                    go.Scatter(
                        x=serie.index,
                        y=serie_values,
                        mode="lines",
                        hoverinfo="text",
                        text=lst_feat[i],
                        line=dict(color=colors_hex[i], width=3),
                        hoverlabel=dict(
                            font_size=12,
                        ),
                    )
                )

            # Layout for this degree value, adjusting y-axis range
            frame_layout = go.Layout(
                yaxis=dict(visible=False, autorange=True),
                plot_bgcolor="white",
                paper_bgcolor="white",
                width=width,
                height=height,
            )

            # Append each frame with its own layout
            frames.append(go.Frame(data=new_figs, name=f"degree_{deg}", layout=frame_layout))

        # Add slider to control the degree parameter
        sliders = [
            {
                "currentvalue": {"prefix": "Degree: "},
                "pad": {"b": 10},
                "steps": [
                    {
                        "args": [
                            [f"degree_{deg}"],
                            {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"},
                        ],
                        "label": str(deg),
                        "method": "animate",
                    }
                    for deg in degree_range
                ],
            }
        ]

        # Add frames and sliders to the figure
        fig.update(frames=frames)
        fig.update_layout(sliders=sliders)

    # Optionally save the plot to a file
    if file_name:
        plot(fig, filename=file_name, auto_open=auto_open)

    return fig


def _apply_bold_formatting(
    global_feat_imp,
    local_imp_lev1,
    local_imp_lev2,
    subset_feat_imp,
    mode,
    features_groups_keys=None,
    inv_features_dict=None,
):
    """Apply bold formatting to feature names for feature groups."""

    def bold_feature_name(index):
        feature_name = str(index)
        if inv_features_dict.get(index) in features_groups_keys:
            return f"<b>{feature_name}"
        return feature_name

    if inv_features_dict is None:
        inv_features_dict = {}
    if features_groups_keys is None:
        features_groups_keys = {}
    global_feat_imp.index = [bold_feature_name(f) for f in global_feat_imp.index]

    if mode == "global-local":
        local_imp_lev1.index = [bold_feature_name(f) for f in global_feat_imp.index]
        local_imp_lev2.index = [bold_feature_name(f) for f in global_feat_imp.index]
    if subset_feat_imp is not None:
        subset_feat_imp.index = [bold_feature_name(f) for f in subset_feat_imp.index]
    return global_feat_imp, local_imp_lev1, local_imp_lev2, subset_feat_imp
