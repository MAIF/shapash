from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_line_break, adjust_title_height, truncate_str


def plot_bar_chart(
    index_value,
    var_dict,
    x_val,
    contrib,
    style_dict,
    features_groups=None,
    x_init=None,
    features_dict=None,
    inv_features_dict=None,
    yaxis_max_label=12,
    subtitle=None,
    width=900,
    height=550,
    file_name=None,
    auto_open=False,
    zoom=False,
):
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
    style_dict: dict
        the different styles used in the different outputs of Shapash
    features_groups : dict, optional (default: None)
        Dictionnary containing features that should be grouped together. This option allows
        to compute and display the contributions and importance of this group of features.
        Features that are grouped together will still be displayed in the webapp when clicking
        on a group.
        >>> {
        ‘feature_group_1’ : ['feature3', 'feature7', 'feature24'],
        ‘feature_group_2’ : ['feature1', 'feature12'],
        }
    x_init: pandas.DataFrame (default: None)
        x_encoded dataset with inverse transformation with eventual postprocessing modifications.
    features_dict: dict (default: None)
        Dictionary mapping technical feature names to domain names.
    inv_features_dict: dict (default: None)
        Inverse features_dict mapping.
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
    zoom: bool (default=False)
        graph is currently zoomed
    Returns
    -------
    plotly bar plot
        A bar plot with selected contributions and
        associated feature values for one observation.
    """
    if features_dict is None:
        features_dict = {}
    if inv_features_dict is None:
        inv_features_dict = {}

    if len(index_value) != 0:
        topmargin = 80
        title = f"Local Explanation - Id: <b>{index_value[0]}</b>"
        # Add subtitle
        if subtitle:
            title += "<br><sup>" + subtitle + "</sup>"
            topmargin += 15
        dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
        dict_xaxis = style_dict["dict_xaxis"] | {"text": "Contribution"}
        dict_yaxis = style_dict["dict_yaxis"] | {"text": None}
        dict_local_plot_colors = style_dict["dict_local_plot_colors"] | {"text": None}

        layout = go.Layout(
            barmode="group",
            template="none",
            width=width,
            height=height,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            yaxis_type="category",
            hovermode="closest",
            margin={"l": 150, "r": 20, "t": topmargin, "b": 70},
        )
        bars = []
        for num, expl in enumerate(zip(var_dict, x_val, contrib)):
            feat_name, x_val_el, contrib_value = expl
            is_grouped = False
            if x_val_el == "":
                ylabel = f"<i>{feat_name}</i>"
                hoverlabel = f"<b>{feat_name}</b>"
            else:
                # If bar is a group of features, hovertext includes the values of the features of the group
                # And color changes
                group_name = inv_features_dict.get(feat_name)
                if features_groups is not None and group_name in features_groups.keys() and len(index_value) > 0:
                    is_grouped = True
                    feat_groups_values = x_init[features_groups[group_name]].loc[index_value[0]]
                    hoverlabel = "<br />".join(
                        [
                            f"<b>{add_line_break(features_dict.get(f_name, f_name), 40, maxlen=120)} :</b>{add_line_break(f_value, 40, maxlen=160)}"
                            for f_name, f_value in feat_groups_values.to_dict().items()
                        ]
                    )
                else:
                    hoverlabel = f"<b>{add_line_break(feat_name, 40, maxlen=120)} :</b><br />{add_line_break(x_val_el, 40, maxlen=160)}"
                trunc_value = truncate_str(feat_name, 45)
                if not zoom:
                    # Truncate value if length is upper than 30
                    trunc_new_value = (
                        trunc_value.replace(trunc_value[24 : len(trunc_value) - 3], "...")
                        if len(trunc_value) > 30
                        else trunc_value
                    )
                else:
                    trunc_new_value = trunc_value
                if len(contrib) <= yaxis_max_label and (
                    features_groups is None
                    # We don't want to display label values for t-sne projected values of groups of features.
                    or (features_groups is not None and group_name not in features_groups.keys())
                ):
                    # ylabel is based on trunc_new_value
                    ylabel = f"<b>{trunc_new_value} :</b><br />{truncate_str(x_val_el, 45)}"
                else:
                    ylabel = f"<b>{trunc_new_value}</b>"
            # colors
            if contrib_value >= 0:
                color = 1 if x_val_el != "" else 0
            else:
                color = -1 if x_val_el != "" else -2

            # If the bar is a group of features we modify the color
            if is_grouped:
                bar_color = style_dict["featureimp_groups"][0] if color == 1 else style_dict["featureimp_groups"][1]
            else:
                bar_color = dict_local_plot_colors[color]["color"]

            barobj = go.Bar(
                x=[contrib_value],
                y=[ylabel],
                customdata=[hoverlabel],
                orientation="h",
                marker=dict_local_plot_colors[color],
                marker_color=bar_color,
                showlegend=False,
                hovertemplate="%{customdata}<br />Contribution: %{x:.4f}<extra></extra>",
            )

            bars.append([color, contrib_value, num, barobj])

        bars.sort()
        fig = go.Figure(data=[x[-1] for x in bars], layout=layout)
        fig.update_yaxes(dtick=1)
        fig.update_yaxes(automargin=True)

        if file_name:
            plot(fig, filename=file_name, auto_open=auto_open)
    else:
        fig = go.Figure()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "Select a valid single sample to display<br />Local Explanation plot.",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )
    return fig
