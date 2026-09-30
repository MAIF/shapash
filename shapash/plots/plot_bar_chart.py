from collections import Counter
from typing import Any, Literal

from plotly import graph_objs as go
from plotly.offline import plot

from shapash.utils.utils import add_line_break, adjust_title_height, compute_digit_number, truncate_str


def plot_bar_chart(
    index_value: list[Any],
    var_dict: list[str],
    x_val: list[Any],
    contrib: list[float],
    style_dict: dict[str, Any],
    features_groups: dict[str, list[str]] | None = None,
    x_init: Any = None,
    features_dict: dict[str, str] | None = None,
    inv_features_dict: dict[str, str] | None = None,
    yaxis_max_label: int = 12,
    subtitle: str | None = None,
    plot_type: Literal["bar", "waterfall"] = "bar",
    base_value: float | None = None,
    width: int = 900,
    height: int = 550,
    file_name: str | None = None,
    auto_open: bool = False,
    zoom: bool = False,
    waterfall_baseline_position: Literal["top", "bottom"] = "bottom",
    waterfall_xaxis_start: float | int | Literal["auto"] | None = None,
    waterfall_tooltips: list[str] | None = None,
) -> go.Figure:
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
    plot_type: string (default: "bar")
        Type of plot to render:
        - "bar": standard local contribution bars
        - "waterfall": cumulative explanation from a baseline value
    base_value: float (default: None)
        Baseline value used as the start of the waterfall when plot_type="waterfall".
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
    waterfall_baseline_position: {"top", "bottom"} (default="bottom")
        Position of the baseline in waterfall mode.
        - "top": baseline is displayed at the top and prediction at the bottom.
        - "bottom": baseline is displayed at the bottom and prediction at the top.
    waterfall_xaxis_start: float, int, "auto" or None (default: None)
        Start value of x-axis in waterfall mode.
        - None: keep default automatic axis behavior
        - "auto": compute an intelligent start based on baseline/prediction scale
        - numeric value: force a manual x-axis start
    waterfall_tooltips: list (default: None)
        Optional per-bar tooltip suffixes for waterfall mode.
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

        def _suffix_duplicate_labels(labels: list[str]) -> list[str]:
            counts = Counter(labels)
            current_count: Counter[str] = Counter()
            output = []
            for label in labels:
                if counts[label] > 1:
                    current_count[label] += 1
                    output.append(f"{label}_{current_count[label]}")
                else:
                    output.append(label)
            return output

        topmargin = 80
        title = f"Local Explanation - Id: <b>{index_value[0]}</b>"
        # Add subtitle
        if subtitle:
            max_chars_single_line = max(40, int((width - 120) / 7))
            if len(subtitle) > max_chars_single_line:
                ratio = max_chars_single_line / max(len(subtitle), 1)
                subtitle_font_size = max(8, min(16, int(16 * ratio)))
                title += (
                    "<br><sup><span style='font-size: " + str(subtitle_font_size) + "px;'>" + subtitle + "</span></sup>"
                )
                topmargin += 15
            else:
                title += "<br><sup>" + subtitle + "</sup>"
                topmargin += 15
        dict_t = style_dict["dict_title"] | {"text": title, "y": adjust_title_height(height)}
        xaxis_title = "Model output" if plot_type == "waterfall" else "Contribution"
        dict_xaxis = style_dict["dict_xaxis"] | {"text": xaxis_title}
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

        display_feature_names = []
        for feat_name, x_val_el in zip(var_dict, x_val, strict=False):
            if x_val_el == "":
                display_feature_names.append(feat_name)
            else:
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
                display_feature_names.append(trunc_new_value)

        display_feature_names = _suffix_duplicate_labels(display_feature_names)

        bars = []
        for num, expl in enumerate(zip(var_dict, x_val, contrib, display_feature_names, strict=False)):
            feat_name, x_val_el, contrib_value, display_feat_name = expl
            is_grouped = False
            if x_val_el == "":
                ylabel = f"<i>{display_feat_name}</i>"
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
                if len(contrib) <= yaxis_max_label and (
                    features_groups is None
                    # We don't want to display label values for t-sne projected values of groups of features.
                    or (features_groups is not None and group_name not in features_groups.keys())
                ):
                    # ylabel is based on shortened and uniquified feature name
                    ylabel = f"<b>{display_feat_name} :</b><br />{truncate_str(x_val_el, 45)}"
                else:
                    ylabel = f"<b>{display_feat_name}</b>"
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

            bars.append([color, contrib_value, num, barobj, ylabel, hoverlabel, bar_color])

        if plot_type == "bar":
            bars.sort()

        if plot_type == "waterfall":
            if base_value is None:
                base_value = 0.0

            final_prediction = base_value + sum(x[1] for x in bars)

            def _format_value(value: float) -> str:
                digit = compute_digit_number(value)
                formatted = f"{value:.{digit}f}"
                if "." in formatted:
                    formatted = formatted.rstrip("0").rstrip(".")
                return formatted

            y_values = ["<i>Baseline</i>"] + [x[4] for x in bars] + ["<i>Prediction</i>"]
            bar_lengths = [base_value]
            bar_bases = [0.0]
            baseline_bar_color = style_dict.get("prediction_plot", {}).get(1, dict_local_plot_colors[1]["color"])
            prediction_bar_color = baseline_bar_color

            bar_colors = [baseline_bar_color]
            custom_data = [f"<b>Baseline</b>: {_format_value(base_value)}"]
            value_text = [_format_value(base_value)]

            cumulative_value = base_value
            for elem in bars:
                bar_lengths.append(elem[1])
                bar_bases.append(cumulative_value)
                bar_colors.append(elem[6])

                cumulative_value += elem[1]
                custom_data.append(f"{elem[5]}<br />Cumulative output: <b>{_format_value(cumulative_value)}</b>")

                value_text.append(_format_value(elem[1]))

            bar_lengths.append(final_prediction)
            bar_bases.append(0.0)
            bar_colors.append(prediction_bar_color)
            custom_data.append(f"<b>Model output</b>: {_format_value(final_prediction)}")
            value_text.append(_format_value(final_prediction))

            if waterfall_tooltips is not None:
                custom_data = [
                    custom + (f"<br />{extra}" if extra else "")
                    for custom, extra in zip(
                        custom_data,
                        waterfall_tooltips,
                        strict=True,
                    )
                ]

            wf = go.Bar(
                orientation="h",
                y=y_values,
                x=bar_lengths,
                base=bar_bases,
                customdata=custom_data,
                text=value_text,
                textposition="none",
                marker_color=bar_colors,
                showlegend=False,
                hovertemplate="%{customdata}<br />Contribution: %{text}<extra></extra>",
            )
            fig = go.Figure(data=[wf], layout=layout)

            if waterfall_baseline_position == "top":
                fig.update_yaxes(autorange="reversed")
            elif waterfall_baseline_position == "bottom":
                fig.update_yaxes(autorange=True)
            else:
                raise ValueError("waterfall_baseline_position must be 'top' or 'bottom'.")

            if waterfall_xaxis_start is not None:
                path_values = [base_value]
                running_value = base_value
                for elem in bars:
                    running_value += elem[1]
                    path_values.append(running_value)

                min_path_value = min(path_values)
                max_path_value = max(path_values)
                path_span = max_path_value - min_path_value
                pad = max(path_span * 0.05, 1e-12)
                xaxis_end = max_path_value + pad

                if waterfall_xaxis_start == "auto":
                    if min_path_value >= 0:
                        xaxis_start = max(0.0, min_path_value - (path_span * 0.15 + pad))
                    elif max_path_value <= 0:
                        xaxis_start = min_path_value - pad
                    else:
                        xaxis_start = min(0.0, min_path_value - pad)
                elif isinstance(waterfall_xaxis_start, (int, float)):
                    xaxis_start = float(waterfall_xaxis_start)
                else:
                    raise ValueError("waterfall_xaxis_start must be None, 'auto', or a numeric value.")

                if xaxis_start >= xaxis_end:
                    xaxis_end = xaxis_start + max(abs(xaxis_start) * 0.05, 1.0)

                fig.update_xaxes(range=[xaxis_start, xaxis_end])
        else:
            fig = go.Figure(data=[x[3] for x in bars], layout=layout)

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
