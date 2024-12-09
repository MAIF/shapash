"""
functions for loading and manipulating colors
"""

import json
import os

import numpy as np

from shapash.utils.utils import convert_string_to_int_keys


def random_color(a: float = 0.6) -> str:
    """
    Generates a random RGBA color string.

    Parameters
    ----------
    a : float, optional, default=0.6
        Alpha (transparency) value for the color. Must be between 0 and 1.

    Returns
    -------
    str
        A string representing a random RGBA color in the format "rgba(r, g, b, a)".
    """
    if not (0 <= a <= 1):
        raise ValueError("Alpha value 'a' must be between 0 and 1.")

    rng = np.random.default_rng()
    r, g, b = rng.integers(0, 256, size=3)  # Generate random RGB values
    return f"rgba({r}, {g}, {b}, {a})"


def colors_loading():
    """
    colors_loading allows shapash to load a json file which contains different
    palettes of colors that can be used in the plot
    Returns
    -------
    dict:
        contains all available pallets
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    jsonfile = os.path.join(current_path, "colors.json")
    with open(jsonfile) as openfile:
        colors_dic = json.load(openfile)
    return colors_dic


def select_palette(colors_dic, palette_name):
    """
    colors_loading allows shapash to load a json file which contains different
    palettes of colors that can be used in the plot
    Parameters
    ----------
    colors_dic : dict
        dictionnary with every palettes
    palette_name : String
        name of the palette
    Returns
    -------
    dict:
        contains colors of one palette
    """
    if palette_name not in colors_dic.keys():
        raise ValueError(f"Palette {palette_name} not found.")
    return colors_dic[palette_name]


def convert_str_color_to_plt_format(txt):
    """
    Converts an rgb string format to a tuple of float (used by matplotlib format)
    Parameters
    ----------
    txt : str
        a string representation of an rgb color (used by plotly)
    Returns
    -------
    A tuple of float used by matplotlib format
    Example
    --------
    >>> convert_str_color_to_plt_format(txt="rgba(244, 192, 0, 1)")
    (0.96, 0.75, 0.0, 1.0)
    """
    txt = txt.replace("rgba", "").replace("rgb", "").replace("(", "").replace(")", "")
    list_txt = txt.split(",")
    if len(list_txt) > 3:
        return [float(list_txt[i]) / 255 for i in range(3)] + [float(list_txt[3])]
    else:
        return [float(x) / 255 for x in list_txt]


def define_style(palette):
    """
    the define_style function is a function that uses a palette
    to define the different styles used in the different outputs
    of Shapash
    Parameters
    ----------
    palette : dict
        contains colors of one palette
    Returns
    -------
    dict :
        contains different style elements
    """
    style_dict = dict()
    style_dict["dict_title"] = {
        "xanchor": "center",
        "yanchor": "middle",
        "x": 0.5,
        "y": 0.9,
        "font": {"size": 24, "family": "Arial", "color": palette["title_color"]},
    }
    style_dict["dict_title_stability"] = {
        "xanchor": "center",
        "x": 0.5,
        "yanchor": "bottom",
        "pad": dict(b=50),
        "font": {"size": 24, "family": "Arial", "color": palette["title_color"]},
    }
    featureimp_bar = convert_string_to_int_keys(palette["featureimp_bar"])
    style_dict["dict_featimp_colors"] = {
        1: {"color": featureimp_bar[1], "line": {"color": palette["featureimp_line"], "width": 0.5}},
        2: {"color": featureimp_bar[2]},
        3: {"color": featureimp_bar[3], "line": {"color": palette["featureimp_line"], "width": 0.5}},
        4: {"color": featureimp_bar[4], "line": {"color": palette["featureimp_line"], "width": 0.5}},
    }
    style_dict["featureimp_groups"] = convert_string_to_int_keys(palette["featureimp_groups"])
    style_dict["feature_contributions_cumulative"] = palette["feature_contributions_cumulative"]
    style_dict["init_contrib_colorscale"] = palette["contrib_colorscale"]
    style_dict["init_confusion_matrix_colorscale"] = palette["confusion_matrix_colorscale"]
    style_dict["report_confusion_matrix"] = palette["report_confusion_matrix"]
    style_dict["report_feature_distribution"] = palette["report_feature_distribution"]
    style_dict["contrib_distribution"] = palette["contrib_distribution"]
    style_dict["violin_area_classif"] = convert_string_to_int_keys(palette["violin_area_classif"])
    style_dict["prediction_plot"] = convert_string_to_int_keys(palette["prediction_plot"])
    style_dict["violin_default"] = palette["violin_default"]
    style_dict["dict_title_compacity"] = {"font": {"size": 14, "family": "Arial", "color": palette["title_color"]}}
    style_dict["dict_xaxis"] = {"font": {"size": 16, "family": "Arial Black", "color": palette["axis_color"]}}
    style_dict["dict_yaxis"] = {"font": {"size": 16, "family": "Arial Black", "color": palette["axis_color"]}}
    localplot_bar = convert_string_to_int_keys(palette["localplot_bar"])
    localplot_line = convert_string_to_int_keys(palette["localplot_line"])
    style_dict["dict_local_plot_colors"] = {
        1: {"color": localplot_bar[1], "line": {"color": localplot_line[1], "width": 0.5}},
        -1: {"color": localplot_bar[-1], "line": {"color": localplot_line[-1], "width": 0.5}},
        0: {"color": localplot_bar[0], "line": {"color": localplot_line[0], "width": 0.5}},
        -2: {"color": localplot_bar[-2], "line": {"color": localplot_line[-2], "width": 0.5}},
    }
    style_dict["dict_compare_colors"] = palette["compare_plot"]
    style_dict["interactions_col_scale"] = palette["interaction_scale"]
    style_dict["interactions_discrete_colors"] = palette["interaction_discrete"]
    style_dict["dict_stability_bar_colors"] = convert_string_to_int_keys(palette["stability_bar"])
    style_dict["dict_compacity_bar_colors"] = convert_string_to_int_keys(palette["compacity_bar"])
    style_dict["webapp_button"] = convert_string_to_int_keys(palette["webapp_button"])
    style_dict["webapp_bkg"] = palette["webapp_bkg"]
    style_dict["webapp_title"] = palette["webapp_title"]

    return style_dict


def get_palette(palette_name):
    """
    Returns a specific palette linked to the input palette_name
    Parameters
    ----------
    palette_name : str
        name of the palette
    Returns
    -------
    dict:
        contains colors of one palette
    """
    if palette_name is None:
        palette_name = list(colors_loading().keys())[0]  # Default palette name
    return select_palette(colors_loading(), palette_name)


def get_pyplot_color(colors):
    """
    Returns the color(s) of the color_name key in the palette in matplotlib format.
    Parameters
    ----------
    colors :  str or dict
        Colors used as a dict or string object
    Returns
    -------
    dict or tuple
        the colors in pyplot format
    """
    if isinstance(colors, str):
        return convert_str_color_to_plt_format(colors)
    elif isinstance(colors, dict):
        dict_color_palette = {k: convert_str_color_to_plt_format(v) for k, v in colors.items()}
        return dict_color_palette
    elif isinstance(colors, list):
        return [convert_str_color_to_plt_format(v) for v in colors]
    else:
        raise ValueError(f"Color type not supported for conversion to pyplot : {type(colors)}")
