"""
functions for loading and manipulating colors
"""
import json
import os

from numpy.core.numeric import _full_like_dispatcher
from shapash.utils.utils import convert_string_to_int_keys

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
    with open(jsonfile, 'r') as openfile:
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
    style_dict['dict_title'] = {
        'xanchor': "center",
        'yanchor': "middle",
        'x': 0.5,
        'y': 0.9,
        'font': {
            'size': 24,
            'family': "Arial",
            'color': palette["title_color"]
        }
    }
    style_dict['dict_title_stability'] = {
        'xanchor': "center",
        'x': 0.5,
        "yanchor": "bottom",
        "pad": dict(b=50),
        'font': {
            'size': 24,
            'family': "Arial",
            'color': palette["title_color"]
        }
    }
    featureimp_bar = convert_string_to_int_keys(palette["featureimp_bar"])
    style_dict['dict_featimp_colors'] = {
        1: { 
            'color': featureimp_bar[1],
            'line': {
                'color': palette["featureimp_line"],
                'width': 0.5
            }
        },
        2: {
            'color': featureimp_bar[2]
        }
    }
    style_dict['featureimp_groups'] = list(palette["featureimp_groups"].values())
    style_dict['init_contrib_colorscale'] = palette["contrib_colorscale"]
    style_dict['violin_area_classif']  = list(palette["violin_area_classif"].values())
    style_dict['violin_default'] = palette["violin_default"]
    style_dict['dict_title_compacity'] = {
        'font': {
            'size': 14,
            'family': "Arial",
            'color': palette["title_color"]
        }
    }
    style_dict['dict_xaxis'] = {
        'font': {
            'size': 16,
            'family': "Arial Black",
            'color': palette["axis_color"]
        }
    }
    style_dict['dict_yaxis'] = {
        'font': {
            'size': 16,
            'family': "Arial Black",
            'color': palette["axis_color"]
        }
    }
    localplot_bar = convert_string_to_int_keys(palette["localplot_bar"])
    localplot_line = convert_string_to_int_keys(palette["localplot_line"])
    style_dict['dict_local_plot_colors'] = {
        1: {
            'color': localplot_bar[1],
            'line': {
                'color': localplot_line[1],
                'width': 0.5
            }
        },
        -1: {
            'color':localplot_bar[-1],
            'line': {
                'color': localplot_line[-1],
                'width': 0.5
            }
        },
        0: {
            'color': localplot_bar[0],
            'line': {
                'color': localplot_line[0],
                'width': 0.5
            }
        },
        -2: {
            'color': localplot_bar[-2],
            'line': {
                'color': localplot_line[-2],
                'width': 0.5
            }
        }
    }
    style_dict['dict_compare_colors'] = palette["compare_plot"]
    style_dict['interactions_col_scale'] = palette["interaction_scale"]
    style_dict['interactions_discrete_colors'] = palette["interaction_discrete"]
    style_dict['dict_stability_bar_colors'] = convert_string_to_int_keys(palette["stability_bar"])
    style_dict['dict_compacity_bar_colors'] = convert_string_to_int_keys(palette["compacity_bar"])
    style_dict['webapp_button'] = convert_string_to_int_keys(palette["webapp_button"])
    style_dict['webapp_bkg'] = palette["webapp_bkg"]
    style_dict['webapp_title'] = palette["webapp_title"]

    return style_dict