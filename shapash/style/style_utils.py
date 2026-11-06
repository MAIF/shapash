"""functions for loading and manipulating colors."""

import copy
import difflib
import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.colors as plotly_colors
from matplotlib.colors import to_rgba
from plotly.exceptions import PlotlyError

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


def colors_loading() -> dict[str, Any]:
    """
    Load the JSON file containing the color palettes available to Shapash.

    Returns
    -------
    dict[str, Any]
        All available palettes, keyed by palette name.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    jsonfile = os.path.join(current_path, "colors.json")
    with open(jsonfile) as openfile:
        colors_dic = json.load(openfile)
    return colors_dic


def select_palette(colors_dic: dict[str, Any], palette_name: str) -> dict[str, Any]:
    """
    Selects a palette from the available palettes.

    Parameters
    ----------
    colors_dic : dict[str, Any]
        Dictionary of palettes keyed by name.
    palette_name : str
        Name of the palette.

    Returns
    -------
    dict[str, Any]
        Colors of the selected palette.
    """
    if palette_name not in colors_dic.keys():
        raise ValueError(f"Palette {palette_name} not found.")
    return colors_dic[palette_name]


def merge_palette(palette: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """
    Return a copy of a palette with color overrides merged on top.

    A nested mapping such as ``featureimp_bar`` is merged key by key, so overriding one of its
    colors keeps the others. Any other value, a list of colors included, is replaced whole.
    Inside a nested mapping whose keys are strings, an override key is matched as a string,
    so ``{1: "red"}`` and ``{"1": "red"}`` override the same color.

    Parameters
    ----------
    palette : dict[str, Any]
        Colors of one palette. It is not modified.
    overrides : dict[str, Any] or None
        Colors to set, with the same keys as the palette.

    Returns
    -------
    dict[str, Any]
        A new palette with the overrides applied.
    """
    merged = copy.deepcopy(palette)
    for key, value in (overrides or {}).items():
        base = merged.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            str_keys = all(isinstance(k, str) for k in base)
            base.update({(str(k) if str_keys else k): copy.deepcopy(v) for k, v in value.items()})
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Palette keys holding a continuous color scale: they also accept a plotly scale name.
CONTINUOUS_SCALE_KEYS = frozenset(
    {
        "contrib_colorscale",
        "confusion_matrix_colorscale",
        "interaction_scale",
        "feature_contributions_cumulative",
        "report_confusion_matrix",
    }
)
# Palette keys holding a list of distinct colors: they also accept a plotly qualitative palette name.
DISCRETE_COLOR_KEYS = frozenset({"interaction_discrete", "compare_plot"})
# Scales also drawing signed values (correlations): a non-diverging scale hides the sign, so it draws a warning.
SIGNED_SCALE_KEYS = frozenset({"contrib_colorscale"})

_RGB_PATTERN = re.compile(r"^\s*rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*(?:,\s*([\d.]+)\s*)?\)\s*$")


def _rgba_components(color: str) -> tuple[int, int, int, float | None]:
    """Return the 0-255 components of a color and its opacity, ``None`` when opaque."""
    if not isinstance(color, str):
        raise ValueError(f"{color!r} is not a valid color: expected a string.")
    match = _RGB_PATTERN.match(color)
    if match:
        r, g, b = (round(float(component)) for component in match.groups()[:3])
        alpha = None if match.group(4) is None else float(match.group(4))
        if max(r, g, b) > 255 or (alpha is not None and alpha > 1):
            raise ValueError(f"{color!r} is not a valid color: components must be in 0-255 and alpha in 0-1.")
        return r, g, b, alpha
    try:
        red, green, blue, opacity = to_rgba(color)
    except ValueError as exc:
        raise ValueError(
            f"{color!r} is not a valid color: use rgb(...), rgba(...), a hex code or a color name."
        ) from exc
    r, g, b = (round(component * 255) for component in (red, green, blue))
    return r, g, b, (None if opacity == 1 else opacity)


def to_rgb_string(color: str) -> str:
    """
    Convert a color to the ``rgb(r, g, b)`` / ``rgba(r, g, b, a)`` format used by Shapash.

    Parameters
    ----------
    color : str
        An ``rgb(...)`` or ``rgba(...)`` string, a hex code (``#1f77b4``, ``#1f77b480``) or a
        color name (``red``).

    Returns
    -------
    str
        ``rgb(r, g, b)`` for an opaque color, ``rgba(r, g, b, a)`` otherwise.

    Raises
    ------
    ValueError
        If ``color`` is not a valid color.
    """
    r, g, b, alpha = _rgba_components(color)
    if alpha is None:
        return f"rgb({r}, {g}, {b})"
    return f"rgba({r}, {g}, {b}, {alpha:g})"


def parse_rgb(color: str) -> tuple[int, int, int]:
    """
    Return the red, green and blue components of a color, ignoring its opacity.

    Parameters
    ----------
    color : str
        Any color accepted by :func:`to_rgb_string`.

    Returns
    -------
    tuple[int, int, int]
        Components in 0-255.

    Raises
    ------
    ValueError
        If ``color`` is not a valid color.
    """
    r, g, b, _ = _rgba_components(color)
    return r, g, b


def _named_lists(module: Any) -> dict[str, list[str]]:
    """Map the lowercased names of a plotly color module to their lists of colors."""
    return {name.lower(): value for name, value in vars(module).items() if isinstance(value, list)}


def _sample_named_scale(key: str, name: str, n_colors: int) -> list[str]:
    """Sample the plotly scale *name* to *n_colors* colors for the palette entry *key*."""
    try:
        colors = plotly_colors.sample_colorscale(name, n_colors)
    except PlotlyError as exc:
        raise ValueError(f"Invalid value for {key!r}: {name!r} is not a plotly color scale name.") from exc
    if key in SIGNED_SCALE_KEYS and name.lower().removesuffix("_r") not in _named_lists(plotly_colors.diverging):
        warnings.warn(
            f"{name!r} is not a diverging scale: the correlation plots, which also use {key!r}, will no "
            "longer show whether a correlation is positive or negative. A diverging scale such as 'RdBu' "
            "keeps the sign visible.",
            UserWarning,
            stacklevel=6,
        )
    return [to_rgb_string(color) for color in colors]


def _named_discrete_colors(key: str, name: str) -> list[str]:
    """Return the colors of the plotly qualitative palette *name* for the palette entry *key*."""
    colors = _named_lists(plotly_colors.qualitative).get(name.lower())
    if colors is None:
        raise ValueError(f"Invalid value for {key!r}: {name!r} is not a plotly qualitative palette name.")
    return [to_rgb_string(color) for color in colors]


def _normalize_entry(key: str, value: Any, default: Any) -> Any:
    """Normalize the override *value* of the palette entry *key*, whose palette value is *default*."""
    try:
        if isinstance(value, str):
            if key in CONTINUOUS_SCALE_KEYS and isinstance(default, list):
                return _sample_named_scale(key, value, len(default))
            if key in DISCRETE_COLOR_KEYS:
                return _named_discrete_colors(key, value)
            return to_rgb_string(value)
        if isinstance(value, dict):
            return {sub_key: to_rgb_string(color) for sub_key, color in value.items()}
        if isinstance(value, list | tuple):
            return [to_rgb_string(color) for color in value]
    except ValueError as exc:
        if str(exc).startswith("Invalid value for"):
            raise
        raise ValueError(f"Invalid value for {key!r}: {exc}") from exc
    raise ValueError(f"Invalid value for {key!r}: expected a color, a list or a dict of colors, got {value!r}.")


def normalize_colors(colors_dict: dict[str, Any] | None, palette: dict[str, Any]) -> dict[str, Any]:
    """
    Validate color overrides and convert them to the format the plots expect.

    Every color becomes an ``rgb(...)`` / ``rgba(...)`` string. A plotly scale name given for a
    continuous scale (``contrib_colorscale``, ...) is sampled to as many colors as the palette
    holds, and a plotly qualitative palette name given for a list of distinct colors
    (``interaction_discrete``, ``compare_plot``) is replaced by its colors.

    Parameters
    ----------
    colors_dict : dict[str, Any] or None
        Color overrides, with the same keys as the palette.
    palette : dict[str, Any]
        The palette the overrides apply to.

    Returns
    -------
    dict[str, Any]
        The normalized overrides.

    Raises
    ------
    ValueError
        If a color, scale name or qualitative palette name is not valid.

    Warns
    -----
    UserWarning
        For a key the palette does not have, and for a non-diverging scale name given for
        ``contrib_colorscale``.
    """
    normalized = {}
    for key, value in (colors_dict or {}).items():
        if key not in palette:
            close = difflib.get_close_matches(key, palette, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            warnings.warn(f"Unknown color key {key!r}: no plot uses it.{hint}", UserWarning, stacklevel=4)
        normalized[key] = _normalize_entry(key, value, palette.get(key))
    return normalized


def resolve_colors(palette_name: str, colors_dict: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Return the colors of a palette with validated overrides merged on top.

    Parameters
    ----------
    palette_name : str
        Name of a palette in ``colors.json``.
    colors_dict : dict[str, Any], optional
        Color overrides, see :func:`normalize_colors` for what each key accepts.

    Returns
    -------
    dict[str, Any]
        The resolved colors, every key of the palette included.
    """
    palette = select_palette(colors_loading(), palette_name)
    return merge_palette(palette, normalize_colors(colors_dict, palette))


def convert_str_color_to_plt_format(txt: str) -> list[float]:
    """
    Converts an rgb string format to a tuple of float (used by matplotlib format).

    Parameters
    ----------
    txt : str
        a string representation of an rgb color (used by plotly)

    Returns
    -------
    list[float]
        Color components in matplotlib format (RGB or RGBA).

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


def define_style(palette: dict[str, Any]) -> dict[str, Any]:
    """
    Define the styles used in the different outputs of Shapash from a palette.

    Parameters
    ----------
    palette : dict[str, Any]
        Colors of one palette.

    Returns
    -------
    dict[str, Any]
        Different style elements.
    """
    style_dict: dict[str, Any] = {}
    style_dict["dict_title"] = {
        "xanchor": "center",
        "yanchor": "middle",
        "x": 0.5,
        "y": 0.9,
        "font": {"size": 24, "family": "Arial, sans-serif", "color": palette["title_color"]},
    }
    style_dict["dict_title_stability"] = {
        "xanchor": "center",
        "x": 0.5,
        "yanchor": "bottom",
        "pad": dict(b=50),
        "font": {"size": 24, "family": "Arial, sans-serif", "color": palette["title_color"]},
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
    style_dict["dict_title_compacity"] = {
        "font": {"size": 14, "family": "Arial, sans-serif", "color": palette["title_color"]}
    }
    style_dict["dict_xaxis"] = {
        "font": {"size": 16, "family": "Arial Black, Arial, sans-serif", "color": palette["axis_color"]}
    }
    style_dict["dict_yaxis"] = {
        "font": {"size": 16, "family": "Arial Black, Arial, sans-serif", "color": palette["axis_color"]}
    }
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
    style_dict["lift_curve_colors"] = {
        "curve": style_dict["prediction_plot"][1],
        "random": style_dict["prediction_plot"][0],
        "perfect": style_dict["dict_stability_bar_colors"][0],
    }
    style_dict["webapp_button"] = convert_string_to_int_keys(palette["webapp_button"])
    style_dict["webapp_bkg"] = palette["webapp_bkg"]
    style_dict["webapp_title"] = palette["webapp_title"]
    style_dict["clustering"] = palette["clustering"]

    return style_dict


def get_palette(palette_name: str | None) -> dict[str, Any]:
    """
    Returns a specific palette linked to the input palette_name.

    Parameters
    ----------
    palette_name : str
        Name of the palette, or None to use the first available palette.

    Returns
    -------
    dict[str, Any]
        Colors of the selected palette.
    """
    if palette_name is None:
        palette_name = list(colors_loading().keys())[0]  # Default palette name
    return select_palette(colors_loading(), palette_name)


@dataclass(frozen=True)
class NlpTheme:
    """Resolved colors the NLP webapp and its plots draw from — a palette with overrides merged in.

    ``xpl_positive``/``xpl_negative`` name the concept "explanation pushes the prediction
    up/down" rather than any one backend's algorithm (``nlp_shap`` and ``nlp_captum_lig`` both
    produce signed per-token contributions, drawn with these two colors).
    """

    header_bkg: str
    header_accent: str
    xpl_positive: str
    xpl_negative: str
    confusion_scale: tuple[str, ...]


def resolve_nlp_theme(palette_name: str = "default", colors_dict: dict[str, str] | None = None) -> NlpTheme:
    """Resolve the NLP webapp's theme: *palette_name* from ``colors.json``, *colors_dict* merged on top.

    The single place every NLP color customization funnels through — :class:`~shapash.webapp.nlp_app.NlpWebApp`
    resolves it once from its ``palette_name``/``colors_dict`` constructor arguments; every plot
    function in :mod:`shapash.plots` that draws a signed contribution defaults its own color
    arguments from :data:`DEFAULT_NLP_THEME` below, so a bare call (a doctest, a notebook via
    ``NlpExplanation.plot``) still renders in the same default colors without going through the
    webapp at all.

    Parameters
    ----------
    palette_name : str
        Name of a palette in ``colors.json``.
    colors_dict : dict[str, str], optional
        Per-key overrides merged on top of the selected palette, same key names as ``colors.json``.

    Returns
    -------
    NlpTheme
    """
    palette = resolve_colors(palette_name, colors_dict)
    return NlpTheme(
        header_bkg=palette["webapp_bkg"],
        header_accent=palette["webapp_title"],
        xpl_positive=palette["nlp_xpl_positive"],
        xpl_negative=palette["nlp_xpl_negative"],
        # A tuple, not the palette's list: NlpTheme is a frozen dataclass, so a mutable field
        # would make it unhashable.
        confusion_scale=tuple(palette["confusion_matrix_colorscale"]),
    )


DEFAULT_NLP_THEME = resolve_nlp_theme()


def get_pyplot_color(
    colors: str | dict[Any, str] | list[str],
) -> list[float] | dict[Any, list[float]] | list[list[float]]:
    """
    Returns the color(s) of the color_name key in the palette in matplotlib format.

    Parameters
    ----------
    colors : str or dict or list
        A single RGB(A) string, a mapping of keys to RGB(A) strings, or a list
        of RGB(A) strings.

    Returns
    -------
    list[float] or dict[Any, list[float]] or list[list[float]]
        RGB(A) components as a list, a mapping of keys to component lists, or
        a list of component lists, respectively.
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
