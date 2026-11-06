"""
Unit test of style_utils
"""
import re
import unittest
import warnings

import pytest

from shapash.style.style_utils import (
    colors_loading,
    convert_str_color_to_plt_format,
    define_style,
    get_pyplot_color,
    merge_palette,
    normalize_colors,
    parse_rgb,
    resolve_colors,
    resolve_nlp_theme,
    select_palette,
    to_rgb_string,
)


class TestStyle_utils(unittest.TestCase):
    """
    Class of Unit test for style_utils
    """

    def test_convert_str_color_to_plt_format(self):
        res = convert_str_color_to_plt_format(txt="rgba(244, 192, 0, 1)")
        assert tuple([round(x, 2) for x in res]) == (0.96, 0.75, 0.0, 1.0)

    def rgb_string_detector(self, string_val):
        """
        check rgb() or rgba() format of a str variable
        """
        matching = re.match(r"^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+(?:\.\d+)?))?\)$", string_val)
        matching = False if matching is None else True
        return matching

    def test_colors_loading(self):
        """
        test of colors_loading
        """
        all_colors = colors_loading()
        for palette in all_colors.keys():
            for cle in all_colors[palette].keys():
                entries = all_colors[palette][cle]
                if isinstance(entries, dict):
                    check_list = list(entries.values())
                elif isinstance(entries, str):
                    check_list = [entries]
                else:
                    check_list = entries
                for colors in check_list:
                    if self.rgb_string_detector(colors) is False:
                        print(colors)
                    assert self.rgb_string_detector(colors) is True

    def test_select_palette(self):
        """
        test of select_palette
        """
        available_palettes = colors_loading()
        default_dict = select_palette(available_palettes, "default")
        assert len(list(default_dict.keys())) > 0
        with self.assertRaises(ValueError):
            select_palette(available_palettes, "ERROR_palette")

    def test_define_style(self):
        """
        test of define_style : check that each entry in json file is ok to define style_dict
        """
        available_palettes = colors_loading()
        for palette_name in available_palettes.keys():
            palette = select_palette(available_palettes, palette_name)
            style_dict = define_style(palette)
            assert len(list(style_dict.keys())) > 0

    def test_define_style_fonts_have_generic_fallback(self):
        """
        Every font family set by define_style must end with a generic family, otherwise the browser
        falls back to its own default (serif) when the named font is not installed (e.g. "Arial Black").
        """
        available_palettes = colors_loading()
        for palette_name in available_palettes.keys():
            style_dict = define_style(select_palette(available_palettes, palette_name))
            for key in ["dict_title", "dict_title_stability", "dict_title_compacity", "dict_xaxis", "dict_yaxis"]:
                family = style_dict[key]["font"]["family"]
                assert family.split(",")[-1].strip() == "sans-serif", f"{key}: {family}"

    def test_get_pyplot_color(self):
        available_palettes = colors_loading()
        for palette_name in available_palettes.keys():
            palette = colors_loading()[palette_name]["report_feature_distribution"]
            colors = get_pyplot_color(colors=palette)
            assert isinstance(colors, dict)
            col1 = colors_loading()[palette_name]["title_color"]
            color = get_pyplot_color(col1)
            assert isinstance(color, list)

    def test_palette_key_parity(self):
        """Every palette must define the same keys.

        A palette missing a key that another one has surfaces as a bare ``KeyError`` wherever that
        key is first read (e.g. deep in a plot call), not at load time. Cheaper to catch here.
        """
        available_palettes = colors_loading()
        key_sets = {name: set(palette.keys()) for name, palette in available_palettes.items()}
        reference_name, reference_keys = next(iter(key_sets.items()))
        for name, keys in key_sets.items():
            assert keys == reference_keys, (
                f"Palette '{name}' differs from '{reference_name}': "
                f"missing {reference_keys - keys}, extra {keys - reference_keys}"
            )

    def test_resolve_nlp_theme(self):
        """Default theme resolves from the 'default' palette; colors_dict overrides win."""
        theme = resolve_nlp_theme()
        default_palette = select_palette(colors_loading(), "default")
        assert theme.header_bkg == default_palette["webapp_bkg"]
        assert theme.header_accent == default_palette["webapp_title"]
        assert theme.xpl_positive == default_palette["nlp_xpl_positive"]
        assert theme.xpl_negative == default_palette["nlp_xpl_negative"]

        overridden = resolve_nlp_theme(colors_dict={"nlp_xpl_positive": "rgb(1, 2, 3)"})
        assert overridden.xpl_positive == "rgb(1, 2, 3)"
        assert overridden.xpl_negative == default_palette["nlp_xpl_negative"]

    def test_merge_palette_nested_partial_override(self):
        """A partial nested override keeps the other colors and leaves the palette untouched."""
        palette = {"bar": {"1": "a", "2": "b"}, "scale": ["x", "y"], "title": "t"}
        merged = merge_palette(palette, {"bar": {"1": "red"}, "scale": ["z"], "title": "u"})
        assert merged == {"bar": {"1": "red", "2": "b"}, "scale": ["z"], "title": "u"}
        assert palette == {"bar": {"1": "a", "2": "b"}, "scale": ["x", "y"], "title": "t"}

    def test_merge_palette_int_keys_match_string_keys(self):
        """An int key overrides the palette's string key instead of adding a second entry."""
        merged = merge_palette({"bar": {"1": "a", "2": "b"}}, {"bar": {1: "red"}})
        assert merged == {"bar": {"1": "red", "2": "b"}}

    def test_merge_palette_without_overrides_returns_a_copy(self):
        """No override returns an independent copy of the palette."""
        palette = {"bar": {"1": "a"}}
        merged = merge_palette(palette, None)
        merged["bar"]["1"] = "red"
        assert palette == {"bar": {"1": "a"}}

    def test_define_style_accepts_partial_nested_override(self):
        """define_style no longer fails on a palette with a partially overridden nested mapping."""
        palette = merge_palette(select_palette(colors_loading(), "default"), {"featureimp_bar": {"1": "rgb(1, 2, 3)"}})
        style = define_style(palette)
        assert style["dict_featimp_colors"][1]["color"] == "rgb(1, 2, 3)"
        assert style["dict_featimp_colors"][2]["color"] == colors_loading()["default"]["featureimp_bar"]["2"]

    def test_to_rgb_string_accepts_every_supported_format(self):
        """rgb/rgba strings, hex codes and color names all come out in the rgb(...) format."""
        assert to_rgb_string("rgb(1,2,3)") == "rgb(1, 2, 3)"
        assert to_rgb_string("rgba(1, 2, 3, 0.80)") == "rgba(1, 2, 3, 0.8)"
        assert to_rgb_string("rgb(68.2, 1.0, 84.4)") == "rgb(68, 1, 84)"
        assert to_rgb_string("#123456") == "rgb(18, 52, 86)"
        assert to_rgb_string("#12345680") == "rgba(18, 52, 86, 0.501961)"
        assert to_rgb_string("red") == "rgb(255, 0, 0)"

    def test_to_rgb_string_rejects_invalid_colors(self):
        """Unknown names, out-of-range components and non-strings raise a ValueError."""
        for color in ("notacolor", "rgb(256, 0, 0)", "rgba(0, 0, 0, 2)", 12):
            with pytest.raises(ValueError, match="is not a valid color"):
                to_rgb_string(color)

    def test_parse_rgb_ignores_opacity(self):
        """parse_rgb returns the components of any supported color, without its opacity."""
        assert parse_rgb("rgba(244, 192, 0, 0.5)") == (244, 192, 0)
        assert parse_rgb("#123456") == (18, 52, 86)

    def test_normalize_colors_named_continuous_scale(self):
        """A scale name is sampled to as many colors as the palette's scale holds."""
        palette = select_palette(colors_loading(), "default")
        normalized = normalize_colors({"confusion_matrix_colorscale": "viridis_r"}, palette)
        colors = normalized["confusion_matrix_colorscale"]
        assert len(colors) == len(palette["confusion_matrix_colorscale"])
        assert colors[0] == "rgb(253, 231, 37)"
        assert colors[-1] == "rgb(68, 1, 84)"

    def test_normalize_colors_warns_on_non_diverging_contribution_scale(self):
        """A sequential scale for signed contributions warns; a diverging one does not."""
        palette = select_palette(colors_loading(), "default")
        with pytest.warns(UserWarning, match="'viridis' is not a diverging scale"):
            normalize_colors({"contrib_colorscale": "viridis"}, palette)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            normalized = normalize_colors({"contrib_colorscale": "rdbu_r"}, palette)
        assert len(normalized["contrib_colorscale"]) == len(palette["contrib_colorscale"])

    def test_normalize_colors_named_qualitative_palette(self):
        """A qualitative palette name is replaced by its colors, hex codes included."""
        palette = select_palette(colors_loading(), "default")
        normalized = normalize_colors({"interaction_discrete": "plotly", "compare_plot": "Set2"}, palette)
        assert normalized["interaction_discrete"][0] == "rgb(99, 110, 250)"
        assert normalized["compare_plot"][0] == "rgb(102, 194, 165)"

    def test_normalize_colors_rejects_invalid_names(self):
        """An unknown scale or palette name, or a color where a scale is expected, raises a ValueError."""
        palette = select_palette(colors_loading(), "default")
        with pytest.raises(ValueError, match="'contrib_colorscale': 'red' is not a plotly color scale name"):
            normalize_colors({"contrib_colorscale": "red"}, palette)
        with pytest.raises(ValueError, match="'compare_plot': 'nope' is not a plotly qualitative palette name"):
            normalize_colors({"compare_plot": "nope"}, palette)

    def test_normalize_colors_names_the_key_of_an_invalid_color(self):
        """An invalid color nested in a list, a dict or a single value names its key."""
        palette = select_palette(colors_loading(), "default")
        for overrides in ({"title_color": "nope"}, {"featureimp_bar": {"1": "nope"}}, {"compare_plot": ["nope"]}):
            key = next(iter(overrides))
            with pytest.raises(ValueError, match=f"Invalid value for '{key}'"):
                normalize_colors(overrides, palette)
        with pytest.raises(ValueError, match="expected a color, a list or a dict of colors"):
            normalize_colors({"title_color": 3}, palette)

    def test_normalize_colors_converts_nested_and_list_colors(self):
        """Colors inside nested mappings and lists are converted too."""
        palette = select_palette(colors_loading(), "default")
        normalized = normalize_colors({"featureimp_bar": {"1": "#000000"}, "compare_plot": ("red", "blue")}, palette)
        assert normalized == {"featureimp_bar": {"1": "rgb(0, 0, 0)"}, "compare_plot": ["rgb(255, 0, 0)", "rgb(0, 0, 255)"]}

    def test_normalize_colors_warns_on_unknown_key(self):
        """An unknown key warns with the closest valid key and is still converted."""
        palette = select_palette(colors_loading(), "default")
        with pytest.warns(UserWarning, match="Unknown color key 'titel_color'.*Did you mean 'title_color'"):
            normalized = normalize_colors({"titel_color": "red"}, palette)
        assert normalized == {"titel_color": "rgb(255, 0, 0)"}

    def test_resolve_colors(self):
        """resolve_colors returns the whole palette with the normalized overrides merged."""
        resolved = resolve_colors("blues", {"title_color": "#000000"})
        blues = select_palette(colors_loading(), "blues")
        assert resolved["title_color"] == "rgb(0, 0, 0)"
        assert resolved["webapp_title"] == blues["webapp_title"]
        assert resolved.keys() == blues.keys()

    def test_resolve_nlp_theme_named_scale_and_hex(self):
        """The NLP theme samples a scale name instead of splitting it, and converts hex colors."""
        theme = resolve_nlp_theme(colors_dict={"confusion_matrix_colorscale": "viridis", "nlp_xpl_positive": "#123456"})
        assert theme.confusion_scale[0] == "rgb(68, 1, 84)"
        assert all(color.startswith("rgb(") for color in theme.confusion_scale)
        assert theme.xpl_positive == "rgb(18, 52, 86)"
