"""
Unit test of style_utils
"""
import re
import unittest

from shapash.style.style_utils import (
    colors_loading,
    convert_str_color_to_plt_format,
    define_style,
    get_pyplot_color,
    resolve_nlp_theme,
    select_palette,
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
