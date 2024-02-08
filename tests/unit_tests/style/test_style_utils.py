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
