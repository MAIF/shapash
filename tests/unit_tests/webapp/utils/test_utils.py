import unittest

import pandas as pd

from shapash.webapp.utils.utils import apply_filter, check_row, get_index_type, round_to_k, split_filter_part


class TestUtils(unittest.TestCase):
    def test_round_to_k_1(self):
        x = 123456789
        expected_r_x = 123000000
        assert round_to_k(x, 3) == expected_r_x

    def test_round_to_k_2(self):
        x = 123456789.123
        expected_r_x = 123000000
        assert round_to_k(x, 3) == expected_r_x

    def test_round_to_k_3(self):
        x = 123456789
        expected_r_x = 100000000
        assert round_to_k(x, 1) == expected_r_x

    def test_round_to_k_4(self):
        x = 123.456789
        expected_r_x = 123
        assert round_to_k(x, 3) == expected_r_x

    def test_round_to_k_5(self):
        x = 0.123456789
        expected_r_x = 0.123
        assert round_to_k(x, 3) == expected_r_x

    def test_round_to_k_6(self):
        x = 0.0000123456789
        expected_r_x = 0.0000123
        assert round_to_k(x, 3) == expected_r_x

    def test_get_index_type_numeric(self):
        df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        assert get_index_type(df) == "number"

    def test_get_index_type_text(self):
        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        assert get_index_type(df) == "text"

    def test_check_row_found(self):
        data = [{"_index_": 0, "a": 1}, {"_index_": 1, "a": 2}]
        assert check_row(data, 1) == 1

    def test_check_row_not_found(self):
        data = [{"_index_": 0, "a": 1}]
        assert check_row(data, 99) is None

    def test_check_row_none_index(self):
        data = [{"_index_": 0, "a": 1}]
        assert check_row(data, None) is None

    def test_split_filter_part_eq_quoted(self):
        assert split_filter_part('{col} eq "abc"') == ("col", "eq", "abc")

    def test_split_filter_part_numeric(self):
        assert split_filter_part("{col} ge 3") == ("col", "ge", 3.0)

    def test_split_filter_part_contains(self):
        assert split_filter_part("{col} contains foo") == ("col", "contains", "foo")

    def test_split_filter_part_no_match(self):
        assert split_filter_part("no operator here") == [None, None, None]

    def test_apply_filter_eq(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        filtered = apply_filter(df, "{col} eq 2")
        assert list(filtered["col"]) == [2]

    def test_apply_filter_contains(self):
        df = pd.DataFrame({"col": ["foo", "bar", "foobar"]})
        filtered = apply_filter(df, "{col} contains foo")
        assert set(filtered["col"]) == {"foo", "foobar"}

    def test_apply_filter_datestartswith(self):
        df = pd.DataFrame({"col": ["2024-01-01T10:00", "2023-05-05T00:00"]})
        filtered = apply_filter(df, "{col} datestartswith 2024-01-01")
        assert list(filtered["col"]) == ["2024-01-01T10:00"]

    def test_apply_filter_multiple_expressions(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        filtered = apply_filter(df, "{a} ge 2 && {b} le 20")
        assert list(filtered.index) == [1]
