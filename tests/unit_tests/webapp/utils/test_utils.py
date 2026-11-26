import unittest

import pandas as pd

from shapash.webapp.utils.utils import (
    IndexLabels,
    apply_filter,
    check_row,
    get_index_level_columns,
    get_index_type,
    round_to_k,
    split_filter_part,
)


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

    def test_index_labels_pass_a_flat_index_through(self):
        index = pd.Index([10, 20])
        labels = IndexLabels(index)
        assert not labels.is_multi
        assert labels.labels(index) is index
        assert labels.to_label(10) == 10
        assert labels.to_index(10) == 10
        assert labels.to_index_list([10, 20]) == [10, 20]

    def test_index_labels_name_each_multiindex_entry(self):
        index = pd.MultiIndex.from_tuples([("FR", 0), ("DE", 1)])
        labels = IndexLabels(index)
        assert labels.labels(index) == ["FR | 0", "DE | 1"]
        assert labels.labels(index[::-1]) == ["DE | 1", "FR | 0"]

    def test_index_labels_read_json_lists_back(self):
        labels = IndexLabels(pd.MultiIndex.from_tuples([("FR", 0), ("DE", 1)]))
        assert labels.to_label(["DE", 1]) == "DE | 1"
        assert labels.to_label(["IT", 2]) is None
        assert labels.to_label("DE | 1") == "DE | 1"
        assert labels.to_index("DE | 1") == ("DE", 1)
        assert labels.to_index(["DE", 1]) == ("DE", 1)
        assert labels.to_index("unknown") is None
        assert labels.to_index("DE") is None
        assert labels.to_index_list(["FR | 0", "DE | 1"]) == [("FR", 0), ("DE", 1)]
        assert labels.to_index_list(None) is None

    def test_index_labels_stay_unique_when_values_contain_the_separator(self):
        index = pd.MultiIndex.from_tuples([("a | b", "c"), ("a", "b | c")])
        labels = IndexLabels(index)
        assert labels.labels(index) == ["a | b | c [0]", "a | b | c [1]"]
        assert labels.to_index("a | b | c [1]") == ("a", "b | c")

    def test_check_row_finds_a_multiindex_label(self):
        index = pd.MultiIndex.from_tuples([("FR", 0), ("DE", 1)])
        data = [{"_index_": label, "a": 1} for label in IndexLabels(index).labels(index)]
        assert check_row(data, "DE | 1") == 1

    def test_index_level_columns_for_a_flat_index(self):
        assert get_index_level_columns(pd.Index([1, 2], name="id"), []) == []

    def test_index_level_columns_prefix_every_level(self):
        index = pd.MultiIndex.from_tuples([("FR", 0)], names=["country", None])
        assert get_index_level_columns(index, ["a", "b"]) == ["index_country", "index_1"]

    def test_index_level_columns_do_not_depend_on_feature_names(self):
        index = pd.MultiIndex.from_tuples([("FR", 0)], names=["a", "b"])
        assert get_index_level_columns(index, ["a", "b"]) == ["index_a", "index_b"]

    def test_index_level_columns_avoid_existing_columns(self):
        index = pd.MultiIndex.from_tuples([("FR", 0, 1)], names=["a", None, 1])
        # The unnamed level 1 and the level named 1 would both be "index_1".
        assert get_index_level_columns(index, ["index_a"]) == ["index_a_0", "index_1", "index_1_2"]

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
