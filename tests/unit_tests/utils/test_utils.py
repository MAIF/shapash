import unittest

import numpy as np
import pandas as pd

from shapash.utils.utils import (
    add_line_break,
    compute_digit_number,
    compute_sorted_variables_interactions_list_indices,
    compute_top_correlations_features,
    inclusion,
    is_nested_list,
    maximum_difference_sort_value,
    truncate_str,
    within_dict,
)


class TestUtils(unittest.TestCase):
    def test_inclusion_1(self):
        x1 = [1, 2, 3]
        x2 = [1, 2, 3, 4]
        assert inclusion(x1, x2)

    def test_inclusion_2(self):
        x1 = [1, 2, 3]
        x2 = [1, 2, 3, 4]
        assert not inclusion(x2, x1)

    def test_inclusion_3(self):
        x1 = 2
        x2 = [1, 2, 3, 4]
        with self.assertRaises(TypeError):
            inclusion(x1, x2)

    def test_within_dict_1(self):
        x = [1, 2, 3]
        d = {1: 10, 2: 12, 3: 13, 4: 14}
        assert within_dict(x, d)

    def test_within_dict_2(self):
        x = [1, 2, 3, 5]
        d = {1: 10, 2: 12, 3: 13, 4: 14}
        assert not within_dict(x, d)

    def test_within_dict_3(self):
        x = 3
        d = {1: 10, 2: 12, 3: 13, 4: 14}
        with self.assertRaises(TypeError):
            within_dict(x, d)

    def test_is_nested_list_1(self):
        x = [1, 2, 3]
        assert not is_nested_list(x)

    def test_is_nested_list_2(self):
        x = [[1, 2, 3], [4, 5, 6]]
        assert is_nested_list(x)

    def test_compute_digit_number_1(self):
        t = compute_digit_number(12)
        assert t == 2

    def test_compute_digit_number_2(self):
        t = compute_digit_number(122344)
        assert t == 0

    def test_compute_digit_number_3(self):
        t = compute_digit_number(0.000044)
        assert t == 8

    def test_truncate_str_1(self):
        t = truncate_str(12)
        assert t == 12

    def test_truncate_str_2(self):
        t = truncate_str("this is a test", 50)
        assert t == "this is a test"

    def test_truncate_str_3(self):
        t = truncate_str("this is a test", 10)
        assert t == "this is a..."

    def test_add_line_break_1(self):
        t = add_line_break(3453, 10)
        assert t == 3453

    def test_add_line_break_2(self):
        t = add_line_break("this is a very long sentence in order to make a very great test", 10)
        expected = "this is a very<br />long sentence<br />in order to make<br />a very great<br />test"
        assert t == expected

    def test_add_line_break_3(self):
        t = add_line_break("this is a very long sentence in order to make a very great test", 15, maxlen=30)
        expected = "this is a very long<br />sentence in order<br />to..."
        assert t == expected

    def test_maximum_difference_sort_value_1(self):
        t = maximum_difference_sort_value([[1, 20, 3], ["feat1", "feat2", "feat3"]])
        assert t == 19

    def test_maximum_difference_sort_value_2(self):
        t = maximum_difference_sort_value([[100, 11, 281, 64, 6000], ["feat1", "feat2"]])
        assert t == 5989

    def test_maximum_difference_sort_value_3(self):
        t = maximum_difference_sort_value([[1], ["feat1"]])
        assert t == 1

    def test_compute_sorted_variables_interactions_list_indices_1(self):
        interaction_values = np.array(
            [
                [[0.1, -0.7, 0.01, -0.9], [-0.1, 0.8, 0.02, 0.7], [0.2, 0.5, 0.04, -0.88], [0.15, 0.6, -0.2, 0.5]],
                [[0.2, -0.1, 0.2, 0.8], [-0.2, 0.6, 0.02, -0.67], [0.1, -0.5, 0.05, 1], [0.3, 0.6, 0.02, -0.9]],
            ]
        )

        expected_output = [
            [3, 1],
            [2, 1],
            [3, 0],
            [2, 0],
            [1, 0],
            [3, 2],
            [3, 3],
            [2, 3],
            [2, 2],
            [1, 3],
            [1, 2],
            [1, 1],
            [0, 3],
            [0, 2],
            [0, 1],
            [0, 0],
        ]

        output = compute_sorted_variables_interactions_list_indices(interaction_values)

        assert np.array_equal(expected_output, output)

    def test_compute_top_correlations_features_1(self):
        """
        Test function with small number of features
        """
        df = pd.DataFrame(np.random.rand(10, 2))

        corr = df.corr()

        list_features = compute_top_correlations_features(corr=corr, max_features=20)
        assert len(list_features) == 2

    def test_compute_top_correlations_features_2(self):
        """
        Test function with high number of features
        """
        df = pd.DataFrame(np.random.rand(10, 30))

        corr = df.corr()

        list_features = compute_top_correlations_features(corr=corr, max_features=5)

        assert len(list_features) == 5
