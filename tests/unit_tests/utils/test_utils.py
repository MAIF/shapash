import unittest
from shapash.utils.utils import inclusion, within_dict, is_nested_list,\
    compute_digit_number, truncate_str, add_line_break


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
        assert t == 7

    def test_truncate_str_1(self):
        t = truncate_str(12)
        assert t == 12

    def test_truncate_str_2(self):
        t = truncate_str("this is a test",50)
        assert t == "this is a test"

    def test_truncate_str_3(self):
        t = truncate_str("this is a test",10)
        assert t == "this is a..."

    def test_add_line_break_1(self):
        t = add_line_break(3453,10)
        assert t == 3453

    def test_add_line_break_2(self):
        t = add_line_break("this is a very long sentence in order to make a very great test",10)
        expected = 'this is a very<br />long sentence<br />in order to make<br />a very great<br />test'
        assert t == expected

    def test_add_line_break_3(self):
        t = add_line_break("this is a very long sentence in order to make a very great test",15,maxlen=30)
        expected = 'this is a very long<br />sentence in order<br />to...'
        assert t == expected