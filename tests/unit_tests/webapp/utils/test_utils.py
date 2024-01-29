import unittest

from shapash.webapp.utils.utils import round_to_k


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
