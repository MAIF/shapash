"""
Unit test Translate
"""
import unittest

from shapash.utils.translate import translate


class TestTranslate(unittest.TestCase):
    """
    Unit test Translate class
    """

    def test_translate(self):
        """
        Unit test translate
        """
        elements = ["X_1", "X_2"]
        mapping = {"X_1": "âge", "X_2": "profession"}
        output = translate(elements, mapping)
        expected = ["âge", "profession"]
        self.assertListEqual(output, expected)
