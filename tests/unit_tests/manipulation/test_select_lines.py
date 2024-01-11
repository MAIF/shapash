"""
Unit test for select lines
"""
import unittest

import pandas as pd

from shapash.manipulation.select_lines import select_lines

DF = pd.DataFrame(
    [["A", "A", -16.4, 12], ["C", "A", 3.4, 0], ["C", "B", 8.4, 9], ["C", "B", -9, -5]],
    columns=["col1", "col2", "col3", "col4"],
)


class TestSelectLines(unittest.TestCase):
    """
    test select lines unit test
    TODO: Docstring
    """

    def test_compute_select_no_lines(self):
        """
        test of compute select no lines
        """
        output = select_lines(DF, "col3 < -3000")
        assert len(output) == 0

    def test_compute_select_simple_index_lines(self):
        """
        test of compute select simple index lines
        """
        output = select_lines(DF, "col3 < col4")
        expected = [0, 2, 3]
        assert output == expected

    def test_compute_select_multiple_index_lines(self):
        """
        test of compute select multiple index lines
        """
        dataframe = DF.set_index(["col1", "col2"])
        output = select_lines(dataframe, "col3 < 0")
        expected = [("A", "A"), ("C", "B")]
        assert output == expected
