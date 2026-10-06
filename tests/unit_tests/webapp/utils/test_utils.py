import unittest

import numpy as np
import pandas as pd

from shapash.webapp.utils.utils import get_datatable_data_and_tooltips, round_to_k


class TestUtils(unittest.TestCase):
    def test_get_datatable_data_and_tooltips_1(self):
        """
        Null values must be displayed as "missing" in the datatable cells and tooltips
        """
        df = pd.DataFrame({"num": [1.5, np.nan], "txt": ["a", None]})
        data, tooltip_data = get_datatable_data_and_tooltips(df)

        assert data[0] == {"num": 1.5, "txt": "a"}
        assert data[1] == {"num": "missing", "txt": "missing"}
        assert tooltip_data[0] == {
            "num": {"value": "1.5", "type": "text"},
            "txt": {"value": "a", "type": "text"},
        }
        assert tooltip_data[1] == {
            "num": {"value": "missing", "type": "text"},
            "txt": {"value": "missing", "type": "text"},
        }

    def test_get_datatable_data_and_tooltips_2(self):
        """
        Tooltips can be built from a different (unrounded) dataframe than the cells
        """
        data_df = pd.DataFrame({"num": [1.5, np.nan]})
        tooltip_df = pd.DataFrame({"num": [1.54321, np.nan]})
        data, tooltip_data = get_datatable_data_and_tooltips(data_df, tooltip_df)

        assert data[0] == {"num": 1.5}
        assert data[1] == {"num": "missing"}
        assert tooltip_data[0] == {"num": {"value": "1.54321", "type": "text"}}
        assert tooltip_data[1] == {"num": {"value": "missing", "type": "text"}}

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
