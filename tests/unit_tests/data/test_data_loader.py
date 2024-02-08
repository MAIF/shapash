"""
Unit test data loader
"""
import unittest

import pandas as pd

from shapash.data.data_loader import data_loading


class Test_load_data(unittest.TestCase):
    def test_load_house(self):
        """
        Unit test house_prices
        """
        house_df, house_dict = data_loading("house_prices")
        assert isinstance(house_df, pd.DataFrame)
        assert isinstance(house_dict, dict)

    def test_load_titanic(self):
        """
        Unit test load_titanic
        """
        titanic_df, titanic_dict = data_loading("titanic")
        assert isinstance(titanic_df, pd.DataFrame)
        assert isinstance(titanic_dict, dict)

    def test_load_telco(self):
        """
        Unit test telco_customer
        """
        telco_df = data_loading("telco_customer_churn")
        assert isinstance(telco_df, pd.DataFrame)

    def test_load_accidents(self):
        """
        Unit test us_car_accident
        """
        accidents_df = data_loading("us_car_accident")
        assert isinstance(accidents_df, pd.DataFrame)
