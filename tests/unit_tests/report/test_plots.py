import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from shapash.plots.plot_univariate import (
    plot_distribution,
    plot_categorical_distribution,
    plot_continuous_distribution,
)

from plotly import graph_objects as go

class TestPlots(unittest.TestCase):
    @patch("shapash.plots.plot_univariate.plot_continuous_distribution")
    @patch("shapash.plots.plot_univariate.plot_categorical_distribution")
    def test_plot_distribution_1(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame(
            {
                "string_data": ["a", "b", "c", "d", "e", np.nan],
                "data_train_test": ["train", "train", "train", "train", "test", "test"],
            }
        )

        plot_distribution(df, "string_data", "data_train_test")
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    @patch("shapash.plots.plot_univariate.plot_continuous_distribution")
    @patch("shapash.plots.plot_univariate.plot_categorical_distribution")
    def test_plot_distribution_2(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame(
            {"int_data": list(range(50)), "data_train_test": ["train", "train", "train", "train", "test"] * 10}
        )

        plot_distribution(df, "int_data", "data_train_test")
        mock_plot_cont.assert_called_once()
        self.assertEqual(mock_plot_cat.call_count, 0)

    @patch("shapash.plots.plot_univariate.plot_continuous_distribution")
    @patch("shapash.plots.plot_univariate.plot_categorical_distribution")
    def test_plot_distribution_3(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame(
            {
                "int_cat_data": [10, 10, 20, 20, 20, 10],
                "data_train_test": ["train", "train", "train", "train", "test", "test"],
            }
        )

        plot_distribution(df, "int_cat_data", "data_train_test")
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    def test_plot_continuous_distribution_1(self):
        df = pd.DataFrame(
            {
                "int_data": [10, 20, 30, 40, 50],
            }
        )
        fig = plot_continuous_distribution(df, "int_data")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"

    def test_plot_continuous_distribution_2(self):
        df = pd.DataFrame(
            {
                "int_data": [10, 20, 30, 40, 50, 30, 20, 0, 10, 20],
                "data_train_test": ["train", "train", "train", "train", "train", "test", "test", "test", "test", "test"],
            }
        )
        fig = plot_continuous_distribution(df, "int_data", "data_train_test")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        assert fig.data[0].type == "scatter"
        assert fig.data[1].type == "scatter"

    def test_plot_categorical_distribution_1(self):
        df = pd.DataFrame(
            {"int_data": [0, 0, 0, 1, 1, 0], "data_train_test": ["train", "train", "train", "train", "test", "test"]}
        )

        fig = plot_categorical_distribution(df, "int_data", "data_train_test")

        assert len(fig.data) == 2
        assert fig.data[0].type == "bar"
        assert fig.data[1].type == "bar"
        assert len(fig.data[0]['x']) == 2

    def test_plot_categorical_distribution_2(self):
        df = pd.DataFrame(
            {"int_data": [0, 0, 0, 1, 1, 0], "data_train_test": ["train", "train", "train", "train", "train", "train"]}
        )

        fig = plot_categorical_distribution(df, "int_data", "data_train_test")

        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert len(fig.data[0]['x']) == 2

    def test_plot_categorical_distribution_3(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame(
            {
                "int_data": [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    9,
                    10,
                    11,
                ],
                "data_train_test": ["train"] * 30,
            }
        )

        fig = plot_categorical_distribution(df, "int_data", "data_train_test", nb_cat_max=7)

        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert len(fig.data[0]['x']) == 8

    def test_plot_categorical_distribution_4(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame(
            {
                "int_data": [
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    8,
                    8,
                    8,
                    9,
                    10,
                    11,
                ],
                "data_train_test": ["train"] * 20 + ["test"] * 10,
            }
        )

        fig = plot_categorical_distribution(df, "int_data", "data_train_test", nb_cat_max=7)

        # Number of bars (multiplied by two as we have train + test for each cat)
        assert len(fig.data) == 2
        assert fig.data[0].type == "bar"
        assert fig.data[1].type == "bar"
        assert len(fig.data[0]['x']) == 8

    def test_plot_categorical_distribution_5(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame({"int_data": [k for k in range(10) for _ in range(k)], "data_train_test": ["train"] * 45})

        fig = plot_categorical_distribution(df, "int_data", "data_train_test", nb_cat_max=7)

        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"
        assert len(fig.data[0]['x']) == 8

    def test_plot_categorical_distribution_6(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame(
            {
                "int_data": [k for k in range(10) for _ in range(k)],
                "data_train_test": ["train"] * 10 + ["test"] * 10 + ["train"] * 25,
            }
        )

        fig = plot_categorical_distribution(df, "int_data", "data_train_test", nb_cat_max=7)

        # Number of bars (multiplied by two as we have train + test for each cat)
        assert len(fig.data) == 2
        assert fig.data[0].type == "bar"
        assert fig.data[1].type == "bar"
        assert len(fig.data[0]['x']) == 8
