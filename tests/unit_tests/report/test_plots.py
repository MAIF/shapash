import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapash.report.plots import generate_fig_univariate, generate_fig_univariate_continuous, \
    generate_fig_univariate_categorical
from shapash.report.common import VarType


class TestPlots(unittest.TestCase):

    @patch('shapash.report.plots.generate_fig_univariate_continuous')
    @patch('shapash.report.plots.generate_fig_univariate_categorical')
    def test_generate_fig_univariate_1(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame({
            "string_data": ["a", "b", "c", "d", "e", np.nan],
            "data_train_test": ['train', 'train', 'train', 'train', 'test', 'test']
        })

        generate_fig_univariate(df, 'string_data', 'data_train_test', type=VarType.TYPE_CAT)
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    @patch('shapash.report.plots.generate_fig_univariate_continuous')
    @patch('shapash.report.plots.generate_fig_univariate_categorical')
    def test_generate_fig_univariate_2(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame({
            "int_data": list(range(50)),
            "data_train_test": ['train', 'train', 'train', 'train', 'test']*10
        })

        generate_fig_univariate(df, 'int_data', 'data_train_test', type=VarType.TYPE_NUM)
        mock_plot_cont.assert_called_once()
        self.assertEqual(mock_plot_cat.call_count, 0)

    @patch('shapash.report.plots.generate_fig_univariate_continuous')
    @patch('shapash.report.plots.generate_fig_univariate_categorical')
    def test_generate_fig_univariate_3(self, mock_plot_cat, mock_plot_cont):
        df = pd.DataFrame({
            "int_cat_data": [10, 10, 20, 20, 20, 10],
            "data_train_test": ['train', 'train', 'train', 'train', 'test', 'test']
        })

        generate_fig_univariate(df, 'int_cat_data', 'data_train_test', type=VarType.TYPE_CAT)
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    def test_generate_fig_univariate_continuous(self):
        df = pd.DataFrame({
            "int_data": [10, 20, 30, 40, 50, 0],
            "data_train_test": ['train', 'train', 'train', 'train', 'test', 'test']
        })
        fig = generate_fig_univariate_continuous(df, 'int_data', 'data_train_test')
        assert isinstance(fig, plt.Figure)

    def test_generate_fig_univariate_categorical_1(self):
        df = pd.DataFrame({
            "int_data": [0, 0, 0, 1, 1, 0],
            "data_train_test": ['train', 'train', 'train', 'train', 'test', 'test']
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test')

        assert len(fig.axes[0].patches) == 4  # Number of bars

    def test_generate_fig_univariate_categorical_2(self):
        df = pd.DataFrame({
            "int_data": [0, 0, 0, 1, 1, 0],
            "data_train_test": ['train', 'train', 'train', 'train', 'train', 'train']
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test')

        assert len(fig.axes[0].patches) == 2  # Number of bars

    def test_generate_fig_univariate_categorical_3(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame({
            "int_data": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10, 11],
            "data_train_test": ['train'] * 30
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test', nb_cat_max=7)

        assert len(fig.axes[0].patches) == 7  # Number of bars

    def test_generate_fig_univariate_categorical_4(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame({
            "int_data": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 10, 11],
            "data_train_test": ['train'] * 20 + ['test']*10
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test', nb_cat_max=7)

        # Number of bars (multiplied by two as we have train + test for each cat)
        assert len(fig.axes[0].patches) == 7*2

    def test_generate_fig_univariate_categorical_5(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame({
            "int_data": [k for k in range(10) for _ in range(k)],
            "data_train_test": ['train'] * 45
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test', nb_cat_max=7)

        assert len(fig.axes[0].patches) == 7  # Number of bars

    def test_generate_fig_univariate_categorical_6(self):
        """
        Test merging small categories into 'other' category
        """
        df = pd.DataFrame({
            "int_data": [k for k in range(10) for _ in range(k)],
            "data_train_test": ['train'] * 10 + ['test'] * 10 + ['train'] * 25
        })

        fig = generate_fig_univariate_categorical(df, 'int_data', 'data_train_test', nb_cat_max=7)

        print(len(fig.axes[0].patches))

        # Number of bars (multiplied by two as we have train + test for each cat)
        assert len(fig.axes[0].patches) == 7*2
