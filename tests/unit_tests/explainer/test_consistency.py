import itertools
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from shapash.explainer.consistency import Consistency


class TestConsistency(unittest.TestCase):
    """
    Unit test consistency
    """

    def setUp(self):

        self.df = pd.DataFrame(
            data=np.array([[1, 2, 3, 0], [2, 4, 6, 1], [2, 2, 1, 0], [2, 4, 1, 1]]), columns=["X1", "X2", "X3", "y"]
        )
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]

        self.w1 = pd.DataFrame(
            np.array([[0.14, 0.04, 0.17], [0.02, 0.01, 0.33], [0.56, 0.12, 0.29], [0.03, 0.01, 0.04]]),
            columns=["X1", "X2", "X3"],
        )
        self.w2 = pd.DataFrame(
            np.array([[0.38, 0.35, 0.01], [0.01, 0.30, 0.05], [0.45, 0.41, 0.12], [0.07, 0.30, 0.21]]),
            columns=["X1", "X2", "X3"],
        )
        self.w3 = pd.DataFrame(
            np.array([[0.49, 0.17, 0.02], [0.25, 0.12, 0.25], [0.01, 0.06, 0.06], [0.19, 0.02, 0.18]]),
            columns=["X1", "X2", "X3"],
        )
        self.contributions = {"contrib_1": self.w1, "contrib_2": self.w2, "contrib_3": self.w3}

        self.cns = Consistency()
        self.cns.compile(contributions=self.contributions, x=self.X)

    def test_compile(self):
        assert isinstance(self.cns.methods, list)
        assert len(self.cns.methods) == len(self.contributions)
        assert isinstance(self.cns.weights, list)
        assert self.cns.weights[0].shape == self.w1.shape
        assert all(x.shape == self.cns.weights[0].shape for x in self.cns.weights)

    def test_check_consistency_contributions(self):
        weights = [self.w1, self.w2, self.w3]

        if weights[0].ndim == 1:
            raise ValueError("Multiple datapoints are required to compute the metric")
        if not all(isinstance(x, pd.DataFrame) for x in weights):
            raise ValueError("Contributions must be pandas DataFrames")
        if not all(x.shape == weights[0].shape for x in weights):
            raise ValueError("Contributions must be of same shape")
        if not all(x.columns.tolist() == weights[0].columns.tolist() for x in weights):
            raise ValueError("Columns names are different between contributions")
        if not all(x.index.tolist() == weights[0].index.tolist() for x in weights):
            raise ValueError("Index names are different between contributions")

    def test_calculate_all_distances(self):
        all_comparisons, mean_distances = self.cns.calculate_all_distances(self.cns.methods, self.cns.weights)

        num_comb = len(list(itertools.combinations(self.cns.methods, 2)))

        assert all_comparisons.shape == (num_comb * self.X.shape[0], 4)
        assert isinstance(mean_distances, pd.DataFrame)
        assert mean_distances.shape == (len(self.cns.methods), len(self.cns.methods))

    def test_calculate_pairwise_distances(self):
        l2_dist = self.cns.calculate_pairwise_distances(self.cns.weights, 0, 1)

        assert l2_dist.shape == (self.X.shape[0],)

    def test_calculate_mean_distances(self):
        _, mean_distances = self.cns.calculate_all_distances(self.cns.methods, self.cns.weights)
        l2_dist = self.cns.calculate_pairwise_distances(self.cns.weights, 0, 1)

        self.cns.calculate_mean_distances(self.cns.methods, mean_distances, 0, 1, l2_dist)

        assert mean_distances.shape == (len(self.cns.methods), len(self.cns.methods))
        assert (
            mean_distances.loc[self.cns.methods[0], self.cns.methods[1]]
            == mean_distances.loc[self.cns.methods[1], self.cns.methods[0]]
        )

    def test_find_examples(self):
        weights = [weight.values for weight in self.cns.weights]
        all_comparisons, mean_distances = self.cns.calculate_all_distances(self.cns.methods, weights)
        method_1, method_2, l2, _, _, _ = self.cns.find_examples(mean_distances, all_comparisons, weights)

        assert isinstance(method_1, list)
        assert isinstance(method_2, list)
        assert isinstance(l2, list)
        assert len(method_1) == len(method_2) == len(l2)
        assert 1 <= len(l2) <= 5

    def test_calculate_coords(self):
        _, mean_distances = self.cns.calculate_all_distances(self.cns.methods, self.cns.weights)
        coords = self.cns.calculate_coords(mean_distances)

        assert coords.shape == (len(self.cns.methods), 2)

    def test_pairwise_consistency_plot(self):
        methods = ["contrib_1", "contrib_3"]
        max_features = 2
        max_points = 100
        output = self.cns.pairwise_consistency_plot(methods=methods, max_features=max_features, max_points=max_points)

        assert len(output.data[0].x) == min(max_points, len(self.X))
        assert len(output.data) == 2 * max_features + 1
