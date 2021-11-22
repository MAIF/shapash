import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import unittest
from shapash.explainer.consistency import Consistency


class TestConsistency(unittest.TestCase):
    """
    Unit test consistency
    """
    def test_compile(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)
        methods = ["shap", "acv", "lime"]

        cns = Consistency()
        cns.compile(x=X, model=model)

        assert isinstance(cns.methods, list)
        assert len(cns.methods) == len(methods)
        assert isinstance(cns.weights, list)
        assert cns.weights[0].shape == X.shape
        assert all(x.shape == cns.weights[0].shape for x in cns.weights)

    def test_compute_contributions(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)
        methods = ["shap", "acv", "lime"]

        cns = Consistency()
        res = cns.compute_contributions(x=X,
                                        model=model,
                                        methods=methods,
                                        preprocessing=None)

        assert isinstance(res, dict)
        assert len(res) == len(methods)
        assert res["shap"].shape == (len(X), X.shape[1])

    def test_check_consistency_contributions(self):
        w1 = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        w2 = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        w3 = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        weights = [w1, w2, w3]

        if weights[0].ndim == 1:
            raise ValueError('Multiple datapoints are required to compute the metric')
        if not all(isinstance(x, pd.DataFrame) for x in weights):
            raise ValueError('Contributions must be pandas DataFrames')
        if not all(x.shape == weights[0].shape for x in weights):
            raise ValueError('Contributions must be of same shape')
        if not all(x.columns.tolist() == weights[0].columns.tolist() for x in weights):
            raise ValueError('Columns names are different between contributions')
        if not all(x.index.tolist() == weights[0].index.tolist() for x in weights):
            raise ValueError('Index names are different between contributions')

    def test_calculate_all_distances(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)

        cns = Consistency()
        cns.compile(x=X, model=model)

        all_comparisons, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)

        num_comb = len(list(itertools.combinations(cns.methods, 2)))

        assert all_comparisons.shape == (num_comb*X.shape[0], 4)
        assert isinstance(mean_distances, pd.DataFrame)
        assert mean_distances.shape == (len(cns.methods), len(cns.methods))

    def test_calculate_pairwise_distances(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)

        cns = Consistency()
        cns.compile(x=X, model=model)

        l2_dist = cns.calculate_pairwise_distances(cns.weights, 0, 1)

        assert l2_dist.shape == (X.shape[0], )

    def test_calculate_mean_distances(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)

        cns = Consistency()
        cns.compile(x=X, model=model)

        _, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)
        l2_dist = cns.calculate_pairwise_distances(cns.weights, 0, 1)

        cns.calculate_mean_distances(cns.methods, mean_distances, 0, 1, l2_dist)

        assert mean_distances.shape == (len(cns.methods), len(cns.methods))
        assert mean_distances.loc[cns.methods[0], cns.methods[1]] == mean_distances.loc[cns.methods[1], cns.methods[0]]

    def test_find_examples(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)

        cns = Consistency()
        cns.compile(x=X, model=model)

        all_comparisons, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)
        method_1, method_2, l2 = cns.find_examples(mean_distances, all_comparisons, cns.weights)

        assert isinstance(method_1, list)
        assert isinstance(method_2, list)
        assert isinstance(l2, list)
        assert len(method_1) == len(method_2) == len(l2)
        assert 1 <= len(l2) <= 5

    def test_calculate_coords(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)

        cns = Consistency()
        cns.compile(x=X, model=model)

        _, mean_distances = cns.calculate_all_distances(cns.methods, cns.weights)
        coords = cns.calculate_coords(mean_distances)

        assert coords.shape == (len(cns.methods), 2)
