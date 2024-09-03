import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from shapash.utils.explanation_metrics import (
    _compute_distance,
    _compute_similarities,
    _df_to_array,
    _get_radius,
    find_neighbors,
    get_distance,
    get_min_nb_features,
    shap_neighbors,
)


class TestExplanationMetrics(unittest.TestCase):
    def test_df_to_array(self):
        df = pd.DataFrame([1, 2, 3], columns=["col"])
        expected = np.array([[1], [2], [3]])
        t = _df_to_array(df)
        assert np.array_equal(t, expected)

    def test_compute_distance(self):
        x1 = np.array([1, 0, 1])
        x2 = np.array([0, 0, 1])
        mean_vector = np.array([2, 1, 3])
        epsilon = 0
        expected = 0.5
        t = _compute_distance(x1, x2, mean_vector, epsilon)
        assert np.isclose(t, expected)

    def test_compute_similarities(self):
        rng = np.random.default_rng(seed=79)
        df = pd.DataFrame(rng.integers(0, 100, size=(5, 4)), columns=list("ABCD")).values
        instance = df[0, :]
        expected_len = 5
        expected_dist = 0
        t = _compute_similarities(instance, df)
        assert len(t) == expected_len
        assert t[0] == expected_dist

    def test_get_radius(self):
        rng = np.random.default_rng(seed=79)
        df = pd.DataFrame(rng.integers(0, 100, size=(5, 4)), columns=list("ABCD")).values
        t = _get_radius(df, n_neighbors=3)
        assert t > 0

    def test_find_neighbors(self):
        rng = np.random.default_rng(seed=79)
        df = pd.DataFrame(rng.integers(0, 100, size=(15, 4)), columns=list("ABCD"))
        selection = [1, 3]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)
        mode = "regression"
        t = find_neighbors(selection, X, model, mode)
        assert len(t) == len(selection)
        assert t[0].shape[1] == X.shape[1] + 2

    def test_shap_neighbors(self):
        rng = np.random.default_rng(seed=79)
        df = pd.DataFrame(rng.integers(0, 100, size=(15, 4)), columns=list("ABCD"))
        contrib = pd.DataFrame(rng.integers(10, size=(15, 4)), columns=list("EFGH"))
        instance = df.iloc[:2, :].values
        extra_cols = np.repeat(np.array([0, 0]), 2).reshape(2, -1)
        instance = np.append(instance, extra_cols, axis=1)
        mode = "regression"
        t = shap_neighbors(instance, df, contrib, mode)
        assert t[0].shape == instance[:, :-2].shape
        assert t[1].shape == (len(df.columns),)
        assert t[2].shape == (len(df.columns),)

    def test_get_min_nb_features(self):
        rng = np.random.default_rng(seed=79)
        contrib = pd.DataFrame(rng.integers(10, size=(15, 4)), columns=list("ABCD"))
        selection = [1, 3]
        distance = 0.1
        mode = "regression"
        t = get_min_nb_features(selection, contrib, mode, distance)
        assert type(t) == list
        assert all(isinstance(x, int) for x in t)
        assert len(t) == len(selection)

    def test_get_distance(self):
        rng = np.random.default_rng(seed=79)
        contrib = pd.DataFrame(rng.integers(10, size=(15, 4)), columns=list("ABCD"))
        selection = [1, 3]
        nb_features = 2
        mode = "regression"
        t = get_distance(selection, contrib, mode, nb_features)
        assert type(t) == np.ndarray
        assert all(isinstance(x, float) for x in t)
        assert len(t) == len(selection)
