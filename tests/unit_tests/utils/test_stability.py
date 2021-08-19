import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from shapash.utils.stability import _df_to_array, _compute_distance,\
    _compute_similarities, _get_radius, find_neighbors, shap_neighbors,\
    get_color_rgb, get_color_hex


class TestStability(unittest.TestCase):

    def test_df_to_array(self):
        df = pd.DataFrame([1, 2, 3], columns=["col"])
        expected = np.array([[1], [2], [3]])
        t = _df_to_array(df)
        assert np.array_equal(t, expected)

    def test_compute_distance(self):
        x1 = np.array([1, 0, 1])
        x2 = np.array([0, 0, 1])
        mean_vector = 2
        epsilon = 0
        expected = 0.5
        t = _compute_distance(x1, x2, mean_vector, epsilon)
        assert t == expected

    def test_compute_similarities(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)), columns=list('ABCD')).values
        instance = df[0, :]
        expected_len = 5
        expected_dist = 0
        t = _compute_similarities(instance, df)
        assert len(t) == expected_len
        assert t[0] == expected_dist

    def test_get_radius(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(5, 4)), columns=list('ABCD')).values
        t = _get_radius(df, n_neighbors=3)
        assert t > 0

    def test_find_neighbors(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        selection = [1, 3]
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = LinearRegression().fit(X, y)
        mode = "regression"
        t = find_neighbors(selection, X, model, mode)
        assert len(t) == len(selection)
        assert t[0].shape[1] == X.shape[1] + 2

    def test_shap_neighbors(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list('ABCD'))
        contrib = pd.DataFrame(np.random.randint(10, size=(15, 4)), columns=list('EFGH'))
        instance = df.values[:2, :]
        extra_cols = np.repeat(np.array([0, 0]), 2).reshape(2, -1)
        instance = np.append(instance, extra_cols, axis=1)
        t = shap_neighbors(instance, df, contrib)
        assert t[0].shape == instance[:, :-2].shape
        assert t[1].shape == (len(df.columns),)
        assert t[2].shape == (len(df.columns),)

    def test_get_color_rgb(self):
        t = get_color_rgb("thermal", 0.1)
        assert type(t) == str
        assert "rgb" in t

    def test_get_color_hex(self):
        expected = '#11306a'
        t = get_color_hex("thermal", 0.1)
        assert t == expected
