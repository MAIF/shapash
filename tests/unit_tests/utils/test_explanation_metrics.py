import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

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

    def test_find_neighbors_with_fewer_rows_than_requested(self):
        rng = np.random.default_rng(seed=79)
        df = pd.DataFrame(rng.integers(0, 100, size=(5, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1] + 100
        model = LinearRegression().fit(X, y)

        neighbors = find_neighbors([1], X, model, "regression", n_neighbors=10)

        assert len(neighbors) == 1
        assert 0 < neighbors[0].shape[0] <= len(X)
        assert neighbors[0].shape[1] == X.shape[1] + 2
        np.testing.assert_array_equal(neighbors[0][0, : X.shape[1]], X.loc[1].values)

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

    def test_shap_neighbors_keeps_instance_first_with_duplicate_features(self):
        x = pd.DataFrame(
            {"A": [1.0, 5.0, 1.0, 1.1, 10.0], "B": [1.0, 5.0, 1.0, 1.0, 10.0]},
            index=["duplicate", "far", "selected", "near", "last"],
        )
        contributions = pd.DataFrame(
            {"A": [9.0, 1.0, 1.0, 5.0, 1.0], "B": [1.0, 1.0, 9.0, 5.0, 1.0]},
            index=x.index,
        )
        model = LinearRegression().fit(x, np.full(len(x), 10.0))

        with patch("shapash.utils.explanation_metrics._get_radius", return_value=np.inf):
            neighbors, positions = find_neighbors(
                ["selected"], x, model, "regression", n_neighbors=2, return_positions=True
            )

        normalized, _, amplitude = shap_neighbors(
            neighbors[0], x, contributions, "regression", neighbor_positions=positions[0]
        )

        assert x.index[positions[0]].tolist() == ["selected", "duplicate", "near"]
        assert normalized.shape == (3, 2)
        assert np.allclose(normalized[0], [0.1, 0.9])
        assert np.allclose(amplitude, [0.1, 0.9])

    def test_find_neighbors_keeps_selected_row_when_distance_ties_exceed_limit(self):
        x = pd.DataFrame({"A": [1.0] * 5, "B": [2.0] * 5})
        model = LinearRegression().fit(x, np.full(len(x), 10.0))

        with patch("shapash.utils.explanation_metrics._get_radius", return_value=np.inf):
            neighbors, positions = find_neighbors([4], x, model, "regression", n_neighbors=2, return_positions=True)

        assert positions[0][0] == 4
        assert len(positions[0]) == len(neighbors[0]) == 3

    def test_find_neighbors_classification_returns_filtered_positions(self):
        x = pd.DataFrame(
            {
                "A": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
                "B": [0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
            },
            index=["low", "selected-low", "high", "selected-high", "higher", "highest"],
        )
        model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x, [0, 0, 1, 1, 1, 1])
        selection = ["selected-high", "selected-low"]

        with patch("shapash.utils.explanation_metrics._get_radius", return_value=np.inf):
            neighbors, positions = find_neighbors(
                selection, x, model, "classification", n_neighbors=3, return_positions=True
            )

        for selected, neighborhood, neighborhood_positions in zip(selection, neighbors, positions, strict=True):
            assert neighborhood_positions[0] == x.index.get_loc(selected)
            assert len(neighborhood_positions) == len(neighborhood)

    def test_shap_neighbors_uses_neighborhood_order_without_positions(self):
        x = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [1.0, 2.0, 3.0]})
        contributions = pd.DataFrame({"A": [9.0, 5.0, 1.0], "B": [1.0, 5.0, 9.0]})
        instance = np.c_[x.iloc[[2, 0, 1]].to_numpy(), np.zeros((3, 2))]

        normalized, _, amplitude = shap_neighbors(instance, x, contributions, "regression")

        assert np.allclose(normalized[0], [0.1, 0.9])
        assert np.allclose(amplitude, [0.1, 0.9])

    def test_shap_neighbors_needs_positions_for_duplicate_features(self):
        x = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 2.0]})
        contributions = pd.DataFrame({"A": [1.0, 9.0], "B": [9.0, 1.0]})
        instance = np.c_[x.iloc[[1, 0]].to_numpy(), np.zeros((2, 2))]

        with self.assertRaisesRegex(ValueError, "pass neighbor_positions"):
            shap_neighbors(instance, x, contributions, "regression")

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
