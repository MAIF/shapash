import numpy as np
import pandas as pd
import pytest

from shapash.backend._precomputed_backend import PrecomputedBackend


@pytest.fixture
def x():
    return pd.DataFrame({"first": [1.0, 2.0], "second": [3.0, 4.0]}, index=["a", "b"])


def test_precomputed_backend_rejects_explanation_computation(x):
    backend = PrecomputedBackend("regression", None)
    with pytest.raises(ValueError, match="must be supplied to compile"):
        backend.run_explainer(x)


@pytest.mark.parametrize(
    "values",
    [
        [["not", "numeric"], ["values", "here"]],
        np.array([[np.inf, 0.0], [1.0, 2.0]]),
        [np.ones((2, 2)), "invalid"],
    ],
)
def test_precomputed_backend_rejects_invalid_contributions(x, values):
    backend = PrecomputedBackend("regression", None)
    with pytest.raises(ValueError, match="finite numeric|DataFrames or numeric"):
        backend.format_and_aggregate_local_contributions(x, values)


@pytest.mark.parametrize(
    "base, expected",
    [
        (0.5, [0.5, 0.5]),
        ([0.2, 0.7], [0.2, 0.7]),
        (np.array([[0.2], [0.7]]), [0.2, 0.7]),
    ],
)
def test_precomputed_regression_baselines(x, base, expected):
    backend = PrecomputedBackend("regression", None)
    np.testing.assert_array_equal(backend.validate_base_values(x, base), expected)


@pytest.mark.parametrize(
    "base, expected",
    [
        ([0.2, 0.8], [[0.2, 0.8], [0.2, 0.8]]),
        ([[0.1, 0.9], [0.7, 0.3]], [[0.1, 0.9], [0.7, 0.3]]),
    ],
)
def test_precomputed_classification_baselines(x, base, expected):
    backend = PrecomputedBackend("classification", [0, 1])
    np.testing.assert_array_equal(backend.validate_base_values(x, base), expected)


@pytest.mark.parametrize(
    "base, message",
    [
        (pd.Series([0.1, 0.2], index=["wrong", "index"]), "same index"),
        (np.array([np.nan, 0.2]), "finite numeric"),
        ([0.1, 0.2, 0.3], "per-row"),
    ],
)
def test_precomputed_backend_rejects_invalid_baselines(x, base, message):
    backend = PrecomputedBackend("regression", None)
    with pytest.raises(ValueError, match=message):
        backend.validate_base_values(x, base)


def test_precomputed_classification_baselines_validate_order_and_shape(x):
    backend = PrecomputedBackend("classification", [0, 1])
    with pytest.raises(ValueError, match="same order"):
        backend.validate_base_values(x, pd.DataFrame([[0.1, 0.9], [0.2, 0.8]], index=x.index, columns=[1, 0]))
    with pytest.raises(ValueError, match="per-row"):
        backend.validate_base_values(x, [0.2, 0.3, 0.5])
