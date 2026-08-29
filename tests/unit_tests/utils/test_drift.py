"""Unit tests for lightweight SmartPredictor schema-drift utilities."""

import pandas as pd

from shapash.utils.drift import compute_schema_distribution, detect_schema_drift


def test_compute_schema_distribution_numeric_and_categorical() -> None:
    x = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, None],
            "segment": ["a", "a", "b", None],
        }
    )

    summary = compute_schema_distribution(x)

    assert summary["age"] == {
        "kind": "numeric",
        "sample_size": 4,
        "missing_rate": 0.25,
        "min": 20.0,
        "q25": 25.0,
        "median": 30.0,
        "q75": 35.0,
        "max": 40.0,
    }
    assert summary["segment"]["kind"] == "categorical"
    assert summary["segment"]["missing_rate"] == 0.25
    assert summary["segment"]["cardinality"] == 2


def test_detect_schema_drift_stable_input_is_empty() -> None:
    x = pd.DataFrame({"age": range(100), "segment": ["a", "b"] * 50})
    reference = compute_schema_distribution(x)

    assert detect_schema_drift(reference, compute_schema_distribution(x.copy())) == {}


def test_detect_schema_drift_numeric_median_shift() -> None:
    reference = compute_schema_distribution(pd.DataFrame({"age": range(100)}))
    current = compute_schema_distribution(pd.DataFrame({"age": range(200, 300)}))

    drift = detect_schema_drift(reference, current)

    assert "age" in drift
    assert any("median shifted" in reason for reason in drift["age"])


def test_detect_schema_drift_missing_rate_change() -> None:
    reference = compute_schema_distribution(pd.DataFrame({"age": range(100)}))
    current = compute_schema_distribution(pd.DataFrame({"age": [None] * 20 + list(range(80))}))

    drift = detect_schema_drift(reference, current)

    assert "age" in drift
    assert any("missing rate changed" in reason for reason in drift["age"])


def test_detect_schema_drift_categorical_frequency_change() -> None:
    reference = compute_schema_distribution(pd.DataFrame({"segment": ["a"] * 80 + ["b"] * 20}))
    current = compute_schema_distribution(pd.DataFrame({"segment": ["a"] * 20 + ["b"] * 80}))

    drift = detect_schema_drift(reference, current)

    assert "segment" in drift
    assert any("total variation" in reason for reason in drift["segment"])


def test_detect_schema_drift_skips_small_batches() -> None:
    """Avoid noisy drift warnings for individual or very small inference requests."""
    reference = compute_schema_distribution(pd.DataFrame({"age": range(100)}))
    current = compute_schema_distribution(pd.DataFrame({"age": range(200, 210)}))

    assert detect_schema_drift(reference, current) == {}
