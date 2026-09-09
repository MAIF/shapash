"""Lightweight schema-distribution summaries and drift detection."""

from typing import Any

import pandas as pd

MISSING_RATE_DELTA_THRESHOLD = 0.10
NUMERIC_MEDIAN_IQR_THRESHOLD = 1.5
CATEGORICAL_TVD_THRESHOLD = 0.20
CATEGORICAL_CARDINALITY_RATIO_THRESHOLD = 1.5
DEFAULT_TOP_K = 10
MIN_DRIFT_SAMPLE_SIZE = 30

DEFAULT_SCHEMA_DRIFT_CONFIG: dict[str, float | int] = {
    "missing_rate_delta_threshold": MISSING_RATE_DELTA_THRESHOLD,
    "numeric_median_iqr_threshold": NUMERIC_MEDIAN_IQR_THRESHOLD,
    "categorical_tvd_threshold": CATEGORICAL_TVD_THRESHOLD,
    "categorical_cardinality_ratio_threshold": CATEGORICAL_CARDINALITY_RATIO_THRESHOLD,
    "top_k": DEFAULT_TOP_K,
    "min_sample_size": MIN_DRIFT_SAMPLE_SIZE,
}


def resolve_schema_drift_config(config: dict[str, float | int] | None = None) -> dict[str, float | int]:
    """Merge user-provided schema-drift settings with validated defaults."""
    resolved = DEFAULT_SCHEMA_DRIFT_CONFIG.copy()
    if config is None:
        return resolved
    if not isinstance(config, dict):
        raise ValueError("schema_drift_config must be a dict.")

    unknown = set(config) - set(resolved)
    if unknown:
        raise ValueError(f"Unknown schema drift configuration keys: {sorted(unknown)}")
    resolved.update(config)

    for key in (
        "missing_rate_delta_threshold",
        "numeric_median_iqr_threshold",
        "categorical_tvd_threshold",
        "categorical_cardinality_ratio_threshold",
    ):
        if not isinstance(resolved[key], (int, float)) or isinstance(resolved[key], bool) or resolved[key] < 0:
            raise ValueError(f"schema_drift_config['{key}'] must be a non-negative number.")
    for key in ("top_k", "min_sample_size"):
        if not isinstance(resolved[key], int) or isinstance(resolved[key], bool) or resolved[key] < 1:
            raise ValueError(f"schema_drift_config['{key}'] must be a positive integer.")
    return resolved


def compute_schema_distribution(x: pd.DataFrame, top_k: int = DEFAULT_TOP_K) -> dict[Any, dict[str, Any]]:
    """Return compact per-column distribution summaries for ``x``."""
    summaries: dict[Any, dict[str, Any]] = {}
    for column in x.columns:
        series = x[column]
        non_missing = series.dropna()
        missing_rate = float(series.isna().mean())

        if pd.api.types.is_numeric_dtype(series.dtype):
            quantiles = non_missing.quantile([0.25, 0.5, 0.75]) if not non_missing.empty else pd.Series(dtype=float)
            summaries[column] = {
                "kind": "numeric",
                "sample_size": int(len(series)),
                "missing_rate": missing_rate,
                "min": _to_float(non_missing.min()) if not non_missing.empty else None,
                "q25": _to_float(quantiles.loc[0.25]) if not quantiles.empty else None,
                "median": _to_float(quantiles.loc[0.5]) if not quantiles.empty else None,
                "q75": _to_float(quantiles.loc[0.75]) if not quantiles.empty else None,
                "max": _to_float(non_missing.max()) if not non_missing.empty else None,
            }
            continue

        frequencies = non_missing.map(repr).value_counts(normalize=True)
        top_frequencies = frequencies.head(top_k)
        summaries[column] = {
            "kind": "categorical",
            "sample_size": int(len(series)),
            "missing_rate": missing_rate,
            "cardinality": int(non_missing.nunique()),
            "top_frequencies": {str(key): float(value) for key, value in top_frequencies.items()},
            "other_frequency": float(max(0.0, 1.0 - top_frequencies.sum())),
        }
    return summaries


def detect_schema_drift(
    reference: dict[Any, dict[str, Any]],
    current: dict[Any, dict[str, Any]],
    config: dict[str, float | int] | None = None,
) -> dict[Any, list[str]]:
    """Return actionable drift reasons keyed by column name."""
    resolved_config = resolve_schema_drift_config(config)
    drift: dict[Any, list[str]] = {}
    for column, reference_summary in reference.items():
        current_summary = current.get(column)
        if current_summary is None or current_summary.get("kind") != reference_summary.get("kind"):
            continue
        if (
            min(int(reference_summary.get("sample_size", 0)), int(current_summary.get("sample_size", 0)))
            < resolved_config["min_sample_size"]
        ):
            continue

        reasons = _missing_rate_reasons(
            reference_summary, current_summary, float(resolved_config["missing_rate_delta_threshold"])
        )
        if reference_summary["kind"] == "numeric":
            reasons.extend(
                _numeric_reasons(
                    reference_summary, current_summary, float(resolved_config["numeric_median_iqr_threshold"])
                )
            )
        else:
            reasons.extend(
                _categorical_reasons(
                    reference_summary,
                    current_summary,
                    float(resolved_config["categorical_tvd_threshold"]),
                    float(resolved_config["categorical_cardinality_ratio_threshold"]),
                )
            )
        if reasons:
            drift[column] = reasons
    return drift


def _missing_rate_reasons(reference: dict[str, Any], current: dict[str, Any], threshold: float) -> list[str]:
    reference_rate = float(reference["missing_rate"])
    current_rate = float(current["missing_rate"])
    if abs(current_rate - reference_rate) < threshold:
        return []
    return [f"missing rate changed from {reference_rate:.3f} to {current_rate:.3f}"]


def _numeric_reasons(reference: dict[str, Any], current: dict[str, Any], threshold: float) -> list[str]:
    reference_median = reference.get("median")
    current_median = current.get("median")
    reference_q25 = reference.get("q25")
    reference_q75 = reference.get("q75")
    if reference_median is None or current_median is None or reference_q25 is None or reference_q75 is None:
        return []

    iqr = float(reference_q75) - float(reference_q25)
    scale = max(abs(iqr), 1e-12)
    normalized_shift = abs(float(current_median) - float(reference_median)) / scale
    if normalized_shift < threshold:
        return []
    return [f"median shifted by {normalized_shift:.2f} reference IQRs"]


def _categorical_reasons(
    reference: dict[str, Any], current: dict[str, Any], tvd_threshold: float, cardinality_ratio_threshold: float
) -> list[str]:
    reference_top = reference.get("top_frequencies", {})
    current_top = current.get("top_frequencies", {})
    categories = set(reference_top)
    current_other = max(0.0, 1.0 - sum(float(current_top.get(key, 0.0)) for key in categories))
    total_variation = 0.5 * (
        sum(abs(float(reference_top.get(key, 0.0)) - float(current_top.get(key, 0.0))) for key in categories)
        + abs(float(reference.get("other_frequency", 0.0)) - current_other)
    )

    reasons: list[str] = []
    if total_variation >= tvd_threshold:
        reasons.append(f"category-frequency total variation is {total_variation:.3f}")

    reference_cardinality = int(reference.get("cardinality", 0))
    current_cardinality = int(current.get("cardinality", 0))
    if reference_cardinality > 0 and current_cardinality / reference_cardinality >= cardinality_ratio_threshold:
        reasons.append(f"cardinality changed from {reference_cardinality} to {current_cardinality}")
    return reasons


def _to_float(value: Any) -> float:
    """Convert numpy and pandas scalar values to a builtin float."""
    return float(value)
