"""Formatting of user-provided contributions without an inference model."""

from typing import Any, Literal

import numpy as np
import pandas as pd

from shapash.backend.base_backend import BaseBackend


class PrecomputedBackend(BaseBackend):
    """Reuse tabular aggregation and importance calculations without computing explanations."""

    name = "precomputed"

    def __init__(
        self,
        task: Literal["regression", "classification"],
        classes: list[int | float | str] | None,
        preprocessing: Any = None,
    ) -> None:
        super().__init__(None, preprocessing, task=task, classes=classes)

    def run_explainer(self, x: pd.DataFrame) -> dict[str, Any]:
        """Reject computation of new contributions without a model."""
        raise ValueError("Without a model, contributions must be supplied to compile().")

    def format_and_aggregate_local_contributions(
        self,
        x: pd.DataFrame,
        contributions: pd.DataFrame | np.ndarray | list[pd.DataFrame] | list[np.ndarray],
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """Validate numeric contributions before applying standard shape checks and aggregation."""
        frames = contributions if isinstance(contributions, list) else [contributions]
        for frame in frames:
            if not isinstance(frame, (pd.DataFrame, np.ndarray)):
                raise ValueError("Precomputed contributions must be DataFrames or numeric NumPy arrays.")
            values = np.asarray(frame)
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError("Precomputed contributions must contain finite numeric values.")
        return super().format_and_aggregate_local_contributions(x, contributions)

    def validate_base_values(self, x: pd.DataFrame, base_values: Any) -> np.ndarray:
        """Normalize explicit baselines without guessing their output scale."""
        if isinstance(base_values, (pd.Series, pd.DataFrame)) and not base_values.index.equals(x.index):
            raise ValueError("base_values and x must have the same index.")
        values = np.asarray(base_values)
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("base_values must contain finite numeric values.")
        if self._case == "regression":
            if values.ndim == 0:
                return np.full(len(x), values.item())
            if values.shape == (len(x), 1):
                return values[:, 0]
            if values.shape == (len(x),):
                return values
        else:
            n_classes = len(self._classes or [])
            if isinstance(base_values, pd.DataFrame) and list(base_values.columns) != self._classes:
                raise ValueError("base_values columns must match classes in the same order.")
            if values.shape == (n_classes,):
                return np.broadcast_to(values, (len(x), n_classes)).copy()
            if values.shape == (len(x), n_classes):
                return values
        raise ValueError(
            "base_values must be scalar/per-row for regression or per-class/per-row-per-class for classification."
        )
