"""Unit tests for shapash.webapp.nlp_components.base: compose_selection and error_positions."""

import unittest

import numpy as np
import pandas as pd

from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_components import compose_selection as _compose_selection
from shapash.webapp.nlp_components import error_positions

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)


def _make_token_data() -> tuple[list[list[str]], list[np.ndarray], np.ndarray]:
    """Synthetic per-token contribution data for 3 samples, 6 classes."""
    rng = np.random.default_rng(42)
    token_strings = [
        ["", "i", "feel", "so", "happy", "today", ""],
        ["", "this", "is", "terrible", "and", "sad", ""],
        ["", "what", "a", "wonderful", "day", ""],
    ]
    values = [rng.uniform(-0.4, 0.4, size=(len(t), N_CLASSES)).astype(np.float32) for t in token_strings]
    base_values = rng.uniform(-0.1, 0.1, size=(3, N_CLASSES)).astype(np.float32)
    return token_strings, values, base_values


def _make_texts() -> pd.Series:
    return pd.Series(
        ["i feel so happy today", "this is terrible and sad", "what a wonderful day"],
        index=pd.RangeIndex(3),
    )


def _make_explanation(
    texts: pd.Series | None = None,
    y_pred: pd.Series | None = None,
    y_prob: pd.DataFrame | None = None,
    y_true: pd.Series | None = None,
    label_names: list[str] | None = LABEL_NAMES,
    backend_name: str = "nlp_shap",
    is_additive: bool = True,
    reference_kind: str = "none",
    output_space: str = "probability",
) -> NlpExplanation:
    """Synthetic ``NlpExplanation`` for 3 samples, 6 classes — the ``explain()`` return value."""
    texts = _make_texts() if texts is None else texts
    token_strings, values, base_values = _make_token_data()
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=(
            pd.Series(["joy", "sadness", "joy"], index=texts.index, name="prediction") if y_pred is None else y_pred
        ),
        y_prob=y_prob,
        y_true=y_true,
        label_names=label_names,
        folds_case=None,
        backend_name=backend_name,
        is_additive=is_additive,
        reference_kind=reference_kind,
        output_space=output_space,
    )


class TestComposeSelection(unittest.TestCase):
    """The three sample filters (scatter box, confusion cell, errors-only) must intersect."""

    def test_nothing_active_returns_none(self):
        self.assertIsNone(_compose_selection(None, None, None))

    def test_scatter_only(self):
        self.assertEqual(_compose_selection([1, 2, 3], None, None), [1, 2, 3])

    def test_cell_only(self):
        self.assertEqual(_compose_selection(None, [4, 5], None), [4, 5])

    def test_errors_only(self):
        self.assertEqual(_compose_selection(None, None, {2, 7}), [2, 7])

    def test_scatter_intersect_cell(self):
        self.assertEqual(_compose_selection([1, 2, 3], [2, 3, 4], None), [2, 3])

    def test_scatter_intersect_errors(self):
        # This is the box-then-errors case: errors *within* the selected points.
        self.assertEqual(_compose_selection([1, 2, 3, 4], None, {2, 4, 9}), [2, 4])

    def test_all_three_intersect(self):
        self.assertEqual(_compose_selection([1, 2, 3, 4], [2, 3, 4], {3, 4}), [3, 4])

    def test_empty_intersection_is_empty_list(self):
        self.assertEqual(_compose_selection([1, 2], None, {8, 9}), [])


class TestErrorPositions(unittest.TestCase):
    """The shared definition of "a model error" every errors-scoped panel filters on."""

    def test_none_without_ground_truth(self):
        self.assertIsNone(error_positions(_make_explanation()))

    def test_compares_as_strings(self):
        texts = _make_texts()
        explanation = _make_explanation(y_true=pd.Series(["joy", "joy", "joy"], index=texts.index))
        # y_pred is ["joy", "sadness", "joy"], so only the middle row disagrees.
        self.assertEqual(error_positions(explanation), {1})
