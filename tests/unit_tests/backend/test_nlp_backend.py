"""Unit tests for the shared NLP backend layer — the punctuation helper.

``NlpExplanation.word_importance`` filtering is covered in ``tests/unit_tests/explainer``; this file
covers :func:`~shapash.backend.nlp_backend.is_punctuation` itself, which word segmentation now
makes load-bearing (punctuation is emitted as its own unit rather than glued to a neighbour).
"""

import unittest

import numpy as np

from shapash.backend.nlp_backend import NlpBackend, is_punctuation
from shapash.backend.nlp_lime_backend import NlpLimeBackend
from shapash.backend.nlp_shap_backend import NlpShapBackend
from shapash.model.base import SupportsEmbeddings, TextModel


class TestIsPunctuation(unittest.TestCase):
    def test_pure_punctuation_units(self):
        for unit in [".", ",", "!", "!!!", "?!", "--", "\u2026", "(", ")", "\u201c", "/"]:
            with self.subTest(unit=unit):
                self.assertTrue(is_punctuation(unit))

    def test_words_are_not_punctuation(self):
        for unit in ["great", "don't", "state-of-the-art", "10/10", "u.s.", "_", "3"]:
            with self.subTest(unit=unit):
                self.assertFalse(is_punctuation(unit))

    def test_blank_is_not_punctuation(self):
        # Blanks are the special-token concern, handled by ``filter_special`` — not this helper.
        for unit in ["", "   ", "\n"]:
            with self.subTest(unit=unit):
                self.assertFalse(is_punctuation(unit))

    def test_surrounding_whitespace_is_ignored(self):
        self.assertTrue(is_punctuation("  !  "))


class TestBackendContractAttributes(unittest.TestCase):
    """``reference_kind`` / ``is_additive`` / ``output_space`` / `requires_model_capabilities`` (A9 / A11)."""

    def test_nlp_shap_backend(self):
        self.assertEqual(NlpShapBackend.reference_kind, "none")
        self.assertTrue(NlpShapBackend.is_additive)
        self.assertEqual(NlpShapBackend.output_space, "probability")
        self.assertEqual(NlpShapBackend.requires_model_capabilities, ())

    def test_nlp_lime_backend(self):
        self.assertEqual(NlpLimeBackend.reference_kind, "none")
        self.assertFalse(NlpLimeBackend.is_additive)
        self.assertEqual(NlpLimeBackend.output_space, "probability")
        self.assertEqual(NlpLimeBackend.requires_model_capabilities, ())


class _PredictOnlyModel(TextModel):
    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))


class _MinimalBackend(NlpBackend):
    """Minimal concrete backend for testing ``NlpBackend.__init__``'s capability guard."""

    name = "test_minimal_backend"
    reference_kind = "none"
    is_additive = True
    requires_model_capabilities: tuple[type, ...] = ()

    def run_explainer(self, x):
        raise NotImplementedError

    def get_local_contributions(self, x, explain_data, subset=None):
        raise NotImplementedError


class _RequiresEmbeddings(_MinimalBackend):
    """Same as ``_MinimalBackend``, but declares a capability requirement."""

    name = "test_requires_embeddings"
    requires_model_capabilities = (SupportsEmbeddings,)


class TestRequiresModelCapabilitiesGuard(unittest.TestCase):
    """``NlpBackend.__init__`` centralizes the capability check (A11 point 2)."""

    def test_raises_when_model_lacks_required_capability(self):
        with self.assertRaises(TypeError):
            _RequiresEmbeddings(_PredictOnlyModel())

    def test_no_check_when_no_capabilities_required(self):
        # The base default is () — a backend that declares no capability requirement
        # (like NlpShapBackend/NlpLimeBackend) accepts any callable, including a bare
        # model with no capability mixins at all.
        self.assertEqual(NlpBackend.requires_model_capabilities, ())
        backend = _MinimalBackend(_PredictOnlyModel())
        self.assertIsInstance(backend, _MinimalBackend)


if __name__ == "__main__":
    unittest.main()
