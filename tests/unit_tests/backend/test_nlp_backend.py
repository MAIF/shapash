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
    """``reference_kind`` / ``is_additive`` / ``output_space`` / ``requires_model_capabilities``."""

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

    def test_every_backend_so_far_is_signed_and_learns_its_baseline_at_fit(self):
        # ``baseline_token_`` depends on the model's tokenizer / masker: an unfitted backend has none
        # to report, rather than a default that reads like an answer.
        for backend_cls in (NlpShapBackend, NlpLimeBackend):
            with self.subTest(backend=backend_cls.name):
                self.assertTrue(backend_cls.is_signed)
                self.assertFalse(hasattr(backend_cls(), "baseline_token_"))

    def test_progress_is_a_display_setting(self):
        self.assertIn("show_progress", NlpBackend.display_params)


class _PredictOnlyModel(TextModel):
    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))


class _MinimalBackend(NlpBackend):
    """Minimal concrete backend for testing ``NlpBackend.fit``'s contract."""

    name = "test_minimal_backend"
    reference_kind = "none"
    is_additive = True
    requires_model_capabilities: tuple[type, ...] = ()

    def __init__(self, setting: int = 1, options: dict | None = None) -> None:
        self.setting = setting
        self.options = options

    def _run_explainer(self, x):
        return x


class _RequiresEmbeddings(_MinimalBackend):
    """Same as ``_MinimalBackend``, but declares a capability requirement."""

    name = "test_requires_embeddings"
    requires_model_capabilities = (SupportsEmbeddings,)


class TestRequiresModelCapabilitiesGuard(unittest.TestCase):
    """``NlpBackend.fit`` centralizes the capability check."""

    def test_raises_when_model_lacks_required_capability(self):
        with self.assertRaisesRegex(TypeError, "SupportsEmbeddings"):
            _RequiresEmbeddings().fit(_PredictOnlyModel())

    def test_no_check_when_no_capabilities_required(self):
        # The base default is () — a backend that declares no capability requirement
        # (like NlpShapBackend/NlpLimeBackend) accepts any callable, including a bare
        # model with no capability mixins at all.
        self.assertEqual(NlpBackend.requires_model_capabilities, ())
        model = _PredictOnlyModel()
        backend = _MinimalBackend().fit(model)
        self.assertIs(backend.model_, model)


class TestSettingsThenFit(unittest.TestCase):
    """A backend holds settings; ``fit`` binds the model and returns ``self``."""

    def test_get_params_are_the_settings_as_given(self):
        options = {"a": 1}
        backend = _MinimalBackend(setting=3, options=options)
        self.assertEqual(backend.get_params(), {"setting": 3, "options": options})
        self.assertIs(backend.get_params()["options"], options)

    def test_fit_sets_no_baseline_token_unless_the_subclass_finds_one(self):
        self.assertIsNone(_MinimalBackend().fit(_PredictOnlyModel()).baseline_token_)

    def test_run_explainer_hands_over_once_fitted(self):
        self.assertEqual(_MinimalBackend().fit(_PredictOnlyModel()).run_explainer(["a"]), ["a"])

    def test_fit_returns_self_with_model_and_label_names(self):
        backend = _MinimalBackend()
        model = _PredictOnlyModel()
        self.assertIs(backend.fit(model, ["neg", "pos"]), backend)
        self.assertIs(backend.model_, model)
        self.assertEqual(backend.label_names_, ["neg", "pos"])

    def test_label_names_default_to_empty(self):
        self.assertEqual(_MinimalBackend().fit(_PredictOnlyModel()).label_names_, [])

    def test_run_explainer_before_fit_raises_not_fitted(self):
        from sklearn.exceptions import NotFittedError

        with self.assertRaises(NotFittedError):
            _MinimalBackend().run_explainer(["a"])

    def test_copy_from_get_params_carries_no_fit(self):
        fitted = _MinimalBackend(setting=2).fit(_PredictOnlyModel())
        fresh = type(fitted)(**fitted.get_params(deep=False))
        self.assertEqual(fresh.setting, 2)
        self.assertFalse(hasattr(fresh, "model_"))

    def test_repr_shows_settings(self):
        self.assertEqual(repr(_MinimalBackend(setting=5)), "_MinimalBackend(setting=5)")


if __name__ == "__main__":
    unittest.main()
