"""Unit tests for ``NlpLimeBackend`` — LIME-based word attribution for NLP models."""

import unittest
from unittest.mock import patch

import numpy as np

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_lime_backend import NlpLimeBackend
from shapash.model.base import TextModel

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)

# Keep num_samples tiny so LIME tests run fast in CI.
_LIME_COMPUTE_ARGS = {"num_samples": 50, "num_features": 5}
_SAMPLE_TEXTS = ["i feel so happy today", "this is terrible and sad"]


def _fake_classifier(texts: list[str]) -> np.ndarray:
    """Deterministic fake classifier returning (n_texts, N_CLASSES) probabilities."""
    rng = np.random.default_rng(0)
    probs = rng.random((len(texts), N_CLASSES)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


class _FakeTextModel(TextModel):
    """Prediction-only ``TextModel`` over :func:`_fake_classifier`."""

    def predict(self, texts: list[str]) -> np.ndarray:
        return _fake_classifier(texts)


def _make_lime_backend() -> NlpLimeBackend:
    return NlpLimeBackend(
        _fake_classifier,
        label_names=LABEL_NAMES,
        explainer_compute_args=_LIME_COMPUTE_ARGS,
    )


class TestNlpLimeBackend(unittest.TestCase):
    def setUp(self):
        self.backend = _make_lime_backend()

    # --- init / config ---

    def test_name(self):
        self.assertEqual(self.backend.name, "nlp_lime")

    def test_inherits_nlp_backend(self):
        self.assertIsInstance(self.backend, NlpBackend)

    def test_label_names_stored(self):
        self.assertEqual(self.backend._classes, LABEL_NAMES)

    def test_mask_string_stored(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            mask_string="[MASK]",
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertEqual(backend.mask_string, "[MASK]")

    def test_explainer_args_forwarded(self):
        backend = NlpLimeBackend(
            _fake_classifier,
            label_names=LABEL_NAMES,
            explainer_args={"bow": False},
            explainer_compute_args=_LIME_COMPUTE_ARGS,
        )
        self.assertFalse(backend.explainer.bow)

    def test_baseline_token_is_none_when_words_are_removed(self):
        # bow=True (LIME's default) drops words outright: there is no substitute token.
        self.assertIsNone(self.backend.baseline_token)

    def test_baseline_token_is_the_mask_string_when_words_are_replaced(self):
        for mask_string, expected in ((None, "UNKWORDZ"), ("[MASK]", "[MASK]")):
            with self.subTest(mask_string=mask_string):
                backend = NlpLimeBackend(
                    _fake_classifier,
                    label_names=LABEL_NAMES,
                    mask_string=mask_string,
                    explainer_args={"bow": False},
                    explainer_compute_args=_LIME_COMPUTE_ARGS,
                )
                self.assertEqual(backend.baseline_token, expected)

    # --- _classifier_fn ---

    def test_classifier_fn_converts_list_to_array(self):
        def list_model(texts):
            return [[0.5] * N_CLASSES for _ in texts]

        backend = NlpLimeBackend(list_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, N_CLASSES))

    def test_classifier_fn_passes_through_array(self):
        arr = np.ones((2, N_CLASSES), dtype=np.float32)

        def array_model(texts):
            return arr

        backend = NlpLimeBackend(array_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["a", "b"])
        self.assertIs(result, arr)

    def test_classifier_fn_converts_hf_pipeline_format(self):
        # HuggingFace pipeline with return_all_scores=True returns list[list[dict]].
        def hf_model(texts):
            return [[{"label": name, "score": 1.0 / N_CLASSES} for name in LABEL_NAMES] for _ in texts]

        backend = NlpLimeBackend(hf_model, label_names=LABEL_NAMES)
        result = backend._classifier_fn(["hello", "world"])
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, N_CLASSES))
        self.assertEqual(result.dtype, np.float64)
        # Each score should be 1/N_CLASSES
        np.testing.assert_allclose(result, 1.0 / N_CLASSES, atol=1e-6)

    # --- run_explainer ---

    def test_run_explainer_returns_nlp_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertIsInstance(raw, NlpContributions)

    def test_run_explainer_contributions_count(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.values), len(_SAMPLE_TEXTS))

    def test_run_explainer_contributions_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        for arr in raw.values:
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[1], N_CLASSES)

    def test_run_explainer_base_values_shape(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(raw.base_values.shape, (len(_SAMPLE_TEXTS), N_CLASSES))

    def test_run_explainer_data_is_word_list(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertEqual(len(raw.token_strings), len(_SAMPLE_TEXTS))
        for word_list in raw.token_strings:
            self.assertIsInstance(word_list, list)
            self.assertTrue(all(isinstance(w, str) for w in word_list))

    def test_run_explainer_sparse_weights(self):
        # LIME fills at most num_features non-zero weights per label column.
        raw = self.backend.run_explainer(_SAMPLE_TEXTS[:1])
        matrix = raw.values[0]
        for col in range(N_CLASSES):
            self.assertLessEqual(
                np.count_nonzero(matrix[:, col]),
                _LIME_COMPUTE_ARGS["num_features"],
            )

    def test_num_features_defaults_to_the_distinct_word_count(self):
        # LIME's own default (10) zero-fills every word past the top 10 — indistinguishable from "no effect".
        text = "one two three four five six seven eight nine ten eleven twelve one"
        backend = NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES, explainer_compute_args={"num_samples": 50})
        with patch.object(backend.explainer, "explain_instance", wraps=backend.explainer.explain_instance) as spy:
            raw = backend.run_explainer([text])
        self.assertEqual(spy.call_args.kwargs["num_features"], 12)
        self.assertEqual(len(raw.token_strings[0]), 12)

    def test_every_word_gets_a_weight_by_default(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        backend = NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES, explainer_compute_args={"num_samples": 50})
        captured = {}
        original = backend.explainer.explain_instance

        def _capture(*args, **kwargs):
            captured["exp"] = original(*args, **kwargs)
            return captured["exp"]

        with patch.object(backend.explainer, "explain_instance", side_effect=_capture):
            backend.run_explainer([text])
        for label_idx in range(N_CLASSES):
            self.assertEqual(len(captured["exp"].local_exp[label_idx]), 12)

    def test_feature_count_follows_the_explainer_settings(self):
        char = NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES, explainer_args={"char_level": True})
        self.assertEqual(char._count_features("abca"), 3)
        positional = NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES, explainer_args={"bow": False})
        self.assertEqual(positional._count_features("a b a"), 3)
        self.assertEqual(self.backend._count_features("a b a"), 2)
        self.assertEqual(self.backend._count_features(""), 1)

    def test_explicit_num_features_is_respected(self):
        with patch.object(self.backend.explainer, "explain_instance", wraps=self.backend.explainer.explain_instance) as spy:
            self.backend.run_explainer(_SAMPLE_TEXTS[:1])
        self.assertEqual(spy.call_args.kwargs["num_features"], _LIME_COMPUTE_ARGS["num_features"])

    def test_default_is_written_into_the_settings(self):
        # Visible on the backend and part of NlpExplainer's cache key, so a result computed under
        # LIME's own top-10 default is never reused for "every word".
        backend = NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES)
        self.assertEqual(backend.explainer_compute_args["num_features"], "all")
        self.assertEqual(self.backend.explainer_compute_args["num_features"], _LIME_COMPUTE_ARGS["num_features"])

    def test_token_strings_are_plain_str(self):
        # LIME's own vocabulary is np.str_ (a str subclass, so isinstance alone would not catch it).
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        self.assertTrue(all(type(w) is str for words in raw.token_strings for w in words))

    def test_show_progress(self):
        self.assertFalse(self.backend.show_progress)
        backend = NlpLimeBackend(
            _fake_classifier, label_names=LABEL_NAMES, explainer_compute_args=_LIME_COMPUTE_ARGS, show_progress=True
        )
        self.assertTrue(backend.show_progress)
        self.assertEqual(len(backend.run_explainer(_SAMPLE_TEXTS).values), len(_SAMPLE_TEXTS))

    def test_caller_settings_dict_is_not_modified(self):
        settings = {"num_samples": 50}
        NlpLimeBackend(_fake_classifier, label_names=LABEL_NAMES, explainer_compute_args=settings)
        self.assertEqual(settings, {"num_samples": 50})

    def test_run_explainer_without_label_names_explains_every_column(self):
        # LIME's own default would explain class 1 only, leaving every other column silently zero.
        backend = NlpLimeBackend(_fake_classifier, explainer_compute_args=_LIME_COMPUTE_ARGS)
        raw = backend.run_explainer(_SAMPLE_TEXTS[:1])
        self.assertEqual(raw.values[0].shape[1], N_CLASSES)
        self.assertEqual(raw.base_values.shape, (1, N_CLASSES))
        self.assertTrue(all(np.any(raw.values[0][:, col]) for col in range(N_CLASSES)))

    # --- TextModel input ---

    def test_accepts_a_text_model_scored_through_predict(self):
        model = _FakeTextModel(LABEL_NAMES)
        backend = NlpLimeBackend(model, explainer_compute_args=_LIME_COMPUTE_ARGS)
        self.assertEqual(backend.model, model.predict)
        raw = backend.run_explainer(_SAMPLE_TEXTS[:1])
        self.assertEqual(raw.values[0].shape[1], N_CLASSES)

    def test_label_names_default_to_the_text_model_s(self):
        backend = NlpLimeBackend(_FakeTextModel(LABEL_NAMES))
        self.assertEqual(backend._classes, LABEL_NAMES)
        self.assertEqual(backend.explainer.class_names, LABEL_NAMES)

    def test_explicit_label_names_override_the_text_model_s(self):
        renamed = [name.upper() for name in LABEL_NAMES]
        backend = NlpLimeBackend(_FakeTextModel(LABEL_NAMES), label_names=renamed)
        self.assertEqual(backend._classes, renamed)

    # --- get_local_contributions ---

    def test_get_local_contributions_returns_nlp_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertIsInstance(contrib, NlpContributions)

    def test_get_local_contributions_token_strings_match_data(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        self.assertEqual(contrib.token_strings, raw.token_strings)

    def test_get_local_contributions_values_match_contributions(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw)
        for got, expected in zip(contrib.values, raw.values):
            np.testing.assert_array_equal(got, expected)

    def test_get_local_contributions_subset(self):
        raw = self.backend.run_explainer(_SAMPLE_TEXTS)
        contrib = self.backend.get_local_contributions(_SAMPLE_TEXTS, raw, subset=[0])
        self.assertEqual(len(contrib.token_strings), 1)
        self.assertEqual(len(contrib.values), 1)
        self.assertEqual(contrib.base_values.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
