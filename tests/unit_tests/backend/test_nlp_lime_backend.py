"""Unit tests for ``NlpLimeBackend`` — LIME-based word attribution for NLP models."""

import unittest

import numpy as np

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_lime_backend import NlpLimeBackend

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
