"""Unit tests for ``NlpShapBackend`` — subword aggregation and explainer wiring.

``_aggregate_subwords`` is pure numpy (no live shap.Explainer needed). ``run_explainer`` is exercised
with a fake explainer injected via ``explainer_args={"explainer": ...}`` so the test stays offline.
"""

import unittest

import numpy as np

from shapash.backend import NlpShapBackend, get_backend_cls_from_name
from shapash.backend.nlp_shap_backend import _aggregate_subwords


class TestNlpShapBackend(unittest.TestCase):
    def test_registered_by_name(self):
        self.assertIs(get_backend_cls_from_name("nlp_shap"), NlpShapBackend)

    def test_backend_name_attribute(self):
        self.assertEqual(NlpShapBackend.name, "nlp_shap")


class TestAggregateSubwords(unittest.TestCase):
    """``_aggregate_subwords`` merges SHAP's segments into words at word/non-word boundaries."""

    def test_merges_subwords_and_drops_specials(self):
        # SHAP's Text masker: a subword glued to the previous piece carries no trailing space;
        # the piece that ends the word does (see shap.maskers.Text.token_segments).
        tokens = ["", "i ", "am ", "hap", "py ", ""]
        contribs = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0], [6.0, 5.0]], dtype=float)
        base = np.array([10.0, 20.0])

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        self.assertEqual(words, ["i", "am", "happy"])
        # "happy" = "hap" + "py ": contributions summed.
        np.testing.assert_allclose(word_contribs, [[2.0, 1.0], [3.0, 2.0], [9.0, 7.0]])
        # Leading/trailing blank (CLS/SEP-equivalent) attribution folded into the baseline.
        np.testing.assert_allclose(new_base, [10.0 + 1.0 + 6.0, 20.0 + 0.0 + 5.0])

    def test_splits_contraction_at_the_apostrophe(self):
        # "don't" is segmented as "don" + "'" + "t " — none carry inter-token whitespace, so the
        # old whitespace-only rule glued them. The word/non-word rule splits at the apostrophe,
        # matching the units the tokenizer's own pre-tokenizer produces (and hence the LIG backend).
        tokens = ["", "i ", "don", "'", "t ", "feel ", "sad", ""]
        contribs = np.arange(8 * 2, dtype=float).reshape(8, 2)
        base = np.zeros(2)

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        self.assertEqual(words, ["i", "don", "'", "t", "feel", "sad"])
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), base + contribs.sum(axis=0))

    def test_punctuation_is_not_glued_to_neighbouring_words(self):
        # Regression: real ``shap.maskers.Text`` output for
        # "The acting was superb!!! I really did enjoy.Overall,I recommend it." — the old rule
        # produced "superb!!!" and "enjoy.Overall,I" because only whitespace ended a word.
        tokens = [
            "",
            "The ",
            "acting ",
            "was ",
            "superb",
            "!",
            "!",
            "! ",
            "I ",
            "really ",
            "did ",
            "enjoy",
            ".",
            "Overall",
            ",",
            "I ",
            "recommend ",
            "it",
            ".",
            "",
        ]
        contribs = np.arange(len(tokens), dtype=float).reshape(-1, 1)

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(1))

        self.assertEqual(
            words,
            [
                "The",
                "acting",
                "was",
                "superb",
                "!",
                "!",
                "!",
                "I",
                "really",
                "did",
                "enjoy",
                ".",
                "Overall",
                ",",
                "I",
                "recommend",
                "it",
                ".",
            ],
        )
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), contribs.sum(axis=0))

    def test_leading_space_regime_does_not_collapse_the_sample(self):
        # SHAP's slow-tokenizer fallback prepends a *leading* space to each token instead of
        # carrying trailing gap text (shap/maskers/_text.py). Under the old trailing-whitespace
        # rule nothing ever flushed, so a whole sample became one "word".
        tokens = ["", "The", " acting", " was", " superb", "!", "!", "!", " I", " enjoy", ""]
        contribs = np.arange(len(tokens), dtype=float).reshape(-1, 1)

        words, _, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))

        self.assertEqual(words, ["The", "acting", "was", "superb", "!", "!", "!", "I", "enjoy"])

    def test_subwords_merge_in_both_segment_regimes(self):
        for tokens in (["up", "dating "], [" up", "dating"]):
            with self.subTest(tokens=tokens):
                contribs = np.array([[1.0], [2.0]])
                words, word_contribs, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
                self.assertEqual(words, ["updating"])
                np.testing.assert_allclose(word_contribs, [[3.0]])

    def test_unsegmented_script_does_not_collapse_into_one_word(self):
        # CJK has no inter-word spaces, so a character-class rule would see one long word run.
        tokens = ["\u6211", "\u559c", "\u6b22"]
        contribs = np.arange(3, dtype=float).reshape(-1, 1)
        words, _, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
        self.assertEqual(words, ["\u6211", "\u559c", "\u6b22"])

    def test_korean_subwords_still_merge(self):
        # Korean *is* space-segmented, so its subword pieces must join like any other script's —
        # Hangul is deliberately excluded from the unsegmented-script list.
        tokens = ["\uc601\ud654", "\ub294 ", "\uc88b", "\uc558\ub2e4"]
        contribs = np.arange(4, dtype=float).reshape(-1, 1)
        words, _, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
        self.assertEqual(words, ["\uc601\ud654\ub294", "\uc88b\uc558\ub2e4"])

    def test_bracket_text_folds_into_baseline_only_without_a_special_set(self):
        # With the tokenizer's own special set, literal "[LAUGHTER]" in the source is a real word.
        tokens = ["ha ", "[LAUGHTER] ", "good "]
        contribs = np.array([[1.0], [2.0], [3.0]])

        words, _, new_base = _aggregate_subwords(
            tokens, contribs, np.zeros(1), special_tokens=frozenset({"[CLS]", "[SEP]"})
        )
        self.assertEqual(words, ["ha", "[LAUGHTER]", "good"])
        np.testing.assert_allclose(new_base, [0.0])

        # Without one (bare callable), the bracket regex still guards against leaking [CLS]/[SEP].
        words, _, new_base = _aggregate_subwords(["[CLS] ", "good ", "[SEP] "], contribs, np.zeros(1))
        self.assertEqual(words, ["good"])
        np.testing.assert_allclose(new_base, [4.0])

    def test_declared_special_token_folds_into_baseline(self):
        tokens = ["<s> ", "good ", "</s> "]
        contribs = np.array([[1.0], [2.0], [3.0]])
        words, word_contribs, new_base = _aggregate_subwords(
            tokens, contribs, np.zeros(1), special_tokens=frozenset({"<s>", "</s>"})
        )
        self.assertEqual(words, ["good"])
        np.testing.assert_allclose(word_contribs, [[2.0]])
        np.testing.assert_allclose(new_base, [4.0])

    def test_preserves_completeness(self):
        tokens = ["", "great ", "mov", "ie ", ""]
        contribs = np.random.default_rng(0).normal(size=(5, 3))
        base = np.array([0.5, -0.5, 1.0])

        _, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        # base + Σ over words must equal the original base + Σ over every token (nothing lost).
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), base + contribs.sum(axis=0))

    def test_all_special_returns_empty_words(self):
        tokens = ["", ""]
        contribs = np.array([[1.0, 2.0], [3.0, 4.0]])
        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(2))
        self.assertEqual(words, [])
        self.assertEqual(word_contribs.shape, (0, 2))
        np.testing.assert_allclose(new_base, [4.0, 6.0])

    def test_last_token_without_trailing_space_is_flushed(self):
        # The final real word before the closing special token often has no trailing space.
        tokens = ["", "hello ", "world", ""]
        contribs = np.array([[0.0], [1.0], [2.0], [0.0]])
        words, word_contribs, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
        self.assertEqual(words, ["hello", "world"])
        np.testing.assert_allclose(word_contribs, [[1.0], [2.0]])


class FakeShapExplanation:
    def __init__(self, data, values, base_values):
        self.data = data
        self.values = values
        self.base_values = base_values


class FakeShapExplainer:
    def __call__(self, x, **kwargs):
        return FakeShapExplanation(
            data=[["", "up", "dating ", ""]],
            values=[np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])],
            base_values=np.array([[0.1, 0.2]]),
        )


class TestRunExplainer(unittest.TestCase):
    def test_aggregates_subwords_end_to_end(self):
        backend = NlpShapBackend(
            model=lambda texts: np.zeros((len(texts), 2)),
            label_names=["neg", "pos"],
            explainer_args={"explainer": FakeShapExplainer},
        )

        raw = backend.run_explainer(["updating"])

        self.assertEqual(raw.token_strings, [["updating"]])
        np.testing.assert_allclose(raw.values[0], [[5.0, 3.0]])
        # CLS ([1,0]) + SEP ([4,3]) folded into the base.
        np.testing.assert_allclose(raw.base_values, [[0.1 + 1.0 + 4.0, 0.2 + 0.0 + 3.0]])


class FakePipeline:
    """Stands in for a ``transformers`` pipeline: callable, with the private batch-size slot."""

    def __init__(self, batch_size=None):
        self._batch_size = batch_size
        self.tokenizer = object()

    def __call__(self, texts, **kwargs):
        return np.zeros((len(texts), 2))


class TestPipelineBatchSize(unittest.TestCase):
    """A pipeline arriving with no batch size scores masked variants one at a time — see __init__."""

    def _backend(self, model, **kwargs):
        return NlpShapBackend(model=model, explainer_args={"explainer": FakeShapExplainer}, **kwargs)

    def test_sets_batch_size_on_unconfigured_pipeline(self):
        pipe = FakePipeline()
        self._backend(pipe)
        self.assertEqual(pipe._batch_size, 64)

    def test_treats_batch_size_of_one_as_unconfigured(self):
        # transformers resolves an unset batch size to 1, so 1 carries no intent.
        pipe = FakePipeline(batch_size=1)
        self._backend(pipe)
        self.assertEqual(pipe._batch_size, 64)

    def test_never_overrides_a_configured_pipeline(self):
        pipe = FakePipeline(batch_size=8)
        self._backend(pipe)
        self.assertEqual(pipe._batch_size, 8)

    def test_explicit_batch_size_is_honoured(self):
        pipe = FakePipeline()
        self._backend(pipe, batch_size=256)
        self.assertEqual(pipe._batch_size, 256)

    def test_none_leaves_the_pipeline_untouched(self):
        pipe = FakePipeline()
        self._backend(pipe, batch_size=None)
        self.assertIsNone(pipe._batch_size)

    def test_bare_callable_is_unaffected(self):
        # A LIME-style scoring function has no _batch_size slot; it must not gain one.
        model = lambda texts: np.zeros((len(texts), 2))  # noqa: E731
        self._backend(model)
        self.assertFalse(hasattr(model, "_batch_size"))


if __name__ == "__main__":
    unittest.main()
