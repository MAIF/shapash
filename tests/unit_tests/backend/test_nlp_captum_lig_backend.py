"""Unit tests for ``NlpCaptumLigBackend`` construction and registration (captum-free).

The capability guard rejects incompatible models *before* captum is imported, so these run without
torch/captum installed. The real LayerIntegratedGradients attribution path is covered by the
integration tests (which require a live transformer).
"""

import unittest

import numpy as np

from shapash.backend import NlpCaptumLigBackend, get_backend_cls_from_name
from shapash.backend.nlp_captum_lig_backend import _aggregate_by_alignment, _aggregate_subwords
from shapash.model.base import SupportsCaptumIG, TextModel


class PredictOnlyModel(TextModel):
    """A model without the Captum attribution surface (no ``SupportsCaptumIG``)."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))


class TestNlpCaptumLigBackend(unittest.TestCase):
    def test_registered_by_name(self):
        self.assertIs(get_backend_cls_from_name("nlp_captum_lig"), NlpCaptumLigBackend)

    def test_backend_name_attribute(self):
        self.assertEqual(NlpCaptumLigBackend.name, "nlp_captum_lig")

    def test_requires_captum_capability(self):
        with self.assertRaises(TypeError):
            NlpCaptumLigBackend(PredictOnlyModel(), label_names=["neg", "pos"])

    def test_backend_contract_attributes(self):
        self.assertEqual(NlpCaptumLigBackend.reference_kind, "point")
        self.assertTrue(NlpCaptumLigBackend.is_additive)
        self.assertEqual(NlpCaptumLigBackend.output_space, "logit")
        self.assertEqual(NlpCaptumLigBackend.requires_model_capabilities, (SupportsCaptumIG,))


class TestAggregateSubwords(unittest.TestCase):
    """``_aggregate_subwords`` is pure numpy — no torch/captum needed."""

    def test_merges_subwords_and_drops_specials(self):
        tokens = ["[CLS]", "i", "am", "hap", "##py", "[SEP]"]
        contribs = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0], [6.0, 5.0]], dtype=float)
        base = np.array([10.0, 20.0])

        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        self.assertEqual(words, ["i", "am", "happy"])
        # "happy" = "hap" + "##py": contributions summed.
        np.testing.assert_allclose(word_contribs, [[2.0, 1.0], [3.0, 2.0], [9.0, 7.0]])
        # [CLS] + [SEP] attribution folded into the baseline.
        np.testing.assert_allclose(new_base, [10.0 + 1.0 + 6.0, 20.0 + 0.0 + 5.0])

    def test_preserves_completeness(self):
        tokens = ["[CLS]", "great", "mov", "##ie", "[SEP]"]
        contribs = np.random.default_rng(0).normal(size=(5, 3))
        base = np.array([0.5, -0.5, 1.0])

        _, word_contribs, new_base = _aggregate_subwords(tokens, contribs, base)

        # base + Σ over words must equal the original base + Σ over every subword (nothing lost).
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), base + contribs.sum(axis=0))

    def test_all_special_returns_empty_words(self):
        tokens = ["[CLS]", "[SEP]"]
        contribs = np.array([[1.0, 2.0], [3.0, 4.0]])
        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(2))
        self.assertEqual(words, [])
        self.assertEqual(word_contribs.shape, (0, 2))
        np.testing.assert_allclose(new_base, [4.0, 6.0])

    def test_byte_bpe_word_start_markers_and_angle_specials(self):
        # RoBERTa/XLM-R: "Ġ" marks word *starts*, specials are "<s>"/"</s>" (not brackets). The first
        # content token ("im") carries no marker but is still a new word.
        tokens = ["<s>", "im", "Ġfeeling", "Ġhappy", "</s>"]
        contribs = np.arange(10, dtype=float).reshape(5, 2)
        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(2))
        self.assertEqual(words, ["im", "feeling", "happy"])  # no "<s>"/"Ġ" leakage
        np.testing.assert_allclose(word_contribs, [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        np.testing.assert_allclose(new_base, [0.0 + 8.0, 1.0 + 9.0])  # <s> + </s> folded

    def test_byte_bpe_midword_piece_is_merged(self):
        # An unmarked non-initial piece ("ing") continues the previous word under byte-BPE.
        tokens = ["<s>", "Ġfeel", "ing", "</s>"]
        contribs = np.array([[1.0], [2.0], [3.0], [4.0]])
        words, word_contribs, _ = _aggregate_subwords(tokens, contribs, np.zeros(1))
        self.assertEqual(words, ["feeling"])
        np.testing.assert_allclose(word_contribs, [[5.0]])  # 2 + 3

    def test_sentencepiece_word_start_marker(self):
        # DeBERTa-v2/v3 / T5: "▁" marks word starts.
        tokens = ["[CLS]", "▁im", "▁feeling", "[SEP]"]
        contribs = np.ones((4, 2))
        words, _, _ = _aggregate_subwords(tokens, contribs, np.zeros(2))
        self.assertEqual(words, ["im", "feeling"])

    def test_explicit_special_set_overrides_regex(self):
        # When the model's own special set is passed, membership decides specials — a token that merely
        # *looks* bracket-y ("<odd>") is content unless it's actually in the set.
        tokens = ["<s>", "hi", "<odd>"]
        contribs = np.array([[1.0], [2.0], [3.0]])
        words, word_contribs, new_base = _aggregate_subwords(tokens, contribs, np.zeros(1), special_tokens={"<s>"})
        self.assertEqual(words, ["hi", "<odd>"])  # "<odd>" kept as content, only "<s>" folded
        np.testing.assert_allclose(new_base, [1.0])


class TestAggregateByAlignment(unittest.TestCase):
    """``_aggregate_by_alignment`` folds exact tokenizer grouping — pure numpy, no torch."""

    def test_sums_words_and_folds_specials(self):
        contribs = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [6.0, 5.0]])
        # words = ["im", "feeling"] over positions [[1], [2, 3]]; specials at 0 and 4.
        alignment = (["im", "feeling"], [[1], [2, 3]], [0, 4])
        words, word_contribs, new_base = _aggregate_by_alignment(contribs, np.array([10.0, 20.0]), alignment)
        self.assertEqual(words, ["im", "feeling"])
        np.testing.assert_allclose(word_contribs, [[2.0, 1.0], [7.0, 5.0]])  # [2,3] summed
        np.testing.assert_allclose(new_base, [10.0 + 1.0 + 6.0, 20.0 + 0.0 + 5.0])

    def test_preserves_completeness(self):
        contribs = np.random.default_rng(1).normal(size=(5, 3))
        base = np.array([0.5, -0.5, 1.0])
        alignment = (["a", "b"], [[1], [2, 3]], [0, 4])
        _, word_contribs, new_base = _aggregate_by_alignment(contribs, base, alignment)
        np.testing.assert_allclose(new_base + word_contribs.sum(axis=0), base + contribs.sum(axis=0))


class TestProgressIter(unittest.TestCase):
    """``_progress_iter`` is captum-free — exercise it via ``object.__new__`` to skip ``__init__``."""

    @staticmethod
    def _backend(show_progress):
        backend = object.__new__(NlpCaptumLigBackend)  # bypass captum import in __init__
        backend.show_progress = show_progress
        return backend

    def test_returns_items_unchanged_when_disabled(self):
        items = ["a", "b", "c"]
        self.assertIs(self._backend(show_progress=False)._progress_iter(items), items)

    def test_iterates_all_items_when_enabled(self):
        # With tqdm installed it wraps in a bar; without it, the plain list — either way the loop must
        # still visit every element.
        items = ["a", "b", "c"]
        self.assertEqual(list(self._backend(show_progress=True)._progress_iter(items)), items)


if __name__ == "__main__":
    unittest.main()
