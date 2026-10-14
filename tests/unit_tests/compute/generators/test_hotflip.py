"""Unit tests for the HotFlip counterfactual generator.

A deterministic linear fake model (2-D embeddings = per-class logit contributions) lets us assert
the full HotFlip contract — replacement selection, prediction flip, minimality — without a real
transformer or torch.
"""

import unittest

import numpy as np

from shapash.compute.generators import Counterfactual, HotFlipGenerator
from shapash.compute.generators.cf_utils import is_prediction_flip, prediction_difference
from shapash.model.base import (
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
)

# Embedding == [neg_logit_contribution, pos_logit_contribution] for each vocab token.
_EMB = {
    "this": [0.1, 0.1],
    "is": [0.1, 0.1],
    "great": [0.0, 2.0],
    "good": [0.0, 1.0],
    "terrible": [3.0, 0.0],
    "awful": [3.0, 0.0],
    "ok": [0.2, 0.2],
}
_VOCAB = list(_EMB)


class LinearFakeModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """Bag-of-embeddings linear classifier: logits = sum of per-token 2-vectors."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])
        self._matrix = np.array([_EMB[t] for t in _VOCAB], dtype=float)

    def _logits(self, text):
        toks = [t for t in text.split() if t in _EMB]
        return np.sum([_EMB[t] for t in toks], axis=0) if toks else np.zeros(2)

    def predict(self, texts):
        out = []
        for t in texts:
            z = self._logits(t)
            e = np.exp(z - z.max())
            out.append(e / e.sum())
        return np.vstack(out)

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return list(_VOCAB), self._matrix

    def embed(self, texts):
        return np.vstack([self._logits(t) for t in texts])

    def token_gradients(self, text, target_class):
        # Linear model ⇒ d logit[target] / d e_i = one-hot(target); identical per token.
        toks = text.split()
        grad = np.zeros(2)
        grad[target_class] = 1.0
        return toks, np.tile(grad, (len(toks), 1))


class TestCfUtils(unittest.TestCase):
    def test_is_prediction_flip_any(self):
        self.assertTrue(is_prediction_flip(np.array([0.2, 0.8]), np.array([0.7, 0.3])))
        self.assertFalse(is_prediction_flip(np.array([0.2, 0.8]), np.array([0.4, 0.6])))

    def test_is_prediction_flip_targeted(self):
        self.assertTrue(is_prediction_flip(np.array([0.8, 0.2]), np.array([0.1, 0.9]), target_class=1))
        self.assertFalse(is_prediction_flip(np.array([0.8, 0.2]), np.array([0.1, 0.9]), target_class=0))

    def test_prediction_difference(self):
        self.assertAlmostEqual(prediction_difference(np.array([0.9, 0.1]), np.array([0.4, 0.6]), 0), 0.5)


class TestHotFlipCompat(unittest.TestCase):
    def test_incompatible_without_gradients(self):
        class PredictOnly(TextModel):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

        self.assertFalse(HotFlipGenerator.is_compatible(PredictOnly()))

    def test_compatible_with_full_model(self):
        self.assertTrue(HotFlipGenerator.is_compatible(LinearFakeModel()))


class TestHotFlipGenerate(unittest.TestCase):
    def setUp(self):
        self.model = LinearFakeModel()
        self.gen = HotFlipGenerator(self.model)

    def test_config_spec_defaults(self):
        spec = self.gen.config_spec()
        self.assertEqual(spec["num_examples"].default, 5)
        self.assertEqual(spec["max_flips"].default, 3)
        self.assertEqual(self.gen.resolve_config({"max_flips": 2})["max_flips"], 2)

    def test_generates_flip(self):
        cfs = self.gen.generate("this is great", config={"num_examples": 3, "max_flips": 1})
        self.assertTrue(cfs, "expected at least one counterfactual")
        cf = cfs[0]
        self.assertIsInstance(cf, Counterfactual)
        self.assertEqual(cf.orig_label, "pos")
        self.assertEqual(cf.new_label, "neg")
        # Replacement must be one of the strongly-negative tokens.
        self.assertTrue(all(new in {"terrible", "awful"} for _, _, new in cf.substitutions))
        # Verify the reported flip actually holds under the model.
        self.assertTrue(
            is_prediction_flip(self.model.predict([cf.original_text])[0], self.model.predict([cf.new_text])[0])
        )

    def test_minimality(self):
        cfs = self.gen.generate("this is great", config={"num_examples": 10, "max_flips": 3})
        sets = [frozenset(cf.flipped_positions) for cf in cfs]
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i != j:
                    self.assertFalse(a < b, "returned a non-minimal (superset) counterfactual")

    def test_tokens_to_ignore(self):
        # Ignoring every content token yields no flips.
        cfs = self.gen.generate("this is great", config={"tokens_to_ignore": ["this", "is", "great"], "max_flips": 2})
        self.assertEqual(cfs, [])

    def test_no_flip_returns_empty(self):
        # Already strongly negative; ignore the negative words so nothing can push further.
        cfs = self.gen.generate("terrible awful", config={"tokens_to_ignore": ["terrible", "awful"]})
        self.assertEqual(cfs, [])


class CountingLinearModel(LinearFakeModel):
    """Linear fake model recording each ``predict`` call, for asserting how the search batches."""

    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def predict(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return super().predict(texts)


class TestHotFlipBatching(unittest.TestCase):
    def test_combination_search_is_batched(self):
        model = CountingLinearModel()
        gen = HotFlipGenerator(model)
        # Five flippable positions and max_flips=2 means 5 + C(5,2) = 20 candidate combinations.
        cfs = gen.generate("this is great good ok", config={"num_examples": 20, "max_flips": 2})
        self.assertTrue(cfs)
        # Budget: one call for the original, one shortlist re-scoring per position, and one per
        # combination *size level* — not one per combination, which is what dominated the cost.
        self.assertLessEqual(len(model.calls), 1 + 5 + 2)
        self.assertGreater(max(len(c) for c in model.calls), 1)


class SentencePieceLinearModel(LinearFakeModel):
    """The same linear classifier, tokenized SentencePiece-style: ``▁`` marks every word *start*.

    Both the tokens *and the vocabulary* carry the marker, as they do for a real DeBERTa-v3 / XLM-R
    tokenizer — which is what makes HotFlip's candidate shortlists empty under the old word test.
    """

    def tokenize(self, text):
        return [f"▁{w}" for w in text.split()]

    def detokenize(self, tokens):
        return " ".join(t.removeprefix("▁") for t in tokens)

    def get_embedding_table(self):
        return [f"▁{t}" for t in _VOCAB], self._matrix

    def token_gradients(self, text, target_class):
        tokens, grads = super().token_gradients(text, target_class)
        return [f"▁{t}" for t in tokens], grads


class TestHotFlipOnMarkedTokenizer(unittest.TestCase):
    """Regression: a word-start-marking tokenizer used to produce no candidates at all.

    Every vocabulary entry is ``"▁word"``, which is not alphabetic, so the old bare ``isalpha`` test
    rejected all of them: every shortlist came back empty, ``replacement`` stayed empty, and
    ``generate`` returned ``[]`` for every input on every SentencePiece model — silently.
    """

    def setUp(self):
        self.model = SentencePieceLinearModel()
        self.gen = HotFlipGenerator(self.model)

    def test_still_compatible(self):
        self.assertTrue(HotFlipGenerator.is_compatible(self.model))

    def test_marked_vocab_entries_are_candidates(self):
        vocab, _ = self.model.get_embedding_table()
        self.assertTrue(all(self.model.is_substitutable(t) for t in vocab))

    def test_generates_flip(self):
        cfs = self.gen.generate("this is great", config={"num_examples": 3, "max_flips": 1})
        self.assertTrue(cfs, "no counterfactual on a SentencePiece tokenizer — the C2 regression")
        cf = cfs[0]
        self.assertEqual((cf.orig_label, cf.new_label), ("pos", "neg"))
        self.assertTrue(all(new in {"▁terrible", "▁awful"} for _, _, new in cf.substitutions))
        # The rebuilt text detokenizes cleanly, and the reported flip holds under the model.
        self.assertNotIn("▁", cf.new_text)
        self.assertTrue(
            is_prediction_flip(self.model.predict([cf.original_text])[0], self.model.predict([cf.new_text])[0])
        )

    def test_tokens_to_ignore_matches_the_bare_word_a_user_types(self):
        cfs = self.gen.generate(
            "this is great", config={"tokens_to_ignore": ["this", "is", "great"], "max_flips": 2}
        )
        self.assertEqual(cfs, [])


class TestHotFlipRequiresTokenization(unittest.TestCase):
    def test_incompatible_without_tokenization(self):
        # HotFlip calls detokenize/is_substitutable, so the capability must be declared, not assumed.
        class GradientsOnly(TextModel, SupportsEmbeddings, SupportsGradients):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

            def get_embedding_table(self):
                return ["a"], np.zeros((1, 2))

            def embed(self, texts):
                return np.zeros((len(texts), 2))

            def token_gradients(self, text, target_class):
                return ["a"], np.zeros((1, 2))

        self.assertFalse(HotFlipGenerator.is_compatible(GradientsOnly()))


if __name__ == "__main__":
    unittest.main()
