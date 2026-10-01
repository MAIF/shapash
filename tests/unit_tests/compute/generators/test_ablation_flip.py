"""Unit tests for the AblationFlip counterfactual generator.

The Captum scoring step (:meth:`AblationFlipGenerator._ablation_scores`) is stubbed with an exact,
captum-free re-implementation (measure the drop in the original class when a token is removed) so the
greedy-flip / minimality logic is exercised without torch or captum. A deterministic bag-of-words fake
model provides predictions. The real Captum path is covered by the integration tests.
"""

import unittest

import numpy as np

from shapash.compute.generators import AblationFlipGenerator, Counterfactual
from shapash.compute.generators.cf_utils import is_prediction_flip
from shapash.model.base import SupportsTokenization, TextModel, is_word_token

# Per-token [neg_logit, pos_logit] contributions; logits = sum over content tokens.
_EMB = {
    "this": [0.1, 0.1],
    "is": [0.1, 0.1],
    "great": [0.0, 2.0],
    "good": [0.0, 1.0],
    "super": [0.0, 2.0],
    "meh": [3.0, 0.0],
    "terrible": [3.0, 0.0],
}


class BagOfWordsModel(TextModel, SupportsTokenization):
    """Prediction-only (plus tokenization) linear bag-of-words classifier."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

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


class SubwordModel(BagOfWordsModel):
    """WordPiece-style fake: some words split into pieces, so a word *head* is bare but not a word.

    ``BagOfWordsModel`` splits on whitespace, which can never reproduce the multi-piece case that made
    both generators emit malformed text (``"grouchy"`` → ``["gr", "##ou", "##chy"]``, flipped to
    ``"superouchy"``).
    """

    _PIECES = {"terrible": ["terr", "##ible"], "grouchy": ["gr", "##ou", "##chy"]}

    def tokenize(self, text):
        pieces = []
        for word in text.split():
            pieces.extend(self._PIECES.get(word, [word]))
        return pieces

    def detokenize(self, tokens):
        out = ""
        for token in tokens:
            if token.startswith("##"):
                out += token[2:]
            else:
                out = f"{out} {token}" if out else token
        return out


def _exact_ablation_scores(self, tokens, content_positions, orig_class):
    """Captum-free stand-in for ``_ablation_scores``: exact drop in ``orig_class`` per removed token."""
    base = self.model.predict([self.model.detokenize(list(tokens))])[0][orig_class]
    scores = []
    for p in content_positions:
        remaining = [t for i, t in enumerate(tokens) if i != p]
        scores.append(base - self.model.predict([self.model.detokenize(remaining)])[0][orig_class])
    return np.array(scores)


def _make_generator():
    gen = AblationFlipGenerator(BagOfWordsModel())
    gen._ablation_scores = _exact_ablation_scores.__get__(gen, AblationFlipGenerator)
    return gen


class TestIsWordToken(unittest.TestCase):
    def test_accepts_words_rejects_subwords_and_specials(self):
        self.assertTrue(is_word_token("happy"))
        self.assertFalse(is_word_token("##ing"))
        self.assertFalse(is_word_token("[CLS]"))
        self.assertFalse(is_word_token("1b"))
        self.assertFalse(is_word_token("!"))


class TestAblationFlipCompat(unittest.TestCase):
    def test_compatible_with_tokenizable_predict_only_model(self):
        self.assertTrue(AblationFlipGenerator.is_compatible(BagOfWordsModel()))

    def test_incompatible_without_tokenization(self):
        class PredictOnly(TextModel):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

        self.assertFalse(AblationFlipGenerator.is_compatible(PredictOnly()))


class TestAblationFlipGenerate(unittest.TestCase):
    def setUp(self):
        self.gen = _make_generator()

    def test_config_spec_defaults(self):
        spec = self.gen.config_spec()
        self.assertEqual(spec["num_examples"].default, 5)
        self.assertEqual(spec["max_ablations"].default, 3)
        self.assertEqual(self.gen.resolve_config({"max_ablations": 2})["max_ablations"], 2)

    def test_generates_flip_by_removing_supportive_token(self):
        cfs = self.gen.generate("this is great", config={"num_examples": 3, "max_ablations": 1})
        self.assertTrue(cfs, "expected at least one counterfactual")
        cf = cfs[0]
        self.assertIsInstance(cf, Counterfactual)
        self.assertEqual(cf.orig_label, "pos")
        self.assertEqual(cf.new_label, "neg")
        # The removed token is the strongly-positive one; removals record an empty replacement.
        self.assertTrue(all(new == "" for _, _, new in cf.substitutions))
        self.assertIn("great", [old for _, old, _ in cf.substitutions])
        self.assertNotIn("great", cf.new_text.split())
        # The reported flip actually holds under the model.
        self.assertTrue(
            is_prediction_flip(self.gen.model.predict([cf.original_text])[0], self.gen.model.predict([cf.new_text])[0])
        )

    def test_minimality(self):
        cfs = self.gen.generate("this is great good", config={"num_examples": 10, "max_ablations": 3})
        sets = [frozenset(cf.flipped_positions) for cf in cfs]
        for i, a in enumerate(sets):
            for j, b in enumerate(sets):
                if i != j:
                    self.assertFalse(a < b, "returned a non-minimal (superset) counterfactual")

    def test_respects_num_examples_cap(self):
        # "meh great super": removing either strong-positive token alone flips, so two size-1
        # counterfactuals exist; the cap stops generation after the first.
        capped = self.gen.generate("meh great super", config={"num_examples": 1, "max_ablations": 1})
        self.assertEqual(len(capped), 1)
        uncapped = self.gen.generate("meh great super", config={"num_examples": 5, "max_ablations": 1})
        self.assertEqual(len(uncapped), 2)

    def test_targeted_flip(self):
        cfs = self.gen.generate("this is great", target_label="neg", config={"max_ablations": 2})
        self.assertTrue(cfs)
        self.assertTrue(all(cf.new_label == "neg" for cf in cfs))

    def test_tokens_to_ignore_blocks_flip(self):
        cfs = self.gen.generate("this is great", config={"tokens_to_ignore": ["great", "good"], "max_ablations": 2})
        self.assertEqual(cfs, [])

    def test_no_content_tokens_returns_empty(self):
        cfs = self.gen.generate("! ? .", config={"max_ablations": 2})
        self.assertEqual(cfs, [])


class SentencePieceModel(BagOfWordsModel):
    """The same classifier, tokenized SentencePiece-style: ``▁`` marks every word *start*.

    Used by DeBERTa-v2/v3, XLM-R, T5 and most multilingual checkpoints.
    """

    def tokenize(self, text):
        return [f"▁{w}" for w in text.split()]

    def detokenize(self, tokens):
        return " ".join(t.removeprefix("▁") for t in tokens)


class CountingBagOfWordsModel(BagOfWordsModel):
    """Bag-of-words model recording each ``predict`` call, for asserting how the search batches."""

    batch_size = 32

    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def predict(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return super().predict(texts)


class TestAblationScores(unittest.TestCase):
    """The scorer is plain leave-one-out now, so it is testable directly — captum is not involved."""

    def setUp(self):
        self.model = CountingBagOfWordsModel()
        self.gen = AblationFlipGenerator(self.model)

    def test_matches_exact_leave_one_out(self):
        tokens = self.model.tokenize("this is great good")
        positions = list(range(len(tokens)))
        expected = _exact_ablation_scores(self.gen, tokens, positions, orig_class=1)
        np.testing.assert_allclose(self.gen._ablation_scores(tokens, positions, orig_class=1), expected)

    def test_supportive_tokens_score_higher(self):
        tokens = self.model.tokenize("this is great")
        scores = self.gen._ablation_scores(tokens, [0, 1, 2], orig_class=1)
        # Removing "great" costs the positive class far more than removing "this".
        self.assertGreater(scores[2], scores[0])

    def test_scores_every_token_in_one_predict_call(self):
        tokens = self.model.tokenize("this is great good")
        self.gen._ablation_scores(tokens, [0, 1, 2, 3], orig_class=1)
        # One call carrying the baseline plus one perturbation per token, not one call per token.
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(len(self.model.calls[0]), 5)

    def test_generate_works_without_stubbing_the_scorer(self):
        # End-to-end on a plain predict-only model: the whole generator now runs on numpy alone.
        cfs = self.gen.generate("this is great", config={"num_examples": 3, "max_ablations": 2})
        self.assertTrue(cfs)
        self.assertEqual(cfs[0].new_label, "neg")
        # Scoring (1 call) plus one call per combination *chunk* — a handful, not one call per combo.
        self.assertLessEqual(len(self.model.calls), 4)


class TestAblationFlipOnMarkedTokenizer(unittest.TestCase):
    """Regression: a word-start-marking tokenizer used to yield zero removal candidates.

    Content tokens are ``"▁great"``, which is not alphabetic, so the old bare ``isalpha`` word test
    rejected all of them — ``content_positions`` came back empty and ``generate`` returned ``[]`` for
    *every* input. Nothing raised; the webapp just reported "No counterfactual found".
    """

    def setUp(self):
        self.gen = AblationFlipGenerator(SentencePieceModel())
        self.gen._ablation_scores = _exact_ablation_scores.__get__(self.gen, AblationFlipGenerator)

    def test_marked_tokens_are_removal_candidates(self):
        model = self.gen.model
        tokens = model.tokenize("this is great")
        self.assertEqual(tokens, ["▁this", "▁is", "▁great"])
        self.assertTrue(all(model.is_substitutable(t) for t in tokens))

    def test_generates_flip(self):
        cfs = self.gen.generate("this is great", config={"num_examples": 3, "max_ablations": 1})
        self.assertTrue(cfs, "no counterfactual on a SentencePiece tokenizer — the C2 regression")
        cf = cfs[0]
        self.assertEqual((cf.orig_label, cf.new_label), ("pos", "neg"))
        self.assertIn("▁great", [old for _, old, _ in cf.substitutions])
        # The rebuilt text is detokenized cleanly (no stray markers) and the flip really holds.
        self.assertNotIn("▁", cf.new_text)
        self.assertTrue(
            is_prediction_flip(self.gen.model.predict([cf.original_text])[0], self.gen.model.predict([cf.new_text])[0])
        )

    def test_tokens_to_ignore_matches_the_bare_word_a_user_types(self):
        # ``tokens_to_ignore`` is typed into the webapp, so it holds plain words. Comparing them
        # against raw tokens means "great" never matches "▁great" and the ignore list silently does
        # nothing; the comparison goes through the token's display form instead.
        cfs = self.gen.generate("this is great", config={"tokens_to_ignore": ["great"], "max_ablations": 1})
        self.assertEqual(cfs, [])


class TestCandidatePositionCap(unittest.TestCase):
    """The cap is one named attribute on the ABC, overridable without patching module globals.

    It was two module constants (``_MAX_ABLATABLE_TOKENS`` / ``_MAX_FLIPPABLE_TOKENS``) holding the
    same value for the same purpose in two files, reachable only by monkeypatching.
    """

    def test_default_is_shared_by_both_generators(self):
        from shapash.compute.generators import HotFlipGenerator

        self.assertEqual(AblationFlipGenerator.max_candidate_positions, 10)
        self.assertEqual(HotFlipGenerator.max_candidate_positions, 10)

    def test_instance_override_limits_the_positions_searched(self):
        gen = _make_generator()
        gen.max_candidate_positions = 1
        # Only the single best removal is offered, so a flip needing any other token cannot be found.
        cfs = gen.generate("this is great and good", config={"num_examples": 5, "max_ablations": 2})
        for cf in cfs:
            self.assertEqual(len(cf.flipped_positions), 1)

    def test_override_does_not_leak_to_other_instances(self):
        gen = _make_generator()
        gen.max_candidate_positions = 2
        self.assertEqual(_make_generator().max_candidate_positions, 10)


class TestMultiPieceWordsAreNotStranded(unittest.TestCase):
    """Dropping the *head* of a multi-piece word strands its continuations into a fragment.

    ``"terrible"`` tokenizes to ``["terr", "##ible"]``. Removing ``"terr"`` leaves ``"ible"`` — a word
    the model never saw, which reads to the search as a large probability drop and so gets selected
    eagerly. The position must be excluded, even at the cost of finding no counterfactual at all.
    """

    def setUp(self):
        self.gen = AblationFlipGenerator(SubwordModel())
        self.gen._ablation_scores = _exact_ablation_scores.__get__(self.gen, AblationFlipGenerator)

    def test_word_head_is_not_an_ablation_candidate(self):
        model = self.gen.model
        tokens = model.tokenize("this is terrible")
        self.assertEqual(tokens, ["this", "is", "terr", "##ible"])
        self.assertTrue(model.is_substitutable(tokens[2]))  # token-level accepts the bare head
        self.assertFalse(model.is_substitutable_at(tokens, 2))  # the position does not

    def test_never_returns_a_stranded_fragment(self):
        # Pre-fix, "terr" was removable and this returned "this is ible but good" as a valid flip.
        cfs = self.gen.generate("this is terrible but good", config={"num_examples": 5, "max_ablations": 2})
        for cf in cfs:
            self.assertNotIn("ible", cf.new_text)
            self.assertNotIn("ouchy", cf.new_text)

    def test_prefers_no_counterfactual_over_a_malformed_one(self):
        # The only flip-enabling removal is the multi-piece word, which cannot be removed piecewise.
        self.assertEqual(self.gen.generate("this is terrible but good", config={"max_ablations": 1}), [])


if __name__ == "__main__":
    unittest.main()
