"""Unit tests for the shared minimal-perturbation search on ``CounterfactualGenerator``.

Both concrete generators end in the same combination search, so it is tested once here, against a
scripted model that records every ``predict`` call. That record is what lets us assert the two
properties the search claims beyond "it finds flips": it **batches** (one call per chunk, not one per
candidate) and it **prunes** (a superset of a success is never even scored).
"""

import unittest

import numpy as np

from shapash.compute.generators.base import (
    DEFAULT_PREDICT_BATCH_SIZE,
    CounterfactualGenerator,
)
from shapash.model.base import SupportsTokenization, TextModel

_REMOVE = lambda _pos: ""  # noqa: E731 - the removal replacement, named for readability


class ScriptedModel(TextModel, SupportsTokenization):
    """Predicts ``pos`` for everything except the texts the test declares as flipping.

    Records each ``predict`` call as one entry of :attr:`calls`, so tests can assert both how many
    calls the search made and exactly which texts were scored.
    """

    def __init__(self, flipping=(), batch_size=None):
        super().__init__(label_names=["neg", "pos"])
        self.flipping = set(flipping)
        self.calls: list[list[str]] = []
        if batch_size is not None:
            self.batch_size = batch_size

    def predict(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return np.array([[0.9, 0.1] if t in self.flipping else [0.1, 0.9] for t in texts])

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    @property
    def scored_texts(self):
        """Every text passed to ``predict``, flattened across calls."""
        return [t for call in self.calls for t in call]


class SearchGenerator(CounterfactualGenerator):
    """Minimal concrete generator — exists only to reach the ABC's shared search."""

    name = "search_only"

    def config_spec(self):
        return {}

    @classmethod
    def is_compatible(cls, model):
        return True

    def generate(self, text, target_label=None, config=None):
        return []


_TEXT = "a b c d e f"
_TOKENS = _TEXT.split()
_POSITIONS = list(range(len(_TOKENS)))
_POS_PROBS = np.array([0.1, 0.9])


def _search(gen, **kwargs):
    """Run the shared search over the whole of ``_TEXT`` with removal as the perturbation."""
    params = {
        "max_size": 2,
        "num_examples": 5,
        "orig_probs": _POS_PROBS,
        **kwargs,
    }
    return gen.search_minimal(_TEXT, _TOKENS, _POSITIONS, params.pop("replacement", _REMOVE), **params)


class TestPredictBatchSize(unittest.TestCase):
    def test_taken_from_the_model(self):
        # The model owns device knowledge, so a CPU adapter's small batch is respected as-is.
        self.assertEqual(SearchGenerator(ScriptedModel(batch_size=8)).predict_batch_size, 8)

    def test_default_when_the_model_declares_none(self):
        self.assertEqual(SearchGenerator(ScriptedModel()).predict_batch_size, DEFAULT_PREDICT_BATCH_SIZE)

    def test_never_below_one(self):
        # A zero would make the chunk loop score nothing at all; floor it instead of looping forever.
        self.assertEqual(SearchGenerator(ScriptedModel(batch_size=0)).predict_batch_size, 1)


class TestSearchMinimalResults(unittest.TestCase):
    def test_finds_the_declared_flip(self):
        gen = SearchGenerator(ScriptedModel(flipping={"b c d e f"}))
        cfs = _search(gen)
        self.assertEqual(len(cfs), 1)
        cf = cfs[0]
        self.assertEqual(cf.original_text, _TEXT)
        self.assertEqual(cf.new_text, "b c d e f")
        self.assertEqual(cf.flipped_positions, [0])
        self.assertEqual(cf.substitutions, [(0, "a", "")])
        self.assertEqual((cf.orig_label, cf.new_label), ("pos", "neg"))

    def test_substitution_replaces_instead_of_removing(self):
        gen = SearchGenerator(ScriptedModel(flipping={"a b X d e f"}))
        cfs = _search(gen, replacement=lambda _pos: "X")
        self.assertEqual(len(cfs), 1)
        self.assertEqual(cfs[0].new_text, "a b X d e f")
        self.assertEqual(cfs[0].substitutions, [(2, "c", "X")])

    def test_results_ordered_by_increasing_size(self):
        # "b c d e f" (size 1) and "a b c d" (size 2) both flip; the smaller must come first.
        gen = SearchGenerator(ScriptedModel(flipping={"b c d e f", "a b c d"}))
        cfs = _search(gen)
        self.assertEqual([len(cf.flipped_positions) for cf in cfs], [1, 2])

    def test_no_flip_returns_empty(self):
        self.assertEqual(_search(SearchGenerator(ScriptedModel())), [])

    def test_target_class_must_match(self):
        gen = SearchGenerator(ScriptedModel(flipping={"b c d e f"}))
        # Flipping towards "pos" (class 1) is impossible: the text is already pos.
        self.assertEqual(_search(gen, target_class=1), [])
        self.assertEqual(len(_search(gen, target_class=0)), 1)

    def test_requires_a_tokenizable_model(self):
        class PredictOnly(TextModel):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

        # Rebuilding a perturbed text goes through detokenize; fail loudly rather than AttributeError.
        with self.assertRaises(TypeError):
            _search(SearchGenerator(PredictOnly()))


class TestSearchMinimalBatching(unittest.TestCase):
    """The batching is an efficiency claim, so it is asserted on the recorded ``predict`` calls."""

    def test_one_call_per_chunk_not_per_candidate(self):
        gen = SearchGenerator(ScriptedModel(batch_size=DEFAULT_PREDICT_BATCH_SIZE))
        _search(gen, max_size=1)
        # Six candidates at size 1, one batch: one call carrying all six texts.
        self.assertEqual(len(gen.model.calls), 1)
        self.assertEqual(len(gen.model.calls[0]), 6)

    def test_chunks_respect_the_batch_size(self):
        gen = SearchGenerator(ScriptedModel(batch_size=4))
        _search(gen, max_size=1)
        self.assertEqual([len(c) for c in gen.model.calls], [4, 2])

    def test_successful_supersets_are_never_scored(self):
        gen = SearchGenerator(ScriptedModel(flipping={"b c d e f"}, batch_size=DEFAULT_PREDICT_BATCH_SIZE))
        _search(gen)
        # Removing "a" flips, so no size-2 removal that also drops "a" may be evaluated: every text
        # scored after the size-1 batch still contains "a". Pruning is what keeps the search tractable,
        # and batching a whole size level must not weaken it.
        level_two = gen.model.calls[1]
        self.assertTrue(all("a" in t.split() for t in level_two))
        self.assertEqual(len(level_two), 10)  # C(6,2) = 15, minus the 5 containing position 0

    def test_num_examples_stops_within_a_level(self):
        # Every single-token removal flips, so the cap is reachable on the first chunk; the search
        # must not score the remaining candidates of that level just because it batches.
        every_removal = {" ".join(_TOKENS[:i] + _TOKENS[i + 1 :]) for i in _POSITIONS}
        gen = SearchGenerator(ScriptedModel(flipping=every_removal, batch_size=2))
        cfs = _search(gen, num_examples=1)
        self.assertEqual(len(cfs), 1)
        self.assertEqual(len(gen.model.calls), 1)
        self.assertEqual(len(gen.model.scored_texts), 2)

    def test_batching_does_not_change_the_outcome(self):
        # Exactness check: batch of 1 (the old per-combination behaviour) against a full-level batch.
        flipping = {"b c d e f", "a b c d", "a c d e f"}
        one_at_a_time = _search(SearchGenerator(ScriptedModel(flipping=flipping, batch_size=1)))
        batched = _search(SearchGenerator(ScriptedModel(flipping=flipping, batch_size=64)))
        self.assertEqual(one_at_a_time, batched)
        self.assertTrue(batched)


if __name__ == "__main__":
    unittest.main()
