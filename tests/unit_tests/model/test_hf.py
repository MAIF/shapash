"""Unit tests for ``HFPipelineModel``'s label-order resolution.

Uses a fake ``pipeline`` callable (plain function, no ``transformers``/``torch``) so these stay fast
unit tests: only the label-name/column-order contract of ``predict`` is under test here.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.nlp

from torch import nn  # noqa: E402

from shapash.model.hf import HFClassifierModel, HFPipelineModel  # noqa: E402


class FakePipeline:
    """Mimics a HuggingFace ``text-classification`` pipeline with ``top_k=None``.

    Returns, per text, a list of ``{"label", "score"}`` dicts in a fixed (non-alphabetical,
    non-id) order to make sure column alignment is name-based, not position-based.
    """

    def __init__(self, rows: list[list[dict]]) -> None:
        self.rows = rows
        self.tokenizer = None

    def __call__(self, texts: list[str], top_k=None):
        assert len(texts) <= len(self.rows)
        return self.rows[: len(texts)]


def _rows(*label_score_pairs_per_row):
    return [[{"label": label, "score": score} for label, score in row] for row in label_score_pairs_per_row]


class TestHFPipelineModelLabelOrder(unittest.TestCase):
    def test_infers_order_and_label_names_from_pipeline_when_none_given(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline)

        probs = model.predict(["great movie"])

        self.assertEqual(model.label_names, ["negative", "positive"])
        self.assertEqual(list(probs[0]), [0.1, 0.9])

    def test_provided_label_names_become_the_column_order(self):
        # Pipeline's own row order is "negative" first; the caller asserts the opposite order.
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["positive", "negative"])

        probs = model.predict(["great movie"])

        self.assertEqual(model.label_names, ["positive", "negative"])
        self.assertEqual(list(probs[0]), [0.9, 0.1])

    def test_order_and_label_names_stay_in_sync_after_predict(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["positive", "negative"])

        model.predict(["great movie"])

        self.assertEqual(model._order, model.label_names)

    def test_mismatched_label_names_raise_instead_of_silently_diverging(self):
        pipeline = FakePipeline(_rows([("negative", 0.1), ("positive", 0.9)]))
        model = HFPipelineModel(pipeline, label_names=["negative", "possitive"])  # typo

        with self.assertRaises(ValueError):
            model.predict(["great movie"])

    def test_label_names_proper_subset_raises(self):
        # Pipeline emits 3 classes; caller only declares 2 — must not silently drop the third.
        pipeline = FakePipeline(_rows([("negative", 0.1), ("neutral", 0.2), ("positive", 0.7)]))
        model = HFPipelineModel(pipeline, label_names=["negative", "positive"])

        with self.assertRaises(ValueError):
            model.predict(["great movie"])

    def test_label_names_valid_regardless_of_first_row_score_sort_order(self):
        # Different rows may come back sorted by score (per-row), not by a stable id order; the
        # caller's declared order must still win via name-based lookup.
        pipeline = FakePipeline(
            _rows(
                [("positive", 0.9), ("negative", 0.1)],
                [("negative", 0.8), ("positive", 0.2)],
            )
        )
        model = HFPipelineModel(pipeline, label_names=["negative", "positive"])

        probs = model.predict(["good", "bad"])

        self.assertEqual(list(probs[0]), [0.1, 0.9])
        self.assertEqual(list(probs[1]), [0.8, 0.2])


class _FakeTokenizer:
    """Minimal tokenizer: maps each word to a small id, pads a batch to its longest sequence."""

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        seqs = [[(hash(w) % 8) + 1 for w in t.split()] for t in texts]
        width = max(len(s) for s in seqs)
        input_ids, attention_mask = [], []
        for s in seqs:
            pad = width - len(s)
            input_ids.append(s + [0] * pad)
            attention_mask.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _TinyClassifier(nn.Module):
    """A tiny sequence-classifier exposing ``pre_classifier`` (pooled) and ``emb`` (token-level) modules."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 4)
        self.pre_classifier = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        h = self.emb(input_ids)  # (B, S, 4) — a token-level layer output
        pooled = torch.relu(self.pre_classifier(h.mean(dim=1)))  # (B, 4) — a pooled layer output
        out = SimpleNamespace(logits=self.classifier(pooled))
        if output_hidden_states:
            out.hidden_states = (h,)
        return out


class TestHFClassifierEmbedSpaces(unittest.TestCase):
    """``embed`` is the single entry point for every representation space."""

    def setUp(self):
        self.model = HFClassifierModel(_TinyClassifier(), _FakeTokenizer(), label_names=["neg", "pos"], batch_size=2)

    def test_decision_is_the_default_space(self):
        self.assertEqual(self.model.embedding_space, "decision")
        # Decision space = input to the final Linear(4, 2), i.e. 4 features, one vector per text.
        self.assertEqual(self.model.embed(["hello world", "a b c", "single"]).shape, (3, 4))

    def test_pooled_space(self):
        self.assertEqual(self.model.embed(["hello world", "x"], "pooled").shape, (2, 4))

    def test_named_submodule_space_is_pooled_when_token_level(self):
        # The embedding output is (B, S, 4); the hook path must mask-mean-pool it to (B, 4).
        self.assertEqual(self.model.embed(["hello world", "x"], "emb").shape, (2, 4))

    def test_masking_makes_embedding_padding_invariant(self):
        # A text embedded alone vs. batched with a longer text (so it gets padded) must match,
        # proving padding tokens are excluded via the attention mask.
        alone = self.model.embed(["hi"], "emb")
        batched = self.model.embed(["hi", "one two three four"], "emb")
        np.testing.assert_allclose(alone[0], batched[0], rtol=1e-5, atol=1e-6)

    def test_unknown_space_raises_valueerror_eagerly(self):
        # At construction: a typo must not wait for the first forward pass to surface.
        with self.assertRaises(ValueError):
            HFClassifierModel(_TinyClassifier(), _FakeTokenizer(), label_names=["neg", "pos"], embedding_space="nope")
        # And on an explicit per-call space.
        with self.assertRaises(ValueError):
            self.model.embed(["hello"], "does_not_exist")

    def test_hook_removed_after_call(self):
        before = len(self.model.classifier.pre_classifier._forward_hooks)
        self.model.embed(["hello world"], "pre_classifier")
        self.assertEqual(len(self.model.classifier.pre_classifier._forward_hooks), before)

    def test_model_id_tracks_configuration(self):
        # Two models differing only in pooling must not share a cache key.
        other = HFClassifierModel(
            _TinyClassifier(), _FakeTokenizer(), label_names=["neg", "pos"], batch_size=2, pool="cls"
        )
        self.assertNotEqual(self.model.model_id, other.model_id)


class _PoolerClassifier(nn.Module):
    """A classifier with a top-level ``pooler`` (BERT/DeBERTa shape) but no ``pre_classifier``."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 4)
        self.pooler = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        h = self.emb(input_ids)  # (B, S, 4)
        pooled = torch.relu(self.pooler(h.mean(dim=1)))
        out = SimpleNamespace(logits=self.classifier(pooled))
        if output_hidden_states:
            out.hidden_states = (h,)
        return out


class _NoPooledClassifier(nn.Module):
    """A classifier with neither ``pre_classifier`` nor ``pooler`` (RoBERTa/XLM-R shape).

    Its ``forward`` honours ``output_hidden_states`` so the mean-pooled-last-hidden-state fallback in
    ``activations`` (via ``embed``) has hidden states to pool.
    """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 4)
        self.classifier = nn.Linear(4, 2)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        h = self.emb(input_ids)  # (B, S, 4)
        out = SimpleNamespace(logits=self.classifier(h.mean(dim=1)))
        if output_hidden_states:
            out.hidden_states = (h,)
        return out


class TestEmbedSpacesAcrossArchitectures(unittest.TestCase):
    """Both keyword spaces work on every head shape — no per-architecture configuration."""

    @staticmethod
    def _model(classifier, **kwargs):
        return HFClassifierModel(classifier, _FakeTokenizer(), label_names=["neg", "pos"], batch_size=2, **kwargs)

    def test_decision_space_on_pre_classifier_head(self):
        self.assertEqual(self._model(_TinyClassifier()).embed(["hello world", "x"]).shape, (2, 4))

    def test_decision_space_on_pooler_head(self):
        self.assertEqual(self._model(_PoolerClassifier()).embed(["hello world", "x"]).shape, (2, 4))

    def test_decision_space_on_headless_architecture(self):
        # No pre_classifier/pooler (RoBERTa/XLM-R shape): the final Linear is still resolvable.
        self.assertEqual(self._model(_NoPooledClassifier()).embed(["hello world", "a b c"]).shape, (2, 4))

    def test_pooled_space_on_every_architecture(self):
        for classifier in (_TinyClassifier(), _PoolerClassifier(), _NoPooledClassifier()):
            with self.subTest(classifier=type(classifier).__name__):
                self.assertEqual(self._model(classifier).embed(["hello world", "x"], "pooled").shape, (2, 4))

    def test_decision_space_does_not_require_label_names(self):
        # The final Linear is located structurally (the head module -> its last Linear), so label_names —
        # a display concern — must not decide which representation a caller gets.
        model = HFClassifierModel(_NoPooledClassifier(), _FakeTokenizer(), label_names=None, batch_size=2)
        self.assertIs(model._resolve_decision_linear(), model.classifier.classifier)
        self.assertEqual(model.embed(["hello world", "a b c"]).shape, (2, 4))

    def test_decision_space_falls_back_audibly_when_no_head_is_found(self):
        # A backbone naming its head outside the known conventions leaves nothing to read a decision
        # space off; the caller asked for "decision" and gets "pooled", so this must be audible.
        model = HFClassifierModel(_OddlyNamedHeadClassifier(), _FakeTokenizer(), batch_size=2)
        self.assertIsNone(model._resolve_decision_linear())
        # pytest.warns, not unittest's assertWarns: the latter walks sys.modules reading
        # __warningregistry__, which blows up on any lazily-imported module (transformers) that a
        # sibling test suite may have loaded into the same session.
        with pytest.warns(UserWarning):
            self.assertEqual(model.embed(["hello world", "a b c"]).shape, (2, 4))

    def test_odd_head_name_is_recoverable_by_overriding_head_module(self):
        # The documented escape hatch for an unconventional layout: two lines on the adapter, and the
        # decision space works again — no change to the base class.
        class _Adapter(HFClassifierModel):
            def _head_module(self):
                return self.backbone.decision_layer

        model = _Adapter(_OddlyNamedHeadClassifier(), _FakeTokenizer(), batch_size=2)
        self.assertIs(model._resolve_decision_linear(), model.backbone.decision_layer)


class _OddlyNamedHeadClassifier(nn.Module):
    """A classifier whose head is named outside the ``head``/``classifier``/``score`` conventions.

    Exercises the structural resolution's miss path: no head can be located, so the decision space must
    degrade audibly to pooled rather than silently returning the wrong representation.
    """

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(9, 4)
        self.decision_layer = nn.Linear(4, 2)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        h = self.emb(input_ids)
        out = SimpleNamespace(logits=self.decision_layer(h.mean(dim=1)))
        if output_hidden_states:
            out.hidden_states = (h,)
        return out


class _Enc(dict):
    """Minimal fast-tokenizer encoding: a dict plus a ``word_ids()`` accessor."""

    def __init__(self, input_ids, word_ids):
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class _FakeFastTokenizer:
    """Fast tokenizer stub exposing ``word_ids()`` for a fixed byte-BPE encoding of any text."""

    is_fast = True
    _tokens = ["<s>", "im", "Ġfeel", "ing", "</s>"]
    _word_ids = [None, 0, 1, 1, None]

    def __call__(self, text, truncation=True):
        return _Enc(list(range(len(self._tokens))), list(self._word_ids))

    def convert_ids_to_tokens(self, ids):
        return [self._tokens[i] for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).replace("Ġ", " ")  # byte-BPE: the marker denotes a space


class TestWordAlignment(unittest.TestCase):
    def test_groups_subwords_and_flags_specials(self):
        model = HFClassifierModel(_TinyClassifier(), _FakeFastTokenizer(), label_names=["neg", "pos"])
        words, word_positions, special_positions = model.word_alignment("anything")
        self.assertEqual(words, ["im", "feeling"])  # "Ġfeel" + "ing" -> "feeling"
        self.assertEqual(word_positions, [[1], [2, 3]])
        self.assertEqual(special_positions, [0, 4])  # <s>, </s>

    def test_returns_none_for_slow_tokenizer(self):
        # _FakeTokenizer has no is_fast/word_ids -> caller falls back to the string heuristic.
        model = HFClassifierModel(_TinyClassifier(), _FakeTokenizer(), label_names=["neg", "pos"])
        self.assertIsNone(model.word_alignment("anything"))


if __name__ == "__main__":
    unittest.main()
