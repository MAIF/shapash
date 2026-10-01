"""Unit tests for the encoded-length diagnostic in ``EncoderClassifierModel``.

The guard pre-empts an out-of-bounds position-id gather (which on CUDA surfaces as an asynchronous
device-side assert, far from its cause). It is pure Python — no torch, transformers or GPU — so it is
exercised here with fake backbones/tokenizers via ``object.__new__``, skipping the real ``__init__``.

The central property under test is that it **never false-positives**: it must not reject a sequence
the model would have accepted, since it raises rather than truncating.
"""

import unittest
from types import SimpleNamespace

from shapash.model.encoder import EncoderClassifierModel, _encoded_length, _position_capacity


class FakeTokenizer:
    """Returns a canned encoding; records the kwargs it was called with."""

    def __init__(self, encoding, model_max_length=512):
        self.encoding = encoding
        self.model_max_length = model_max_length
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        return self.encoding


def _backbone(max_position_embeddings=512, **config_extra):
    config = SimpleNamespace(max_position_embeddings=max_position_embeddings, **config_extra)
    return SimpleNamespace(config=config)


def _model(backbone, tokenizer, max_length=None):
    """Build the adapter without running ``__init__`` (which needs a real torch backbone)."""
    model = object.__new__(EncoderClassifierModel)
    model.backbone = backbone
    model.tokenizer = tokenizer
    model.max_length = max_length
    return model


class TestEncodedLength(unittest.TestCase):
    """``_encoded_length`` must read every shape ``_tokenize`` can return."""

    def test_reads_tensor_like_shape(self):
        self.assertEqual(_encoded_length({"input_ids": SimpleNamespace(shape=(1, 1092))}), 1092)

    def test_reads_batched_tensor_shape(self):
        self.assertEqual(_encoded_length({"input_ids": SimpleNamespace(shape=(8, 256))}), 256)

    def test_reads_single_id_list(self):
        self.assertEqual(_encoded_length({"input_ids": [5, 6, 7]}), 3)

    def test_reads_batched_id_lists_as_the_longest_row(self):
        self.assertEqual(_encoded_length({"input_ids": [[1, 2], [1, 2, 3, 4]]}), 4)

    def test_empty_encoding_is_zero(self):
        self.assertEqual(_encoded_length({"input_ids": []}), 0)


class TestPositionCapacity(unittest.TestCase):
    """``_position_capacity`` returns ``None`` (meaning "do not check") whenever it is unsure."""

    def test_reads_max_position_embeddings(self):
        self.assertEqual(_position_capacity(_backbone(1025)), 1025)

    def test_none_when_config_absent(self):
        self.assertIsNone(_position_capacity(SimpleNamespace()))

    def test_none_when_max_position_embeddings_absent(self):
        # A rotary/ALiBi model has no absolute position table.
        self.assertIsNone(_position_capacity(SimpleNamespace(config=SimpleNamespace())))

    def test_none_for_deberta_style_relative_positions(self):
        # DeBERTa v1/v2/v3 ship position_biased_input=False and add no absolute position embeddings,
        # so their max_position_embeddings is not an indexing bound — checking it would false-positive.
        self.assertIsNone(_position_capacity(_backbone(512, position_biased_input=False)))

    def test_applies_when_position_biased_input_is_true(self):
        self.assertEqual(_position_capacity(_backbone(1025, position_biased_input=True)), 1025)

    def test_none_for_nonsense_values(self):
        for bad in (0, -5, None, "512", True):
            self.assertIsNone(_position_capacity(_backbone(bad)), f"capacity={bad!r}")


class TestTokenizeLengthGuard(unittest.TestCase):
    def test_raises_when_sequence_exceeds_capacity(self):
        # The camembertv2 case: 1092 tokens against a 1025-wide position buffer.
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 1092))}, model_max_length=10**30)
        model = _model(_backbone(1025), tokenizer)
        with self.assertRaises(ValueError) as ctx:
            model._tokenize("a very long review")
        message = str(ctx.exception)
        self.assertIn("1092", message)
        self.assertIn("1025", message)
        self.assertIn("max_length", message)  # tells the caller what to do

    def test_allows_sequence_exactly_at_capacity(self):
        # Boundary: capacity positions are indices 0..capacity-1 for an arange scheme, so a sequence of
        # exactly `capacity` tokens fits. Rejecting it would be a false positive.
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 512))})
        model = _model(_backbone(512), tokenizer)
        model._tokenize("text")  # must not raise

    def test_allows_short_sequence(self):
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 7))})
        model = _model(_backbone(512), tokenizer)
        model._tokenize("text")

    def test_does_not_fire_for_deberta_style_model(self):
        # Relative-position model: a longer-than-mpe sequence must pass through untouched.
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 900))})
        model = _model(_backbone(512, position_biased_input=False), tokenizer)
        model._tokenize("text")

    def test_does_not_fire_when_capacity_unknown(self):
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 9999))})
        model = _model(SimpleNamespace(config=SimpleNamespace()), tokenizer)
        model._tokenize("text")

    def test_explicit_max_length_is_still_forwarded_to_the_tokenizer(self):
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 512))})
        model = _model(_backbone(1025), tokenizer, max_length=512)
        model._tokenize("text")
        self.assertEqual(tokenizer.calls[-1]["max_length"], 512)
        self.assertTrue(tokenizer.calls[-1]["truncation"])

    def test_no_max_length_sends_truncation_only(self):
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 8))})
        model = _model(_backbone(512), tokenizer)
        model._tokenize("text")
        self.assertNotIn("max_length", tokenizer.calls[-1])

    def test_message_surfaces_the_sentinel_model_max_length(self):
        # The actionable root cause: a tokenizer reporting the ~1e30 "unset" sentinel makes
        # truncation=True a silent no-op, which is how an unbounded sequence gets here at all.
        sentinel = 1000000000000000019884624838656
        tokenizer = FakeTokenizer({"input_ids": SimpleNamespace(shape=(1, 2000))}, model_max_length=sentinel)
        model = _model(_backbone(1025), tokenizer)
        with self.assertRaises(ValueError) as ctx:
            model._tokenize("text")
        self.assertIn("model_max_length", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
