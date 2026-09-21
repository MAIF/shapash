"""Unit tests for the external-head presets and the shared encoder spine.

These exercise ``EncoderClassifierModel`` via ``TorchClassifierModel`` (a body + pool + head fused into
a backbone) with tiny torch modules — no ``transformers``/``sentence_transformers`` needed. They lock
in that the fused backbone honours the backbone contract and that every capability comes out of the
shared spine unchanged.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.nlp

from torch import nn  # noqa: E402

from shapash.model import (  # noqa: E402
    SupportsCaptumIG,
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TorchClassifierModel,
    has_capabilities,
)
from shapash.model.encoder import EncoderClassifierModel, _pool_hidden  # noqa: E402
from shapash.model.torch_models import build_encoder_head_backbone  # noqa: E402


class _FakeTokenizer:
    """Minimal tokenizer: word -> small id, batch-padded; single-text path returns a batch of one.

    Honours ``max_length`` (and records the calls) so tests can assert the model threads its truncation
    settings through *every* tokenization path rather than only the batched one.
    """

    is_fast = False
    model_max_length = 512

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", max_length=None):
        self.calls.append({"truncation": truncation, "max_length": max_length})
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        seqs = [[(hash(w) % 8) + 1 for w in t.split()] for t in batch]
        if truncation and max_length is not None:
            seqs = [s[:max_length] for s in seqs]
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

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
        return [0] * len(ids)  # this fake has no special tokens


class _Body(nn.Module):
    """A HF-``AutoModel``-like encoder: returns ``.last_hidden_state`` and accepts ``inputs_embeds``."""

    def __init__(self, vocab=9, hidden=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        hidden = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(last_hidden_state=hidden)


def _model(pool="mean"):
    return TorchClassifierModel(_Body(), nn.Linear(4, 2), _FakeTokenizer(), label_names=["neg", "pos"], pool=pool)


class TestPoolHidden(unittest.TestCase):
    """``_pool_hidden`` is pure torch — mask-aware mean/max, cls, and callable."""

    def setUp(self):
        # Two texts, seq len 3, hidden 2; second position of row 2 is padding.
        self.hidden = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]], [[2.0, 0.0], [4.0, 8.0], [0.0, 0.0]]]
        )
        self.mask = torch.tensor([[1, 1, 1], [1, 1, 0]])

    def test_mean_is_mask_aware(self):
        pooled = _pool_hidden(self.hidden, self.mask, "mean")
        np.testing.assert_allclose(pooled.numpy(), [[3.0, 3.0], [3.0, 4.0]])  # row2 ignores padding

    def test_cls_takes_first_position(self):
        pooled = _pool_hidden(self.hidden, self.mask, "cls")
        np.testing.assert_allclose(pooled.numpy(), [[1.0, 1.0], [2.0, 0.0]])

    def test_max_ignores_padding(self):
        pooled = _pool_hidden(self.hidden, self.mask, "max")
        np.testing.assert_allclose(pooled.numpy(), [[5.0, 5.0], [4.0, 8.0]])  # padded row2 pos excluded

    def test_callable_pool_is_used(self):
        pooled = _pool_hidden(self.hidden, self.mask, lambda h, m: h.sum(dim=1))
        np.testing.assert_allclose(pooled.numpy(), [[9.0, 9.0], [6.0, 8.0]])


class TestFusedBackbone(unittest.TestCase):
    """``build_encoder_head_backbone`` honours the backbone contract used by EncoderClassifierModel."""

    def setUp(self):
        self.backbone = build_encoder_head_backbone(_Body(), nn.Linear(4, 2), "mean")

    def test_logits_shape_from_ids(self):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.long)
        self.assertEqual(self.backbone(input_ids=ids, attention_mask=mask).logits.shape, (1, 2))

    def test_hidden_states_exposed_on_request(self):
        ids = torch.tensor([[1, 2, 3]])
        out = self.backbone(input_ids=ids, attention_mask=torch.ones(1, 3, dtype=torch.long), output_hidden_states=True)
        self.assertEqual(out.hidden_states[-1].shape, (1, 3, 4))

    def test_inputs_embeds_path_matches_ids_path(self):
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.long)
        via_ids = self.backbone(input_ids=ids, attention_mask=mask).logits
        embeds = self.backbone.get_input_embeddings()(ids)
        via_embeds = self.backbone(inputs_embeds=embeds, attention_mask=mask).logits
        np.testing.assert_allclose(via_ids.detach().numpy(), via_embeds.detach().numpy(), rtol=1e-6)

    def test_get_input_embeddings_is_body_embedding(self):
        self.assertIsInstance(self.backbone.get_input_embeddings(), nn.Embedding)


class TestTorchClassifierModelCapabilities(unittest.TestCase):
    def test_declares_full_capability_surface(self):
        model = _model()
        self.assertTrue(
            has_capabilities(
                model,
                SupportsTokenization,
                SupportsEmbeddings,
                SupportsGradients,
                SupportsCaptumIG,
            )
        )

    def test_predict_shape_and_normalisation(self):
        probs = _model().predict(["hello world", "a b c"])
        self.assertEqual(probs.shape, (2, 2))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=1e-6)

    def test_embed_shape(self):
        self.assertEqual(_model().embed(["hello world", "x"]).shape, (2, 4))

    def test_pooled_space_shape(self):
        # The fused backbone exposes hidden_states, so the universal "pooled" space works here too.
        self.assertEqual(_model().embed(["hello world", "a b c"], "pooled").shape, (2, 4))

    def test_model_id_includes_head_weights(self):
        # Same body + pooling, different head weights => different cache identity. Keying on the class
        # name (or on the body alone) would make these two silently share a cached bank.
        a, b = _model(), _model()
        with torch.no_grad():
            b.head.weight.add_(1.0)
        self.assertNotEqual(a.model_id, b.model_id)

    def test_token_gradients_return_aligned_rows(self):
        tokens, grads = _model().token_gradients("hello world", target_class=1)
        self.assertEqual(len(tokens), grads.shape[0])
        self.assertEqual(grads.shape[1], 4)  # hidden dim

    def test_shap_callable_is_predict_for_external_head(self):
        model = _model()
        # No transformers pipeline (backbone isn't a PreTrainedModel): SHAP wraps the plain scorer.
        # (The companion shap_masker over a real HF tokenizer is covered by the integration tests.)
        self.assertEqual(model.shap_callable, model.predict)


class TestNormalize(unittest.TestCase):
    """``normalize=True`` L2-normalizes the pooled vector in embed *and* before the head."""

    def test_embed_rows_are_unit_norm(self):
        model = TorchClassifierModel(
            _Body(), nn.Linear(4, 2), _FakeTokenizer(), label_names=["neg", "pos"], pool="mean", normalize=True
        )
        emb = model.embed(["hello world", "a b c d"])
        np.testing.assert_allclose(np.linalg.norm(emb, axis=1), [1.0, 1.0], rtol=1e-5)

    def test_normalize_changes_predictions(self):
        # Same body/head/tokens, only the normalize step differs -> the head sees different inputs.
        body, head, tok = _Body(), nn.Linear(4, 2), _FakeTokenizer()
        plain = TorchClassifierModel(body, head, tok, label_names=["neg", "pos"], normalize=False)
        normed = TorchClassifierModel(body, head, tok, label_names=["neg", "pos"], normalize=True)
        self.assertFalse(np.allclose(plain.predict(["hello world"]), normed.predict(["hello world"])))

    def test_default_is_no_normalize(self):
        model = _model()
        self.assertFalse(model.normalize)
        self.assertFalse(model.backbone.normalize)


class TestEmbeddingSpace(unittest.TestCase):
    """``embed`` returns the configured representation; the default reaches into the head."""

    @staticmethod
    def _deep_head_model(**kwargs):
        # Two-layer head: pooled(4) -> Linear(4,3) -> relu -> Linear(3,2)=logits. The decision space is
        # the 3-dim input to the final Linear; the pooled space is 4-dim, so the two are distinguishable.
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        return TorchClassifierModel(_Body(), head, _FakeTokenizer(), label_names=["neg", "pos"], **kwargs)

    def test_default_space_is_decision(self):
        self.assertEqual(self._deep_head_model().embedding_space, "decision")

    def test_decision_reaches_into_head(self):
        # Decision space = input to the final classification Linear (3-dim), not the pooled encoder (4).
        emb = self._deep_head_model(embedding_space="decision").embed(["hello world", "a b c"])
        self.assertEqual(emb.shape, (2, 3))

    def test_pooled_space_is_encoder_output(self):
        emb = self._deep_head_model(embedding_space="pooled").embed(["hello world", "a b c"])
        self.assertEqual(emb.shape, (2, 4))  # the encoder hidden size, head untouched

    def test_resolves_final_classification_linear(self):
        # The last Linear *inside the head* is the one producing logits.
        model = self._deep_head_model()
        linear = model._resolve_decision_linear()
        self.assertIsInstance(linear, nn.Linear)
        self.assertIs(linear, model.head[-1])

    def test_decision_space_does_not_require_label_names(self):
        # Resolution is structural (head -> last Linear), so label_names — a display concern — must not
        # decide which representation a caller gets. Without it the decision space still works.
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        model = TorchClassifierModel(_Body(), head, _FakeTokenizer(), label_names=None)
        self.assertIs(model._resolve_decision_linear(), head[-1])
        self.assertEqual(model.embed(["hello world"]).shape, (1, 3))  # decision, not pooled(4)

    def test_head_with_no_linear_falls_back_to_pooled_audibly(self):
        # A head containing no nn.Linear leaves nothing to read a decision space off. The caller asked
        # for "decision" and gets "pooled", so the degradation must be audible rather than silent.
        model = TorchClassifierModel(_Body(), nn.Identity(), _FakeTokenizer(), label_names=["neg", "pos"])
        self.assertIsNone(model._resolve_decision_linear())
        # pytest.warns, not unittest's assertWarns: the latter walks sys.modules reading
        # __warningregistry__, which blows up on any lazily-imported module (transformers) that a
        # sibling test suite may have loaded into the same session.
        with pytest.warns(UserWarning):
            self.assertEqual(model.embed(["hello world"]).shape, (1, 4))  # pooled encoder hidden size

    def test_unreachable_submodule_space_raises_a_named_error(self):
        # Space validation only checks that a submodule *name exists* — a module can exist on the
        # backbone and still never run (a dead branch, or one this input routes around). The hook then
        # captures nothing, and the failure has to name the unreachable space rather than surface as a
        # bare IndexError off the empty capture list.
        class _BodyWithDeadBranch(_Body):
            def __init__(self):
                super().__init__()
                self.dead = nn.Linear(4, 4)  # registered, but _Body.forward never calls it

        model = TorchClassifierModel(
            _BodyWithDeadBranch(),
            nn.Linear(4, 2),
            _FakeTokenizer(),
            label_names=["neg", "pos"],
            embedding_space="body.dead",  # accepted at construction: the name does exist
        )
        with self.assertRaisesRegex(RuntimeError, "body.dead"):
            model.embed(["hello world"])

    def test_resolution_is_memoized(self):
        model = self._deep_head_model()
        self.assertIs(model._resolve_decision_linear(), model._resolve_decision_linear())


class _FastEnc(dict):
    """A fast-tokenizer encoding: a plain ids mapping plus the ``word_ids()`` accessor."""

    def __init__(self, input_ids, word_ids):
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class _FastFakeTokenizer(_FakeTokenizer):
    """Fast variant of :class:`_FakeTokenizer`: one word per token, with ``word_ids()``.

    Needed because ``word_alignment`` returns ``None`` outright for a slow tokenizer, so the slow fake
    can never exercise its truncation.
    """

    is_fast = True

    def __call__(self, texts, padding=True, truncation=True, return_tensors=None, max_length=None):
        self.calls.append({"truncation": truncation, "max_length": max_length})
        words = texts.split() if isinstance(texts, str) else list(texts)
        if truncation and max_length is not None:
            words = words[:max_length]
        return _FastEnc(list(range(len(words))), list(range(len(words))))

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


class TestMaxLength(unittest.TestCase):
    """``max_length`` reaches every tokenization path, so long texts truncate consistently."""

    @staticmethod
    def _model(max_length=None, tokenizer=None):
        return TorchClassifierModel(
            _Body(),
            nn.Linear(4, 2),
            tokenizer or _FakeTokenizer(),
            label_names=["neg", "pos"],
            max_length=max_length,
        )

    def test_defaults_to_none_so_the_tokenizer_decides(self):
        model = self._model()
        model.predict(["hello world"])
        self.assertIsNone(model.tokenizer.calls[-1]["max_length"])

    def test_batched_path_truncates(self):
        model = self._model(max_length=3)
        long_text = " ".join(f"w{i}" for i in range(20))
        model.predict([long_text])
        self.assertEqual(model.tokenizer.calls[-1]["max_length"], 3)

    def test_every_tokenization_path_uses_it(self):
        # encode/token_gradients/word_alignment must agree with the batched path: word_alignment returns
        # positions into the same subword axis as encode, so a mismatch misaligns highlights silently.
        long_text = " ".join(f"w{i}" for i in range(20))
        paths = {
            "predict": lambda m: m.predict([long_text]),
            "encode": lambda m: m.encode(long_text),
            "token_gradients": lambda m: m.token_gradients(long_text, target_class=0),
            "word_alignment": lambda m: m.word_alignment(long_text),
        }
        for name, run in paths.items():
            with self.subTest(path=name):
                # word_alignment short-circuits on a slow tokenizer, so give it a fast one.
                tokenizer = _FastFakeTokenizer() if name == "word_alignment" else _FakeTokenizer()
                model = self._model(max_length=3, tokenizer=tokenizer)
                run(model)
                self.assertTrue(model.tokenizer.calls, f"{name} performed no tokenization")
                for call in model.tokenizer.calls:
                    self.assertEqual(call["max_length"], 3)
                    self.assertTrue(call["truncation"])

    def test_word_alignment_positions_stay_within_the_truncated_axis(self):
        # The concrete failure the shared tokenization helper prevents: word positions indexing past the
        # end of the (truncated) token axis that encode/attribution produce.
        model = self._model(max_length=3, tokenizer=_FastFakeTokenizer())
        words, word_positions, _ = model.word_alignment(" ".join(f"w{i}" for i in range(20)))
        self.assertEqual(len(words), 3)
        self.assertTrue(all(p < 3 for pos in word_positions for p in pos))

    def test_encode_and_gradients_agree_on_length(self):
        model = self._model(max_length=4)
        long_text = " ".join(f"w{i}" for i in range(20))
        input_ids, _, tokens = model.encode(long_text)
        grad_tokens, grads = model.token_gradients(long_text, target_class=0)
        self.assertEqual(input_ids.shape[1], 4)
        self.assertEqual(len(tokens), len(grad_tokens))
        self.assertEqual(grads.shape[0], len(grad_tokens))


class TestResolveSpace(unittest.TestCase):
    """``resolve_space`` is the single source of truth for "which space will ``embed`` use"."""

    def setUp(self):
        head = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
        self.model = TorchClassifierModel(_Body(), head, _FakeTokenizer(), label_names=["neg", "pos"])

    def test_none_resolves_to_the_configured_default(self):
        self.assertEqual(self.model.resolve_space(), "decision")
        self.model.embedding_space = "pooled"
        self.assertEqual(self.model.resolve_space(), "pooled")

    def test_explicit_space_is_returned_and_validated(self):
        self.assertEqual(self.model.resolve_space("pooled"), "pooled")
        with self.assertRaises(ValueError):
            self.model.resolve_space("no_such_space")

    def test_assigning_an_unknown_space_raises_at_assignment(self):
        # The setter validates, so a typo surfaces here rather than inside the next forward pass.
        with self.assertRaises(ValueError):
            self.model.embedding_space = "no_such_space"
        self.assertEqual(self.model.embedding_space, "decision")  # unchanged by the failed assignment

    def test_assignment_changes_what_embed_returns(self):
        # Switching the space post-construction moves every consumer (scatter + retrieval) at once.
        self.assertEqual(self.model.embed(["hello world"]).shape, (1, 3))  # decision
        self.model.embedding_space = "pooled"
        self.assertEqual(self.model.embed(["hello world"]).shape, (1, 4))  # encoder hidden size


class TestEncoderModelValidation(unittest.TestCase):
    def test_rejects_unknown_pool_mode(self):
        with self.assertRaises(ValueError):
            EncoderClassifierModel(_Body(), _FakeTokenizer(), pool="median")

    def test_rejects_unknown_embedding_space_at_construction(self):
        with self.assertRaises(ValueError):
            EncoderClassifierModel(_Body(), _FakeTokenizer(), embedding_space="no_such_space")


def _dropout_head(p=0.5):
    """A head that is *not* a no-op in training mode, so eval/train is observable in the output."""
    return nn.Sequential(nn.Linear(4, 4), nn.Dropout(p), nn.Linear(4, 2))


class TestInferenceMode(unittest.TestCase):
    """Adapters must pin their backbone to eval mode — dropout/batchnorm can only corrupt analysis.

    Assigning a submodule to an ``nn.Module`` does not touch that submodule's ``training`` flag, so a
    head the caller built fresh stays in *training* mode and silently randomises every forward pass:
    predictions, contributions, embeddings and counterfactual flips all differ run to run, with nothing
    raising. ``from_pretrained`` returns eval-mode models, which made this look fine on the HF path only.
    """

    def test_fused_backbone_is_eval_mode(self):
        backbone = build_encoder_head_backbone(_Body(), _dropout_head(), "mean")
        self.assertFalse(backbone.training)
        self.assertFalse(backbone.head.training)  # eval() recurses into head...
        self.assertFalse(backbone.body.training)  # ...and body

    def test_fused_backbone_forward_is_deterministic(self):
        backbone = build_encoder_head_backbone(_Body(), _dropout_head(), "mean")
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones(1, 3, dtype=torch.long)
        first = backbone(input_ids=ids, attention_mask=mask).logits.detach().numpy()
        for _ in range(4):
            repeat = backbone(input_ids=ids, attention_mask=mask).logits.detach().numpy()
            np.testing.assert_allclose(repeat, first, rtol=1e-7)

    def test_torch_classifier_predict_is_deterministic_with_dropout_head(self):
        # The end-to-end symptom: identical text, different probabilities on every call.
        model = TorchClassifierModel(_Body(), _dropout_head(), _FakeTokenizer(), label_names=["neg", "pos"])
        first = model.predict(["hello world"])
        for _ in range(4):
            np.testing.assert_allclose(model.predict(["hello world"]), first, rtol=1e-7)

    def test_embed_is_deterministic_with_dropout_head(self):
        # Non-determinism here would also destabilise the scatter and similar-example neighbours.
        model = TorchClassifierModel(_Body(), _dropout_head(), _FakeTokenizer(), label_names=["neg", "pos"])
        first = model.embed(["hello world"])
        np.testing.assert_allclose(model.embed(["hello world"]), first, rtol=1e-7)

    def test_encoder_model_evals_any_backbone(self):
        # The guard lives in EncoderClassifierModel too, so it covers a hand-passed backbone that did
        # not come from build_encoder_head_backbone.
        backbone = _Body()
        backbone.train()
        EncoderClassifierModel(backbone, _FakeTokenizer(), embedding_space="pooled")
        self.assertFalse(backbone.training)

    def test_backbone_without_eval_is_tolerated(self):
        # The guard is best-effort: a non-nn.Module backbone honouring the contract must still build.
        class _PlainBackbone:
            device = torch.device("cpu")

            def named_modules(self):
                return []

            def get_input_embeddings(self):
                return nn.Embedding(9, 4)

        EncoderClassifierModel(_PlainBackbone(), _FakeTokenizer(), embedding_space="pooled")


if __name__ == "__main__":
    unittest.main()
