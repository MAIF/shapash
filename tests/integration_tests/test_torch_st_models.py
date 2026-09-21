"""Integration tests for the external-head presets against real encoders.

Covers the two adapters whose backbone is a fused ``body -> pool -> head`` rather than a single
``AutoModelForSequenceClassification``:

* :class:`SentenceTransformerModel` over a real ``sentence_transformers`` model + a Linear head.
* :class:`TorchClassifierModel` over a raw HuggingFace ``AutoModel`` body + a Linear head.

The heads are randomly initialised — predictions are meaningless, but every *capability* (predict,
embed, gradients, activations, similar-example retrieval) and both attribution backends (SHAP via the
explicit Text masker, and Captum LIG) must run and return the right shapes. Skipped when
``transformers`` / ``torch`` / ``sentence_transformers`` / ``captum`` are unavailable.
"""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
st = pytest.importorskip("sentence_transformers")
pytest.importorskip("captum")
pytestmark = pytest.mark.nlp

from torch import nn  # noqa: E402

from shapash.backend import NlpCaptumLigBackend  # noqa: E402
from shapash.explainer.nlp_explainer import NlpExplainer  # noqa: E402
from shapash.model import (  # noqa: E402
    SentenceTransformerModel,
    SupportsCaptumIG,
    SupportsEmbeddings,
    SupportsGradients,
    TorchClassifierModel,
    has_capabilities,
)

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Tiny random-weight DistilBert: only shapes/capabilities are asserted below (the head is random
# too), so there's no need for the ~250MB trained checkpoint. ST_MODEL stays full-size because
# several tests below pin behaviour specific to its actual config (mean pooling, normalize=True,
# the 512 vs 256 tokenizer/max_seq_length mismatch).
HF_BODY = "hf-internal-testing/tiny-random-DistilBertModel"
LABELS = ["neg", "pos", "neutral"]
TEXTS = ["i am so happy today", "this is terrible and sad", "an ordinary grey afternoon"]


@pytest.fixture(scope="module")
def st_model():
    """A SentenceTransformerModel: cached MiniLM sentence-transformer + a random 3-class head."""
    try:
        sentence_model = st.SentenceTransformer(ST_MODEL, device="cpu")
    except Exception as exc:  # network / cache miss
        pytest.skip(f"sentence-transformers model unavailable: {exc}")
    hidden = sentence_model.get_sentence_embedding_dimension()
    head = nn.Linear(hidden, len(LABELS))
    return SentenceTransformerModel(sentence_model, head, label_names=LABELS)


@pytest.fixture(scope="module")
def torch_model():
    """A TorchClassifierModel: a tiny random-weight HF encoder body + a random 3-class head + fast tokenizer."""
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(HF_BODY, use_fast=True)
        body = transformers.AutoModel.from_pretrained(HF_BODY)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"HF body unavailable: {exc}")
    body.eval()
    head = nn.Linear(body.config.hidden_size, len(LABELS))
    return TorchClassifierModel(body, head, tokenizer, label_names=LABELS, pool="mean")


# --------------------------------------------------------------------------------------------------
# SentenceTransformerModel
# --------------------------------------------------------------------------------------------------


def test_st_extracts_body_tokenizer_and_pool(st_model):
    """Extraction pulls the HF transformer body, its tokenizer, and the configured pooling mode."""
    assert isinstance(st_model.body, transformers.PreTrainedModel)
    assert st_model.tokenizer is not None
    assert st_model.pool == "mean"  # all-MiniLM-L6-v2 pools by mean
    assert st_model.normalize is True  # all-MiniLM-L6-v2 ends in a Normalize module


def test_st_embed_reproduces_encode(st_model):
    """With Normalize detected, embed() must match the sentence-transformer's own encode()."""
    ours = st_model.embed(TEXTS)
    reference = st_model.st_model.encode(TEXTS, convert_to_numpy=True)
    np.testing.assert_allclose(ours, reference, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(ours, axis=1), np.ones(len(TEXTS)), atol=1e-5)


def test_st_adopts_max_seq_length(st_model):
    """The ST model's own truncation length is adopted, so the served length is explicit."""
    assert st_model.max_length == st_model.st_model.max_seq_length


def test_st_embed_reproduces_encode_on_long_text(st_model):
    """Agreement with encode() must also hold past the truncation boundary, not just for short texts.

    The short-text check above passes regardless of how truncation is configured, so on its own it says
    nothing about long inputs. This exercises a text well past ``max_seq_length``.
    """
    long_text = " ".join(TEXTS * 40)  # comfortably past the 256-token limit
    assert len(st_model.tokenizer.tokenize(long_text)) > st_model.max_length
    ours = st_model.embed([long_text])
    reference = st_model.st_model.encode([long_text], convert_to_numpy=True)
    np.testing.assert_allclose(ours, reference, atol=1e-5)


def test_st_declares_full_capability_surface(st_model):
    assert has_capabilities(st_model, SupportsEmbeddings, SupportsGradients, SupportsCaptumIG)


def test_st_predict_is_normalised(st_model):
    probs = st_model.predict(TEXTS)
    assert probs.shape == (len(TEXTS), len(LABELS))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(TEXTS)), atol=1e-5)


def test_st_embed_spaces_shapes(st_model):
    hidden = st_model.backbone.get_input_embeddings().weight.shape[1]
    # Default "decision" space: the input to the head's final linear.
    assert st_model.embed(TEXTS).shape[0] == len(TEXTS)
    # "pooled" is the universal space every architecture provides: the pooled last hidden state.
    assert st_model.embed(TEXTS, "pooled").shape == (len(TEXTS), hidden)


def test_st_token_gradients_align(st_model):
    tokens, grads = st_model.token_gradients(TEXTS[0], target_class=1)
    assert grads.shape[0] == len(tokens)
    assert grads.shape[1] == st_model.backbone.get_input_embeddings().weight.shape[1]


def test_st_shap_backend_runs(st_model):
    """SHAP works via the explicit Text masker (no transformers pipeline needed)."""
    xpl = NlpExplainer(st_model, label_names=LABELS)
    explanation = xpl.explain(TEXTS[:2])
    words = explanation.token_strings[0]
    assert len(words) > 0
    # word-level highlights must not leak special tokens or subword markers.
    assert not any(w.startswith(("##", "Ġ", "▁")) or w in ("[CLS]", "[SEP]") for w in words)


def test_st_lig_backend_runs(st_model):
    backend = NlpCaptumLigBackend(st_model, label_names=LABELS)
    raw = backend.run_explainer(TEXTS[:2])
    assert len(raw.values) == 2
    for contrib in raw.values:
        assert contrib.shape[1] == len(LABELS)


# --------------------------------------------------------------------------------------------------
# TorchClassifierModel
# --------------------------------------------------------------------------------------------------


def test_torch_declares_full_capability_surface(torch_model):
    assert has_capabilities(torch_model, SupportsEmbeddings, SupportsGradients, SupportsCaptumIG)


def test_torch_predict_and_embed(torch_model):
    probs = torch_model.predict(TEXTS)
    assert probs.shape == (len(TEXTS), len(LABELS))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(TEXTS)), atol=1e-5)
    assert torch_model.embed(TEXTS).shape == (len(TEXTS), torch_model.body.config.hidden_size)


def test_torch_max_length_controls_long_text_truncation():
    """``max_length`` is what keeps a hand-wired body+tokenizer faithful to how the body is served.

    A raw ``AutoTokenizer`` for ``all-MiniLM-L6-v2`` reports ``model_max_length=512``, but the model is
    served at 256 (that is what ``SentenceTransformer`` truncates at). Pairing the two by hand — the
    natural ``TorchClassifierModel`` wiring — therefore feeds the body twice the context it expects on
    long inputs, and the sentence embedding drifts from the reference. This pins both directions.
    """
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(ST_MODEL, use_fast=True)
        body = transformers.AutoModel.from_pretrained(ST_MODEL)
        reference = st.SentenceTransformer(ST_MODEL, device="cpu")
    except Exception as exc:  # network / cache miss
        pytest.skip(f"model unavailable: {exc}")
    body.eval()

    long_text = " ".join(TEXTS * 40)
    assert len(tokenizer.tokenize(long_text)) > reference.max_seq_length
    assert tokenizer.model_max_length > reference.max_seq_length  # 512 vs 256: the trap

    head = nn.Linear(body.config.hidden_size, len(LABELS))
    kwargs = {"label_names": LABELS, "normalize": True}
    naive = TorchClassifierModel(body, head, tokenizer, **kwargs)
    fixed = TorchClassifierModel(body, head, tokenizer, max_length=reference.max_seq_length, **kwargs)
    target = reference.encode([long_text], convert_to_numpy=True)

    # Without max_length the embedding drifts; with it, it matches the reference exactly.
    assert np.abs(naive.embed([long_text], "pooled") - target).max() > 1e-3
    np.testing.assert_allclose(fixed.embed([long_text], "pooled"), target, atol=1e-5)


def test_torch_word_alignment_with_fast_tokenizer(torch_model):
    """A fast tokenizer yields exact subword->word grouping (used by the LIG highlight path)."""
    alignment = torch_model.word_alignment(TEXTS[0])
    assert alignment is not None
    words, positions, specials = alignment
    assert len(words) == len(positions)
    assert len(specials) >= 2  # [CLS] / [SEP]


def test_torch_shap_and_lig_backends_run(torch_model):
    xpl = NlpExplainer(torch_model, label_names=LABELS)
    explanation = xpl.explain(TEXTS[:2])
    assert len(explanation.token_strings[0]) > 0

    backend = NlpCaptumLigBackend(torch_model, label_names=LABELS)
    raw = backend.run_explainer(TEXTS[:2])
    assert len(raw.values) == 2


def test_torch_pool_mode_changes_embedding(torch_model):
    """The pool knob is honoured: cls-pooling differs from mean-pooling on the same text."""
    mean_emb = torch_model.embed([TEXTS[0]])
    cls_model = TorchClassifierModel(
        torch_model.body, torch_model.head, torch_model.tokenizer, label_names=LABELS, pool="cls"
    )
    cls_emb = cls_model.embed([TEXTS[0]])
    assert not np.allclose(mean_emb, cls_emb)
