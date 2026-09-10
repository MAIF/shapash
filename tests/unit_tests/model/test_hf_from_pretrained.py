"""Unit tests for ``HFClassifierModel.from_pretrained`` and its loader helpers.

These cover the checkpoint-loading logic that used to live in ``demo/serve_nlp.py``: tokenizer-source
resolution (including the base-model fallback for a directory saved without its tokenizer), the
``model_max_length`` no-op trap, the fast/slow tokenizer fallback, the label-name arity check, and the
end-to-end classmethod. ``transformers`` is faked throughout so the tests stay fast and need no network,
no real weights, and no GPU.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shapash.model import hf
from shapash.model.hf import (
    _FALLBACK_MAX_LENGTH,
    _UNSET_TOKENIZER_MAX_LENGTH,
    HFClassifierModel,
    _load_tokenizer,
    _resolve_max_length,
    _resolve_tokenizer_source,
)


# ── _resolve_tokenizer_source ───────────────────────────────────────────────────────────────────────


def test_explicit_tokenizer_name_wins_over_everything(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": "base-model"}))
    assert _resolve_tokenizer_source(str(tmp_path), tokenizer_name="my/tokenizer") == "my/tokenizer"


def test_hub_id_is_passed_through_unchanged():
    # Not a local directory → assumed a hub repo, which always ships its own tokenizer.
    assert _resolve_tokenizer_source("distilbert-base-uncased") == "distilbert-base-uncased"


def test_local_dir_with_tokenizer_files_loads_from_the_dir(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    assert _resolve_tokenizer_source(str(tmp_path)) == str(tmp_path)


def test_local_dir_without_tokenizer_falls_back_to_config_base_model(tmp_path):
    # A save_pretrained() dir: weights + config.json but no tokenizer files.
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": "the-base-model"}))
    assert _resolve_tokenizer_source(str(tmp_path)) == "the-base-model"


def test_local_dir_with_no_tokenizer_and_no_base_model_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_labels": 2}))  # no _name_or_path
    with pytest.raises(RuntimeError, match="no base model to fall back on"):
        _resolve_tokenizer_source(str(tmp_path))


def test_local_dir_with_no_files_at_all_raises(tmp_path):
    with pytest.raises(RuntimeError, match="ships no tokenizer files"):
        _resolve_tokenizer_source(str(tmp_path))


# ── _resolve_max_length ─────────────────────────────────────────────────────────────────────────────


def _backbone(max_position_embeddings=None, position_biased_input=True):
    return SimpleNamespace(
        config=SimpleNamespace(
            max_position_embeddings=max_position_embeddings,
            position_biased_input=position_biased_input,
        )
    )


def test_real_model_max_length_is_trusted():
    tok = SimpleNamespace(model_max_length=256)
    assert _resolve_max_length(tok, _backbone(max_position_embeddings=512)) == 256


def test_unset_sentinel_falls_back_to_backbone_capacity():
    tok = SimpleNamespace(model_max_length=int(1e30))  # transformers' "no limit" sentinel
    assert tok.model_max_length > _UNSET_TOKENIZER_MAX_LENGTH
    assert _resolve_max_length(tok, _backbone(max_position_embeddings=512)) == 512


def test_missing_model_max_length_falls_back_to_capacity():
    tok = SimpleNamespace(model_max_length=None)
    assert _resolve_max_length(tok, _backbone(max_position_embeddings=1024)) == 1024


def test_sentinel_with_no_position_capacity_uses_hardcoded_default():
    tok = SimpleNamespace(model_max_length=int(1e30))
    # A rotary/ALiBi backbone exposes no absolute-position table → capacity unknowable.
    assert _resolve_max_length(tok, _backbone(max_position_embeddings=None)) == _FALLBACK_MAX_LENGTH


# ── _load_tokenizer (fast → slow fallback) ────────────────────────────────────────────────────────────


class _FakeAutoTokenizer:
    """Records calls; can be told to fail the fast path, the slow path, or both."""

    def __init__(self, fail_fast=False, fail_slow=False):
        self.fail_fast = fail_fast
        self.fail_slow = fail_slow
        self.calls = []
        self.kwargs = []

    def from_pretrained(self, source, use_fast, **kwargs):
        self.calls.append(use_fast)
        self.kwargs.append(kwargs)
        if use_fast and self.fail_fast:
            raise OSError("no tokenizer.json")
        if not use_fast and self.fail_slow:
            raise OSError("no slow tokenizer either")
        return SimpleNamespace(kind="fast" if use_fast else "slow")


def test_load_tokenizer_prefers_fast():
    auto = _FakeAutoTokenizer()
    tok = _load_tokenizer("some-model", SimpleNamespace(AutoTokenizer=auto))
    assert tok.kind == "fast"
    assert auto.calls == [True]


def test_load_tokenizer_falls_back_to_slow():
    auto = _FakeAutoTokenizer(fail_fast=True)
    tok = _load_tokenizer("some-model", SimpleNamespace(AutoTokenizer=auto))
    assert tok.kind == "slow"
    assert auto.calls == [True, False]


def test_load_tokenizer_both_fail_suggests_trust_remote_code():
    auto = _FakeAutoTokenizer(fail_fast=True, fail_slow=True)
    with pytest.raises(RuntimeError, match="retry with trust_remote_code=True"):
        _load_tokenizer("some-model", SimpleNamespace(AutoTokenizer=auto))


def test_load_tokenizer_both_fail_with_custom_code_allowed_points_elsewhere():
    # Suggesting trust_remote_code=True to someone who already passed it would misdiagnose the failure.
    auto = _FakeAutoTokenizer(fail_fast=True, fail_slow=True)
    with pytest.raises(RuntimeError, match="not a trust_remote_code problem"):
        _load_tokenizer("some-model", SimpleNamespace(AutoTokenizer=auto), trust_remote_code=True)


def test_load_tokenizer_forwards_kwargs_to_both_attempts():
    auto = _FakeAutoTokenizer(fail_fast=True)
    _load_tokenizer("some-model", SimpleNamespace(AutoTokenizer=auto), trust_remote_code=True, revision="v2")
    assert auto.kwargs == [
        {"trust_remote_code": True, "revision": "v2"},
        {"trust_remote_code": True, "revision": "v2"},
    ]


# ── label_names arity check in __init__ ───────────────────────────────────────────────────────────────


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        seqs = [[(hash(w) % 8) + 1 for w in t.split()] for t in ([texts] if isinstance(texts, str) else texts)]
        width = max(len(s) for s in seqs)
        return {
            "input_ids": torch.tensor([s + [0] * (width - len(s)) for s in seqs], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], dtype=torch.long),
        }


class _TinyClassifier(nn.Module):
    """A tiny 2-class sequence classifier carrying a realistic ``config``."""

    def __init__(self, num_labels=2, id2label=None):
        super().__init__()
        self.emb = nn.Embedding(9, 4)
        self.classifier = nn.Linear(4, num_labels)
        self.config = SimpleNamespace(
            num_labels=num_labels,
            id2label=id2label,
            max_position_embeddings=512,
            position_biased_input=True,
            _name_or_path="tiny-base",
        )

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        h = self.emb(input_ids)
        out = SimpleNamespace(logits=self.classifier(h.mean(dim=1)))
        if output_hidden_states:
            out.hidden_states = (h,)
        return out


def test_explicit_label_names_wrong_length_raises():
    with pytest.raises(ValueError, match="for a 2-class model"):
        HFClassifierModel(_TinyClassifier(), _FakeTokenizer(), label_names=["neg", "neutral", "pos"])


def test_explicit_label_names_right_length_is_accepted():
    model = HFClassifierModel(_TinyClassifier(), _FakeTokenizer(), label_names=["neg", "pos"])
    assert model.label_names == ["neg", "pos"]


def test_label_names_read_from_config_never_trip_the_arity_check():
    clf = _TinyClassifier(id2label={0: "neg", 1: "pos"})
    model = HFClassifierModel(clf, _FakeTokenizer())  # label_names=None → read from config
    assert model.label_names == ["neg", "pos"]


# ── from_pretrained (end-to-end, faked transformers) ──────────────────────────────────────────────────


class _FakeTransformers:
    """A stand-in ``transformers`` module: hands back a fake tokenizer and a real tiny nn.Module."""

    def __init__(self, tokenizer, classifier):
        self._tokenizer = tokenizer
        self._classifier = classifier
        self.model_loaded_from = None
        self.model_kwargs = None
        self.tokenizer_kwargs = None
        self.AutoTokenizer = SimpleNamespace(from_pretrained=self._load_tokenizer)
        self.AutoModelForSequenceClassification = SimpleNamespace(from_pretrained=self._load_model)

    def _load_tokenizer(self, source, use_fast, **kwargs):
        self.tokenizer_kwargs = kwargs
        return self._tokenizer

    def _load_model(self, source, **kwargs):
        self.model_loaded_from = source
        self.model_kwargs = kwargs
        return self._classifier


@pytest.fixture
def fake_transformers(monkeypatch):
    """Patch ``import_optional_module`` so ``from_pretrained`` gets our fake transformers module."""
    tokenizer = _FakeTokenizer()
    tokenizer.model_max_length = int(1e30)  # unset sentinel → exercises the capacity fallback
    classifier = _TinyClassifier(id2label={0: "neg", 1: "pos"})
    fake = _FakeTransformers(tokenizer, classifier)
    monkeypatch.setattr(hf, "import_optional_module", lambda name, extra="": fake)
    return fake


def test_from_pretrained_builds_a_working_adapter(fake_transformers):
    model = HFClassifierModel.from_pretrained("some/checkpoint")

    assert isinstance(model, HFClassifierModel)
    assert model.classifier is fake_transformers._classifier
    assert model.label_names == ["neg", "pos"]  # read from the model config
    assert fake_transformers.model_loaded_from == "some/checkpoint"
    probs = model.predict(["a good movie", "a bad one"])
    assert probs.shape == (2, 2)


def test_from_pretrained_auto_resolves_the_unset_max_length_sentinel(fake_transformers):
    # The fake tokenizer reports the ~1e30 sentinel; "auto" must substitute the backbone capacity (512).
    model = HFClassifierModel.from_pretrained("some/checkpoint", max_length="auto")
    assert model.max_length == 512


def test_from_pretrained_none_max_length_is_passed_through(fake_transformers):
    model = HFClassifierModel.from_pretrained("some/checkpoint", max_length=None)
    assert model.max_length is None


def test_from_pretrained_explicit_int_max_length_wins(fake_transformers):
    model = HFClassifierModel.from_pretrained("some/checkpoint", max_length=128)
    assert model.max_length == 128


def test_from_pretrained_rejects_a_bad_max_length(fake_transformers):
    with pytest.raises(ValueError, match="max_length must be"):
        HFClassifierModel.from_pretrained("some/checkpoint", max_length="longish")


def test_from_pretrained_label_names_override_is_arity_checked(fake_transformers):
    with pytest.raises(ValueError, match="for a 2-class model"):
        HFClassifierModel.from_pretrained("some/checkpoint", label_names=["only-one"])


def test_from_pretrained_forwards_model_kwargs(fake_transformers):
    model = HFClassifierModel.from_pretrained("some/checkpoint", batch_size=7)
    assert model.batch_size == 7
    # ``**model_kwargs`` configures the adapter — it must never leak into the checkpoint loaders.
    assert "batch_size" not in fake_transformers.model_kwargs
    assert "batch_size" not in fake_transformers.tokenizer_kwargs


def test_from_pretrained_passes_trust_remote_code_to_both_loaders(fake_transformers):
    # A custom architecture (GTE-v1.5, Jina, Nomic) ships both a model and a tokenizer implementation,
    # so the flag has to reach both loads or the pair fails halfway.
    HFClassifierModel.from_pretrained("custom/checkpoint", trust_remote_code=True)
    assert fake_transformers.model_kwargs["trust_remote_code"] is True
    assert fake_transformers.tokenizer_kwargs["trust_remote_code"] is True


def test_from_pretrained_sends_trust_remote_code_false_explicitly(fake_transformers):
    # Explicit False, not omitted: left unset, transformers prompts for confirmation on an interactive
    # terminal, which would hang a webapp or batch job on a prompt nobody sees.
    HFClassifierModel.from_pretrained("some/checkpoint")
    assert fake_transformers.model_kwargs["trust_remote_code"] is False
    assert fake_transformers.tokenizer_kwargs["trust_remote_code"] is False


def test_from_pretrained_forwards_load_kwargs_to_both_loaders(fake_transformers):
    HFClassifierModel.from_pretrained("some/checkpoint", load_kwargs={"revision": "abc123"})
    assert fake_transformers.model_kwargs["revision"] == "abc123"
    assert fake_transformers.tokenizer_kwargs["revision"] == "abc123"


def test_from_pretrained_load_kwargs_can_override_trust_remote_code(fake_transformers):
    HFClassifierModel.from_pretrained("some/checkpoint", load_kwargs={"trust_remote_code": True})
    assert fake_transformers.model_kwargs["trust_remote_code"] is True


def test_from_pretrained_skips_the_tokenizer_load_when_one_is_supplied(fake_transformers):
    # An already-loaded tokenizer is used verbatim, so nothing is forwarded to a tokenizer loader.
    tokenizer = _FakeTokenizer()
    HFClassifierModel.from_pretrained("some/checkpoint", tokenizer=tokenizer, trust_remote_code=True)
    assert fake_transformers.tokenizer_kwargs is None
    assert fake_transformers.model_kwargs["trust_remote_code"] is True


def test_from_pretrained_moves_classifier_to_requested_device(fake_transformers, monkeypatch):
    moved_to = []
    monkeypatch.setattr(fake_transformers._classifier, "to", lambda device: (moved_to.append(device) or fake_transformers._classifier))
    HFClassifierModel.from_pretrained("some/checkpoint", device="cpu")
    assert moved_to == ["cpu"]
