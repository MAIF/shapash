"""Backend units across tokenizer families: placed in the source text, and comparable across backends.

Each backend cuts a text into units its own way, and each tokenizer family moves the cuts again
(WordPiece splits every punctuation mark, byte-level BPE glues symbols together, SentencePiece splits
on whitespace only, uncased tokenizers rewrite the text). These tests run the real SHAP, LIG and LIME
``run_explainer`` through ``NlpExplainer.explain`` on one checkpoint per family, against text built
to break string matching (accents, decomposed accents, emoji, curly quotes, full-width characters,
double spaces, newlines), and check the invariants alignment relies on:

- every unit is placed: it has spans, and its string is the source text at its span;
- no unit is whitespace;
- SHAP and LIG report the same units, since both read the model's own encoding;
- SHAP is additive against the text itself: it masks token ids, so the model never sees a rebuilt string;
- SHAP in probability space (its default) compares with LIME, which explains probabilities too;
- aligning backends drops nothing: each backend's values sum to the same total before and after.

The models are randomly initialised from each checkpoint's config at a tiny size, so only the config
and tokenizer are downloaded; the numbers mean nothing, the units are real. Skipped when the ``nlp``
or ``lime`` extra is missing or a checkpoint cannot be fetched.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")
pytest.importorskip("captum")
pytest.importorskip("lime")
pytestmark = pytest.mark.nlp

from shapash.backend import NlpCaptumLigBackend, NlpLimeBackend, NlpShapBackend  # noqa: E402
from shapash.explainer.nlp_comparison import align_row, corpus_agreement, name_explanations  # noqa: E402
from shapash.explainer.nlp_explainer import NlpExplainer  # noqa: E402
from shapash.model import HFClassifierModel  # noqa: E402

CHECKPOINTS = {
    "wordpiece-uncased": "distilbert-base-uncased",
    "byte-bpe": "bhadresh-savani/roberta-base-emotion",
    "bpe-modernbert": "Alibaba-NLP/gte-modernbert-base",
    "sentencepiece-unigram": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "sentencepiece-uncased": "bhadresh-savani/albert-base-v2-emotion",
    "sentencepiece-deberta": "microsoft/deberta-v3-small",
}

TEXTS = [
    "I didn't like it, but she's sure it won't fail!",
    "Le café était très bon, naïve and naive.",
    "Le café était très bon.",  # decomposed accents (NFD)
    "Superb!!!Really enjoy.Overall,I loved it…",
    "Well… it’s “fine” — I love it \U0001f60d\U0001f44d\U0001f3fd",
    "Hello  world\n\nThis is\tgreat.",
    "A state-of-the-art ﬁnal: ＡＢＣ costs $3.50 on 2024-01-05.",
    "Die Straße ist groß. İstanbul, 我喜欢 snake_case.",
]
LABELS = ["a", "b"]

# Shrinks every common config to one tiny layer; attributes a family does not have are skipped.
_TINY = {
    "num_hidden_layers": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "embedding_size": 32,
    "pooler_hidden_size": 32,
    "n_layers": 1,
    "dim": 32,
    "hidden_dim": 64,
    "n_heads": 2,
}


def _tiny_model(checkpoint: str) -> HFClassifierModel:
    try:
        config = transformers.AutoConfig.from_pretrained(checkpoint)
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"{checkpoint} unavailable: {exc}")
    for key, value in _TINY.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.num_labels = len(LABELS)
    torch.manual_seed(0)
    classifier = (
        transformers.AutoModelForSequenceClassification.from_config(config).float().eval()
    )  # some configs pin fp16
    return HFClassifierModel(classifier, tokenizer, label_names=LABELS, max_length=128)


@pytest.fixture(scope="module", params=list(CHECKPOINTS), ids=list(CHECKPOINTS))
def explanations(request):
    model = _tiny_model(CHECKPOINTS[request.param])
    backends = {
        "shap": NlpShapBackend(
            model, label_names=LABELS, output_space="logit", explainer_compute_args={"max_evals": 500}
        ),
        "shap_prob": NlpShapBackend(model, label_names=LABELS, explainer_compute_args={"max_evals": 500}),
        "lig": NlpCaptumLigBackend(model, label_names=LABELS, explainer_compute_args={"n_steps": 2}),
        "lime": NlpLimeBackend(model, label_names=LABELS, explainer_compute_args={"num_samples": 50}),
    }
    out = {
        name: NlpExplainer(model, label_names=LABELS, backend=backend).explain(TEXTS)
        for name, backend in backends.items()
    }
    out["model"] = model
    assert backends["shap"].masking == "token_ids"
    return out


@pytest.mark.parametrize("backend", ["shap", "lig", "lime"])
def test_every_unit_is_placed_and_reads_as_written(explanations, backend):
    exp = explanations[backend]
    for row, text in enumerate(TEXTS):
        for word, spans in zip(exp.token_strings[row], exp.token_spans[row], strict=True):
            assert spans, f"{word!r} has no span in {text!r}"
            assert word.strip() == word and word, f"whitespace in unit {word!r}"
            for start, end in spans:
                assert text[start:end] == word
        if backend != "lime":  # a sequence backend: one span per unit, in order, disjoint
            flat = [unit[0] for unit in exp.token_spans[row]]
            assert all(len(unit) == 1 for unit in exp.token_spans[row])
            assert all(a[1] <= b[0] for a, b in zip(flat, flat[1:], strict=False))


def test_shap_and_lig_report_the_same_units(explanations):
    shap, lig = explanations["shap"], explanations["lig"]
    for row, text in enumerate(TEXTS):
        assert shap.token_spans[row] == lig.token_spans[row], f"row {row}: {text!r}"
        assert shap.token_strings[row] == lig.token_strings[row]


def test_shap_is_exactly_additive_against_the_text_itself(explanations):
    exp, model = explanations["shap"], explanations["model"]
    logits = model.predict_logits(TEXTS)
    for row in range(len(TEXTS)):
        np.testing.assert_allclose(exp.base_values[row] + exp.values[row].sum(axis=0), logits[row], atol=1e-4)


def test_shap_and_lig_measure_against_the_same_reference(explanations):
    # Both hide every content token behind the model's reference id, so both baselines are the
    # reference's logits — except where a whitespace-only word (a byte-level "Ġ"/"Ċ") folds its own,
    # method-specific value into them: compare on texts without one.
    shap, lig = explanations["shap"], explanations["lig"]
    for row, text in enumerate(TEXTS):
        if any(c in text for c in "\n\t") or "  " in text:
            continue
        np.testing.assert_allclose(shap.base_values[row], lig.base_values[row], atol=1e-4)


def test_shap_and_lime_compare_in_probability_space(explanations):
    prob, logit, lime = explanations["shap_prob"], explanations["shap"], explanations["lime"]
    assert (prob.output_space, lime.output_space) == ("probability", "probability")
    # The output space changes the values, never the units.
    assert prob.token_spans == logit.token_spans and prob.token_strings == logit.token_strings
    probs = explanations["model"].predict(TEXTS)
    for row in range(len(TEXTS)):
        np.testing.assert_allclose(prob.base_values[row] + prob.values[row].sum(axis=0), probs[row], atol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # same space: no cross-space warning
        df = corpus_agreement(prob, {"lime": lime}, label_idx=1)
    assert len(df) == len(TEXTS)
    assert np.isfinite(df.spearman.fillna(0)).all()
    # LIME splits words its own way, so coverage is partial, but it is the same against either space.
    pd.testing.assert_series_equal(
        df.coverage.reset_index(drop=True),
        corpus_agreement(logit, {"lime": lime}, label_idx=1).coverage.reset_index(drop=True),
    )
    for row in range(len(TEXTS)):
        units, aligned = align_row(name_explanations(prob, {"lime": lime}), row, 1)
        assert np.isfinite(aligned["shap"]).all()
        assert np.nansum(aligned["shap"]) == pytest.approx(prob.values[row][:, 1].sum(), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("label_idx", [0, 1])
def test_alignment_drops_nothing(explanations, label_idx):
    named = name_explanations(explanations["shap"], {"lig": explanations["lig"], "lime": explanations["lime"]})
    for row in range(len(TEXTS)):
        units, aligned = align_row(named, row, label_idx)
        text = TEXTS[row]
        assert all(unit and unit == unit.strip() for unit in units)
        # Both sequence backends attribute every group: same tokenizer, same characters.
        assert np.isfinite(aligned["shap"]).all() and np.isfinite(aligned["lig"]).all(), text
        for name, key in (("shap", "shap"), ("lig", "lig")):
            total = explanations[key].values[row][:, label_idx].sum()
            assert np.nansum(aligned[name]) == pytest.approx(total, rel=1e-6, abs=1e-9)
        # LIME's one weight counts at every occurrence.
        lime = explanations["lime"]
        occurrences = np.array([len(spans) for spans in lime.token_spans[row]])
        total = (lime.values[row][:, label_idx] * occurrences).sum()
        assert np.nansum(aligned["lime"]) == pytest.approx(total, rel=1e-6, abs=1e-9)


def test_corpus_agreement_runs_on_every_family(explanations):
    df = corpus_agreement(explanations["shap"], {"lig": explanations["lig"], "lime": explanations["lime"]}, label_idx=1)
    shap_lig = df[(df.backend_a == "shap") & (df.backend_b == "lig")]
    assert len(shap_lig) == len(TEXTS)
    assert (shap_lig.coverage == 1.0).all()


def test_a_truncated_backend_shows_as_partial_coverage():
    model = _tiny_model(CHECKPOINTS["wordpiece-uncased"])
    model.max_length = 8  # SHAP and LIG explain the encoding, cut at 6 words; LIME splits the whole text
    text = ["one two three four five six seven eight nine ten eleven twelve"]
    explain = {
        name: NlpExplainer(model, label_names=LABELS, backend=backend).explain(text)
        for name, backend in {
            "shap": NlpShapBackend(
                model, label_names=LABELS, output_space="logit", explainer_compute_args={"max_evals": 200}
            ),
            "lig": NlpCaptumLigBackend(model, label_names=LABELS, explainer_compute_args={"n_steps": 2}),
            "lime": NlpLimeBackend(model, label_names=LABELS, explainer_compute_args={"num_samples": 50}),
        }.items()
    }
    assert explain["shap"].token_strings == explain["lig"].token_strings == [text[0].split()[:6]]
    def coverage(other):
        return corpus_agreement(explain["shap"], {other: explain[other]}, label_idx=0).coverage.iloc[0]

    assert coverage("lig") == 1.0
    assert coverage("lime") == pytest.approx(0.5)


@pytest.mark.parametrize("missing", ["offsets", "alignment and offsets"])
def test_lig_without_offsets_locates_its_words(monkeypatch, missing):
    # A slow tokenizer reports no offsets (and no word ids): words keep their token-built strings
    # and are found in the text by search. On a plain lowercase text that gives the offsets path's units
    # and values.
    model = _tiny_model(CHECKPOINTS["wordpiece-uncased"])
    text = ["i love unbelievably good tea"]
    backend = NlpCaptumLigBackend(model, label_names=LABELS, explainer_compute_args={"n_steps": 2})
    exact = backend.run_explainer(text)
    monkeypatch.setattr(model, "token_offsets", lambda text: None)
    if missing == "alignment and offsets":
        monkeypatch.setattr(model, "word_alignment", lambda text: None)
    located = backend.run_explainer(text)
    assert exact.token_strings == located.token_strings == [text[0].split()]
    assert exact.token_spans == located.token_spans == [[((0, 1),), ((2, 6),), ((7, 19),), ((20, 24),), ((25, 28),)]]
    np.testing.assert_allclose(located.values[0], exact.values[0], atol=1e-6)
    np.testing.assert_allclose(located.base_values, exact.base_values, atol=1e-6)
