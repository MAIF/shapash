"""Backend units across tokenizer families: placed in the source text, and comparable across backends.

Each backend cuts a text into units its own way, and each tokenizer family moves the cuts again
(WordPiece splits every punctuation mark, byte-level BPE glues symbols together, SentencePiece splits
on whitespace only, uncased tokenizers rewrite the text). These tests run the real SHAP, LIG and LIME
``run_explainer`` through ``NlpExplainer.explain`` on one checkpoint per family, against text built
to break string matching (accents, decomposed accents, emoji, curly quotes, full-width characters,
double spaces, newlines), and check the invariants alignment relies on:

- every unit is placed: it has spans, and its string is the source text at its span;
- no unit is whitespace;
- SHAP and LIG cover the same characters, since both read the same tokenizer's offsets;
- aligning backends drops nothing: each backend's values sum to the same total before and after.

The models are randomly initialised from each checkpoint's config at a tiny size, so only the config
and tokenizer are downloaded; the numbers mean nothing, the units are real. Skipped when the ``nlp``
or ``lime`` extra is missing or a checkpoint cannot be fetched.
"""

import numpy as np
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
        "lig": NlpCaptumLigBackend(model, label_names=LABELS, explainer_compute_args={"n_steps": 2}),
        "lime": NlpLimeBackend(model, label_names=LABELS, explainer_compute_args={"num_samples": 50}),
    }
    out = {
        name: NlpExplainer(model, label_names=LABELS, backend=backend).explain(TEXTS)
        for name, backend in backends.items()
    }
    out["model"] = model
    out["shap_masker"] = backends["shap"].explainer.masker
    return out


def _shap_input(masker, text: str) -> str:
    """The string SHAP's ``Text`` masker hands the model when nothing is masked.

    Not always ``text``: the masker rebuilds it from its segments, so it collapses whitespace and,
    where tokens share a character (an emoji split by byte-level BPE), repeats it. An upstream
    behaviour of ``shap.maskers.Text``, outside what alignment can fix.
    """
    segments, _ = masker.token_segments(text)
    return str(masker(np.ones(len(segments), dtype=bool), text)[0][0])


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


def test_shap_and_lig_cover_the_same_characters(explanations):
    for row, text in enumerate(TEXTS):
        covered = {}
        for backend in ("shap", "lig"):
            chars = set()
            for ((start, end),) in explanations[backend].token_spans[row]:
                chars.update(range(start, end))
            covered[backend] = chars
        assert covered["shap"] == covered["lig"], f"row {row}: {text!r}"
        visible = {i for i, c in enumerate(text) if not c.isspace()}
        assert covered["shap"] <= visible


def test_shap_stays_exactly_additive_after_merging(explanations):
    # Against the string SHAP explains (see _shap_input): merging units must not move the total.
    exp, model = explanations["shap"], explanations["model"]
    logits = model.predict_logits([_shap_input(explanations["shap_masker"], text) for text in TEXTS])
    for row in range(len(TEXTS)):
        np.testing.assert_allclose(exp.base_values[row] + exp.values[row].sum(axis=0), logits[row], atol=1e-4)


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
    model.max_length = 8  # LIG truncates; SHAP's masker segments the whole text
    text = ["one two three four five six seven eight nine ten eleven twelve"]
    shap = NlpExplainer(
        model,
        label_names=LABELS,
        backend=NlpShapBackend(
            model, label_names=LABELS, output_space="logit", explainer_compute_args={"max_evals": 200}
        ),
    ).explain(text)
    lig = NlpExplainer(
        model,
        label_names=LABELS,
        backend=NlpCaptumLigBackend(model, label_names=LABELS, explainer_compute_args={"n_steps": 2}),
    ).explain(text)
    assert len(lig.token_strings[0]) < len(shap.token_strings[0])
    row = corpus_agreement(shap, {"lig": lig}, label_idx=0).iloc[0]
    assert 0 < row.coverage < 1
