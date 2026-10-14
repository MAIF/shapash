"""Integration tests for the NLP attribution backends against a real transformer.

One section per backend — SHAP (:class:`NlpShapBackend`), Layer Integrated Gradients
(:class:`NlpCaptumLigBackend`) and LIME (:class:`NlpLimeBackend`) — plus the cross-backend checks, all
on the emotion distilbert model used by the demos. Skipped automatically when the ``nlp`` extra
(``transformers`` / ``torch`` / ``captum``) is not installed or the model is unavailable; the LIME
section additionally needs the separate ``lime`` extra.
"""

import importlib.util

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("captum")
pytestmark = pytest.mark.nlp

from shapash.backend import NlpCaptumLigBackend, NlpLimeBackend, NlpShapBackend  # noqa: E402
from shapash.explainer.nlp_explainer import NlpExplainer  # noqa: E402
from shapash.explainer.nlp_explanation import NlpExplanation  # noqa: E402
from shapash.model import HFClassifierModel  # noqa: E402
from shapash.webapp.nlp_app import NlpWebApp  # noqa: E402

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ``lime`` is its own extra rather than part of ``nlp``, so only its section is gated on it.
requires_lime = pytest.mark.skipif(importlib.util.find_spec("lime") is None, reason="needs the lime extra")


@pytest.fixture(scope="module")
def model():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"model unavailable: {exc}")
    return HFClassifierModel(classifier, tokenizer, label_names=LABELS)


# ---------------------------------------------------------------------------
# SHAP: raw-logit explanation space
# ---------------------------------------------------------------------------

_SPACE_TEXTS = ["i am so happy today", "i feel terrified and alone"]
_SHAP_FAST = {"max_evals": 60}


def test_shap_logit_space_is_exactly_additive_against_predict_logits(model):
    backend = NlpShapBackend(model, label_names=LABELS, output_space="logit", explainer_compute_args=_SHAP_FAST)
    raw = backend.run_explainer(_SPACE_TEXTS)
    logits = model.predict_logits(_SPACE_TEXTS)
    for i in range(len(_SPACE_TEXTS)):
        np.testing.assert_allclose(raw.base_values[i] + raw.values[i].sum(axis=0), logits[i], atol=1e-4)


def test_shap_logit_space_does_not_cancel_across_classes(model):
    # Probability-space contributions sum to ~0 over classes for every word (the payoff sums to 1 for
    # every coalition); logit-space ones do not — the reason the space exists.
    probability = NlpShapBackend(model, label_names=LABELS, explainer_compute_args=_SHAP_FAST)
    logit = NlpShapBackend(model, label_names=LABELS, output_space="logit", explainer_compute_args=_SHAP_FAST)
    prob_sum = np.abs(probability.run_explainer(_SPACE_TEXTS[:1]).values[0].sum(axis=1)).max()
    logit_sum = np.abs(logit.run_explainer(_SPACE_TEXTS[:1]).values[0].sum(axis=1)).max()
    assert prob_sum < 1e-5
    assert logit_sum > 1e-2


def test_explainer_records_logit_space_and_keys_it_apart(model, tmp_path):
    probability = NlpExplainer(
        model, label_names=LABELS, backend=NlpShapBackend(model, label_names=LABELS, explainer_compute_args=_SHAP_FAST)
    )
    logit = NlpExplainer(
        model,
        label_names=LABELS,
        backend=NlpShapBackend(model, label_names=LABELS, output_space="logit", explainer_compute_args=_SHAP_FAST),
    )
    assert probability._compute_key(_SPACE_TEXTS) != logit._compute_key(_SPACE_TEXTS)
    explanation = logit.explain(_SPACE_TEXTS, cache_dir=tmp_path)
    assert (explanation.output_space, explanation.baseline_token) == ("logit", "[MASK]")
    assert probability.explain(_SPACE_TEXTS, cache_dir=tmp_path).output_space == "probability"


# ---------------------------------------------------------------------------
# Captum LIG
# ---------------------------------------------------------------------------


def test_captum_ig_surface(model):
    """The classifier adapter exposes the LayerIntegratedGradients surface with sane shapes."""
    input_ids, attention_mask, tokens = model.encode("i am so happy today")
    assert input_ids.shape == attention_mask.shape
    assert len(tokens) == input_ids.shape[1]
    ref_ids = model.reference_ids(input_ids)
    assert ref_ids.shape == input_ids.shape
    # Special tokens are preserved in the baseline; content ids are replaced.
    assert ref_ids[0, 0].item() == input_ids[0, 0].item()  # [CLS]
    assert ref_ids[0, -1].item() == input_ids[0, -1].item()  # [SEP]
    logits = model.logits(input_ids, attention_mask)
    assert logits.shape == (1, len(LABELS))


def test_lig_backend_contributions(model):
    """LIG produces per-token, per-class contributions that satisfy the completeness relation."""
    backend = NlpCaptumLigBackend(model, label_names=LABELS)
    raw = backend.run_explainer(["i am so happy today", "i feel terrified"])

    assert len(raw.values) == 2
    assert raw.base_values.shape == (2, len(LABELS))
    for values, tokens in zip(raw.values, raw.token_strings):
        assert values.shape == (len(tokens), len(LABELS))
        assert np.isfinite(values).all()
        # Subwords are merged to whole words and specials dropped — no [CLS]/[SEP]/## leak into highlights.
        assert all(not (t.startswith("##") or (t.startswith("[") and t.endswith("]"))) for t in tokens)

    # LIG is a completion method: base + sum(attributions) ≈ logits(x) for each class.
    input_ids, attention_mask, _ = model.encode("i am so happy today")
    logits_x = model.logits(input_ids, attention_mask)[0].detach().cpu().numpy()
    recon = raw.base_values[0] + raw.values[0].sum(axis=0)
    np.testing.assert_allclose(recon, logits_x, atol=0.2)


def test_lig_backend_through_explainer_and_word_importance(model):
    """The LIG backend drives NlpExplainer.explain and the shared word-importance aggregation."""
    xpl = NlpExplainer(model, label_names=LABELS, backend=NlpCaptumLigBackend(model, label_names=LABELS))
    explanation = xpl.explain(["i am so happy today", "i feel terrified and alone"])
    assert len(explanation) == 2
    joy_idx = LABELS.index("joy")
    word_imp = explanation.word_importance(joy_idx, n_top=5)
    assert len(word_imp) > 0  # some words survive special-token filtering


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

_LIME_TEXTS = ["i am so happy and excited today", "i feel sad and lonely tonight"]
# Seeded so runs are reproducible; enough samples for a stable top word on these short texts.
_LIME_ARGS = {"random_state": 0}
_LIME_COMPUTE = {"num_samples": 300}


def _lime_backend(scorer, **kwargs):
    return NlpLimeBackend(scorer, explainer_args=_LIME_ARGS, explainer_compute_args=_LIME_COMPUTE, **kwargs)


@requires_lime
def test_lime_text_model_input_explains_every_class_with_the_model_s_labels(model):
    backend = _lime_backend(model)  # a TextModel, no label_names: both come from the model
    assert backend._classes == LABELS

    raw = backend.run_explainer(_LIME_TEXTS)

    assert raw.base_values.shape == (len(_LIME_TEXTS), len(LABELS))
    for text, tokens, values in zip(_LIME_TEXTS, raw.token_strings, raw.values, strict=True):
        assert values.shape == (len(tokens), len(LABELS))
        assert np.isfinite(values).all()
        assert set(tokens) <= set(text.split())  # LIME explains the text's own words
        assert all(np.any(values[:, col]) for col in range(len(LABELS)))  # no silently-empty class


@requires_lime
def test_lime_attributes_the_obvious_word(model):
    """A sanity check on the numbers themselves, not just their shapes."""
    raw = _lime_backend(model).run_explainer(_LIME_TEXTS[:1])
    joy = raw.values[0][:, LABELS.index("joy")]
    assert raw.token_strings[0][int(joy.argmax())] in {"happy", "excited"}


@requires_lime
def test_lime_seeded_runs_are_reproducible(model):
    first = _lime_backend(model).run_explainer(_LIME_TEXTS[:1]).values[0]
    second = _lime_backend(model).run_explainer(_LIME_TEXTS[:1]).values[0]
    np.testing.assert_array_equal(first, second)


@requires_lime
def test_lime_pipeline_input_matches_the_text_model_input(model):
    """The ``list[list[dict]]`` pipeline adapter must reorder scores by label, not by position."""
    pipe = transformers.pipeline(
        "text-classification", model=model.classifier, tokenizer=model.tokenizer, top_k=None, device=-1
    )
    via_pipeline = _lime_backend(pipe, label_names=LABELS).run_explainer(_LIME_TEXTS[:1])
    via_model = _lime_backend(model).run_explainer(_LIME_TEXTS[:1])
    assert via_pipeline.token_strings == via_model.token_strings
    np.testing.assert_allclose(via_pipeline.values[0], via_model.values[0], atol=1e-4)


@requires_lime
def test_lime_explainer_artifact_and_round_trip(model, tmp_path):
    xpl = NlpExplainer(model, backend=_lime_backend(model))
    explanation = xpl.explain(_LIME_TEXTS, y=["joy", "sadness"])

    assert explanation.backend_name == "nlp_lime"
    assert explanation.is_additive is False
    assert explanation.output_space == "probability"
    assert explanation.baseline_token is None  # bow=True removes words, substitutes nothing
    assert list(explanation.y_pred) == [LABELS[i] for i in model.predict(_LIME_TEXTS).argmax(axis=1)]

    path = tmp_path / "lime.xpl"
    explanation.save(path)
    reloaded = NlpExplanation.load(path)
    assert reloaded.backend_name == "nlp_lime" and reloaded.is_additive is False
    for before, after in zip(explanation.values, reloaded.values, strict=True):
        np.testing.assert_allclose(before, after)


@requires_lime
def test_lime_live_explain_text(model):
    xpl = NlpExplainer(model, backend=_lime_backend(model))
    contributions, label, probabilities = xpl.explain_text("i am thrilled about this")
    assert label in LABELS
    assert set(probabilities) == set(LABELS)
    assert contributions.values[0].shape == (len(contributions.token_strings[0]), len(LABELS))


@requires_lime
def test_webapp_hides_the_waterfall_for_lime(model):
    xpl = NlpExplainer(model, backend=_lime_backend(model))
    app = NlpWebApp(xpl.explain(_LIME_TEXTS), engine=xpl)
    assert app._tab_groups["lower-right-tabs"] == ["highlight"]
    assert "editor" in app._tab_groups["left-tabs"]  # the What-if Lab still mounts with a live TextModel


# ---------------------------------------------------------------------------
# Cross-backend
# ---------------------------------------------------------------------------


def test_shap_and_lig_share_the_mask_reference(model):
    shap_backend = NlpShapBackend(model, label_names=LABELS, output_space="logit")
    lig_backend = NlpCaptumLigBackend(model, label_names=LABELS)
    assert shap_backend.baseline_token == lig_backend.baseline_token == model.tokenizer.mask_token
    input_ids, _, _ = model.encode("i am so happy today")
    ref = model.reference_ids(input_ids)[0, 1:-1]
    assert (ref == model.tokenizer.mask_token_id).all()
