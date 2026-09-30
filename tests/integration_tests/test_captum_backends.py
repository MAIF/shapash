"""Integration tests for the Captum-backed NLP features against a real transformer.

Exercises the LayerIntegratedGradients attribution backend (:class:`NlpCaptumLigBackend`) and the
token-removal counterfactual generator (:class:`AblationFlipGenerator`) end to end on the emotion
distilbert model used by the demos. Skipped automatically when ``transformers`` / ``torch`` / ``captum``
are not installed (or the model is unavailable).

``AblationFlipGenerator`` no longer needs captum (it scores leave-one-out through the model's own
``predict``), so its tests are gated more tightly here than they strictly need to be — they share this
module's real-transformer fixture.
"""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")
pytest.importorskip("captum")
pytestmark = pytest.mark.nlp

from shapash.backend import NlpCaptumLigBackend, NlpShapBackend  # noqa: E402
from shapash.compute.generators import AblationFlipGenerator  # noqa: E402
from shapash.explainer.nlp_explainer import NlpExplainer  # noqa: E402
from shapash.model import HFClassifierModel  # noqa: E402

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@pytest.fixture(scope="module")
def model():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"model unavailable: {exc}")
    return HFClassifierModel(classifier, tokenizer, label_names=LABELS)


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


def test_ablation_flip_finds_flip(model):
    """FeatureAblation-scored token removal flips a confidently classified sample, minimally."""
    gen = AblationFlipGenerator(model)
    cfs = gen.generate("i am so happy today", config={"num_examples": 3, "max_ablations": 3})
    assert cfs, "AblationFlip found no counterfactual on a confident sample"
    for cf in cfs:
        assert cf.new_label != cf.orig_label
        assert cf.new_text != cf.original_text
        assert all(new == "" for _, _, new in cf.substitutions)  # removals record an empty replacement
        # Removed tokens really are gone and the reported flip holds under the model.
        assert model.predict([cf.new_text])[0].argmax() != model.predict([cf.original_text])[0].argmax()
    # minimality: no returned removal set is a strict superset of another
    sets = [frozenset(cf.flipped_positions) for cf in cfs]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j:
                assert not (b < a)


def test_ablation_flip_auto_bound_for_pipeline(model):
    """A prediction-only pipeline model gets AblationFlip auto-bound (HotFlip cannot apply)."""
    pipe = transformers.pipeline("text-classification", model=model.classifier, tokenizer=model.tokenizer, top_k=None)
    from shapash.model import HFPipelineModel

    pm = HFPipelineModel(pipe, label_names=LABELS)
    xpl = NlpExplainer(pm, label_names=LABELS)
    assert xpl.can_counterfactual()
    assert isinstance(xpl.cf_generator, AblationFlipGenerator)


# ---------------------------------------------------------------------------
# Explanation space: raw-logit SHAP and the shared [MASK] baseline
# (docs/architecture/explanation-space.md §8, phases 1b and 3)
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


def test_shap_and_lig_share_the_mask_reference(model):
    shap_backend = NlpShapBackend(model, label_names=LABELS, output_space="logit")
    lig_backend = NlpCaptumLigBackend(model, label_names=LABELS)
    assert shap_backend.baseline_token == lig_backend.baseline_token == model.tokenizer.mask_token
    input_ids, _, _ = model.encode("i am so happy today")
    ref = model.reference_ids(input_ids)[0, 1:-1]
    assert (ref == model.tokenizer.mask_token_id).all()


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
