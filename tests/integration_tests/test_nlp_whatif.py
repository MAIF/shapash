"""Integration tests for the NLP what-if stack against a real transformer.

Exercises the full-capability HuggingFace adapter (predict / embeddings / gradients) and the
HotFlip generator on the emotion distilbert model used by the demos. Skipped automatically when
``transformers``/``torch`` are not installed.
"""

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")
pytestmark = pytest.mark.nlp

from shapash.compute.generators import AblationFlipGenerator, HotFlipGenerator, TokenListField  # noqa: E402
from shapash.explainer.interactive import InteractiveEngine  # noqa: E402
from shapash.explainer.nlp_explainer import NlpExplainer  # noqa: E402
from shapash.model import HFClassifierModel, HFPipelineModel  # noqa: E402
from shapash.model.base import SupportsEmbeddings, SupportsGradients, has_capabilities  # noqa: E402
from shapash.webapp.nlp_app import NlpWebApp  # noqa: E402

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@pytest.fixture(scope="module")
def hf():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    except Exception as exc:  # network / cache miss
        pytest.skip(f"model unavailable: {exc}")
    pipe = transformers.pipeline("text-classification", model=classifier, tokenizer=tokenizer, top_k=None)
    return classifier, tokenizer, pipe


def test_pipeline_adapter_predict_and_capabilities(hf):
    _, _, pipe = hf
    model = HFPipelineModel(pipe, label_names=LABELS)
    probs = model.predict(["i am so happy today", "i feel terrified and alone"])
    assert probs.shape == (2, len(LABELS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert not has_capabilities(model, SupportsGradients)


def test_classifier_adapter_full_capabilities(hf):
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)
    assert has_capabilities(model, SupportsGradients, SupportsEmbeddings)

    probs = model.predict(["i am so happy today"])
    assert probs.shape == (1, len(LABELS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    vocab, matrix = model.get_embedding_table()
    assert matrix.shape[0] == tokenizer.vocab_size == len(vocab)
    assert matrix.shape[1] == classifier.config.dim

    emb = model.embed(["i am so happy today", "hello world"])
    assert emb.shape == (2, classifier.config.dim)

    tokens, grads = model.token_gradients("i am so happy today", int(probs[0].argmax()))
    assert len(tokens) == grads.shape[0] > 0
    assert np.isfinite(grads).all()
    assert (np.abs(grads).sum(axis=1) > 0).all()
    assert all(not (t.startswith("[") and t.endswith("]")) for t in tokens)


def test_hotflip_finds_flip(hf):
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)
    gen = HotFlipGenerator(model)
    cfs = gen.generate("i am so happy today", config={"num_examples": 3, "max_flips": 2})
    assert cfs, "HotFlip found no counterfactual on a confidently-classified sample"
    for cf in cfs:
        assert cf.new_label != cf.orig_label
        assert cf.new_text != cf.original_text
    # minimality: no returned set is a strict superset of another
    sets = [frozenset(cf.flipped_positions) for cf in cfs]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j:
                assert not (b < a)


def test_hotflip_counterfactuals_are_wellformed_words(hf):
    """Replacements are real words, not sub-word pieces — and are model-verified, not linear guesses.

    Regression test for two HotFlip bugs: (1) committing to the single first-order-best candidate per
    position, which is an unreliable proxy that often does not flip; and (2) admitting sub-word
    (``##``) / non-word tokens as replacements, which produced malformed text like ``"i beforewu"``.
    The generator now shortlists top-K word candidates and re-scores them against the model.
    """
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)
    gen = HotFlipGenerator(model)

    cfs = gen.generate("i feel great", config={"num_examples": 5, "max_flips": 2})
    assert cfs, "HotFlip found no counterfactual on a confidently-classified sample"
    for cf in cfs:
        # Every substituted replacement is a plausible standalone word (no '##' pieces, no specials).
        for _pos, _old, new in cf.substitutions:
            assert new.isalpha(), f"replacement {new!r} is not a well-formed word token"
        # Rebuilt text stays well-formed and the reported flip really holds under the model.
        assert "##" not in cf.new_text and "[" not in cf.new_text
        assert cf.new_label != cf.orig_label
        cf_probs = model.predict([cf.new_text])[0]
        assert LABELS[int(cf_probs.argmax())] == cf.new_label


def test_explainer_interactive_engine_with_classifier(hf):
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)
    xpl = NlpExplainer(model, label_names=LABELS)

    assert isinstance(xpl, InteractiveEngine)
    assert xpl.can_edit()
    assert xpl.can_counterfactual()  # HotFlip auto-built from the gradient-capable model
    assert isinstance(xpl.cf_generator, HotFlipGenerator)

    label, probs = xpl.predict("i am so happy today")
    assert label in LABELS
    assert abs(sum(probs.values()) - 1.0) < 1e-4

    explanation = xpl.explain(["i am so happy today", "i feel terrified and alone"], y=["joy", "fear"])
    assert list(explanation.y_true) == ["joy", "fear"]
    contribs, elabel, eprobs = xpl.explain_text("i am furious about this")
    assert len(contribs.token_strings) == 1
    assert elabel in LABELS
    assert set(eprobs.keys()) == set(LABELS)

    cfs = xpl.generate_counterfactuals("i am so happy today", config={"num_examples": 2, "max_flips": 2})
    assert all(cf.new_label != cf.orig_label for cf in cfs)


def test_explainer_pipeline_gets_ablation_flip_but_not_hotflip(hf):
    """A predict-only pipeline can't run gradient-based HotFlip, but AblationFlip (forward-pass-only,
    needs only tokenization) auto-binds as the fallback — so it still gets a What-if Lab.
    """
    _, _, pipe = hf
    xpl = NlpExplainer(pipe, label_names=LABELS)  # predict-only pipeline
    assert xpl.can_edit()
    assert xpl.can_counterfactual()
    assert isinstance(xpl.cf_generator, AblationFlipGenerator)
    assert [name for name, _ in xpl.available_cf_generators()] == ["ablation_flip"]
    assert set(xpl.cf_config_spec()) == {"num_examples", "max_ablations", "tokens_to_ignore"}


def _find_callback_key(app, output_substr):
    """Return the callback_map key whose outputs mention ``output_substr``."""
    for key in app.callback_map:
        if output_substr in key:
            return key
    raise KeyError(output_substr)


def _post_callback(app, output_key, inputs, state):
    """Drive one Dash callback via the Flask test client, returning the parsed response."""
    spec = app.callback_map[output_key]
    raw_out = spec["output"]
    raw_out = raw_out if isinstance(raw_out, list) else [raw_out]
    outputs = [{"id": o.component_id, "property": o.component_property} for o in raw_out]
    body = {
        "output": output_key,
        "outputs": outputs if len(outputs) > 1 else outputs[0],
        "inputs": inputs,
        "state": state,
        "changedPropIds": [f"{i['id']}.{i['property']}" for i in inputs],
    }
    resp = app.server.test_client().post("/_dash-update-component", json=body)
    return resp


def test_whatif_app_callbacks_end_to_end(hf):
    """Drive the real Predict and Generate callbacks in-process against the live model."""
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)
    xpl = NlpExplainer(model, label_names=LABELS)
    explanation = xpl.explain(["i am so happy today", "i feel terrified and alone"], y=["joy", "fear"])
    webapp = NlpWebApp(explanation, engine=xpl)
    app = webapp.app

    # Predict: edited text -> probability figure + current-datapoint (drives the shared highlight).
    predict_key = _find_callback_key(app, "data-editor-prob")
    r = _post_callback(
        app,
        predict_key,
        inputs=[{"id": "data-editor-predict-btn", "property": "n_clicks", "value": 1}],
        state=[
            {"id": "data-editor-input", "property": "value", "value": "i am furious about this"},
        ],
    )
    assert r.status_code == 200
    payload = r.get_json()["response"]
    assert "figure" in payload["data-editor-prob"]
    assert payload["current-datapoint"]["data"]["text"] == "i am furious about this"
    assert payload["current-datapoint"]["data"]["tokens"]

    # Generate: current datapoint text -> counterfactual results table + store of new texts.
    # The classifier model binds both HotFlip and AblationFlip (see test_explainer_interactive_engine_
    # with_classifier), so the counterfactual panel renders a method selector plus one config-control
    # group per generator, each namespaced "counterfactual-cfg-{generator}-{field}". The callback's
    # State list mirrors that layout exactly (selector, then every generator's fields in order) — build
    # it from the live config specs rather than hardcoding ids, so it keeps matching the panel's layout.
    gen_names = [name for name, _ in xpl.available_cf_generators()]
    assert "hotflip" in gen_names
    overrides = {"hotflip": {"num_examples": 3, "max_flips": 2, "tokens_to_ignore": ""}}
    state = [
        {"id": "current-datapoint", "property": "data", "value": {"text": "i am so happy today"}},
        {"id": "counterfactual-generator", "property": "value", "value": "hotflip"},
    ]
    for gen in gen_names:
        for field_name, field in xpl.cf_config_spec(gen).items():
            default = ",".join(field.default) if isinstance(field, TokenListField) else field.default
            value = overrides.get(gen, {}).get(field_name, default)
            state.append({"id": f"counterfactual-cfg-{gen}-{field_name}", "property": "value", "value": value})

    gen_key = _find_callback_key(app, "counterfactual-results")
    r = _post_callback(
        app,
        gen_key,
        inputs=[{"id": "counterfactual-generate-btn", "property": "n_clicks", "value": 1}],
        state=state,
    )
    assert r.status_code == 200
    payload = r.get_json()["response"]
    assert payload["counterfactual-results"]["children"] is not None
    assert isinstance(payload["counterfactual-store"]["data"], list)


def test_label_noise_detection_with_real_probabilities(hf):
    """Confident learning end to end on real model outputs, with neighbour corroboration.

    The corpus deliberately mislabels two obviously-emotional sentences, so a working pipeline has to
    surface them; the assertions stay on structure and on the planted rows rather than on an exact
    count, since the estimated per-cell counts depend on the model's real calibration.
    """
    classifier, tokenizer, _ = hf
    model = HFClassifierModel(classifier, tokenizer, label_names=LABELS)

    texts = [
        "i am so happy today",
        "i feel terrified and alone",
        "i am furious about this",
        "what a wonderful and joyful day",
        "i am scared of the dark",
        "this fills me with rage",
    ]
    truth = ["joy", "fear", "anger", "joy", "fear", "anger"]
    # Plant two label errors: rows 0 and 2 keep their text but get someone else's label.
    noisy = list(truth)
    noisy[0], noisy[2] = "anger", "joy"

    xpl = NlpExplainer(model, label_names=LABELS).fit(X_reference=texts, y=truth)
    explanation = xpl.explain(texts, y=noisy)

    assert xpl.can_detect_label_noise(explanation)
    assert xpl.can_probe_labels()
    report = xpl.detect_label_noise(explanation, top_n=5)

    assert report.n_samples == len(texts)
    assert report.label_names == LABELS
    assert report.noise_matrix.shape == (len(LABELS), len(LABELS))
    np.testing.assert_allclose(report.noise_matrix.sum(), 1.0, atol=1e-6)
    assert 0.0 <= report.noise_rate <= 1.0
    assert report.thresholds.shape == (len(LABELS),)

    flagged = {issue.index for issue in report.issues}
    assert {0, 2} <= flagged, f"planted mislabels missed; flagged {flagged}"
    for issue in report.issues:
        assert issue.given_label in LABELS
        assert issue.suggested_label in LABELS
        assert issue.suggested_label != issue.given_label
        assert issue.text == texts[issue.index]
        assert issue.probe is not None
        assert issue.probe.top_label in LABELS
        assert 0.0 <= issue.probe.given_prob <= 1.0

    # The probe is fit on the reference corpus, which here carries the *clean* labels — so on the two
    # planted errors it should reject the (wrong) label the batch gave them. Note this reference
    # corpus deliberately reuses the batch texts to keep the assertion deterministic; in real use it
    # must be a separate corpus, or the probe would be scoring its own training rows.
    planted = {issue.index: issue for issue in report.issues if issue.index in {0, 2}}
    for index, issue in planted.items():
        assert not issue.probe.backs_given, f"row {index}: probe defended a planted wrong label"
        assert issue.probe.top_label == truth[index]

    # Without a reference corpus there is no second opinion, and detection still runs.
    bare = NlpExplainer(model, label_names=LABELS)
    bare_explanation = bare.explain(texts, y=noisy)
    assert not bare.can_probe_labels()
    assert all(issue.probe is None for issue in bare.detect_label_noise(bare_explanation, top_n=5).issues)


def test_label_noise_unavailable_without_ground_truth(hf):
    classifier, tokenizer, _ = hf
    xpl = NlpExplainer(HFClassifierModel(classifier, tokenizer, label_names=LABELS), label_names=LABELS)
    explanation = xpl.explain(["i am so happy today", "i feel terrified and alone"])
    assert not xpl.can_detect_label_noise(explanation)
    with pytest.raises(RuntimeError, match="ground-truth labels"):
        xpl.detect_label_noise(explanation)
