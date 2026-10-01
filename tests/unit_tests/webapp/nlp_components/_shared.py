"""Fixtures and helpers shared by 2+ of the ``tests/unit_tests/webapp/nlp_components`` test modules."""

from dataclasses import replace

import numpy as np
import pandas as pd

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.diagnostics.label_noise import detect_label_issues
from shapash.compute.diagnostics.label_probe import LabelProbe
from shapash.compute.generators.base import Counterfactual, IntField, TokenListField
from shapash.compute.retrieval.similar_examples import Neighbor
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_components.base import AppContext
from shapash.webapp.nlp_components.datapoint import pack_datapoint

LABEL_NAMES = ["neg", "pos"]


def _ctx(explanation, engine=None, coords=None) -> AppContext:
    """The context ``NlpWebApp`` builds, for calling a component directly."""
    return AppContext(explanation, engine=engine, coords=coords)


# A tiny lexically separable corpus for the independent label probe. "bad"/"awful" sit in "neg",
# so a row labelled "neg" gets backed and a row labelled "pos" gets rejected.
_PROBE_CORPUS = (
    [
        "i am happy",
        "happy and glad",
        "so glad today",
        "a happy glad day",
        "this is bad",
        "bad and awful",
        "so awful today",
        "a bad awful day",
    ],
    ["pos"] * 4 + ["neg"] * 4,
)


def _contributions() -> NlpContributions:
    token_strings = [["i", "am", "happy"], ["this", "is", "bad"]]
    values = [np.random.randn(3, 2), np.random.randn(3, 2)]
    base_values = np.zeros((2, 2))
    return NlpContributions(token_strings=token_strings, values=values, base_values=base_values)


class FakeEngine:
    """Minimal explainer/engine stand-in exposing the InteractiveEngine surface + compiled data."""

    def __init__(
        self,
        can_edit: bool,
        can_cf: bool,
        can_similar: bool = False,
        has_labels: bool = False,
        probe_corpus: tuple[list[str], list[str]] | None = None,
    ):
        self.probe_corpus = probe_corpus
        self._can_edit = can_edit
        self._can_cf = can_cf
        self._can_similar = can_similar
        # A retriever-like handle the SimilarExamples panel reads its layer caption from.
        self._retriever = type("R", (), {"layer": "pre_classifier"})() if can_similar else None
        self.label_names = LABEL_NAMES
        self.texts = pd.Series(["i am happy", "this is bad"], index=pd.RangeIndex(2))
        self.contributions = _contributions()
        self.y_pred = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="prediction")
        self.y_prob = pd.DataFrame({"neg": [0.2, 0.8], "pos": [0.8, 0.2]}, index=pd.RangeIndex(2))
        self.y_true = None
        if has_labels:
            # A batch with exactly one planted label error. With two samples and two classes the
            # arrangement is forced: both classes must carry a label (or the unlabelled one has no
            # estimable threshold and can never be suggested), so sample 0 is labelled "neg" while
            # the model confidently says "pos", and sample 1 is labelled "pos" and agrees.
            self.y_prob = pd.DataFrame({"neg": [0.1, 0.1], "pos": [0.9, 0.9]}, index=pd.RangeIndex(2))
            self.y_pred = pd.Series(["pos", "pos"], index=pd.RangeIndex(2), name="prediction")
            self.y_true = pd.Series(["neg", "pos"], index=pd.RangeIndex(2), name="ground_truth")
        # Ground truth is off by default so the existing layout tests, which pin the exact tab
        # groups, keep seeing neither the Error Analysis nor the Label Noise tab.
        self.detect_calls = []

    def to_explanation(self) -> NlpExplanation:
        """The ``NlpExplanation`` a real ``explain()`` call would have produced for this batch.

        ``NlpWebApp`` holds an explanation, not the engine — this is what tests pass as the first
        argument, while ``self`` (the ``FakeEngine``) is passed separately as ``engine=``.
        """
        return NlpExplanation(
            texts=self.texts,
            token_strings=self.contributions.token_strings,
            values=self.contributions.values,
            base_values=self.contributions.base_values,
            y_pred=self.y_pred,
            y_prob=self.y_prob,
            y_true=self.y_true,
            label_names=self.label_names,
            folds_case=None,
            backend_name="fake",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def can_detect_label_noise(self, explanation=None):
        return self.y_true is not None

    def can_probe_labels(self):
        return self.probe_corpus is not None

    def detect_label_noise(self, explanation, top_n=50, score="self_confidence", probe=True):
        self.detect_calls.append({"top_n": top_n, "score": score, "probe": probe})
        report = detect_label_issues(
            self.y_prob.to_numpy(dtype=float),
            [str(v) for v in self.y_true.tolist()],
            [str(t) for t in self.texts.tolist()],
            list(self.y_prob.columns),
            top_n=top_n,
            score=score,
        )
        if probe and self.can_probe_labels() and report.issues:
            verdicts = LabelProbe(*self.probe_corpus).verdicts(
                [i.text for i in report.issues], [i.given_label for i in report.issues]
            )
            report = replace(
                report,
                issues=[replace(i, probe=v) for i, v in zip(report.issues, verdicts, strict=True)],
            )
        return report

    def can_edit(self):
        return self._can_edit

    def can_counterfactual(self):
        return self._can_cf

    def can_find_similar(self):
        return self._can_similar

    def find_similar(self, text, top_k=5):
        return [
            Neighbor(index=0, score=0.99, text="i am joyful", label="pos"),
            Neighbor(index=1, score=0.80, text="this is awful", label="neg"),
        ][:top_k]

    def find_similar_threshold(self, text, threshold=0.95, limit=50):
        all_neighbors = [
            Neighbor(index=0, score=0.99, text="i am joyful", label="pos"),
            Neighbor(index=1, score=0.96, text="so glad today", label="pos"),
            Neighbor(index=2, score=0.80, text="this is awful", label="neg"),
        ]
        matches = [n for n in all_neighbors if n.score > threshold]
        return matches[:limit], len(matches)

    def available_cf_generators(self):
        return [("hotflip", "HotFlip"), ("ablation_flip", "Ablation")]

    def cf_config_spec(self, generator=None):
        max_field = "max_ablations" if generator == "ablation_flip" else "max_flips"
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            max_field: IntField(label="Max token edits", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    def predict(self, text):
        return "pos", {"neg": 0.3, "pos": 0.7}

    def explain_text(self, text):
        c = _contributions()
        return c, "pos", {"neg": 0.3, "pos": 0.7}

    def generate_counterfactuals(self, text, config=None, generator=None):
        return [
            Counterfactual(
                original_text=text,
                new_text=text.replace("happy", "bad"),
                tokens=["i", "am", "happy"],
                flipped_positions=[2],
                substitutions=[(2, "happy", "bad")],
                orig_label="pos",
                new_label="neg",
                orig_prob=0.8,
                new_prob=0.7,
                prob_delta=0.5,
            )
        ]


def _collect_ids(node, found):
    """Recursively collect all string component ids in a Dash layout tree.

    Also descends into RadioItems/Checklist ``options[].label`` — components nested there (e.g. a
    numeric input inline with its radio button) aren't under ``.children``, but Dash still renders
    and wires them, since ``label`` is documented to accept a component.
    """
    cid = getattr(node, "id", None)
    if isinstance(cid, str):
        found.add(cid)
    options = getattr(node, "options", None)
    if isinstance(options, (list, tuple)):
        for opt in options:
            label = opt.get("label") if isinstance(opt, dict) else None
            if isinstance(label, (list, tuple)):
                for item in label:
                    _collect_ids(item, found)
            elif label is not None:
                _collect_ids(label, found)
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for ch in children:
            _collect_ids(ch, found)
    else:
        _collect_ids(children, found)


def _callback_binding_ids(app, output_substr):
    """Return the (id, property) pairs bound as inputs+state for the callback with this output."""
    for key, spec in app.app.callback_map.items():
        if output_substr in key:
            pairs = [(i["id"], i["property"]) for i in spec["inputs"]]
            pairs += [(s["id"], s["property"]) for s in spec.get("state", [])]
            return pairs
    raise KeyError(output_substr)


def _callback(app, out_substr):
    for key, spec in app.app.callback_map.items():
        if out_substr in key:
            return getattr(spec["callback"], "__wrapped__", spec["callback"])
    raise KeyError(out_substr)


def _make_local_panel_explanation():
    """Two samples, two classes — enough for the sentence-highlight / waterfall panels."""
    texts = pd.Series(["i am happy", "so sad"])
    return NlpExplanation(
        texts=texts,
        token_strings=[["i", "am", "happy"], ["so", "sad"]],
        values=[np.array([[0.1, -0.1], [0.2, -0.2], [0.4, -0.4]]), np.array([[-0.3, 0.3], [-0.5, 0.5]])],
        base_values=np.array([[0.0, 0.0], [0.0, 0.0]]),
        y_pred=pd.Series(["pos", "neg"], index=texts.index, name="prediction"),
        y_prob=None,
        y_true=None,
        label_names=LABEL_NAMES,
        folds_case=True,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


def _make_local_panel_datapoint(label="pos"):
    return pack_datapoint(
        text="i am happy",
        orig_idx=0,
        tokens=["i", "am", "happy"],
        values=np.array([[0.1, -0.1], [0.2, -0.2], [0.4, -0.4]]),
        base_values=np.array([0.0, 0.0]),
        label=label,
    )
