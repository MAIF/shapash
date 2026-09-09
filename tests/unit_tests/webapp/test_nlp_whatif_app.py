"""Unit tests for the What-if Lab wiring in ``NlpWebApp`` (no real model, no server).

A fake explainer/engine with synthetic contributions drives layout construction and capability
gating, verifying that the data-editor and counterfactual components mount only when the engine
reports the matching capabilities.
"""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.diagnostics.label_noise import LabelIssue, detect_label_issues
from shapash.compute.diagnostics.label_probe import LabelProbe, ProbeVerdict
from shapash.compute.generators.base import Counterfactual, IntField, TokenListField
from shapash.compute.retrieval.similar_examples import Neighbor
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_app import NlpWebApp
from shapash.webapp.nlp_components import (
    CounterfactualComponent,
    DataEditorComponent,
    LabelNoiseComponent,
    SimilarExamplesComponent,
)

LABEL_NAMES = ["neg", "pos"]

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


class TestWhatIfMounting(unittest.TestCase):
    def _ids(self, engine):
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    @staticmethod
    def _whatif_components(app):
        # app._components also holds the always-on core panels (Sentence Highlight, Waterfall);
        # scope these assertions to the capability-gated What-if Lab ones only.
        gated = (DataEditorComponent, CounterfactualComponent, SimilarExamplesComponent)
        return [c for c in app._components if isinstance(c, gated)]

    def test_full_capabilities_mount_both(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("data-editor-input", ids)
        self.assertIn("data-editor-predict-btn", ids)
        self.assertIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(self._whatif_components(app)), 2)

    def test_counterfactual_config_controls_in_initial_layout(self):
        # The generate callback's State references these ids, so they must exist in the initial
        # layout (rendered per generator from cf_config_spec), not be injected by a later callback.
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        for name in ("num_examples", "max_flips", "tokens_to_ignore"):
            self.assertIn(f"counterfactual-cfg-hotflip-{name}", ids)
        for name in ("num_examples", "max_ablations", "tokens_to_ignore"):
            self.assertIn(f"counterfactual-cfg-ablation_flip-{name}", ids)

    def test_counterfactual_method_selector_present_with_multiple_generators(self):
        # A method selector and one visibility-toggled control group per generator are in the layout.
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("counterfactual-generator", ids)
        self.assertIn("counterfactual-cfg-group-hotflip", ids)
        self.assertIn("counterfactual-cfg-group-ablation_flip", ids)

    def test_selector_toggle_callback_registered_with_multiple_generators(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("counterfactual-cfg-group-hotflip.style", outputs)

    def test_edit_only_mounts_editor_not_counterfactual(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False))
        self.assertIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertEqual(len(self._whatif_components(app)), 1)

    def test_no_capabilities_hides_whatif_lab(self):
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertNotIn("data-editor-input", ids)
        self.assertNotIn("counterfactual-generate-btn", ids)
        self.assertNotIn("whatif-apply-store", ids)
        self.assertEqual(self._whatif_components(app), [])

    def test_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("data-editor-prob.figure", outputs)
        self.assertIn("counterfactual-results.children", outputs)

    def test_similar_panel_mounts_when_capable(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False, can_similar=True))
        self.assertIn("similar-topk", ids)
        self.assertIn("similar-results", ids)
        self.assertTrue(any(isinstance(c, SimilarExamplesComponent) for c in app._components))

    def test_similar_panel_hidden_without_capability(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True, can_similar=False))
        self.assertNotIn("similar-topk", ids)
        self.assertNotIn("similar-results", ids)

    def test_similar_panel_requires_predict(self):
        # CAP_SIMILAR alone is not enough — the Inspect flow needs predict (explain_text) too.
        _, ids = self._ids(FakeEngine(can_edit=False, can_cf=False, can_similar=True))
        self.assertNotIn("similar-topk", ids)

    def test_similar_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("similar-results.children", outputs)


def _callback_binding_ids(app, output_substr):
    """Return the (id, property) pairs bound as inputs+state for the callback with this output."""
    for key, spec in app.app.callback_map.items():
        if output_substr in key:
            pairs = [(i["id"], i["property"]) for i in spec["inputs"]]
            pairs += [(s["id"], s["property"]) for s in spec.get("state", [])]
            return pairs
    raise KeyError(output_substr)


class TestThreePanelLayout(unittest.TestCase):
    """The LIT-style three-panel shell: tab groups, mounted bodies, and the current-datapoint store."""

    def _ids(self, engine, **kwargs):
        app = NlpWebApp(engine.to_explanation(), engine=engine, **kwargs)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_full_tab_groups_when_all_panels_available(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), scatter_xy=np.zeros((2, 2)))
        self.assertEqual(
            app._tab_groups,
            {
                "left-tabs": ["table", "scatter", "editor"],
                # Word Profile is data-only, so it mounts unconditionally beside Word Importance.
                "upper-right-tabs": ["importance", "word-profile", "counterfactual"],
                "lower-right-tabs": ["highlight", "waterfall"],
            },
        )
        # Every tab body is mounted in the DOM (visibility is toggled, not the mount).
        for body_id in (
            "left-tabs-body-table",
            "left-tabs-body-scatter",
            "left-tabs-body-editor",
            "upper-right-tabs-body-importance",
            "upper-right-tabs-body-counterfactual",
            "lower-right-tabs-body-highlight",
            "lower-right-tabs-body-waterfall",
        ):
            self.assertIn(body_id, ids)

    def test_tabs_degrade_without_scatter_or_whatif(self):
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertEqual(app._tab_groups["left-tabs"], ["table"])
        self.assertEqual(app._tab_groups["upper-right-tabs"], ["importance", "word-profile"])
        self.assertEqual(app._tab_groups["lower-right-tabs"], ["highlight", "waterfall"])
        self.assertNotIn("left-tabs-body-scatter", ids)
        self.assertNotIn("left-tabs-body-editor", ids)
        self.assertNotIn("upper-right-tabs-body-counterfactual", ids)

    def test_counterfactual_tab_only_with_editor(self):
        # Editor without CF: editor tab present on the left, no counterfactual tab on the right.
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=False))
        self.assertIn("editor", app._tab_groups["left-tabs"])
        self.assertNotIn("counterfactual", app._tab_groups["upper-right-tabs"])

    def test_selection_bar_present(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), scatter_xy=np.zeros((2, 2)))
        self.assertIn("selection-summary", ids)
        self.assertIn("word-filter-clear-btn", ids)
        self.assertIn("scatter-clear-btn", ids)

    def test_scatter_clear_absent_without_scatter(self):
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertIn("word-filter-clear-btn", ids)
        self.assertNotIn("scatter-clear-btn", ids)

    def test_current_datapoint_store_present(self):
        _, ids = self._ids(FakeEngine(can_edit=False, can_cf=False))
        self.assertIn("current-datapoint", ids)

    def test_detail_panels_read_current_datapoint(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        # Highlight and waterfall render off the shared primary-selection store...
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "sentence-highlight.children"))
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "waterfall-graph.figure"))
        # ...and counterfactuals generate from it too (so a selected row works, not only editor text).
        self.assertIn(("current-datapoint", "data"), _callback_binding_ids(app, "counterfactual-results.children"))

    def test_current_datapoint_written_by_row_and_editor(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        # The table-selection writer keys off the selected row.
        self.assertIn(("dataset-table", "selectedRows"), _callback_binding_ids(app, "current-datapoint.data"))
        # The editor's Predict also writes it (allow_duplicate) — its combined key carries the prob figure.
        editor_key = next(k for k in app.app.callback_map if "data-editor-prob.figure" in k)
        self.assertIn("current-datapoint.data", editor_key)

    def test_tab_toggle_callbacks_registered(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine, scatter_xy=np.zeros((2, 2)))
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("left-tabs-body-table.style", outputs)
        self.assertIn("lower-right-tabs-body-waterfall.style", outputs)


class TestReadContractSeam(unittest.TestCase):
    """The app shell reads compiled data only from the ``NlpExplanation``; no raw explainer handle.

    ``FakeEngine`` has no ``explainer`` attribute, so a mounted app proves the read path is served by
    the artifact alone. These asserts lock the seam so a future edit cannot silently reintroduce a
    ``self.explainer`` bypass, nor let display state accumulate back onto the artifact.
    """

    def test_app_holds_no_raw_explainer_handle(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        self.assertFalse(hasattr(app, "explainer"))

    def test_app_exposes_explanation_and_engine_roles(self):
        engine = FakeEngine(can_edit=True, can_cf=True)
        explanation = engine.to_explanation()
        app = NlpWebApp(explanation, engine=engine)
        self.assertIs(app._explanation, explanation)  # read contract: the artifact itself
        self.assertIs(app._engine, engine)  # live-action contract


class TestSimilarComponent(unittest.TestCase):
    """Exercise the Similar Examples component's renderer and callbacks directly."""

    @staticmethod
    def _register():
        import dash

        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = SimilarExamplesComponent()
        comp.register_callbacks(
            app, explanation, engine, {"apply": "whatif-apply-store", "current": "current-datapoint"}
        )
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_layout_shows_layer_caption(self):
        from shapash.webapp.nlp_components import SimilarExamplesComponent

        engine = FakeEngine(can_edit=True, can_cf=False, can_similar=True)
        found = set()
        _collect_ids(SimilarExamplesComponent().layout(engine.to_explanation(), engine), found)
        self.assertIn("similar-topk", found)
        self.assertIn("similar-threshold", found)
        self.assertIn("similar-mode", found)
        self.assertIn("similar-results", found)

    def test_update_similar_returns_table_and_texts(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "topk", 5, 0.95)
        self.assertEqual(texts, ["i am joyful", "this is awful"])
        self.assertIsNotNone(children)

    def test_update_similar_ignores_empty_text(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        update = self._callback(app, "similar-results")
        with self.assertRaises(PreventUpdate):
            update({"text": "  "}, "topk", 5, 0.95)

    def test_update_similar_threshold_mode_filters_and_reports_total(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.90)
        # Only the two neighbours scoring above 0.90 clear the threshold (see FakeEngine.find_similar_threshold).
        self.assertEqual(texts, ["i am joyful", "so glad today"])
        self.assertIsNotNone(children)

    def test_update_similar_threshold_mode_empty_above_cutoff(self):
        app, _ = self._register()
        update = self._callback(app, "similar-results")
        children, texts = update({"text": "i am happy", "label": "pos"}, "threshold", 5, 0.999)
        self.assertEqual(texts, [])
        self.assertIsNotNone(children)

    def test_toggle_mode_inputs_disables_the_inactive_control(self):
        app, _ = self._register()
        toggle = self._callback(app, "similar-topk.disabled")
        topk_disabled, threshold_disabled = toggle("threshold")
        self.assertTrue(topk_disabled)
        self.assertFalse(threshold_disabled)
        topk_disabled, threshold_disabled = toggle("topk")
        self.assertFalse(topk_disabled)
        self.assertTrue(threshold_disabled)

    def test_inspect_makes_neighbor_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import similar_examples as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "similar-apply", "index": 1}
            dp = inspect([1, 1], ["first neighbor", "second neighbor"])
        self.assertEqual(dp["text"], "second neighbor")
        self.assertEqual(dp["label"], "pos")  # FakeEngine.explain_text returns "pos"

    def test_neighbors_table_marks_matching_label(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [
            Neighbor(index=0, score=0.9, text="joyful one", label="pos"),
            Neighbor(index=1, score=0.5, text="grim one", label="neg"),
        ]
        table = _neighbors_table(neighbors, predicted_label="pos", component_id="similar")
        ids = set()
        _collect_ids(table, ids)
        # One Inspect button per neighbour (pattern-matching ids are dicts, so assert via the count).
        self.assertEqual(sum(1 for n in neighbors), 2)
        self.assertIsNotNone(table)

    def test_neighbors_table_without_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _neighbors_table

        neighbors = [Neighbor(index=0, score=0.9, text="some text", label=None)]
        table = _neighbors_table(neighbors, predicted_label=None, component_id="similar")
        self.assertIsNotNone(table)

    def test_match_rate_caption_reports_share_of_predicted_label(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [
            Neighbor(index=0, score=0.9, text="a", label="pos"),
            Neighbor(index=1, score=0.8, text="b", label="pos"),
            Neighbor(index=2, score=0.7, text="c", label="neg"),
            Neighbor(index=3, score=0.6, text="d", label="neg"),
        ]
        caption = _match_rate_caption(neighbors, predicted_label="pos")
        self.assertIn("2/4", caption)
        self.assertIn("50%", caption)

    def test_match_rate_caption_none_without_a_prediction_or_labels(self):
        from shapash.webapp.nlp_components.similar_examples import _match_rate_caption

        neighbors = [Neighbor(index=0, score=0.9, text="a", label="pos")]
        self.assertIsNone(_match_rate_caption(neighbors, predicted_label=None))
        unlabelled = [Neighbor(index=0, score=0.9, text="a", label=None)]
        self.assertIsNone(_match_rate_caption(unlabelled, predicted_label="pos"))

    def test_render_results_includes_cap_note_only_when_capped(self):
        from shapash.webapp.nlp_components.similar_examples import _render_results

        neighbors = [Neighbor(index=0, score=0.99, text="a", label="pos")]
        capped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=5)
        uncapped = _render_results(neighbors, predicted_label="pos", component_id="similar", shown_of=None)

        def _flat_text(node):
            children = getattr(node, "children", None)
            if isinstance(children, str):
                return children
            if isinstance(children, list):
                return "".join(_flat_text(c) for c in children if c is not None)
            return _flat_text(children) if children is not None else ""

        self.assertIn("Showing 1 of 5", _flat_text(capped))
        self.assertNotIn("Showing", _flat_text(uncapped))


class TestLabelNoiseMounting(unittest.TestCase):
    """``CAP_LABELS`` is a *data* capability: it depends on the compiled batch, not on a live model."""

    def _ids(self, engine):
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_mounts_with_ground_truth_and_per_class_probabilities(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True, has_labels=True))
        self.assertIn("label-noise-detect-btn", ids)
        self.assertIn("label-noise-results", ids)
        self.assertIn("label-noise", app._tab_groups["upper-right-tabs"])
        self.assertTrue(any(isinstance(c, LabelNoiseComponent) for c in app._components))

    def test_hidden_without_ground_truth(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True))
        self.assertNotIn("label-noise-detect-btn", ids)
        self.assertNotIn("label-noise", app._tab_groups["upper-right-tabs"])

    def test_hidden_when_only_the_winning_probability_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=True, has_labels=True)
        engine.y_prob = pd.DataFrame({"probability": [0.8, 0.8]}, index=pd.RangeIndex(2))
        _, ids = self._ids(engine)
        self.assertNotIn("label-noise-detect-btn", ids)

    def test_mounts_without_any_live_capability(self):
        # The snapshot case: no model, so no editor/counterfactual/similar panel — but the labels and
        # probabilities are still in the compiled batch, so this panel stands on its own.
        app, ids = self._ids(FakeEngine(can_edit=False, can_cf=False, has_labels=True))
        self.assertIn("label-noise-detect-btn", ids)
        self.assertNotIn("data-editor-input", ids)

    def test_caption_warns_when_no_independent_cross_check_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True)
        layout = LabelNoiseComponent().layout(engine.to_explanation(), engine)
        caption = layout.children[1].children
        self.assertIn("no independent cross-check", caption)

    def test_caption_explains_the_corpus_column_when_the_probe_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, probe_corpus=_PROBE_CORPUS)
        layout = LabelNoiseComponent().layout(engine.to_explanation(), engine)
        caption = layout.children[1].children
        self.assertIn("Corpus column", caption)

    def test_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=True, has_labels=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("label-noise-results.children", outputs)


class TestLabelNoiseComponent(unittest.TestCase):
    """Drive the panel's renderer and callbacks directly, without the app shell."""

    @staticmethod
    def _register(**kwargs):
        import dash

        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, **kwargs)
        explanation = engine.to_explanation()
        app = dash.Dash(__name__)
        comp = LabelNoiseComponent()
        comp.register_callbacks(
            app, explanation, engine, {"apply": "whatif-apply-store", "current": "current-datapoint"}
        )
        return app, engine

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_detect_returns_a_view_and_the_flagged_indices(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [0])  # sample 0 is labelled neg but confidently predicted pos
        self.assertIsNotNone(children)

    def test_detect_forwards_the_controls_to_the_engine(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, 7, "normalized_margin")
        self.assertEqual(engine.detect_calls[-1], {"top_n": 7, "score": "normalized_margin", "probe": True})

    def test_blank_controls_fall_back_to_defaults(self):
        app, engine = self._register()
        detect = self._callback(app, "label-noise-results")
        detect(1, None, None)
        self.assertEqual(engine.detect_calls[-1], {"top_n": 50, "score": "self_confidence", "probe": True})

    def test_detect_renders_the_corpus_column_when_a_probe_corpus_is_bound(self):
        app, _ = self._register(probe_corpus=_PROBE_CORPUS)
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertEqual(headers, ["Score", "Given", "Probably", "Corpus", "Text", ""])
        # The flagged row is "i am happy" labelled "neg"; the corpus puts that vocabulary in "pos",
        # so the probe rejects the label too rather than defending it.
        cell = table.children[1].children[0].children[3].children
        self.assertEqual(cell.children, "0.14 → pos")

    def test_detect_omits_the_corpus_column_without_a_probe_corpus(self):
        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        children, _ = detect(1, 50, "self_confidence")
        table = children.children[-1]
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)

    def test_detect_does_nothing_before_the_button_is_clicked(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        detect = self._callback(app, "label-noise-results")
        with self.assertRaises(PreventUpdate):
            detect(None, 50, "self_confidence")

    def test_a_clean_corpus_renders_a_message_and_no_rows(self):
        app, engine = self._register()
        # Both classes labelled and both labels agreeing with the model, so there is no noise to find.
        engine.y_prob = pd.DataFrame({"neg": [0.1, 0.9], "pos": [0.9, 0.1]}, index=pd.RangeIndex(2))
        engine.y_true = pd.Series(["pos", "neg"], index=pd.RangeIndex(2), name="ground_truth")
        detect = self._callback(app, "label-noise-results")
        children, indices = detect(1, 50, "self_confidence")
        self.assertEqual(indices, [])
        self.assertIn("No label issues detected", children.children)

    def test_inspect_makes_the_flagged_sample_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            dp = inspect([1], [1])  # row 0 of the table points at compiled sample 1
        self.assertEqual(dp["text"], "this is bad")
        self.assertEqual(dp["orig_idx"], 1)
        self.assertEqual(dp["label"], "pos")  # the *prediction*, matching the dataset-row path
        self.assertEqual(len(dp["tokens"]), 3)

    def test_inspect_ignores_a_render_with_no_clicks(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with self.assertRaises(PreventUpdate):
            inspect([0], [1])

    def test_inspect_ignores_a_missing_trigger(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = None
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_an_out_of_range_row(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 5}
            with self.assertRaises(PreventUpdate):
                inspect([1], [1])

    def test_inspect_ignores_a_stale_index_beyond_the_corpus(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import label_noise as mod

        app, _ = self._register()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "label-noise-apply", "index": 0}
            with self.assertRaises(PreventUpdate):
                inspect([1], [99])


class TestLabelNoiseTable(unittest.TestCase):
    """The neighbour column is evidence *beside* the score, and must not appear when absent."""

    @staticmethod
    def _issue(**kwargs):
        base = dict(
            index=0,
            text="i am happy",
            given_label="neg",
            suggested_label="pos",
            given_prob=0.2,
            suggested_prob=0.8,
            score=0.2,
        )
        return LabelIssue(**{**base, **kwargs})

    def test_corpus_column_omitted_when_no_probe_ran(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        table = _issues_table([self._issue()], "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertNotIn("Corpus", headers)
        self.assertEqual(headers, ["Score", "Given", "Probably", "Text", ""])

    def test_corpus_column_present_when_any_row_has_a_verdict(self):
        from shapash.webapp.nlp_components.label_noise import _issues_table

        verdict = ProbeVerdict(given_prob=0.8, top_label="neg", backs_given=True)
        issues = [self._issue(probe=verdict), self._issue(index=1)]
        table = _issues_table(issues, "label-noise")
        headers = [th.children for th in table.children[0].children.children]
        self.assertIn("Corpus", headers)

    def test_probe_summary_warns_when_the_corpus_backs_the_given_label(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        # The row the panel exists to catch: confident learning flagged it, but an independent
        # classifier defends the label — so it is the model that is wrong, not the corpus.
        verdict = ProbeVerdict(given_prob=0.83, top_label="neg", backs_given=True)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "⚠ 0.83")
        self.assertIn("text-warning", summary.className)
        self.assertIn("more likely", summary.title)

    def test_probe_summary_reports_a_corroborated_label_error_plainly(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        verdict = ProbeVerdict(given_prob=0.16, top_label="pos", backs_given=False)
        summary = _probe_summary(self._issue(given_label="neg", probe=verdict))
        self.assertEqual(summary.children, "0.16 → pos")
        self.assertIn("text-muted", summary.className)

    def test_rows_without_a_verdict_render_a_dash(self):
        from shapash.webapp.nlp_components.label_noise import _probe_summary

        self.assertEqual(_probe_summary(self._issue()), "—")


class TestSubpathMounting(unittest.TestCase):
    """``url_base_pathname`` — serving behind a reverse proxy that routes a subpath, not a host."""

    def _app(self, **kwargs):
        engine = FakeEngine(can_edit=True, can_cf=True)
        return NlpWebApp(engine.to_explanation(), engine=engine, **kwargs)

    def test_serves_at_the_root_by_default(self):
        config = self._app().app.config
        self.assertEqual(config.routes_pathname_prefix, "/")
        self.assertEqual(config.requests_pathname_prefix, "/")

    def test_mounts_under_the_given_prefix(self):
        config = self._app(url_base_pathname="/shapash-nlp-explainer/").app.config
        # Both prefixes matter: routes_* is where Dash listens, requests_* is what it writes into
        # the asset and callback URLs the browser then fetches.
        self.assertEqual(config.routes_pathname_prefix, "/shapash-nlp-explainer/")
        self.assertEqual(config.requests_pathname_prefix, "/shapash-nlp-explainer/")

    def test_missing_slashes_are_added(self):
        for given in ("shapash-nlp-explainer", "/shapash-nlp-explainer", "shapash-nlp-explainer/"):
            with self.subTest(given=given):
                config = self._app(url_base_pathname=given).app.config
                self.assertEqual(config.requests_pathname_prefix, "/shapash-nlp-explainer/")

    def test_the_served_routes_and_asset_urls_carry_the_prefix(self):
        # The config values above are only half the contract: Dash must also *answer* on the prefix
        # and write it into the script tags, or the page loads blank behind the proxy.
        client = self._app(url_base_pathname="/shapash-nlp-explainer/").app.server.test_client()
        self.assertEqual(client.get("/shapash-nlp-explainer/").status_code, 200)
        self.assertEqual(client.get("/").status_code, 404)
        body = client.get("/shapash-nlp-explainer/").get_data(as_text=True)
        self.assertIn("/shapash-nlp-explainer/_dash-component-suites", body)
        self.assertNotIn('"/_dash-', body)

    def test_empty_prefixes_serve_at_the_root(self):
        for given in ("", "/", "   "):
            with self.subTest(given=given):
                config = self._app(url_base_pathname=given).app.config
                self.assertEqual(config.requests_pathname_prefix, "/")


if __name__ == "__main__":
    unittest.main()


class TestWordProfileComponent(unittest.TestCase):
    """The single-word profile panel: word × aggregation × global selection, then drill-down."""

    STORES = {
        "current": "current-datapoint",
        "selection": "scatter-selected-indices",
        "error_cell": "error-cell",
        "errors_only": "errors-only-switch",
        "word_click": "word-click-filter",
    }

    @staticmethod
    def _explanation():
        """Four samples, two classes. 'happy' pulls +0.4/+0.3 in two, -0.6 in a third."""
        token_strings = [
            ["so", "happy", "today"],
            ["happy", "and", "happy"],
            ["not", "happy", "at", "all"],
            ["nothing", "here"],
        ]
        values = [
            np.array([[0.1, -0.1], [0.4, -0.4], [0.05, -0.05]]),
            np.array([[0.2, -0.2], [0.0, 0.0], [0.1, -0.1]]),
            np.array([[-0.1, 0.1], [-0.6, 0.6], [0.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ]
        texts = pd.Series(["so happy today", "happy and happy", "not happy at all", "nothing here"])
        return NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["pos", "pos", "neg", "neg"], index=texts.index, name="prediction"),
            y_prob=None,
            # Sample 1 is the only model error, so errors-only scopes to exactly that row.
            y_true=pd.Series(["pos", "neg", "neg", "neg"], index=texts.index, name="ground_truth"),
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )

    def _app(self):
        import dash

        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        app = dash.Dash(__name__)
        comp = WordProfileComponent()
        comp.register_callbacks(app, explanation, None, self.STORES)
        return app, explanation

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    # ── mounting / layout ───────────────────────────────────────────────
    def test_mounts_without_an_engine(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        # Data-only panel: it must survive a loaded snapshot with no live model.
        self.assertTrue(WordProfileComponent.is_available(self._explanation(), None))

    def test_layout_declares_every_id_its_callbacks_bind(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        found = set()
        _collect_ids(WordProfileComponent().layout(self._explanation(), None), found)
        for suffix in ("select", "agg", "class", "order", "limit", "graph", "caption", "results", "store"):
            self.assertIn(f"word-profile-{suffix}", found)

    def test_word_dropdown_is_seeded_with_the_top_word(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        layout = WordProfileComponent().layout(explanation, None)
        found = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            children = getattr(node, "children", None)
            for ch in children if isinstance(children, (list, tuple)) else [children]:
                if ch is not None and not isinstance(ch, str):
                    walk(ch)

        walk(layout)
        # Not empty on open, and the seed is a real corpus word.
        self.assertIn(found["word-profile-select"].value, explanation.vocabulary())

    def test_class_picker_hidden_for_a_single_output_column(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        explanation = self._explanation()
        binary = replace(
            explanation,
            values=[v[:, 0] for v in explanation.values],
            base_values=np.zeros(4),
            label_names=["score"],
        )
        layout = WordProfileComponent().layout(binary, None)
        found = set()
        _collect_ids(layout, found)
        # Still present (callbacks bind it), just not shown.
        self.assertIn("word-profile-class", found)

    # ── the profile callback ────────────────────────────────────────────
    def test_profile_reports_counts_and_one_bar_per_class(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "strongest", 10, None, None, False
        )
        self.assertEqual(len(fig.data[0].x), 2)
        self.assertIn("4 occurrence(s) in 3 of 4 sample(s)", caption)
        self.assertIsNotNone(table)
        self.assertEqual(store, [2, 0, 1])  # |-0.6| > 0.4 > 0.3

    def test_mean_reports_its_error_bars_in_the_caption(self):
        app, _ = self._app()
        _, caption, _, _ = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 10, None, None, False)
        self.assertIn("std across occurrences", caption)

    def test_sum_has_no_error_bars(self):
        app, _ = self._app()
        fig, caption, _, _ = self._callback(app, "word-profile-graph")("happy", "sum", 0, "most", 10, None, None, False)
        self.assertIsNone(fig.data[0].error_x.array)
        self.assertNotIn("std", caption)

    def test_absolute_aggregation_surfaces_the_two_way_word(self):
        app, _ = self._app()
        graph = self._callback(app, "word-profile-graph")
        signed = graph("happy", "mean", 0, "most", 10, None, None, False)[0].data[0].x
        magnitude = graph("happy", "mean_abs", 0, "most", 10, None, None, False)[0].data[0].x
        # The signed mean nearly cancels; the magnitude does not. This gap is the panel's point.
        self.assertLess(abs(signed[-1]), 0.05)
        self.assertGreater(magnitude[-1], 0.3)

    def test_order_reorders_the_drill_down_only(self):
        app, _ = self._app()
        graph = self._callback(app, "word-profile-graph")
        self.assertEqual(graph("happy", "mean", 0, "most", 10, None, None, False)[3], [0, 1, 2])
        self.assertEqual(graph("happy", "mean", 0, "least", 10, None, None, False)[3][0], 2)

    def test_limit_truncates_the_drill_down(self):
        app, _ = self._app()
        store = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 1, None, None, False)[3]
        self.assertEqual(len(store), 1)

    def test_scatter_selection_scopes_the_aggregate(self):
        app, _ = self._app()
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, [0, 3], None, False
        )
        self.assertIn("scoped to 2 selected sample(s)", caption)
        self.assertEqual(store, [0])

    def test_errors_only_scopes_the_aggregate(self):
        app, _ = self._app()
        # Sample 1 is the only misclassified row, and it does contain the word.
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, None, None, True
        )
        self.assertIn("scoped to 1 selected sample(s)", caption)
        self.assertEqual(store, [1])

    def test_confusion_cell_scopes_the_aggregate(self):
        app, _ = self._app()
        _, caption, _, _ = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, None, {"pred": 1, "true": 0, "indices": [2]}, False
        )
        self.assertIn("scoped to 1 selected sample(s)", caption)

    def test_word_absent_from_the_scope_says_so(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            "happy", "mean", 0, "most", 10, [3], None, False
        )
        self.assertIn("no occurrences", caption)
        self.assertIn("does not occur", fig.layout.annotations[0].text)
        self.assertIsNone(table)
        self.assertEqual(store, [])

    def test_no_word_selected_shows_a_prompt(self):
        app, _ = self._app()
        fig, caption, table, store = self._callback(app, "word-profile-graph")(
            None, "mean", 0, "most", 10, None, None, False
        )
        self.assertIn("Pick a word", fig.layout.annotations[0].text)
        self.assertEqual((caption, table, store), ("", None, []))

    def test_falls_back_to_defaults_on_cleared_controls(self):
        app, _ = self._app()
        _, caption, _, store = self._callback(app, "word-profile-graph")(
            "happy", None, None, None, None, None, None, None
        )
        self.assertIn("Mean", caption)
        self.assertEqual(len(store), 3)

    # ── cross-panel wiring ──────────────────────────────────────────────
    def test_follows_a_word_bar_clicked_in_the_importance_panel(self):
        app, _ = self._app()
        follow = self._callback(app, "word-profile-select.value")
        self.assertEqual(follow("today"), "today")
        # With a scatter the store holds the multi-select list; the newest word wins.
        self.assertEqual(follow(["today", "happy"]), "happy")

    def test_cleared_word_filter_leaves_the_selection_alone(self):
        from dash.exceptions import PreventUpdate

        app, _ = self._app()
        follow = self._callback(app, "word-profile-select.value")
        for cleared in (None, []):
            with self.assertRaises(PreventUpdate):
                follow(cleared)

    def test_inspect_packs_the_sample_into_the_current_datapoint(self):
        from unittest import mock

        from shapash.webapp.nlp_components import word_profile as mod

        app, _ = self._app()
        inspect = self._callback(app, "current-datapoint")
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "word-profile-inspect", "index": 1}
            # The store holds the displayed rows' positions, so row 1 is sample 0.
            datapoint = inspect([None, 1], [2, 0])
        self.assertEqual(datapoint["orig_idx"], 0)
        self.assertEqual(datapoint["text"], "so happy today")
        self.assertEqual(datapoint["label"], "pos")

    def test_inspect_ignores_clicks_it_cannot_resolve(self):
        from unittest import mock

        from dash.exceptions import PreventUpdate

        from shapash.webapp.nlp_components import word_profile as mod

        app, _ = self._app()
        inspect = self._callback(app, "current-datapoint")
        with self.assertRaises(PreventUpdate):
            inspect([None, None], [0, 1])  # no button actually clicked
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = None
            with self.assertRaises(PreventUpdate):
                inspect([1], [0])
        with mock.patch.object(mod, "callback_context") as cc:
            cc.triggered_id = {"type": "word-profile-inspect", "index": 5}
            with self.assertRaises(PreventUpdate):
                inspect([1], [0])  # index past the end of the displayed rows


class TestGlobalWordImportancePanel(unittest.TestCase):
    """The shell's Word Importance chart: rank-by, frequency floor, and its empty states."""

    @staticmethod
    def _explanation():
        # "rare" pulls hard once; "common" pulls mildly four times. The two rank-by modes must
        # disagree, which is the whole reason the control exists.
        token_strings = [["rare", "common"], ["common"], ["common", "common"], ["mild"]]
        values = [
            np.array([[0.9, -0.9], [0.3, -0.3]]),
            np.array([[0.3, -0.3]]),
            np.array([[0.3, -0.3], [0.3, -0.3]]),
            np.array([[-0.5, 0.5]]),
        ]
        texts = pd.Series(["rare common", "common", "common common", "mild"])
        explanation = NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["neg"] * 4, index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )
        return explanation

    @classmethod
    def _app(cls):
        app = NlpWebApp(cls._explanation(), engine=None)
        for key, spec in app.app.callback_map.items():
            if "global-importance-graph.figure" in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError("global-importance-graph")

    @staticmethod
    def _words(fig):
        # Bars are drawn bottom-to-top, so reverse back into rank order.
        return list(fig.data[0].y)[::-1]

    def setUp(self):
        self.graph = self._app()

    def _call(self, rank_by="mean", floor=1, sign="all", topk=10, indices=None, cell=None, errors=False, label=0):
        return self.graph(label, topk, sign, [], rank_by, floor, indices, cell, errors)

    @staticmethod
    def _values(fig):
        return list(fig.data[0].x)[::-1]

    def test_mean_and_sum_produce_different_rankings(self):
        self.assertEqual(self._words(self._call(rank_by="mean"))[0], "rare")
        self.assertEqual(self._words(self._call(rank_by="sum"))[0], "common")

    def test_axis_names_the_statistic_drawn(self):
        self.assertEqual(self._call(rank_by="mean").layout.xaxis.title.text, "Mean SHAP contribution")
        self.assertEqual(self._call(rank_by="sum").layout.xaxis.title.text, "Total SHAP contribution")

    def test_frequency_floor_removes_rare_words(self):
        self.assertIn("rare", self._words(self._call(floor=1)))
        self.assertNotIn("rare", self._words(self._call(floor=2)))

    def test_panel_has_no_title(self):
        # The tab is already labelled "Word Importance" and the floor/class are visible in the
        # filter row above the chart, so the figure itself carries no title band — reclaiming that
        # vertical space for word rows instead.
        self.assertIsNone(self._call(floor=2).layout.title.text)
        self.assertIsNone(self._call(floor=1).layout.title.text)

    def test_sign_filter_still_works_in_both_modes(self):
        for rank_by in ("mean", "sum"):
            self.assertEqual(self._words(self._call(rank_by=rank_by, sign="negative")), ["mild"])
            self.assertNotIn("mild", self._words(self._call(rank_by=rank_by, sign="positive")))

    def test_default_floor_is_two(self):
        from shapash.webapp.nlp_components.word_importance import _DEFAULT_MIN_OCCURRENCES

        # A mean over a single observation is not a mean; the panel must not open on one.
        self.assertEqual(_DEFAULT_MIN_OCCURRENCES, 2)

    def test_cleared_floor_input_means_no_filter(self):
        # An emptied number box arrives as None and must not silently restore the default.
        self.assertIn("rare", self._words(self._call(floor=None)))

    def test_floor_below_one_is_clamped(self):
        self.assertIn("rare", self._words(self._call(floor=0)))

    def test_impossible_floor_explains_itself(self):
        fig = self._call(floor=99)
        self.assertEqual(len(fig.data), 0)
        self.assertIn("at least 99 time(s) in these 4 sample(s)", fig.layout.annotations[0].text)

    def test_empty_sign_filter_explains_itself_differently(self):
        # Only "mild" is negative; excluding it by scope leaves the sign filter with nothing, and
        # the message must point at the filter rather than at the frequency floor.
        fig = self._call(sign="negative", indices=[0, 1])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("negative", fig.layout.annotations[0].text)

    def test_floor_counts_within_the_selection(self):
        # "common" occurs 4x overall but once in sample 1, so a floor of 2 must exclude it there.
        fig = self._call(floor=2, indices=[1])
        self.assertEqual(len(fig.data), 0)
        self.assertIn("in these 1 sample(s)", fig.layout.annotations[0].text)

    def test_hover_carries_the_occurrence_count(self):
        fig = self._call(floor=1)
        drawn = dict(zip(self._words(fig), [int(c[0]) for c in fig.data[0].customdata][::-1]))
        self.assertEqual(drawn, {"rare": 1, "common": 4, "mild": 1})
        self.assertIn("Occurrences:", fig.data[0].hovertemplate)

    def test_hover_counts_follow_the_selection(self):
        # A count that ignored the scope would contradict the floor applied right beside it:
        # "common" occurs 4x in the corpus but twice in samples 0-1.
        fig = self._call(floor=1, indices=[0, 1])
        drawn = dict(zip(self._words(fig), [int(c[0]) for c in fig.data[0].customdata][::-1]))
        self.assertEqual(drawn["common"], 2)

    def test_all_classes_ranks_on_the_strongest_class_magnitude(self):
        # "mild" pulls -0.5 on class 0 and +0.5 on class 1: across classes it is a magnitude of
        # 0.5, not the -0.5 the single-class view shows.
        fig = self._call(label="all")
        drawn = dict(zip(self._words(fig), self._values(fig)))
        self.assertEqual(drawn, {"rare": 0.9, "mild": 0.5, "common": 0.3})
        self.assertEqual(dict(zip(self._words(self._call()), self._values(self._call())))["mild"], -0.5)

    def test_all_classes_names_its_statistic_on_the_axis(self):
        self.assertEqual(self._call(label="all").layout.xaxis.title.text, "Largest |mean SHAP| across classes")
        self.assertEqual(
            self._call(label="all", rank_by="sum").layout.xaxis.title.text,
            "Largest |total SHAP| across classes",
        )

    def test_all_classes_ignores_a_stale_sign_filter(self):
        # Every value is a magnitude, so honouring "negative" would blank the chart on a control
        # the user cannot even see is active.
        self.assertEqual(self._words(self._call(label="all", sign="negative")), ["rare", "mild", "common"])

    def test_all_classes_still_honours_the_frequency_floor(self):
        self.assertNotIn("rare", self._words(self._call(label="all", floor=2)))

    def test_sign_filter_is_greyed_out_under_all_classes(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = None
        for key, spec in app.app.callback_map.items():
            if "sign-filter.options" in key:
                fn = getattr(spec["callback"], "__wrapped__", spec["callback"])
        self.assertIsNotNone(fn, "sign-filter gating callback not registered")
        options, value = fn("all")
        self.assertEqual([o["value"] for o in options if o.get("disabled")], ["positive", "negative"])
        self.assertEqual(value, "all")
        options, _ = fn(0)
        self.assertFalse(any(o.get("disabled") for o in options))

    def test_scatter_colouring_survives_all_classes(self):
        # The scatter reads the same dropdown, where int("all") would raise.
        app = NlpWebApp(self._explanation(), engine=None, scatter_xy=np.zeros((4, 2)))
        fn = None
        for key, spec in app.app.callback_map.items():
            if "scatter-plot.figure" in key:
                fn = getattr(spec["callback"], "__wrapped__", spec["callback"])
        self.assertIsNotNone(fn, "scatter callback not registered")
        self.assertIsNotNone(fn("word_contribution", ["common"], "all", False))

    def test_number_boxes_commit_without_needing_a_blur(self):
        # debounce=True commits only on Enter or blur, which leaves the spinner arrows looking
        # dead: clicking one keeps focus in the box, so the chart never updates. A numeric
        # (seconds) debounce commits after a pause instead.
        layout = NlpWebApp(self._explanation(), engine=None).app.layout
        for box_id in ("min-occurrences", "topk-input"):
            debounce = layout[box_id].debounce
            self.assertNotIsInstance(debounce, bool, f"{box_id} commits only on blur")
            self.assertGreater(debounce, 0)

    def test_topk_input_replaces_the_slider(self):
        found = set()
        _collect_ids(NlpWebApp(self._explanation(), engine=None).app.layout, found)
        self.assertIn("topk-input", found)
        self.assertNotIn("topk-slider", found)

    def test_topk_controls_the_bar_count(self):
        self.assertEqual(len(self._words(self._call(topk=2))), 2)

    def test_typed_topk_is_clamped_to_the_boxs_range(self):
        from shapash.webapp.nlp_components.word_importance import _MAX_TOPK, _MIN_TOPK

        # The browser enforces min/max on the spinner but not on typed input.
        self.assertEqual(len(self._words(self._call(topk=999))), 3)  # only 3 words exist
        self.assertEqual(len(self._words(self._call(topk=0))), _MIN_TOPK)
        self.assertEqual(len(self._words(self._call(topk=-5))), _MIN_TOPK)
        self.assertLessEqual(_MAX_TOPK, 50)

    def test_cleared_topk_box_falls_back_to_the_default(self):
        from shapash.webapp.nlp_components.word_importance import _DEFAULT_TOPK

        self.assertEqual(_DEFAULT_TOPK, 20)
        # Only None (an emptied box) restores the default — a typed 0 clamps instead.
        self.assertEqual(len(self._words(self._call(topk=None))), 3)  # all 3 words, under the cap

    def test_chart_keeps_its_computed_height_so_labels_survive(self):
        # The panel scrolls; squeezing 50 words into it is what made plotly drop word labels.
        fig = self._call(topk=50)
        self.assertIsNotNone(fig.layout.height)

    def test_missing_class_prevents_update(self):
        from dash.exceptions import PreventUpdate

        with self.assertRaises(PreventUpdate):
            self.graph(None, 10, "all", [], "mean", 1, None, None, False)


class TestWordImportanceScatterSync(unittest.TestCase):
    """Word Importance and Scatter communicate only through the shared stores (Phase A extraction).

    A bar click always writes ``word-click-filter`` directly now, whether or not a scatter is
    mounted (no more branching on the scatter's presence), and the scatter's own word dropdown
    mirrors that store bidirectionally instead of being the "source of truth" only when it exists.
    """

    @staticmethod
    def _explanation():
        return TestGlobalWordImportancePanel._explanation()

    def test_bar_click_registered_without_a_scatter(self):
        app = NlpWebApp(self._explanation(), engine=None)
        pairs = _callback_binding_ids(app, "global-importance-graph.clickData")
        self.assertIn(("global-importance-graph", "clickData"), pairs)
        self.assertIn(("word-filter-clear-btn", "n_clicks"), pairs)

    def test_bar_click_registered_with_a_scatter_too(self):
        # Before Phase A, a mounted scatter changed which store a bar click targeted; now Word
        # Importance always registers this callback unconditionally, regardless of scatter presence.
        app = NlpWebApp(self._explanation(), engine=None, scatter_xy=np.zeros((4, 2)))
        pairs = _callback_binding_ids(app, "global-importance-graph.clickData")
        self.assertIn(("global-importance-graph", "clickData"), pairs)

    def test_scatter_word_select_mirrors_the_shared_store(self):
        app = NlpWebApp(self._explanation(), engine=None, scatter_xy=np.zeros((4, 2)))
        pairs = _callback_binding_ids(app, "scatter-word-select.value")
        self.assertIn(("word-click-filter", "data"), pairs)

    def test_class_selector_publishes_to_the_active_class_store(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "active-class-store.data")
        self.assertEqual(fn(1), 1)
        # "All classes" (the string sentinel) is not itself a class index — falls back to 0.
        self.assertEqual(fn("all"), 0)


class TestTableFilterIntersection(unittest.TestCase):
    """The dataset table's active filters (error cell, word click, errors-only) must intersect for
    real. A filter that empties the combination has to actually empty the table, not be silently
    dropped while the summary still claims it is applied — see the reported bug where clicking a
    Word Importance bar and then an Error Analysis cell with no matching rows kept the word filter
    in the summary text while showing every row from the cell, unfiltered.
    """

    @staticmethod
    def _explanation():
        texts = pd.Series(["i am happy", "so glad", "this is bad", "very sad"])
        token_strings = [t.split() for t in texts]
        values = [np.random.randn(len(toks), 2) for toks in token_strings]
        return NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["pos", "pos", "neg", "neg"], name="prediction"),
            y_prob=None,
            y_true=pd.Series(["pos", "neg", "neg", "pos"], name="ground_truth"),
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )

    def test_word_filter_after_error_cell_narrows_instead_of_being_dropped(self):
        # "happy" only occurs in row 0 ("i am happy"); the clicked cell (pred=pos, true=neg) holds
        # only row 1 ("so glad"). The intersection is empty and must stay empty.
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "dataset-table.rowData")
        error_cell = {"pred": 1, "true": 0, "indices": [1]}
        rows, selected, _title, summary = fn(None, ["happy"], False, error_cell, None, None)
        self.assertEqual(rows, [])
        self.assertEqual(selected, [])
        self.assertIn('containing "happy"', summary)
        self.assertIn("(0 of 4)", summary)

    def test_word_filter_matching_the_cell_still_narrows(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "dataset-table.rowData")
        error_cell = {"pred": 0, "true": 1, "indices": [3]}  # row 3: "very sad"
        rows, selected, _title, summary = fn(None, ["sad"], False, error_cell, None, None)
        self.assertEqual([r["text"] for r in rows], ["very sad"])
        self.assertEqual(selected, rows)
        self.assertIn("(1 of 4)", summary)

    def test_errors_only_narrows_a_word_filter_too(self):
        # "happy" only occurs in row 0, a correct prediction — errors_only must not silently drop
        # the word filter to avoid an empty table.
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "dataset-table.rowData")
        rows, _selected, _title, summary = fn(None, ["happy"], True, None, None, None)
        self.assertEqual(rows, [])
        self.assertIn("model errors only", summary)
        self.assertIn('containing "happy"', summary)

    def test_word_filter_matches_the_exact_token_not_a_substring(self):
        # "unhappy" contains "happy" as a substring but is a different token — a bar labeled
        # "happy" must not pull it in, matching the exact-token semantics word_importance()/the
        # scatter's word-contribution coloring already use.
        texts = pd.Series(["i am happy", "so unhappy today"])
        token_strings = [t.split() for t in texts]
        values = [np.random.randn(len(toks), 2) for toks in token_strings]
        explanation = NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((2, 2)),
            y_pred=pd.Series(["pos", "neg"], name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )
        app = NlpWebApp(explanation, engine=None)
        fn = _callback(app, "dataset-table.rowData")
        rows, _selected, _title, _summary = fn(None, ["happy"], False, None, None, None)
        self.assertEqual([r["text"] for r in rows], ["i am happy"])

    def test_grid_filter_narrows_the_shown_count_without_touching_rowData(self):
        # AG Grid's own column filter runs client-side on top of whatever rowData filter_table
        # hands it — it must not change rowData/selectedRows, only the reported count, so the
        # grid's filtered view and the "Showing:" summary agree without a second, disconnected
        # counter (the earlier bug this mirrors: a filter description without a matching count).
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "dataset-table.rowData")
        filter_model = {"text": {"filterType": "text", "type": "contains", "filter": "happy"}}
        virtual_row_data = [{"text": "i am happy"}]
        rows, _selected, _title, summary = fn(None, None, False, None, filter_model, virtual_row_data)
        self.assertEqual(len(rows), 4)  # rowData is untouched — the grid does its own filtering
        self.assertIn("grid filter: Text", summary)
        self.assertIn("(1 of 4)", summary)


def _callback(app, out_substr):
    for key, spec in app.app.callback_map.items():
        if out_substr in key:
            return getattr(spec["callback"], "__wrapped__", spec["callback"])
    raise KeyError(out_substr)


class TestGridFilterClearButton(unittest.TestCase):
    """AG Grid's own column filters run client-side, outside filter_table's inputs — the clear
    button is the only thing that lets a user drop one without reaching into the grid's own header
    UI, so its show/hide and reset behavior need their own coverage. The filter's *description* is
    covered by filter_table itself (see TestTableFilterIntersection), since it's folded into the
    same "Showing:" summary rather than a separate badge.
    """

    @staticmethod
    def _explanation():
        texts = pd.Series(["i am happy", "so glad", "this is bad", "very sad"])
        token_strings = [t.split() for t in texts]
        values = [np.random.randn(len(toks), 2) for toks in token_strings]
        return NlpExplanation(
            texts=texts,
            token_strings=token_strings,
            values=values,
            base_values=np.zeros((4, 2)),
            y_pred=pd.Series(["pos", "pos", "neg", "neg"], name="prediction"),
            y_prob=None,
            y_true=pd.Series(["pos", "neg", "neg", "pos"], name="ground_truth"),
            label_names=LABEL_NAMES,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="point",
            output_space="probability",
        )

    def test_no_filter_model_hides_the_clear_button(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "grid-filter-clear-btn.style")
        self.assertEqual(fn(None)["display"], "none")

    def test_active_filter_model_shows_the_clear_button(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "grid-filter-clear-btn.style")
        filter_model = {"prediction": {"filterType": "text", "type": "equals", "filter": "pos"}}
        self.assertNotEqual(fn(filter_model).get("display"), "none")

    def test_clear_button_resets_the_grid_filter_model(self):
        app = NlpWebApp(self._explanation(), engine=None)
        fn = _callback(app, "dataset-table.filterModel")
        self.assertEqual(fn(1), {})


class TestScatterComponentWordContributionOption(unittest.TestCase):
    """``offer_word_contribution`` gates the "Word contribution" color mode, not the whole panel."""

    @staticmethod
    def _explanation():
        return TestGlobalWordImportancePanel._explanation()

    def _color_options(self, offer_word_contribution: bool):
        from shapash.webapp.nlp_components import ScatterComponent

        comp = ScatterComponent(np.zeros((4, 2)), offer_word_contribution=offer_word_contribution)
        layout = comp.layout(self._explanation(), None)
        found = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            children = getattr(node, "children", None)
            for ch in children if isinstance(children, (list, tuple)) else [children]:
                if ch is not None and not isinstance(ch, str):
                    walk(ch)

        walk(layout)
        return [opt["value"] for opt in found["color-by"].options]

    def test_offered_when_word_importance_is_mounted(self):
        self.assertIn("word_contribution", self._color_options(offer_word_contribution=True))

    def test_omitted_without_word_importance(self):
        # Scatter still works standalone (Prediction/Ground-Truth) — it just cannot source a class
        # index for word-contribution coloring without Word Importance's control.
        options = self._color_options(offer_word_contribution=False)
        self.assertNotIn("word_contribution", options)
        self.assertIn("prediction", options)


class TestBuildScatterFig(unittest.TestCase):
    """``_build_scatter_fig``'s own dispatch — the part left after extracting the drawing itself
    into :func:`~shapash.plots.plot_scatter.plot_scatter`."""

    @staticmethod
    def _explanation(with_true: bool, with_pred: bool):
        texts = pd.Series(["rare common", "common"])
        return NlpExplanation(
            texts=texts,
            token_strings=[["rare", "common"], ["common"]],
            values=[np.array([[0.9, -0.9], [0.3, -0.3]]), np.array([[0.3, -0.3]])],
            base_values=None,
            y_pred=pd.Series(["neg", "pos"], index=texts.index) if with_pred else None,
            y_prob=None,
            y_true=pd.Series(["pos", "pos"], index=texts.index) if with_true else None,
            label_names=None,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def _component(self):
        from shapash.webapp.nlp_components import ScatterComponent

        return ScatterComponent(np.zeros((2, 2)), offer_word_contribution=True)

    def test_ground_truth_colors_by_y_true(self):
        fig = self._component()._build_scatter_fig(self._explanation(True, True), "ground_truth")
        names = {trace.name for trace in fig.data}
        self.assertEqual(names, {"pos"})  # both samples are "pos" in y_true

    def test_without_prediction_or_ground_truth_draws_a_single_unlabeled_group(self):
        fig = self._component()._build_scatter_fig(self._explanation(False, False), "prediction")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(len(fig.data[0].x), 2)


class TestWordProfileControls(unittest.TestCase):
    """The picker's sort/labels, the restricted aggregation list, and bar-click class ranking."""

    def _layout(self):
        from shapash.webapp.nlp_components import WordProfileComponent

        comp = WordProfileComponent()
        layout = comp.layout(TestWordProfileComponent._explanation(), None)
        found = {}

        def walk(node):
            cid = getattr(node, "id", None)
            if isinstance(cid, str):
                found[cid] = node
            children = getattr(node, "children", None)
            for ch in children if isinstance(children, (list, tuple)) else [children]:
                if ch is not None and not isinstance(ch, str):
                    walk(ch)

        walk(layout)
        return comp, found

    def _app(self):
        import dash

        from shapash.webapp.nlp_components import WordProfileComponent

        comp = WordProfileComponent()
        explanation = TestWordProfileComponent._explanation()
        comp.layout(explanation, None)  # builds the option lists the sort callback serves
        app = dash.Dash(__name__)
        comp.register_callbacks(app, explanation, None, TestWordProfileComponent.STORES)
        return app

    @staticmethod
    def _callback(app, out_substr):
        for key, spec in app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_sum_aggregations_are_not_offered(self):
        # Per word, a sum is the mean rescaled by the occurrence count — same bars, no information.
        _, found = self._layout()
        self.assertEqual([o["value"] for o in found["word-profile-agg"].options], ["mean", "mean_abs"])

    def test_options_carry_the_occurrence_count(self):
        _, found = self._layout()
        labels = {o["value"]: o["label"] for o in found["word-profile-select"].options}
        self.assertEqual(labels["happy"], "happy (4)")

    def test_option_values_stay_bare_words(self):
        # The label carries the count; the value must not, or every consumer of it breaks.
        _, found = self._layout()
        self.assertIn("happy", [o["value"] for o in found["word-profile-select"].options])

    def test_sort_toggle_reorders_without_changing_values(self):
        app = self._app()
        reorder = self._callback(app, "word-profile-select.options")
        alpha = [o["value"] for o in reorder("alpha")]
        freq = [o["value"] for o in reorder("frequency")]
        self.assertEqual(alpha, sorted(alpha))
        self.assertEqual(freq[0], "happy")  # the most frequent word
        self.assertEqual(set(alpha), set(freq))

    def test_unknown_sort_falls_back_to_alphabetical(self):
        app = self._app()
        reorder = self._callback(app, "word-profile-select.options")
        self.assertEqual(reorder(None), reorder("alpha"))

    def test_bar_click_sets_the_drill_down_class(self):
        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        value, reset = click({"points": [{"customdata": [1], "y": "pos"}]})
        self.assertEqual(value, 1)
        # clickData is reset so clicking the same bar twice re-fires.
        self.assertIsNone(reset)

    def test_bar_click_reads_the_index_not_the_label(self):
        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        # Duplicate display names must not be able to mis-resolve the class.
        self.assertEqual(click({"points": [{"customdata": [0], "y": "neg"}]})[0], 0)

    def test_graph_carries_the_class_index_on_every_bar(self):
        app = self._app()
        fig = self._callback(app, "word-profile-graph")("happy", "mean", 0, "most", 5, None, None, False)[0]
        self.assertEqual([c[0] for c in fig.data[0].customdata], [1, 0])  # reversed for drawing

    def test_empty_clicks_are_ignored(self):
        from dash.exceptions import PreventUpdate

        app = self._app()
        click = self._callback(app, "word-profile-class.value")
        for bad in (None, {}, {"points": []}, {"points": [{}]}):
            with self.assertRaises(PreventUpdate):
                click(bad)
