"""Unit tests for ``shapash.webapp.nlp_app``: the app shell — three-panel layout, capability-gated
What-if Lab mounting, subpath mounting, table/grid filtering, and construction/run wiring.
"""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import jupyter_dash
from dash.exceptions import PreventUpdate

from shapash.compute.embeddings import Embedding
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.webapp.nlp_app import NlpWebApp
from shapash.webapp.nlp_components import (
    CounterfactualComponent,
    DataEditorComponent,
    LabelNoiseComponent,
    SimilarExamplesComponent,
)
from shapash.webapp.nlp_components import error_analysis as error_analysis_module
from shapash.webapp.nlp_components.error_analysis import ErrorAnalysisComponent, _cell_from_click
from shapash.webapp.utils.launch import RunningApp

from tests.unit_tests.webapp.nlp_components._shared import (
    FakeEngine,
    _PROBE_CORPUS,
    _callback,
    _callback_binding_ids,
    _collect_ids,
    _ctx,
)
from tests.unit_tests.webapp.nlp_components._shared import LABEL_NAMES as _WHATIF_LABEL_NAMES

# The 6-class label set used by the TestNlpWebApp / TestNlpWebAppRun fixtures below (moved here
# from tests/unit_tests/nlp/test_nlp_prototype.py). This intentionally shadows, for this module
# only, the 2-class ``LABEL_NAMES`` used by the What-if Lab classes further down (imported above
# as ``_WHATIF_LABEL_NAMES``) — the two source files each defined their own module-level
# ``LABEL_NAMES`` with different values, which collide once both land in one file. See the
# migration report for details.
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)


def _make_token_data() -> tuple[list[list[str]], list[np.ndarray], np.ndarray]:
    """Synthetic per-token contribution data for 3 samples, 6 classes."""
    rng = np.random.default_rng(42)
    token_strings = [
        ["", "i", "feel", "so", "happy", "today", ""],
        ["", "this", "is", "terrible", "and", "sad", ""],
        ["", "what", "a", "wonderful", "day", ""],
    ]
    values = [rng.uniform(-0.4, 0.4, size=(len(t), N_CLASSES)).astype(np.float32) for t in token_strings]
    base_values = rng.uniform(-0.1, 0.1, size=(3, N_CLASSES)).astype(np.float32)
    return token_strings, values, base_values


def _make_texts() -> pd.Series:
    return pd.Series(
        ["i feel so happy today", "this is terrible and sad", "what a wonderful day"],
        index=pd.RangeIndex(3),
    )


def _make_explanation(
    texts: pd.Series | None = None,
    y_pred: pd.Series | None = None,
    y_prob: pd.DataFrame | None = None,
    y_true: pd.Series | None = None,
    label_names: list[str] | None = LABEL_NAMES,
    backend_name: str = "nlp_shap",
    is_additive: bool = True,
    reference_kind: str = "none",
    output_space: str = "probability",
) -> NlpExplanation:
    """Synthetic ``NlpExplanation`` for 3 samples, 6 classes — the ``explain()`` return value."""
    texts = _make_texts() if texts is None else texts
    token_strings, values, base_values = _make_token_data()
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=(
            pd.Series(["joy", "sadness", "joy"], index=texts.index, name="prediction") if y_pred is None else y_pred
        ),
        y_prob=y_prob,
        y_true=y_true,
        label_names=label_names,
        folds_case=None,
        backend_name=backend_name,
        is_additive=is_additive,
        reference_kind=reference_kind,
        output_space=output_space,
    )


def _make_explainer() -> NlpExplainer:
    """Bare ``NlpExplainer`` (engine role only), bypassing ``__init__`` to avoid ``shap.Explainer(None)``.

    Per ``fit``/``explain``, the explainer never holds compiled results — tests that need a computed
    batch build an :class:`NlpExplanation` directly with :func:`_make_explanation` instead.
    """
    xpl = object.__new__(NlpExplainer)
    xpl.model = None
    xpl.label_names = LABEL_NAMES
    xpl.backend = None
    return xpl


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


class TestThreePanelLayout(unittest.TestCase):
    """The LIT-style three-panel shell: tab groups, mounted bodies, and the current-datapoint store."""

    def _ids(self, engine, **kwargs):
        app = NlpWebApp(engine.to_explanation(), engine=engine, **kwargs)
        found = set()
        _collect_ids(app.app.layout, found)
        return app, found

    def test_full_tab_groups_when_all_panels_available(self):
        app, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), projection=np.zeros((2, 2)))
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
        _, ids = self._ids(FakeEngine(can_edit=True, can_cf=True), projection=np.zeros((2, 2)))
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
        app = NlpWebApp(engine.to_explanation(), engine=engine, projection=np.zeros((2, 2)))
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
        layout = LabelNoiseComponent().layout(_ctx(engine.to_explanation(), engine))
        caption = layout.children[1].children
        self.assertIn("no independent cross-check", caption)

    def test_caption_explains_the_corpus_column_when_the_probe_is_available(self):
        engine = FakeEngine(can_edit=True, can_cf=False, has_labels=True, probe_corpus=_PROBE_CORPUS)
        layout = LabelNoiseComponent().layout(_ctx(engine.to_explanation(), engine))
        caption = layout.children[1].children
        self.assertIn("Corpus column", caption)

    def test_callbacks_registered_when_mounted(self):
        engine = FakeEngine(can_edit=True, can_cf=True, has_labels=True)
        app = NlpWebApp(engine.to_explanation(), engine=engine)
        outputs = " ".join(app.app.callback_map.keys())
        self.assertIn("label-noise-results.children", outputs)


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
            label_names=_WHATIF_LABEL_NAMES,
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
            label_names=_WHATIF_LABEL_NAMES,
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
            label_names=_WHATIF_LABEL_NAMES,
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


class TestNlpWebApp(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()
        self.explanation = _make_explanation()
        self.webapp = NlpWebApp(self.explanation, engine=self.xpl)

    def test_layout_built(self):
        self.assertIsNotNone(self.webapp.app.layout)

    def test_class_selector_options(self):
        # Class selector is now two independent dropdowns: "local-class-selector" (Sentence
        # Highlight / Waterfall) and "global-class-selector" (Word Importance / Embeddings).
        # Only the global one offers the cross-class overview — a single sample's highlight has no
        # such aggregate.
        local = self._find_component(self.webapp.app.layout, "local-class-selector")
        self.assertIsNotNone(local, "local-class-selector dropdown not found in layout")
        self.assertEqual([opt["label"] for opt in local.options], LABEL_NAMES)

        glob = self._find_component(self.webapp.app.layout, "global-class-selector")
        self.assertIsNotNone(glob, "global-class-selector dropdown not found in layout")
        self.assertEqual(len(glob.options), N_CLASSES + 1)
        self.assertEqual([opt["label"] for opt in glob.options], ["All classes", *LABEL_NAMES])
        # Its value must never collide with a real class index.
        self.assertEqual(glob.options[0]["value"], "all")
        self.assertEqual([opt["value"] for opt in glob.options[1:]], list(range(N_CLASSES)))

    def test_local_class_selector_defaults_to_predicted_class(self):
        # Defaults to the predicted class of the initially selected row (row 0).
        dropdown = self._find_component(self.webapp.app.layout, "local-class-selector")
        row0_prediction = self.webapp._full_table_records[0]["prediction"]
        self.assertEqual(dropdown.value, LABEL_NAMES.index(row0_prediction))

    def test_dataset_table_populated(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        self.assertIsNotNone(table, "dataset-table not found in layout")
        self.assertEqual(len(table.rowData), 3)
        self.assertIn("text", table.rowData[0])
        self.assertIn("prediction", table.rowData[0])

    def test_graph_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("global-importance-graph", ids)
        self.assertIn("dataset-table", ids)
        self.assertIn("local-class-selector", ids)
        self.assertIn("global-class-selector", ids)
        # token bar chart removed; sentence-highlight replaced it
        self.assertNotIn("local-contributions-graph", ids)

    def test_control_ids_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("topk-input", ids)
        self.assertIn("sign-filter", ids)
        self.assertIn("word-filter", ids)

    def test_sentence_highlight_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("sentence-highlight", ids)

    def test_waterfall_controls_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        # Waterfall is now a tab (no show/hide switch); the threshold slider + graph live in it.
        self.assertIn("waterfall-threshold", ids)
        self.assertIn("waterfall-graph", ids)

    def test_dataset_table_no_ground_truth_by_default(self):
        table = self._find_component(self.webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertNotIn("ground_truth", col_fields)

    def test_dataset_table_with_y_true(self):
        y_true = pd.Series(["sadness", "joy", "sadness"], index=pd.RangeIndex(3), name="ground_truth")
        explanation = _make_explanation(y_true=y_true)
        webapp = NlpWebApp(explanation, engine=self.xpl)
        table = self._find_component(webapp.app.layout, "dataset-table")
        col_fields = [c["field"] for c in table.columnDefs]
        self.assertIn("ground_truth", col_fields)
        self.assertEqual(table.rowData[0]["ground_truth"], "sadness")

    def test_scatter_store_always_present(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertIn("scatter-selected-indices", ids)

    def test_scatter_absent_when_no_xy(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("scatter-plot", ids)
        self.assertNotIn("color-by", ids)

    def test_scatter_present_when_a_projection_is_given(self):
        webapp = NlpWebApp(self.explanation, engine=self.xpl, projection=np.zeros((3, 2)))
        ids = self._collect_ids(webapp.app.layout)
        self.assertIn("scatter-plot", ids)
        self.assertIn("color-by", ids)

    def test_scatter_mounts_from_a_matching_embedding(self):
        projection = Embedding(np.zeros((3, 2)), "m", "s", corpus_id=self.explanation.corpus_id, reducer_tag="pca")
        ids = self._collect_ids(NlpWebApp(self.explanation, engine=self.xpl, projection=projection).app.layout)
        self.assertIn("scatter-plot", ids)

    def test_an_embedding_of_other_texts_is_refused_at_construction(self):
        projection = Embedding(np.zeros((3, 2)), "m", "s", corpus_id="other-texts", reducer_tag="pca")
        with self.assertRaises(ValueError):
            NlpWebApp(self.explanation, engine=self.xpl, projection=projection)

    def test_scatter_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            NlpWebApp(self.explanation, engine=self.xpl, projection=np.zeros((5, 2)))  # 5 rows but only 3 samples

    # ── Error Analysis tab (confusion matrix) ─────────────────────────

    def _make_webapp_with_gt(self) -> NlpWebApp:
        # y_pred is ["joy", "sadness", "joy"]; make sample 0 a sadness→joy error, others correct.
        y_true = pd.Series(["sadness", "sadness", "joy"], index=pd.RangeIndex(3), name="ground_truth")
        explanation = _make_explanation(y_true=y_true)
        return NlpWebApp(explanation, engine=self.xpl)

    def test_error_analysis_absent_without_ground_truth(self):
        ids = self._collect_ids(self.webapp.app.layout)
        self.assertNotIn("confusion-matrix-graph", ids)
        self.assertNotIn("error-pred-importance", ids)

    def test_error_cell_store_always_present(self):
        # The store is created unconditionally so cross-panel callbacks can read it even without gt.
        self.assertIn("error-cell", self._collect_ids(self.webapp.app.layout))

    def test_error_analysis_present_with_ground_truth(self):
        ids = self._collect_ids(self._make_webapp_with_gt().app.layout)
        for cid in ("confusion-matrix-graph", "error-pred-importance", "error-true-importance", "cm-normalize"):
            self.assertIn(cid, ids)

    @staticmethod
    def _error_analysis_comp(webapp: NlpWebApp) -> ErrorAnalysisComponent:
        return next(c for c in webapp._components if isinstance(c, ErrorAnalysisComponent))

    def test_confusion_matrix_counts(self):
        webapp = self._make_webapp_with_gt()
        cm = self._error_analysis_comp(webapp)._cm
        # LABEL_NAMES index: sadness=0, joy=1. Rows=true, cols=pred.
        self.assertEqual(cm[0, 1], 1)  # true sadness predicted joy (the error)
        self.assertEqual(cm[0, 0], 1)  # true sadness predicted sadness
        self.assertEqual(cm[1, 1], 1)  # true joy predicted joy
        self.assertEqual(cm.sum(), 3)

    def test_confusion_matrix_index_arrays(self):
        webapp = self._make_webapp_with_gt()
        comp = self._error_analysis_comp(webapp)
        self.assertEqual(comp._cm_true_idx.tolist(), [0, 0, 1])  # sadness, sadness, joy
        self.assertEqual(comp._cm_pred_idx.tolist(), [1, 0, 1])  # joy, sadness, joy

    def test_confusion_matrix_figure_customdata_orientation(self):
        webapp = self._make_webapp_with_gt()
        graph = self._find_component(webapp.app.layout, "confusion-matrix-graph")
        cd = np.asarray(graph.figure.data[0].customdata)
        self.assertEqual(list(cd[0, 1]), [1, 0])  # cell (true=0, pred=1) → [pred_idx, true_idx]

    def test_cell_from_click_uses_label_names_when_no_customdata(self):
        # Heatmap clicks may omit customdata; the x (pred) / y (true) labels must still resolve.
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness"}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (1, 0))

    def test_cell_from_click_prefers_customdata(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        click = {"points": [{"x": "joy", "y": "sadness", "customdata": [2, 3]}]}
        self.assertEqual(_cell_from_click(click, name_to_idx), (2, 3))

    def test_cell_from_click_none_on_empty_or_unknown(self):
        name_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}
        self.assertIsNone(_cell_from_click(None, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": []}, name_to_idx))
        self.assertIsNone(_cell_from_click({"points": [{"x": "??", "y": "??"}]}, name_to_idx))

    @staticmethod
    def _callback(webapp, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in webapp.app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_update_confusion_matrix_normalizes_on_recall(self):
        webapp = self._make_webapp_with_gt()
        update = self._callback(webapp, "confusion-matrix-graph.figure")
        fig = update("recall")
        # The container drives sizing for this half-column panel, not the figure itself.
        self.assertIsNone(fig.layout.width)
        self.assertIsNone(fig.layout.height)

    def test_set_error_cell_resolves_click_to_cell_and_clears_clickdata(self):
        webapp = self._make_webapp_with_gt()
        set_cell = self._callback(webapp, "error-cell.data")
        click = {"points": [{"x": "joy", "y": "sadness"}]}
        with patch.object(error_analysis_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "confusion-matrix-graph.clickData"}]
            data, click_out = set_cell(click, None)
        self.assertEqual(data, {"pred": 1, "true": 0, "indices": [0]})
        self.assertIsNone(click_out)

    def test_set_error_cell_prevents_update_on_unresolved_click(self):
        webapp = self._make_webapp_with_gt()
        set_cell = self._callback(webapp, "error-cell.data")
        click = {"points": [{"x": "??", "y": "??"}]}
        with patch.object(error_analysis_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "confusion-matrix-graph.clickData"}]
            with self.assertRaises(PreventUpdate):
                set_cell(click, None)

    def test_set_error_cell_clear_button_resets_store(self):
        webapp = self._make_webapp_with_gt()
        set_cell = self._callback(webapp, "error-cell.data")
        with patch.object(error_analysis_module, "callback_context") as cc:
            cc.triggered = [{"prop_id": "error-cell-clear-btn.n_clicks"}]
            data, click_out = set_cell(None, 1)
        self.assertIsNone(data)
        self.assertIsNone(click_out)

    def test_toggle_error_cell_clear_button_visibility(self):
        webapp = self._make_webapp_with_gt()
        toggle = self._callback(webapp, "error-cell-clear-btn.style")
        self.assertEqual(toggle(None)["display"], "none")
        self.assertEqual(toggle({"pred": 1, "true": 0, "indices": [0]})["display"], "inline")

    def test_update_error_word_charts_empty_without_a_selected_cell(self):
        webapp = self._make_webapp_with_gt()
        update = self._callback(webapp, "error-cell-caption.children")
        fig_pred, fig_true, caption = update(None)
        self.assertIsInstance(fig_pred, go.Figure)
        self.assertIsInstance(fig_true, go.Figure)
        self.assertIn("Click a cell", caption)

    def test_update_error_word_charts_empty_when_cell_has_no_samples(self):
        webapp = self._make_webapp_with_gt()
        update = self._callback(webapp, "error-cell-caption.children")
        fig_pred, fig_true, caption = update({"pred": 0, "true": 1, "indices": []})
        self.assertIsInstance(fig_pred, go.Figure)
        self.assertIn("0 samples", caption)

    def test_update_error_word_charts_off_diagonal_cell_flags_few_samples(self):
        webapp = self._make_webapp_with_gt()
        update = self._callback(webapp, "error-cell-caption.children")
        # pred=joy(1), true=sadness(0): the single planted error, row 0.
        fig_pred, fig_true, caption = update({"pred": 1, "true": 0, "indices": [0]})
        self.assertIsInstance(fig_pred, go.Figure)
        self.assertIsInstance(fig_true, go.Figure)
        self.assertIn("predicted joy", caption)
        self.assertIn("true sadness", caption)
        self.assertIn("1 sample", caption)
        self.assertIn("few samples", caption)

    def test_update_error_word_charts_diagonal_cell_marks_correct_predictions(self):
        webapp = self._make_webapp_with_gt()
        update = self._callback(webapp, "error-cell-caption.children")
        # pred=joy(1), true=joy(1): a correct prediction, row 2.
        fig_pred, fig_true, caption = update({"pred": 1, "true": 1, "indices": [2]})
        self.assertIsInstance(fig_pred, go.Figure)
        self.assertIsInstance(fig_true, go.Figure)
        self.assertIn("correct predictions (diagonal)", caption)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_component(self, node, component_id):
        """Depth-first search for a Dash component by id."""
        if hasattr(node, "id") and node.id == component_id:
            return node
        children = getattr(node, "children", None)
        if children is None:
            return None
        if not isinstance(children, list):
            children = [children]
        for child in children:
            result = self._find_component(child, component_id)
            if result is not None:
                return result
        return None

    def _collect_ids(self, node) -> set:
        """Collect all component ids in the layout tree."""
        ids = set()
        if hasattr(node, "id") and isinstance(node.id, str):
            ids.add(node.id)
        children = getattr(node, "children", None)
        if children is None:
            return ids
        if not isinstance(children, list):
            children = [children]
        for child in children:
            ids |= self._collect_ids(child)
        return ids


class TestNlpWebAppRun(unittest.TestCase):
    """``NlpWebApp.run`` / ``NlpExplainer.run_app`` — delegate to ``dash.Dash.run``; a stop handle in notebooks."""

    def setUp(self):
        self.xpl = _make_explainer()
        self.explanation = _make_explanation()

    @staticmethod
    def _jupyter_active(active):
        return patch.object(type(jupyter_dash), "active", new_callable=PropertyMock, return_value=active)

    def test_outside_notebook_delegates_to_dash_and_returns_none(self):
        webapp = NlpWebApp(self.explanation, engine=self.xpl)
        with self._jupyter_active(False), patch.object(webapp.app, "run") as mock_run:
            result = webapp.run(port=8123, debug=True)
        mock_run.assert_called_once_with(port=8123, debug=True, host="127.0.0.1")
        self.assertIsNone(result)

    def test_in_notebook_returns_handle_on_dash_registered_server(self):
        webapp = NlpWebApp(self.explanation, engine=self.xpl)
        server = MagicMock()

        def register(port, debug, host):
            jupyter_dash._servers[(host, port)] = server

        with self._jupyter_active(True), patch.object(webapp.app, "run", side_effect=register):
            app = webapp.run(port=8123)
        self.assertIsInstance(app, RunningApp)
        app.kill()
        server.shutdown.assert_called_once_with()
        self.assertNotIn(("127.0.0.1", 8123), jupyter_dash._servers)

    def test_run_app_forwards_the_handle(self):
        sentinel = object()
        with patch.object(NlpWebApp, "run", return_value=sentinel) as mock_run:
            result = self.xpl.run_app(self.explanation, port=8123)
        mock_run.assert_called_once_with(port=8123, debug=False, host="127.0.0.1")
        self.assertIs(result, sentinel)


class TestRunAppMountPath(unittest.TestCase):
    """``run_app`` forwards the reverse-proxy mount point to the webapp it builds."""

    def test_url_base_pathname_reaches_the_webapp(self):
        xpl = object.__new__(NlpExplainer)
        explanation = object()
        with patch("shapash.explainer.nlp_explainer.NlpWebApp") as web_app:
            xpl.run_app(explanation, url_base_pathname="/shapash-nlp-explainer/")

        self.assertIs(web_app.call_args.args[0], explanation)
        self.assertEqual(
            web_app.call_args.kwargs,
            {
                "engine": xpl,
                "projection": None,
                "url_base_pathname": "/shapash-nlp-explainer/",
                "palette_name": "default",
                "colors_dict": None,
                "info": {},
            },
        )

    def test_defaults_to_no_prefix(self):
        xpl = object.__new__(NlpExplainer)
        with patch("shapash.explainer.nlp_explainer.NlpWebApp") as web_app:
            xpl.run_app(object())
        self.assertIsNone(web_app.call_args.kwargs["url_base_pathname"])
