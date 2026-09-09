"""Minimal Dash webapp for NLP text classification explanations.

Prototype bridge toward Phase 5b (composable WebappComponents). Panels are being extracted into
``WebappComponent``s one at a time (see ``shapash/webapp/nlp_components/``); the dataset table and
the persistent selection bar are the only pieces still built inline — both are mandatory shell UI,
not optional panels, so there is nothing left to extract them into. All tabular-only SmartApp panels
(violin, cluster, scatter prediction picking) beyond what NLP needs are absent rather than disabled.
"""

from __future__ import annotations

import logging

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import MATCH, Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from shapash.explainer.interactive import InteractiveEngine
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.style.style_utils import NlpTheme, resolve_nlp_theme
from shapash.webapp.nlp_components import (
    CounterfactualComponent,
    DataEditorComponent,
    ErrorAnalysisComponent,
    LabelNoiseComponent,
    ScatterComponent,
    SentenceHighlightComponent,
    SimilarExamplesComponent,
    WaterfallComponent,
    WebappComponent,
    WordImportanceComponent,
    WordProfileComponent,
    compose_selection,
    error_positions,
    pack_datapoint,
)
from shapash.webapp.utils.launch import RunningApp, run_in_background

_APPLY_STORE = "whatif-apply-store"
_CURRENT_STORE = "current-datapoint"

_HIDDEN = {"display": "none"}
# The visible tab body is a flex-column item that fills the bodies container (which itself fills the
# card). This unbroken flex chain is what lets a body's inner content use height:100% / flex:1 (e.g. the
# dataset grid) to fill the panel; a plain `display:block` here would collapse to content height.
# overflowX is pinned to "hidden" rather than left at its "visible" default: per the CSS spec, an axis
# left "visible" next to a sibling axis that isn't computes to "auto" instead — so without this, any
# child a stray pixel wider than the panel (a border, a scrollbar's own width) silently grows a
# horizontal scrollbar here, even though nothing in this layout is meant to scroll sideways.
_VISIBLE = {
    "display": "flex",
    "flexDirection": "column",
    "flex": "1 1 auto",
    "minHeight": "0",
    "overflowY": "auto",
    "overflowX": "hidden",
}

# Flex/border idioms repeated verbatim across the header, selection bar, tab columns and
# `_tabbed_card` — named here so a layout tweak to one of them doesn't have to be hunted down
# across every dict literal that copied it.
_FLEX_FILL = {"flex": "1 1 auto", "minHeight": "0"}
_FLEX_FIXED = {"flex": "0 0 auto"}
_PANEL_BORDER = {"border": "1px solid #dee2e6", "borderRadius": "4px"}


def _normalize_base_pathname(path: str | None) -> str | None:
    """Return *path* spelled the one way Dash accepts as ``url_base_pathname``.

    Dash rejects a prefix that does not both start and end with ``/``, and a mount point copied
    from an nginx ``location`` or typed on a CLI routinely arrives missing one. ``/x/`` is the only
    valid spelling of the same intent, so the slashes are added rather than raised over.

    Parameters
    ----------
    path : str or None
        The requested mount point, in any slash spelling.

    Returns
    -------
    str or None
        ``None`` when the app should serve at the server root (``None``, ``""`` or ``"/"``),
        otherwise the prefix with both slashes.
    """
    if path is None:
        return None
    stripped = path.strip().strip("/")
    return f"/{stripped}/" if stripped else None


def _clear_button(button_id: str, label: str, *, margin: str = "me-3") -> dbc.Button:
    """A small "x clear ..." link button, hidden until its own callback reveals it.

    Shared by every selection-bar clear control (scatter / error-cell / word-filter / grid-filter
    — see `_build_selection_bar`) — same look and same hidden-until-active behaviour, differing
    only in id, label, and whether a right-margin is needed before the next control.
    """
    className = f"text-muted p-0 {margin}".rstrip()
    return dbc.Button(
        label,
        id=button_id,
        n_clicks=0,
        color="link",
        size="sm",
        className=className,
        style={"display": "none", "fontSize": "0.8em"},
    )


class NlpWebApp:
    """Minimal Dash webapp driven by an ``NlpExplainer``.

    The layout follows an "overview → filter → detail" funnel:

    Top row — global "where to look" panels that filter the table below:
    - **Global word importance** — mean SHAP contribution per unique word for
      the selected class (updates on any control change or scatter selection).
      Its own controls (top-K slider, positive/negative sign filter, corpus
      word multi-select for manual stopword exclusion) live inside this panel.
    - **Sample scatter** *(optional)* — 2-D projection of the text embeddings,
      coloured by prediction or ground-truth label.  Draw a box or lasso to
      filter the table and word importance to the selected subset.

    Hub — full-width:
    - **Dataset table** — text samples with predicted (and optional ground-truth)
      labels; click a row to populate the local contribution panel.  Filtered
      to the scatter selection and/or a clicked word importance bar.

    Detail-on-demand — full-width:
    - **Local contributions** — inline sentence highlight as the primary view.
      An optional waterfall chart (toggle with a radio button) groups tokens
      below a configurable contribution threshold into a single "other" bar.

    Parameters
    ----------
    explanation : NlpExplanation
        The result of ``NlpExplainer.explain()`` (or a loaded one — see
        ``NlpExplanation.load()``) to serve.
    engine : InteractiveEngine, optional
        Live engine for what-if actions (re-predicting edited text, generating
        counterfactuals, similar-example retrieval, label-noise probing) — normally the
        ``NlpExplainer`` that produced ``explanation``. ``None`` for a snapshot loaded via
        ``NlpExplanation.load()``, which carries no model: components self-disable via
        their ``requires`` when the engine is absent or lacks a capability.
    scatter_xy : np.ndarray, optional
        Pre-computed 2-D projection, shape ``(n_samples, 2)``.  When provided,
        a scatter panel is added to the layout.  Compute with PaCMAP, UMAP,
        t-SNE, PCA, etc. and pass the result here — Shapash does not perform
        the projection itself to avoid heavy optional dependencies.
    url_base_pathname : str, optional
        Mount the app under a URL prefix instead of the server root, for serving behind a reverse
        proxy that routes a subpath (e.g. ``"/shapash-nlp-explainer/"``) to this process. Dash
        rewrites its own routes *and* every asset/callback URL it emits, so the prefix must match
        the proxied path exactly. ``None`` (the default) serves at ``/``.

        Missing slashes are added — ``"shapash-nlp-explainer"`` and ``"/shapash-nlp-explainer"``
        both become ``"/shapash-nlp-explainer/"`` — since that is the only spelling Dash accepts
        and a prefix copied from an nginx ``location`` easily arrives without one.
    palette_name : str, optional
        Name of a palette in ``shapash/style/colors.json`` (the same file the tabular
        ``SmartExplainer`` webapp draws its theme from — see ``smart_app.py``). Selects the
        header's band color (``webapp_bkg``) and accent color (``webapp_title``). Defaults to
        ``"default"``, the same palette ``SmartExplainer`` falls back to.
    colors_dict : dict[str, str], optional
        Per-key overrides merged on top of *palette_name*, same shape and key names as
        ``colors.json`` and as ``SmartExplainer``'s own ``colors_dict`` (see
        ``smart_explainer.py``'s ``define_style``). Only ``webapp_bkg`` (header band) and
        ``webapp_title`` (accent color) are read today; other keys are accepted but unused.
    info : dict[str, str], optional
        Extra ``label -> value`` facts shown in the header's "ⓘ" popover, layered on top of what
        ``explanation`` already carries (``model_id``, ``architecture`` — each omitted when the
        value is ``None`` — plus ``n_samples`` and ``n_classes``/``backend_name``,
        always shown). A key already filled in from ``explanation`` is overridden by *info*; any
        other key is appended after — except ``"Train samples"``, which is placed right after "Test
        samples" since the two are naturally read together. ``NlpExplainer.run_app`` uses this key to
        add it (only known post-``fit()``, so it can never live on the model-free ``explanation``) —
        see there for the full picture when going through it rather than constructing ``NlpWebApp``
        directly.
    """

    def __init__(
        self,
        explanation: NlpExplanation,
        engine: InteractiveEngine | None = None,
        scatter_xy: np.ndarray | None = None,
        url_base_pathname: str | None = None,
        palette_name: str = "default",
        colors_dict: dict[str, str] | None = None,
        info: dict[str, str] | None = None,
    ) -> None:
        if scatter_xy is not None:
            scatter_xy = np.asarray(scatter_xy)
            n = len(explanation.texts)
            if scatter_xy.shape != (n, 2):
                raise ValueError(f"scatter_xy must have shape ({n}, 2), got {scatter_xy.shape}")
        self._scatter_xy: np.ndarray | None = scatter_xy

        # What-if Lab: the immutable artifact + a separate, possibly-absent live engine. The
        # artifact is only ever read here — display state lives in the `dcc.Store`s declared in
        # `_build_layout`, never on the explanation.
        # Components self-disable via their `requires` when the engine is None or lacks a capability
        # (e.g. a loaded snapshot has no model at all). `self._components` itself is assembled later
        # in `_build_layout` (it needs `_full_table_records`, not ready yet here).
        self._explanation = explanation
        self._engine = engine

        # See the `info` docstring above: everything `explanation` can say for itself, in reading
        # order, with *info* (from the caller, or NlpExplainer.run_app's "Train samples") layered on
        # top — overriding a key already here, or appended after for anything new.
        self._info: dict[str, str] = {}
        if explanation.model_id:
            self._info["Model"] = explanation.model_id
        if explanation.architecture:
            self._info["Model architecture"] = explanation.architecture
        self._info["Test samples"] = str(explanation.n_samples)
        if info and "Train samples" in info:
            # Reserve the slot here, right after "Test samples", so the update() below fills an
            # existing key in place instead of appending it at the end like any other extra fact.
            self._info["Train samples"] = info["Train samples"]
        self._info["Number of classes"] = str(explanation.n_classes)
        self._info["Explainer backend"] = explanation.backend_name
        self._info.update(info or {})

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            url_base_pathname=_normalize_base_pathname(url_base_pathname),
        )
        self.app.title = "Shapash - NLP Explainer"

        self._theme: NlpTheme = resolve_nlp_theme(palette_name, colors_dict)

        # Pin the document to the viewport so only the inner panels scroll, never the page. The
        # 100vh shell needs html/body at full height with no default margin; overflow:hidden is the
        # guarantee against any residual sub-pixel overflow producing a page scrollbar.
        self.app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { height: 100%; margin: 0; overflow: hidden; }
            #react-entry-point { height: 100%; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""
        self._build_layout()
        self._register_callbacks()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Orchestrate the shell: table → columns → selection bar → components/tabs → container.

        Each step below is a focused builder (see the methods that follow) — this method's job is
        only the assembly order and the handful of values threaded between them.
        """
        # Predicted-label name → class index, shared by the confusion matrix and (via
        # NlpExplanation.label_to_idx) by SentenceHighlightComponent's predicted-class sync callback.
        self._label_to_idx = self._explanation.label_to_idx

        has_prob = self._build_table_records()
        column_defs = self._build_column_defs(has_prob)
        # Field → header-name lookup for the grid-filter badge in `_register_callbacks` (AG Grid's
        # own column filters report field names via filterModel, but the badge should read like the UI).
        self._column_header_by_field = {c["field"]: c["headerName"] for c in column_defs}
        text_samples_body = self._build_text_samples_body(column_defs)

        selection_bar = self._build_selection_bar()

        # ── Assemble the three panels as tab groups (all bodies stay mounted) ──
        self._tab_groups: dict[str, list[str]] = {}
        components = self._build_components()
        left_tabs, upper_right_tabs, lower_right_tabs = self._build_tabs(components, text_samples_body)

        left_column = html.Div(
            [selection_bar, self._tabbed_card("left-tabs", left_tabs)],
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "height": "100%"},
        )
        right_column = html.Div(
            [
                self._tabbed_card("upper-right-tabs", upper_right_tabs),
                self._tabbed_card("lower-right-tabs", lower_right_tabs),
            ],
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "height": "100%"},
        )

        self.app.layout = dbc.Container(
            [
                # The Class selector lives with the panel it drives: Word Importance (global) and
                # Sentence Highlight (local, see highlight_body) each get their own, so switching
                # one no longer silently reinterprets the other.
                self._build_header(),
                # ── Three-panel body: left (data) | right (global over local) ──
                dbc.Row(
                    [
                        dbc.Col(left_column, width=5, style={"height": "100%"}),
                        dbc.Col(right_column, width=7, style={"height": "100%"}),
                    ],
                    # gx-3 = horizontal gutter only. A vertical gutter (g-3) adds a 1rem net excess
                    # that pushes the 100vh shell past the viewport and makes the whole page scroll.
                    className="gx-3",
                    style=_FLEX_FILL,
                ),
                *self._build_stores(),
            ],
            fluid=True,
            # Full-viewport shell: panels scroll internally, the page itself never scrolls. The 4px
            # paddingBottom is breathing room below the last panel — box-sizing is border-box
            # (Bootstrap's reboot), so it comes out of the 100vh instead of pushing the page taller.
            style={
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
                "paddingBottom": "4px",
            },
        )

    def _build_table_records(self) -> bool:
        """Build the dataset table's row records.

        Caches ``self._full_table_records`` (scatter-driven re-filtering in callbacks reads it) and
        ``self._has_gt``. Returns whether a "probability" column was included, the one piece the
        caller still needs to build the matching column defs.
        """
        n = self._explanation.n_samples
        texts = self._explanation.texts
        assert texts is not None  # noqa: S101 - the webapp requires an already-compiled explainer
        records: dict[str, list] = {
            "_orig_idx": list(range(n)),
            "text": texts.tolist(),
            "prediction": (self._explanation.y_pred.tolist() if self._explanation.y_pred is not None else [""] * n),
        }
        if self._explanation.y_true is not None:
            records["ground_truth"] = self._explanation.y_true.tolist()

        # Single "Probability" column — confidence of the predicted class.
        # max(axis=1) works for both binary and multiclass since the predicted
        # label is always the argmax, regardless of how columns are named.
        y_prob: pd.DataFrame | None = self._explanation.y_prob
        if y_prob is not None:
            records["probability"] = y_prob.max(axis=1).tolist()

        table_df = pd.DataFrame(records)
        self._full_table_records: list[dict] = table_df.to_dict("records")
        self._has_gt = "ground_truth" in table_df.columns
        return y_prob is not None

    def _build_column_defs(self, has_prob: bool) -> list[dict]:
        """AG Grid column defs for the dataset table — ground truth/probability columns are optional."""
        column_defs: list[dict] = [
            {"field": "text", "headerName": "Text", "flex": 3, "tooltipField": "text", "filter": "agTextColumnFilter"},
            {"field": "prediction", "headerName": "Prediction", "flex": 1, "filter": "agTextColumnFilter"},
        ]
        if self._has_gt:
            column_defs.append(
                {"field": "ground_truth", "headerName": "Ground Truth", "flex": 1, "filter": "agTextColumnFilter"}
            )
        if has_prob:
            column_defs.append(
                {
                    "field": "probability",
                    "headerName": "Probability",
                    "flex": 1,
                    "filter": "agNumberColumnFilter",
                    "valueFormatter": {"function": "params.value != null ? params.value.toFixed(3) : ''"},
                }
            )
        return column_defs

    def _build_text_samples_body(self, column_defs: list[dict]) -> html.Div:
        """Dataset table body (left-panel default tab): title row + the AG Grid itself."""
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H6(
                                "Text Samples — click a row to inspect",
                                className="fw-bold mb-0",
                                id="table-title",
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                    ],
                    className="align-items-center mb-2",
                    style=_FLEX_FIXED,
                ),
                # Grid grows to fill the tab body (now that the editor lives on its own tab).
                dag.AgGrid(
                    id="dataset-table",
                    rowData=self._full_table_records,
                    columnDefs=column_defs,
                    defaultColDef={"resizable": True, "sortable": True},
                    dashGridOptions={
                        # AG Grid v32.2+ (dash-ag-grid ≥31) replaced the string form
                        # ("single"/"multiple") with an object. In the new API a row click no
                        # longer selects by default — you must opt in via enableClickSelection,
                        # otherwise `selectedRows` never populates and every click 204s.
                        "rowSelection": {
                            "mode": "singleRow",
                            "checkboxes": False,
                            "enableClickSelection": True,
                        },
                        "tooltipShowDelay": 300,
                        "rowHeight": 38,
                    },
                    selectedRows=[self._full_table_records[0]],
                    style={**_FLEX_FILL, "--ag-font-size": "15px", "--ag-header-font-size": "14px"},
                    className="ag-theme-alpine",
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

    def _build_selection_bar(self) -> html.Div:
        """Persistent, always-visible filter state + clear buttons.

        Lives above the left tabs so a scatter/word selection stays visible whichever left tab
        (table or embeddings) is active. Clear buttons are consolidated here from the individual
        panels they used to live in.
        """
        selection_children: list = [
            html.Span("Showing:", className="text-muted small me-2"),
            # AG Grid's own column filters (the funnel icon in each header) act on the grid
            # client-side and don't flow through filter_table's inputs the way the other filters
            # do — filter_table folds them into this same summary via a filterModel/virtualRowData
            # Input pair, so a grid-level filter is described here too, not just inside the grid.
            html.Span("all samples", id="selection-summary", className="small fw-bold me-3"),
        ]
        if self._scatter_xy is not None:
            selection_children.append(_clear_button("scatter-clear-btn", "× clear selection"))
        if self._has_gt:
            selection_children.append(_clear_button("error-cell-clear-btn", "× clear cell"))
        selection_children.append(_clear_button("word-filter-clear-btn", "× clear word filter"))
        # No trailing margin: it is followed by the errors-only switch, pushed to the far right
        # by that switch's own `ms-auto`, not by spacing here.
        selection_children.append(_clear_button("grid-filter-clear-btn", "× clear grid filter", margin=""))
        # Right-aligned (ms-auto) and here — rather than inside the Dataset tab body — so it stays
        # usable from the Embeddings tab too, where it now also drives point highlighting.
        selection_children.append(
            dbc.Switch(
                id="errors-only-switch",
                label="Model Errors",
                value=False,
                className="small mb-0 ms-auto",
                style={} if self._has_gt else _HIDDEN,
            )
        )
        return html.Div(
            selection_children,
            style={
                **_PANEL_BORDER,
                "padding": "6px 12px",
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                **_FLEX_FIXED,
            },
        )

    def _build_components(self) -> list[WebappComponent]:
        """Instantiate every capability-gated component, plus the scatter panel if data was given.

        Assigns ``self._components`` — read again by ``_register_callbacks`` once the layout exists
        — and returns the same list for ``_build_tabs`` to map onto tab groups.
        """
        # Local class picker default: the predicted class of the initially selected row. It is reset
        # to the newly-selected text's prediction by a sync callback owned by
        # SentenceHighlightComponent — see sync_local_class_to_prediction — so switching sentences
        # always starts on "why did the model predict this", while still letting the user override it
        # for the current sentence.
        predicted_label = self._full_table_records[0].get("prediction")
        default_local_class = self._label_to_idx.get(predicted_label, 0) if isinstance(predicted_label, str) else 0
        components: list[WebappComponent] = [
            comp
            for comp in (
                SentenceHighlightComponent(default_local_class, theme=self._theme),
                WaterfallComponent(theme=self._theme),
                DataEditorComponent(),
                CounterfactualComponent(),
                SimilarExamplesComponent(),
                LabelNoiseComponent(),
                ErrorAnalysisComponent(theme=self._theme),
                WordProfileComponent(theme=self._theme),
                WordImportanceComponent(theme=self._theme),
            )
            if type(comp).is_available(self._explanation, self._engine)
        ]
        # Mounting is decided by the caller-supplied array, not by `is_available` — a pre-computed
        # projection is arbitrary data handed to NlpWebApp, not an explanation/engine capability.
        # Word Importance is always mounted at this point (it has no `requires`), so the scatter's
        # word-contribution color mode can always be offered.
        if self._scatter_xy is not None:
            components.append(ScatterComponent(self._scatter_xy, offer_word_contribution=True))
        self._components = components
        return components

    def _build_tabs(self, components: list[WebappComponent], text_samples_body: html.Div) -> tuple[list, list, list]:
        """Map *components* onto the shell's three tab groups: left, upper-right, lower-right."""
        highlight_comp = next(c for c in components if isinstance(c, SentenceHighlightComponent))
        waterfall_comp = next(c for c in components if isinstance(c, WaterfallComponent))
        word_importance_comp = next(c for c in components if isinstance(c, WordImportanceComponent))
        editor_comp = next((c for c in components if isinstance(c, DataEditorComponent)), None)
        cf_comp = next((c for c in components if isinstance(c, CounterfactualComponent)), None)
        similar_comp = next((c for c in components if isinstance(c, SimilarExamplesComponent)), None)
        noise_comp = next((c for c in components if isinstance(c, LabelNoiseComponent)), None)
        error_analysis_comp = next((c for c in components if isinstance(c, ErrorAnalysisComponent)), None)
        word_profile_comp = next((c for c in components if isinstance(c, WordProfileComponent)), None)
        scatter_comp = next((c for c in components if isinstance(c, ScatterComponent)), None)

        left_tabs: list = [("table", "Dataset", text_samples_body)]
        if scatter_comp is not None:
            left_tabs.append(("scatter", "Embeddings", scatter_comp.layout(self._explanation, self._engine)))
        if editor_comp is not None:
            left_tabs.append(("editor", "Data Editor", editor_comp.layout(self._explanation, self._engine)))

        # Error Analysis sits beside Word Importance: it *is* an aggregated word-importance view
        # (per confusion-matrix cell), so it belongs with the other global "why" panels on the right.
        upper_right_tabs: list = [
            ("importance", "Word Importance", word_importance_comp.layout(self._explanation, self._engine))
        ]
        # Immediately after Word Importance, because it is the same question asked the other way
        # round (one word across all classes, instead of one class across the top words) and the two
        # are read together — a bar clicked there arrives preselected here.
        if word_profile_comp is not None:
            upper_right_tabs.append(
                ("word-profile", "Word Profile", word_profile_comp.layout(self._explanation, self._engine))
            )
        if error_analysis_comp is not None:
            upper_right_tabs.append(
                ("errors", "Error Analysis", error_analysis_comp.layout(self._explanation, self._engine))
            )
        # Next to Error Analysis by design: same ground-truth prerequisite, and it answers the
        # question that panel leaves open — whether the *label*, not the model, is what is wrong.
        if noise_comp is not None:
            upper_right_tabs.append(("label-noise", "Label Noise", noise_comp.layout(self._explanation, self._engine)))
        if cf_comp is not None:
            upper_right_tabs.append(
                ("counterfactual", "Counterfactuals", cf_comp.layout(self._explanation, self._engine))
            )
        if similar_comp is not None:
            upper_right_tabs.append(
                ("similar", "Similar Examples", similar_comp.layout(self._explanation, self._engine))
            )

        lower_right_tabs: list = [
            ("highlight", "Sentence", highlight_comp.layout(self._explanation, self._engine)),
            ("waterfall", "Waterfall", waterfall_comp.layout(self._explanation, self._engine)),
        ]
        return left_tabs, upper_right_tabs, lower_right_tabs

    def _build_stores(self) -> list[dcc.Store]:
        """Shared ``dcc.Store``s the shell and its components read/write cross-panel state through."""
        stores: list = [
            # current-datapoint is the app's primary selection: the one text every
            # per-instance panel (highlight, waterfall, counterfactuals) reads from.
            dcc.Store(id=_CURRENT_STORE, data=None),
            dcc.Store(id="scatter-selected-indices", data=None),
            dcc.Store(id="word-click-filter", data=None),
            # Selected confusion-matrix cell: {"pred": idx, "true": idx, "indices": [...]} or None.
            dcc.Store(id="error-cell", data=None),
            # Word Importance's class selector, published so Scatter's word-contribution color mode
            # can read it without referencing Word Importance's DOM id directly (see
            # nlp_components/scatter.py).
            dcc.Store(id="active-class-store", data=0),
        ]
        # Only the What-if Lab (editor + counterfactual) reads/writes the apply store; the always-on
        # core panels (highlight, waterfall) never do, so their presence alone shouldn't create it.
        needs_apply_store = any(isinstance(c, (DataEditorComponent, CounterfactualComponent)) for c in self._components)
        if needs_apply_store:
            stores.append(dcc.Store(id=_APPLY_STORE, data=None))
        return stores

    def _build_header(self) -> html.Div:
        """Top band: clickable logo + title (links to the repo) and the run-info popover."""
        return html.Div(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    # "fond fonce" — the dark-background mark, matching
                                    # the gray band it sits on here.
                                    src=self.app.get_asset_url("shapash-fond-fonce.png"),
                                    style={"height": "32px"},
                                ),
                                width="auto",
                            ),
                            dbc.Col(
                                html.H4(
                                    "Shapash - NLP Explainer",
                                    className="mb-0",
                                    style={"color": self._theme.header_accent},
                                )
                            ),
                        ],
                        align="center",
                        className="g-2",
                    ),
                    href="https://github.com/MAIF/shapash",
                    target="_blank",
                    style={"textDecoration": "none", "display": "block"},
                ),
                html.Div(
                    [
                        html.Span(
                            html.I("info", className="material-icons"),
                            id="app-info-btn",
                            title="Run info",
                            style={"cursor": "pointer", "color": self._theme.header_accent, "fontSize": "22px"},
                        ),
                        dbc.Popover(
                            dbc.PopoverBody(
                                [
                                    html.Div(
                                        [html.Span(f"{label}: ", className="fw-bold"), html.Span(value)],
                                        className="small",
                                    )
                                    for label, value in self._info.items()
                                ]
                            ),
                            target="app-info-btn",
                            trigger="click",
                            placement="bottom-end",
                        ),
                    ],
                    style={**_FLEX_FIXED, "alignSelf": "center"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "margin": "0 0 6px 0",
                "padding": "12px 16px",
                "backgroundColor": self._theme.header_bkg,
                **_FLEX_FIXED,
            },
        )

    def _tabbed_card(self, tabs_id: str, tabs: list) -> html.Div:
        """Wrap ``(tab_id, label, body)`` tuples into a card with a tab header.

        Every body stays mounted in the DOM; a registered callback toggles ``display`` so
        cross-panel callbacks that target a component on an inactive tab keep working (Dash
        errors if an Output/Input component is absent). The tab ids are recorded on
        ``self._tab_groups`` for that toggle callback.
        """
        active = tabs[0][0]
        self._tab_groups[tabs_id] = [tid for tid, _, _ in tabs]
        headers = [dbc.Tab(label=label, tab_id=tid) for tid, label, _ in tabs]
        bodies = [
            html.Div(body, id=f"{tabs_id}-body-{tid}", style=(_VISIBLE if tid == active else _HIDDEN))
            for tid, _, body in tabs
        ]
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Tabs(headers, id=tabs_id, active_tab=active, style={"flex": "1 1 auto", "minWidth": "0"}),
                        # Magnifies whichever tab is active — see the `panel-card`/`panel-expand-btn`
                        # clientside callback in `_register_callbacks`. A CSS class toggle (not the
                        # browser Fullscreen API) so it works the same for every panel regardless of
                        # what the active tab's own layout does internally.
                        html.Span(
                            html.I("fullscreen", className="material-icons"),
                            id={"type": "panel-expand-btn", "id": tabs_id},
                            n_clicks=0,
                            title="Expand panel",
                            style={"cursor": "pointer", "flex": "0 0 auto", "padding": "0 2px", "color": "#6c757d"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "6px"},
                ),
                # Flex column so the single visible body (the others are display:none) fills the height.
                html.Div(
                    bodies,
                    style={
                        "flex": "1 1 auto",
                        "minHeight": "0",
                        "paddingTop": "8px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
            ],
            id={"type": "panel-card", "id": tabs_id},
            style={
                "border": "1px solid #dee2e6",
                "borderRadius": "4px",
                "padding": "12px",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
                "flex": "1 1 0",
                "minHeight": "0",
                "backgroundColor": "white",
            },
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _register_callbacks(self) -> None:
        explanation = self._explanation
        full_records = self._full_table_records
        has_gt = self._has_gt
        column_header_by_field = self._column_header_by_field

        def _compose_indices(selected_indices, error_cell, errors_only=False):
            """Combine the scatter selection, the confusion-cell selection, and the errors toggle.

            All three are optional filters that intersect: the embedding selection *within* the
            chosen confusion cell, further restricted to misclassified samples when ``errors_only``
            is set. Returns a list of original sample indices, or ``None`` when nothing is active.
            """
            cell_indices = error_cell.get("indices") if error_cell else None
            errors = error_positions(explanation) if (errors_only and has_gt) else None
            return compose_selection(selected_indices, cell_indices, errors)

        # ── Primary selection: selected table row → current-datapoint ────
        # The editor's Predict callback also writes this store (see DataEditorComponent),
        # so every per-instance panel reads one source regardless of where it came from.
        @self.app.callback(
            Output(_CURRENT_STORE, "data"),
            Input("dataset-table", "selectedRows"),
        )
        def set_current_from_row(selected_rows):
            if not selected_rows:
                raise PreventUpdate
            pos = int(selected_rows[0]["_orig_idx"])
            base_values = explanation.base_values
            base = base_values[pos] if base_values is not None else None
            return pack_datapoint(
                text=selected_rows[0].get("text", ""),
                orig_idx=pos,
                tokens=explanation.token_strings[pos],
                values=explanation.values[pos],
                base_values=base,
                label=selected_rows[0].get("prediction"),
            )

        # Local class picker sync + sentence highlight + waterfall are registered by
        # SentenceHighlightComponent / WaterfallComponent (see the component loop at the end of this
        # method) — they only need the current-datapoint store, already shared via `stores["current"]`.

        # Word-bar click / clear → table word filter, and the coloring/selection scatter callbacks,
        # are registered by WordImportanceComponent / ScatterComponent respectively (see the
        # component loop at the end of this method) — both read/write the shared
        # `stores["word_click"]`/`stores["selection"]` stores, already wired above.

        # ── Unified table filter (scatter + word-bar click + errors-only) ──
        # Also writes the selection-summary readout: it already knows the resulting row count, so the
        # count and the filter description stay consistent (no separate, drift-prone callback).
        @self.app.callback(
            [
                Output("dataset-table", "rowData"),
                Output("dataset-table", "selectedRows"),
                Output("table-title", "children"),
                Output("selection-summary", "children"),
            ],
            [
                Input("scatter-selected-indices", "data"),
                Input("word-click-filter", "data"),
                Input("errors-only-switch", "value"),
                Input("error-cell", "data"),
                Input("dataset-table", "filterModel"),
                Input("dataset-table", "virtualRowData"),
            ],
        )
        def filter_table(selected_indices, word_filter, errors_only, error_cell, filter_model, virtual_row_data):
            # word_filter may be a list of words (multi-select in the scatter) or None.
            words = word_filter if isinstance(word_filter, list) else ([word_filter] if word_filter else [])
            effective_indices = _compose_indices(selected_indices, error_cell)
            if effective_indices is None:
                recs = full_records
            else:
                idx_set = set(effective_indices)
                recs = [r for r in full_records if r["_orig_idx"] in idx_set] or full_records
            if errors_only and has_gt:
                # True intersection — must not silently drop this filter when it empties the
                # selection, or the summary below (which still says "model errors only") would
                # describe a filter that was not actually applied.
                recs = [r for r in recs if str(r.get("prediction", "")) != str(r.get("ground_truth", ""))]
            if words:
                lowers = {w.lower() for w in words}
                # Rows containing ANY of the selected words as an exact token — matched against
                # explanation.token_strings, the same tokenization word_importance()/vocabulary()
                # rank on, so a bar labeled "happy" cannot pull in "unhappy"/"happier" via a raw
                # substring match on the sentence text.
                # True intersection, same reasoning as errors_only above — e.g. a word clicked in
                # Word Importance followed by an Error Analysis cell that contains none of that
                # word's rows must show 0 rows, not silently ignore the word and claim it in the
                # summary anyway.
                recs = [r for r in recs if lowers & {t.lower() for t in explanation.token_strings[r["_orig_idx"]]}]

            total = len(full_records)
            parts = []
            if error_cell:
                names = explanation.label_names or []
                pred_name = names[error_cell["pred"]] if error_cell["pred"] < len(names) else str(error_cell["pred"])
                true_name = names[error_cell["true"]] if error_cell["true"] < len(names) else str(error_cell["true"])
                parts.append(f"predicted {pred_name} · true {true_name}")
            if selected_indices is not None:
                parts.append(f"{len(selected_indices)} selected in embeddings")
            if words:
                parts.append("containing " + ", ".join(f'"{w}"' for w in words))
            if errors_only and has_gt:
                parts.append("model errors only")

            # AG Grid's own column filters (the funnel icon in each header) run entirely
            # client-side and never pass through the filtering above — fold them into the same
            # count here rather than reporting a second, disconnected number, so "Showing:" always
            # names the row count actually on screen. virtualRowData is the grid's rowData after
            # those inline filters are applied, so its length is that final on-screen count.
            grid_filtered = bool(filter_model)
            shown = len(virtual_row_data) if (grid_filtered and virtual_row_data is not None) else len(recs)
            if grid_filtered:
                names = ", ".join(column_header_by_field.get(f, f) for f in filter_model)
                parts.append(f"grid filter: {names}")

            if parts:
                summary = " · ".join(parts) + f" ({shown} of {total})"
                title = "Text Samples — filtered"
            else:
                summary = f"all {total} samples"
                title = "Text Samples — click a row to inspect"
            return recs, ([recs[0]] if recs else []), title, summary

        @self.app.callback(
            Output("grid-filter-clear-btn", "style"),
            Input("dataset-table", "filterModel"),
        )
        def toggle_grid_filter_clear_btn(filter_model):
            return {"fontSize": "0.8em"} if filter_model else {"display": "none"}

        @self.app.callback(
            Output("dataset-table", "filterModel"),
            Input("grid-filter-clear-btn", "n_clicks"),
            prevent_initial_call=True,
        )
        def clear_grid_filter(_n_clicks):
            return {}

        # ── Tab visibility: toggle display of always-mounted bodies ──────
        # Bodies stay in the DOM (see _tabbed_card); only their `display` flips so the
        # cross-panel callbacks above keep firing for panels on inactive tabs.
        for tabs_id, tab_ids in self._tab_groups.items():

            @self.app.callback(
                [Output(f"{tabs_id}-body-{tid}", "style") for tid in tab_ids],
                Input(tabs_id, "active_tab"),
            )
            def _toggle_tab_bodies(active, _tab_ids=tab_ids):
                return [(_VISIBLE if tid == active else _HIDDEN) for tid in _tab_ids]

        # (The selection-summary readout is written by filter_table, which knows the row count.)

        # ── Panel magnify: one clientside toggle for every `_tabbed_card` ─────────────────
        # A single MATCH callback handles all three panels (Dataset/Embeddings, the global
        # Word Importance group, Sentence/Waterfall) — no per-graph wiring needed, since it just
        # toggles a CSS class on the whole card (see `.fullscreen-overlay` in style.css) rather
        # than resizing any one plot: the card's own flex layout does the rest. Plotly figures
        # inside don't auto-track a CSS-only resize, so a manual `Plotly.Plots.resize` nudges
        # them once the browser has reflowed the class change.
        self.app.clientside_callback(
            """
            function(n_clicks) {
                if (!n_clicks) { return window.dash_clientside.no_update; }
                const expand = (n_clicks % 2 === 1);
                setTimeout(function() {
                    document.querySelectorAll('.js-plotly-plot').forEach(function(gd) {
                        if (window.Plotly) { window.Plotly.Plots.resize(gd); }
                    });
                }, 60);
                return expand ? 'fullscreen-overlay' : '';
            }
            """,
            Output({"type": "panel-card", "id": MATCH}, "className"),
            Input({"type": "panel-expand-btn", "id": MATCH}, "n_clicks"),
        )

        # ── Registered components (always-on core panels + capability-gated What-if Lab) ──
        stores = {
            "apply": _APPLY_STORE,
            "current": _CURRENT_STORE,
            "error_cell": "error-cell",
            "error_cell_clear": "error-cell-clear-btn",
            # The shell-owned filters a global panel has to honour, plus the clicked-word store and
            # its clear button and the published active class, handed over by id so a component
            # never hard-codes the shell's ids.
            "selection": "scatter-selected-indices",
            "selection_clear": "scatter-clear-btn",
            "errors_only": "errors-only-switch",
            "word_click": "word-click-filter",
            "word_click_clear": "word-filter-clear-btn",
            "active_class": "active-class-store",
        }
        for comp in self._components:
            comp.register_callbacks(self.app, self._explanation, self._engine, stores)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, port: int = 8050, debug: bool = False, host: str = "127.0.0.1") -> RunningApp | None:
        """Launch the Dash server.

        Parameters
        ----------
        port : int
            Port for the server.
        debug : bool
            Enable Dash debug mode (hot reload, error overlay). Debug mode uses Dash's own
            blocking dev server and reloader, so it cannot be handed back as a killable handle —
            stop it by interrupting the kernel/process. Defaults to ``False``.
        host : str
            Host to bind the server to.

        Returns
        -------
        RunningApp or None
            When ``debug`` is ``False`` (the default), the server runs on a background thread and
            this call returns immediately with a handle to it — call ``.kill()`` on it (or use it
            as a context manager) to stop the server. Returns ``None`` when ``debug=True``, since
            that path blocks instead.
        """
        if debug:
            self.app.run(port=port, debug=debug, host=host)
            return None
        running_app = run_in_background(self.app.server, host, port)
        logging.info(f"Your Shapash application run on {running_app.url}")
        logging.info("Use the method .kill() to stop your app.")
        return running_app
