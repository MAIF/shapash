"""Label-noise panel: which ground-truth labels are probably wrong, and where the noise concentrates.

Error analysis stops at "the model got this one wrong"; this panel asks the next question — whether
the *label* is wrong instead. It runs confident learning
(:mod:`shapash.compute.diagnostics.label_noise`) over the probabilities and ground truth already
held from ``compile()``, and shows two views of the answer: an estimated noise matrix for the
corpus-level read (which class pairs are contaminated) and a ranked table for row-level triage.

Confident learning alone cannot answer the question the panel is named for. It flags rows where the
model confidently disagrees with the label, and a genuine label error looks exactly like a
confidently-wrong model. On the emotion demo every flagged row is also a model error, by
construction — so read on its own, the ranking silently invites you to "fix" labels the model simply
got wrong.

The **Corpus check** column is what separates the two. It is a bag-of-words classifier fit on the
reference corpus (:mod:`shapash.compute.diagnostics.label_probe`) — the one signal in this panel that
does not come from the audited model — asked how much probability it puts on the label the row
already carries. High, and the corpus backs the label: suspect the model. Low, and both models agree
the label is off.

That column replaced an earlier one showing the labels of the row's nearest neighbours. Retrieval
ranks in the model's decision space, so those neighbours restated the prediction 98.6% of the time
and backed the *wrong* prediction on 87% of model errors — it read as confirmation while carrying no
information. Representation neighbours are still one click away and honestly framed in the Similar
Examples panel, which follows the current datapoint. See implementation-log C.27.

*Inspect* makes a flagged row the current datapoint so its token contributions render in the shared
Sentence/Waterfall panels — the words that pushed the model away from the given label. Unlike the
counterfactual and similar-example panels, the sample is already compiled, so this costs no
re-explanation.

Gated by ``CAP_LABELS`` alone: the underlying method needs no model, so in principle this panel
should also work on a loaded snapshot (no live engine). **Temporarily not the case**: since the
``fit``/``explain`` split moved ``detect_label_noise`` onto the live ``NlpExplainer`` (it also
attaches the optional label-probe corroboration, which is fit-time engine state), a snapshot loaded
via ``NlpExplanation.load()`` has no engine to call it on. The Detect button therefore shows a
message instead of running when ``engine`` is ``None`` — see :meth:`LabelNoiseComponent.register_callbacks`.
Restoring snapshot support needs ``detect_label_noise``'s core computation split out as a pure
function over the artifact (data-only), with the probe corroboration staying engine-only; deferred
to a later retrofit.
The Corpus check column self-disables when no labelled reference corpus is bound.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_noise_matrix import plot_noise_matrix
from shapash.webapp.nlp_components.base import CAP_LABELS, WebappComponent
from shapash.webapp.nlp_components.datapoint import pack_datapoint

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}
_INLINE = {"display": "flex", "alignItems": "center", "gap": "0.4rem"}

_DEFAULT_TOP_N = 50
_DEFAULT_SCORE = "self_confidence"
_SCORE_OPTIONS: list[dcc.Dropdown.Options] = [
    {"label": "Self-confidence", "value": _DEFAULT_SCORE},
    {"label": "Normalized margin", "value": "normalized_margin"},
]


class LabelNoiseComponent(WebappComponent):
    """Ranked probably-mislabelled samples plus an estimated noise matrix (confident learning)."""

    id = "label-noise"
    name = "Label Noise"
    scope = "global"
    # Data-only: no engine capability, so this also mounts on a snapshot explainer.
    requires = frozenset({CAP_LABELS})

    def layout(self, explanation, engine=None) -> html.Div:
        """Return the panel: detection controls over an initially empty results area."""
        can_probe = engine is not None and engine.can_probe_labels()
        caption = (
            "Confident learning over the model's probabilities and the ground-truth labels. "
            "Assumes those probabilities are out-of-sample — on a model's own training split "
            "it under-reports."
        )
        caption += (
            " Every flagged row is one the model disagrees with, so check the Corpus column: a high "
            "score there means an independent classifier backs the existing label, and the model is "
            "the more likely culprit."
            if can_probe
            else " No labelled reference corpus is bound, so flagged rows carry no independent "
            "cross-check — a row may be a label error or simply a model error."
        )
        return html.Div(
            [
                html.H6("Label-Noise Detection", className="fw-bold mb-2"),
                html.Small(caption, className="text-muted d-block mb-2"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Top-N", className="small fw-bold mb-0"),
                                dcc.Input(
                                    id=f"{self.id}-top-n",
                                    type="number",
                                    min=1,
                                    max=500,
                                    step=1,
                                    value=_DEFAULT_TOP_N,
                                    style={"width": "80px"},
                                ),
                            ],
                            style=_INLINE,
                        ),
                        html.Div(
                            [
                                html.Label("Rank by", className="small fw-bold mb-0"),
                                dcc.Dropdown(
                                    id=f"{self.id}-score",
                                    options=_SCORE_OPTIONS,
                                    value=_DEFAULT_SCORE,
                                    clearable=False,
                                    style={"width": "190px"},
                                ),
                            ],
                            style=_INLINE,
                        ),
                        dbc.Button("Detect", id=f"{self.id}-detect-btn", color="warning", size="sm"),
                    ],
                    className="d-flex flex-wrap align-items-center gap-3 mb-2",
                ),
                dcc.Loading(html.Div(id=f"{self.id}-results")),
                # Positional indices of the displayed rows, so an Inspect click resolves to a sample.
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Run detection on demand; wire each row's Inspect into the shared current datapoint."""
        current_store = stores["current"]

        @app.callback(
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(f"{self.id}-detect-btn", "n_clicks"),
            State(f"{self.id}-top-n", "value"),
            State(f"{self.id}-score", "value"),
        )
        def detect(n_clicks, top_n, score):
            if not n_clicks:
                raise PreventUpdate
            if engine is None:
                # See the module docstring: detect_label_noise lives on the live engine for now,
                # so a loaded snapshot (no engine) cannot run it yet.
                return html.Div(
                    "Label-noise detection needs a live explainer — unavailable on a loaded snapshot.",
                    className="text-muted",
                ), []
            report = engine.detect_label_noise(
                explanation,
                top_n=int(top_n) if top_n else _DEFAULT_TOP_N,
                score=score or _DEFAULT_SCORE,
            )
            if not report.issues:
                return html.Div(
                    "No label issues detected — every label is consistent with the model's confident predictions.",
                    className="text-muted",
                ), []
            return _report_view(report, self.id), [issue.index for issue in report.issues]

        # Inspect: the sample is already in the compiled batch, so it packs straight from the stored
        # contributions — no explain_text round-trip, unlike the counterfactual/similar panels whose
        # texts are novel. Mirrors the app shell's set_current_from_row.
        @app.callback(
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-apply", "index": ALL}, "n_clicks"),
            State(f"{self.id}-store", "data"),
            prevent_initial_call=True,
        )
        def inspect(n_clicks_list, sample_indices):
            if not any(n_clicks_list or []):
                raise PreventUpdate
            triggered = callback_context.triggered_id
            if not triggered:
                raise PreventUpdate
            row = triggered["index"]
            if not sample_indices or not (0 <= row < len(sample_indices)):
                raise PreventUpdate
            pos = int(sample_indices[row])
            texts = explanation.texts
            if texts is None or not (0 <= pos < len(texts)):
                raise PreventUpdate
            base_values = explanation.base_values
            y_pred = explanation.y_pred
            return pack_datapoint(
                text=str(texts.iloc[pos]),
                orig_idx=pos,
                tokens=explanation.token_strings[pos],
                values=explanation.values[pos],
                base_values=(base_values[pos] if base_values is not None else None),
                label=(str(y_pred.iloc[pos]) if y_pred is not None else None),
            )


def _report_view(report, component_id):
    """Assemble the summary line, the noise-matrix heatmap and the ranked issue table."""
    rate = report.noise_rate
    summary = (
        f"Estimated {rate:.1%} of {report.n_samples} labels are wrong "
        f"({report.n_issues} flagged, showing {len(report.issues)})."
    )
    fig = plot_noise_matrix(report.noise_matrix, report.label_names, title="")
    # Let the container height drive size — this panel is only half a column tall.
    fig.layout.width = None
    fig.layout.height = None
    return html.Div(
        [
            html.Div(summary, className="small fw-bold mb-1"),
            html.Small(
                "Rows are the given label, columns the class the model points to instead. "
                "The diagonal (correctly labelled) is blanked so the off-diagonal noise stays visible. "
                "⚠ in the Corpus column marks rows an independent classifier says are model errors, "
                "not label errors.",
                className="text-muted d-block mb-2",
            ),
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "320px"}),
            _issues_table(report.issues, component_id),
        ]
    )


def _issues_table(issues, component_id):
    """Render the ranked issues, with the Corpus column only when the probe produced verdicts."""
    show_probe = any(issue.probe is not None for issue in issues)
    head_cells = [html.Th("Score"), html.Th("Given"), html.Th("Probably")]
    if show_probe:
        head_cells.append(html.Th("Corpus", title="Independent classifier's probability for the given label"))
    head_cells.extend([html.Th("Text"), html.Th("")])
    header = html.Thead(html.Tr(head_cells))

    rows = []
    for i, issue in enumerate(issues):
        cells = [
            html.Td(f"{issue.score:.3f}"),
            html.Td([html.B(issue.given_label), html.Span(f" {issue.given_prob:.2f}", className="text-muted")]),
            html.Td([html.B(issue.suggested_label), html.Span(f" {issue.suggested_prob:.2f}", className="text-muted")]),
        ]
        if show_probe:
            cells.append(html.Td(_probe_summary(issue), style={"fontSize": "0.85em"}))
        cells.append(html.Td(issue.text, style={"fontSize": "0.85em"}))
        cells.append(
            html.Td(
                dbc.Button(
                    "Inspect",
                    id={"type": f"{component_id}-apply", "index": i},
                    color="link",
                    size="sm",
                    className="p-0",
                    title="Show this sample's token contributions on the right",
                )
            )
        )
        rows.append(html.Tr(cells))
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0")


def _probe_summary(issue):
    """The independent probe's probability for the *given* label, flagged when it backs it.

    ``⚠ 0.83`` means a classifier that never saw the audited model puts 0.83 on the label already
    on the row — the corpus is with the label, so the model is the likelier culprit and this is not
    a row to relabel. ``0.16 → love`` means the probe rejects the given label too, which is the
    corroborated label error worth acting on.
    """
    probe = issue.probe
    if probe is None:
        return "—"
    if probe.backs_given:
        return html.Span(
            f"⚠ {probe.given_prob:.2f}",
            className="text-warning fw-bold",
            title=(
                f"An independent classifier backs '{issue.given_label}' — the model is more likely "
                "wrong here than the label."
            ),
        )
    return html.Span(
        f"{probe.given_prob:.2f} → {probe.top_label}",
        className="text-muted",
        title=f"The independent classifier also rejects '{issue.given_label}'; it would pick '{probe.top_label}'.",
    )
