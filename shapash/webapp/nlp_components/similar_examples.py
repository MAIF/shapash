"""Similar-examples component: reference examples most like the selected/edited text.

Reads the shared ``current-datapoint`` store — so it serves a clicked dataset row, an edited sentence,
*and* an applied counterfactual through the same code path — and renders similar reference-corpus
examples (cosine similarity in the model's similarity layer). Two mutually exclusive retrieval modes,
picked with a radio toggle whose two options each carry their own numeric field inline (Top-K's count,
Threshold's cutoff — the inactive one fades out via its ``disabled`` prop): **Top-K** (a fixed neighbour
count, any score) and **Threshold** (every reference above a cutoff, ranked descending and capped at
``_THRESHOLD_DISPLAY_LIMIT`` for display). The modes are kept exclusive rather than combined, since
"top-k above a threshold" is ambiguous about which bound wins and can silently under-report how many
references really clear the threshold. Each neighbour shows its similarity, its label, whether that
label matches the current prediction, its text, and an *Inspect* button that (mirroring the
counterfactual panel) re-explains that example live and makes it the current datapoint, so its token
contributions render immediately in the Sentence/Waterfall panels. Between the filter controls and the
table, a summary line reports what fraction of the *displayed* neighbours share the current prediction's
label.

The lookup is one forward pass against a precomputed embedding bank, so it runs live on every
selection change. Gated by ``CAP_SIMILAR`` + ``CAP_PREDICT``: mounts only when the explainer holds a
reference corpus and a live, embedding-capable model (self-disables on a snapshot or a
prediction-only pipeline).
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.webapp.nlp_components.base import CAP_PREDICT, CAP_SIMILAR, WebappComponent
from shapash.webapp.nlp_components.datapoint import datapoint_from_contributions

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}
_DEFAULT_TOP_K = 5
_DEFAULT_THRESHOLD = 0.95
# A low threshold can match most of the corpus; cap what a live, per-keystroke panel renders.
_THRESHOLD_DISPLAY_LIMIT = 100


class SimilarExamplesComponent(WebappComponent):
    """Nearest reference examples for the current datapoint (representation similarity)."""

    id = "similar"
    name = "Similar Examples"
    scope = "local"
    # Needs predict too: the Inspect button re-explains a chosen example via engine.explain_text.
    requires = frozenset({CAP_SIMILAR, CAP_PREDICT})

    def _layer_name(self, engine) -> str | None:
        """The similarity layer the retriever compares in, for a transparency caption (or ``None``)."""
        retriever = getattr(engine, "_retriever", None)
        return getattr(retriever, "layer", None) if retriever is not None else None

    def layout(self, ctx) -> html.Div:
        """Return the panel: a top-k control plus a results area fed by the current datapoint."""
        engine = ctx.engine
        layer = self._layer_name(engine)
        caption = "Train examples most similar to the selected text in the model's decision space"
        if layer:
            caption += f" (layer: {layer})"
        caption += ". Click Inspect to explain one here."
        mode_options: list[dcc.RadioItems.Options] = [
            {
                "label": html.Span(
                    [
                        "Top-K ",
                        dcc.Input(
                            id=f"{self.id}-topk",
                            type="number",
                            min=1,
                            max=50,
                            step=1,
                            value=_DEFAULT_TOP_K,
                            style={"width": "70px"},
                        ),
                    ],
                    className="d-inline-flex align-items-center gap-1",
                ),
                "value": "topk",
            },
            {
                "label": html.Span(
                    [
                        "Threshold ",
                        dcc.Input(
                            id=f"{self.id}-threshold",
                            type="number",
                            min=0,
                            max=1,
                            step=0.001,
                            value=_DEFAULT_THRESHOLD,
                            disabled=True,
                            style={"width": "70px"},
                        ),
                    ],
                    className="d-inline-flex align-items-center gap-1",
                ),
                "value": "threshold",
            },
        ]
        return html.Div(
            [
                html.H6("Similar Examples", className="fw-bold mb-2"),
                html.Small(caption, className="text-muted d-block mb-2"),
                # Each radio option's label carries its own numeric input, so the button and the
                # field it governs sit on one line; a nested <input> inside a <label> still gets
                # its own clicks (typing in it does not also toggle the radio), so this is safe.
                dcc.RadioItems(
                    id=f"{self.id}-mode",
                    options=mode_options,
                    value="topk",
                    inline=True,
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "20px", "display": "inline-flex", "alignItems": "center"},
                    className="mb-2",
                ),
                dcc.Loading(html.Div(id=f"{self.id}-results")),
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, ctx, stores) -> None:
        """Recompute neighbours on selection/mode/filter change; wire per-row Inspect into the current store."""
        engine = ctx.engine
        current_store = stores["current"]

        @app.callback(
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(current_store, "data"),
            Input(f"{self.id}-mode", "value"),
            Input(f"{self.id}-topk", "value"),
            Input(f"{self.id}-threshold", "value"),
        )
        def update_similar(datapoint, mode, top_k, threshold):
            text = (datapoint or {}).get("text", "")
            if not text or not text.strip():
                raise PreventUpdate
            predicted = (datapoint or {}).get("label")
            if mode == "threshold":
                t = float(threshold) if threshold is not None else _DEFAULT_THRESHOLD
                neighbors, total = engine.find_similar_threshold(text, threshold=t, limit=_THRESHOLD_DISPLAY_LIMIT)
                if not neighbors:
                    return html.Div("No train examples above the threshold.", className="text-muted"), []
                shown_of = total if total > len(neighbors) else None
            else:
                k = int(top_k) if top_k else _DEFAULT_TOP_K
                neighbors = engine.find_similar(text, top_k=k)
                if not neighbors:
                    return html.Div("No train examples available.", className="text-muted"), []
                shown_of = None
            return _render_results(neighbors, predicted, self.id, shown_of=shown_of), [n.text for n in neighbors]

        @app.callback(
            Output(f"{self.id}-topk", "disabled"),
            Output(f"{self.id}-threshold", "disabled"),
            Input(f"{self.id}-mode", "value"),
        )
        def toggle_mode_inputs(mode):
            is_threshold = mode == "threshold"
            return is_threshold, not is_threshold

        # "Inspect" re-explains the chosen reference example and makes it the current datapoint, so its
        # token contributions render in the shared Sentence/Waterfall panels — the same flow the
        # counterfactual panel's Inspect uses (current-datapoint is a shared, allow_duplicate output).
        @app.callback(
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-apply", "index": ALL}, "n_clicks"),
            State(f"{self.id}-store", "data"),
            prevent_initial_call=True,
        )
        def inspect(n_clicks_list, neighbor_texts):
            if not any(n_clicks_list or []):
                raise PreventUpdate
            triggered = callback_context.triggered_id
            if not triggered:
                raise PreventUpdate
            index = triggered["index"]
            if not neighbor_texts or not (0 <= index < len(neighbor_texts)):
                raise PreventUpdate
            text = neighbor_texts[index]
            contributions, label, _probs = engine.explain_text(text)
            return datapoint_from_contributions(text, contributions, label)


def _render_results(neighbors, predicted_label, component_id, shown_of=None):
    """Precede the neighbours table with an optional cap note and a same-label match-rate summary.

    Both sit between the filter controls and the table, since they summarize *what was filtered*
    rather than a per-row detail. ``shown_of`` is the total count that cleared the threshold before
    capping to the displayed ``neighbors`` (``None`` in top-k mode, where the count is fixed by
    construction).
    """
    children = []
    if shown_of is not None:
        children.append(
            html.Small(
                f"Showing {len(neighbors)} of {shown_of} train examples above the threshold.",
                className="text-muted d-block mb-1",
            )
        )
    match_rate = _match_rate_caption(neighbors, predicted_label)
    if match_rate:
        children.append(html.Div(match_rate, className="text-muted d-block mb-2"))
    children.append(_neighbors_table(neighbors, predicted_label, component_id))
    return html.Div(children)


def _match_rate_caption(neighbors, predicted_label) -> str | None:
    """``m/n (p%) share the predicted label`` over the labelled, currently displayed neighbours."""
    if predicted_label is None:
        return None
    labelled = [n for n in neighbors if n.label is not None]
    if not labelled:
        return None
    matches = sum(1 for n in labelled if n.label == predicted_label)
    pct = 100.0 * matches / len(labelled)
    return f'{matches}/{len(labelled)} ({pct:.0f}%) share the predicted label "{predicted_label}".'


def _neighbors_table(neighbors, predicted_label, component_id):
    """Render neighbours as a table with a match flag and a per-row Inspect button."""
    show_label = any(n.label is not None for n in neighbors)
    head_cells = [html.Th("Cosine")]
    if show_label:
        head_cells.append(html.Th("Label"))
    head_cells.extend([html.Th("Text"), html.Th("")])
    header = html.Thead(html.Tr(head_cells))

    rows = []
    for i, n in enumerate(neighbors):
        cells = [html.Td(f"{n.score:.3f}")]
        if show_label:
            matches = predicted_label is not None and n.label == predicted_label
            cells.append(
                html.Td(
                    [
                        html.B(n.label if n.label is not None else "—"),
                        html.Span(" ✓", title="matches the current prediction") if matches else "",
                    ]
                )
            )
        cells.append(html.Td(n.text, style={"fontSize": "0.85em"}))
        cells.append(
            html.Td(
                dbc.Button(
                    "Inspect",
                    id={"type": f"{component_id}-apply", "index": i},
                    color="link",
                    size="sm",
                    className="p-0",
                    title="Explain this example here and show its token contributions on the right",
                )
            )
        )
        rows.append(html.Tr(cells))
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0")
