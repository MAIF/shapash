"""Data-editor component: edit a sample's text and see a live prediction + token highlight.

Prefills from the selected table row (or an applied counterfactual), re-predicts and re-explains the
edited text on demand through the :class:`~shapash.explainer.interactive.InteractiveEngine`, and
renders the result with the existing pure ``plot_sentence_highlight`` renderer.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.webapp.nlp_components.base import CAP_PREDICT, WebappComponent
from shapash.webapp.nlp_components.datapoint import datapoint_from_contributions

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}


def _prob_bar(probs: dict[str, float]) -> go.Figure:
    """Horizontal bar of class probabilities, highest on top."""
    items = sorted(probs.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color="#1f77b4",
            text=[f"{v:.2f}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=6, b=6),
        height=max(120, 26 * len(labels)),
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


class DataEditorComponent(WebappComponent):
    """Editable text box with live re-prediction and token-contribution highlight."""

    id = "data-editor"
    name = "Data Editor"
    scope = "local"
    requires = frozenset({CAP_PREDICT})

    def layout(self, explanation, engine=None) -> html.Div:
        """Return the editor card: textarea, Predict button, class-probability bar.

        Token contributions for the edited text are shown by the shared lower-right panel (it reads the
        ``current-datapoint`` store this component writes on Predict), so they are not duplicated here.
        """
        return html.Div(
            [
                html.Small(
                    "Edit the text below and click Predict. "
                    "Token contributions appear in the Sentence panel on the right.",
                    className="text-muted d-block mb-2",
                ),
                dcc.Textarea(
                    id=f"{self.id}-input",
                    style={"width": "100%", "height": "90px", "fontSize": "0.95em"},
                ),
                dbc.Button("Predict", id=f"{self.id}-predict-btn", color="warning", size="sm", className="mt-2"),
                # Hidden until the first Predict, so an empty plot area is not shown on tab open.
                html.Div(
                    [
                        html.Hr(style={"margin": "12px 0 8px"}),
                        dcc.Graph(id=f"{self.id}-prob", config={"displayModeBar": False}, style={"height": "150px"}),
                    ],
                    id=f"{self.id}-result",
                    style={"display": "none"},
                ),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire prefill (row/apply → textarea) and Predict (textarea → prediction + current datapoint)."""
        apply_store = stores["apply"]
        current_store = stores["current"]

        @app.callback(
            Output(f"{self.id}-input", "value"),
            Input("dataset-table", "selectedRows"),
            Input(apply_store, "data"),
        )
        def prefill(selected_rows, applied_text):
            trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
            if apply_store in trigger and applied_text:
                return applied_text
            if selected_rows:
                return selected_rows[0].get("text", "")
            raise PreventUpdate

        # Predict publishes the edited text as the *current datapoint* (a scratch point with
        # ``orig_idx=None``), so the shared highlight/waterfall/counterfactual panels all update to
        # reflect it. The editor only renders the class-probability bar locally.
        @app.callback(
            Output(f"{self.id}-prob", "figure"),
            Output(f"{self.id}-result", "style"),
            Output(current_store, "data", allow_duplicate=True),
            Input(f"{self.id}-predict-btn", "n_clicks"),
            State(f"{self.id}-input", "value"),
            prevent_initial_call=True,
        )
        def predict(n_clicks, text):
            if not n_clicks or not text or not text.strip():
                raise PreventUpdate
            contributions, label, probs = engine.explain_text(text)
            return _prob_bar(probs), {"display": "block"}, datapoint_from_contributions(text, contributions, label)
