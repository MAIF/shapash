"""Waterfall component: token contributions grouped into a threshold-gated "other" bar."""

from __future__ import annotations

from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_waterfall import plot_waterfall
from shapash.style.style_utils import DEFAULT_NLP_THEME, NlpTheme
from shapash.webapp.nlp_components.base import WebappComponent
from shapash.webapp.nlp_components.datapoint import unpack_datapoint


class WaterfallComponent(WebappComponent):
    """Waterfall chart for the current datapoint, sharing the Sentence panel's class picker.

    Reads ``local-class-selector`` (owned by :class:`~shapash.webapp.nlp_components.sentence_highlight.SentenceHighlightComponent`)
    by literal id — the same cross-component reference style already used by
    :class:`~shapash.webapp.nlp_components.data_editor.DataEditorComponent` for ``dataset-table``.
    """

    id = "waterfall-panel"
    name = "Waterfall"
    scope = "local"

    def __init__(self, theme: NlpTheme = DEFAULT_NLP_THEME) -> None:
        self._theme = theme

    def layout(self, explanation, engine=None) -> html.Div:
        """Return the grouping-threshold slider + waterfall graph."""
        return html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Group tokens below (% of max contribution)",
                            className="small fw-bold mb-1 d-block",
                        ),
                        dcc.Slider(
                            id="waterfall-threshold",
                            min=0,
                            max=50,
                            step=1,
                            value=10,
                            marks={0: "0%", 10: "10%", 25: "25%", 50: "50%"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    className="mb-2",
                ),
                dcc.Graph(id="waterfall-graph", config={"displayModeBar": False}),
            ],
            style={"height": "100%"},
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the waterfall figure to the current datapoint, class picker, and threshold."""
        current_store = stores["current"]

        @app.callback(
            Output("waterfall-graph", "figure"),
            [
                Input(current_store, "data"),
                Input("local-class-selector", "value"),
                Input("waterfall-threshold", "value"),
            ],
        )
        def update_waterfall(datapoint, label_idx, threshold_pct):
            if not datapoint or label_idx is None:
                raise PreventUpdate
            label_idx = int(label_idx)
            tokens, vals, base_value, _ = unpack_datapoint(datapoint, label_idx)
            label_name = (explanation.label_names or [])[label_idx] if explanation.label_names else str(label_idx)
            min_pct = (threshold_pct if threshold_pct is not None else 10) / 100.0
            return plot_waterfall(
                tokens=tokens,
                values=vals,
                base_value=base_value,
                min_pct=min_pct,
                title=f"Token contributions — {label_name}",
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
            )
