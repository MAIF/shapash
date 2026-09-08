"""Sentence-highlight component: inline token-contribution rendering for the current datapoint."""

from __future__ import annotations

from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.style.style_utils import DEFAULT_NLP_THEME, NlpTheme
from shapash.webapp.nlp_components.base import WebappComponent
from shapash.webapp.nlp_components.datapoint import unpack_datapoint


class SentenceHighlightComponent(WebappComponent):
    """Inline sentence highlight for the current datapoint, with its own class picker.

    Owns the ``local-class-selector`` dropdown, which the sibling :class:`WaterfallComponent` also
    reads (by literal id) since both render the same selected sentence and must agree on the class.
    """

    id = "sentence-highlight-panel"
    name = "Sentence"
    scope = "local"

    def __init__(self, default_class_idx: int = 0, theme: NlpTheme = DEFAULT_NLP_THEME) -> None:
        self._default_class_idx = default_class_idx
        self._theme = theme

    def layout(self, explanation, engine=None) -> html.Div:
        """Return the class picker + sentence-highlight placeholder div."""
        label_names = explanation.label_names or [str(i) for i in range(explanation.n_classes)]
        class_options: list[dcc.Dropdown.Options] = [{"label": name, "value": i} for i, name in enumerate(label_names)]
        return html.Div(
            [
                # One inline phrase instead of a title + a same-info dropdown: the class name would
                # otherwise appear twice (once written out, once as the dropdown's selected value).
                html.Div(
                    [
                        html.Span("Token Contributions for", className="fw-bold small me-2"),
                        dcc.Dropdown(
                            id="local-class-selector",
                            options=class_options,
                            value=self._default_class_idx,
                            clearable=False,
                            style={"width": "180px"},
                        ),
                    ],
                    className="d-flex align-items-center mb-2",
                ),
                # Min-height + vertical centring so short samples still give the
                # panel presence instead of collapsing to a thin strip.
                html.Div(
                    id="sentence-highlight",
                    style={
                        "minHeight": "150px",
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center",
                    },
                ),
            ],
            style={"height": "100%"},
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the predicted-class sync and the sentence-highlight render."""
        current_store = stores["current"]

        # Fires only on current-datapoint changes (row click, editor Predict, counterfactual Apply)
        # — not on manual dropdown edits — so a user's in-place class override survives until they
        # actually switch sentences.
        @app.callback(
            Output("local-class-selector", "value"),
            Input(current_store, "data"),
        )
        def sync_local_class_to_prediction(datapoint):
            if not datapoint or datapoint.get("label") is None:
                raise PreventUpdate
            label_idx = explanation.label_to_idx.get(str(datapoint["label"]))
            if label_idx is None:
                raise PreventUpdate
            return label_idx

        @app.callback(
            Output("sentence-highlight", "children"),
            [
                Input(current_store, "data"),
                Input("local-class-selector", "value"),
            ],
        )
        def update_sentence_highlight(datapoint, label_idx):
            if not datapoint or label_idx is None:
                raise PreventUpdate
            tokens, vals, base_value, _ = unpack_datapoint(datapoint, int(label_idx))
            return plot_sentence_highlight(
                tokens=tokens,
                values=vals,
                base_value=base_value,
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
            )
