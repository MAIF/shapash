"""Scatter (Embeddings) panel: 2-D projection of the text embeddings, box/lasso-selectable.

Mounting is decided by the caller-supplied ``scatter_xy`` array, not by ``requires`` — a pre-computed
projection is arbitrary data handed to ``NlpWebApp``, not an explanation/engine capability, so
``NlpWebApp`` only ever constructs this component when it has one (see ``ScatterComponent.__init__``).

The "Word contribution" color mode is a **structural, not capability, dependency** on
:class:`~shapash.webapp.nlp_components.word_importance.WordImportanceComponent`: coloring by a word's
contribution needs a class index, and that control lives on the Word Importance panel. Rather than
reading that panel's DOM id directly (cross-component data flows only through ``stores`` — see
``nlp_components/base.py``), this component reads the shared ``stores["active_class"]`` store that
Word Importance publishes to, and the shell tells this component at construction time
(``offer_word_contribution``) whether to even offer the mode — omitted rather than left silently
non-functional when Word Importance isn't mounted.

The word list (``scatter-word-select``) syncs bidirectionally with the shared ``stores["word_click"]``
store: a Word Importance bar click updates it here too, and editing the dropdown here updates the
shared store back — same end-user behavior as before extraction, just no longer dependent on which of
the two panels happens to be mounted or the order callbacks fire in.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from dash import Input, Output, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.explainer.nlp_explanation import word_contributions_by_sample
from shapash.plots.plot_scatter import plot_scatter
from shapash.webapp.nlp_components.base import WebappComponent, error_mask


class ScatterComponent(WebappComponent):
    """2-D embeddings scatter, coloured by prediction, ground truth, or word SHAP contribution."""

    id = "scatter"
    name = "Embeddings"
    scope = "global"
    requires = frozenset()

    def __init__(self, scatter_xy: np.ndarray, offer_word_contribution: bool) -> None:
        self._scatter_xy = scatter_xy
        self._offer_word_contribution = offer_word_contribution

    def layout(self, explanation, engine=None) -> html.Div:
        """Build the color-by/word-select controls and the scatter graph itself."""
        word_options: list[dcc.Dropdown.Options] = [{"label": w, "value": w} for w in explanation.vocabulary()]
        color_options: list[dcc.Dropdown.Options] = [{"label": "Prediction", "value": "prediction"}]
        if explanation.y_true is not None:
            color_options.append({"label": "Ground Truth", "value": "ground_truth"})
        if self._offer_word_contribution:
            color_options.append({"label": "Word contribution", "value": "word_contribution"})

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H6("Sample Space", className="fw-bold mb-0"),
                            width="auto",
                            className="align-self-center",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="color-by",
                                options=color_options,
                                value="prediction",
                                clearable=False,
                                style={"width": "160px", "fontSize": "0.9em"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="scatter-word-select",
                                options=word_options,
                                value=[],
                                multi=True,
                                clearable=True,
                                placeholder="Select words…",
                                style={"display": "none", "minWidth": "180px", "fontSize": "0.9em"},
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                    ],
                    className="align-items-center mb-2",
                ),
                html.Small(
                    "Box/lasso or click to filter. Click a word bar to color by its SHAP contribution.",
                    className="text-muted d-block mb-2",
                ),
                dcc.Graph(
                    id="scatter-plot",
                    figure=self._build_scatter_fig(explanation, "prediction"),
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["autoScale2d", "resetScale2d"],
                        "responsive": True,
                    },
                    # Grow to fill the card's remaining height so the scatter
                    # matches the taller word-importance panel beside it.
                    style={"flex": "1 1 auto", "minHeight": "340px"},
                ),
            ],
            # Flex column so the graph above can stretch to the panel height.
            style={"display": "flex", "flexDirection": "column", "height": "100%"},
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire coloring, selection, and the bidirectional word-list sync with the shared store."""
        selection_store = stores["selection"]
        selection_clear_btn = stores["selection_clear"]
        errors_only_switch = stores["errors_only"]
        word_click_store = stores["word_click"]
        active_class_store = stores["active_class"]

        @app.callback(
            Output("scatter-plot", "figure"),
            [
                Input("color-by", "value"),
                Input("scatter-word-select", "value"),
                Input(active_class_store, "data"),
                Input(errors_only_switch, "value"),
            ],
        )
        def update_scatter_color(color_by, words, label_idx, errors_only):
            return self._build_scatter_fig(
                explanation,
                color_by or "prediction",
                words=words or [],
                label_idx=label_idx if isinstance(label_idx, int) else 0,
                errors_only=bool(errors_only),
            )

        @app.callback(
            Output("color-by", "value"),
            Input("scatter-word-select", "value"),
        )
        def sync_color_by(words):
            if not words:
                raise PreventUpdate
            return "word_contribution"

        # Bidirectional sync with the shared word-click store: a Word Importance bar click lands
        # here too, and editing/clearing the dropdown here updates the shared store back — neither
        # panel "wins", both stay in sync regardless of which one changed first.
        @app.callback(
            Output("scatter-word-select", "value"),
            Input(word_click_store, "data"),
        )
        def follow_word_click(word_filter):
            words = word_filter if isinstance(word_filter, list) else ([word_filter] if word_filter else [])
            return words

        @app.callback(
            Output(word_click_store, "data", allow_duplicate=True),
            Input("scatter-word-select", "value"),
            prevent_initial_call=True,
        )
        def word_filter_from_scatter(words):
            return words or None

        @app.callback(
            Output("scatter-word-select", "style"),
            Input("color-by", "value"),
        )
        def toggle_word_select(color_by):
            base = {"minWidth": "180px", "fontSize": "0.9em"}
            return base if color_by == "word_contribution" else {**base, "display": "none"}

        @app.callback(
            Output(selection_store, "data"),
            Input("scatter-plot", "selectedData"),
            Input("scatter-plot", "clickData"),
            Input(selection_clear_btn, "n_clicks"),
        )
        def update_scatter_selection(selected_data, click_data, _clear_clicks):
            trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
            if selection_clear_btn in trigger:
                return None
            if "clickData" in trigger:
                if not click_data or not click_data.get("points"):
                    return None
                return [int(click_data["points"][0]["customdata"][0])]
            # An empty selectedData here is almost always plotly re-emitting on a figure recolor
            # (color-by / word / errors-only toggle), NOT a user deselect — ignore it so the box
            # survives. Genuine clears go through the clear button or a point click above.
            if not selected_data or not selected_data.get("points"):
                raise PreventUpdate
            return [int(pt["customdata"][0]) for pt in selected_data["points"]]

        @app.callback(
            Output(selection_clear_btn, "style"),
            Input(selection_store, "data"),
        )
        def toggle_clear_button(selected_indices):
            visible = {"display": "inline", "fontSize": "0.8em"}
            hidden = {"display": "none", "fontSize": "0.8em"}
            return visible if selected_indices else hidden

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scatter_fig(
        self,
        explanation,
        color_by: str,
        words: list[str] | None = None,
        label_idx: int = 0,
        errors_only: bool = False,
    ) -> go.Figure:
        """Slice ``explanation`` for the requested coloring and hand it to :func:`plot_scatter`.

        ``color_by``/``words``/``label_idx`` decide *which* arrays to slice (a webapp-control
        concern); the actual figure construction is the same pure function notebook/script callers
        get via ``explanation.plot.scatter`` — see
        :meth:`~shapash.explainer.nlp_plotter.NlpPlotter.scatter`.
        """
        err_mask = error_mask(explanation) if errors_only else None

        if color_by == "word_contribution" and words:
            contributions = word_contributions_by_sample(explanation, words, label_idx)
            colorbar_title = " + ".join(f'"{w}"' for w in words) if len(words) <= 3 else f"{len(words)} words"
            return plot_scatter(
                self._scatter_xy,
                explanation.texts,
                contributions=contributions,
                colorbar_title=colorbar_title,
                error_mask=err_mask,
            )

        if color_by == "ground_truth" and explanation.y_true is not None:
            labels = [str(label) for label in explanation.y_true.tolist()]
        elif explanation.y_pred is not None:
            labels = [str(label) for label in explanation.y_pred.tolist()]
        else:
            labels = [""] * explanation.n_samples

        label_names = explanation.label_names or sorted(set(labels))
        return plot_scatter(
            self._scatter_xy,
            explanation.texts,
            labels=labels,
            label_names=label_names,
            error_mask=err_mask,
        )
