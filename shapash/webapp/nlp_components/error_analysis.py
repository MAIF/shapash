"""Error Analysis panel: confusion matrix + per-(predicted, true) word importance.

Clicking a confusion-matrix cell is a *global* selection driver, on a par with the embeddings-scatter
selection: it writes the shared ``error-cell`` store (``stores["error_cell"]``), which the app shell
composes with the scatter selection and the errors-only switch (see ``nlp_app._compose_selection``) to
filter the dataset table and the global Word Importance panel, and drives the two per-cell word charts
below the matrix in this panel.

The "clear cell" button lives outside this component's own layout — in the app shell's persistent
selection bar, alongside the other filter clears — so its id is handed in via
``stores["error_cell_clear"]`` rather than created here.

Gated on ``CAP_GROUND_TRUTH`` alone (not ``CAP_LABELS``): a confusion matrix only needs true/predicted
labels, not usable probabilities, so it must not be denied to a model with unusable/absent probabilities
the way Label Noise (confident learning) rightly is.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_word_importance import empty_word_figure, plot_word_importance
from shapash.style.style_utils import DEFAULT_NLP_THEME, NlpTheme
from shapash.webapp.nlp_components.base import CAP_GROUND_TRUTH, WebappComponent

_NORMALIZE_OPTIONS: list[dcc.RadioItems.Options] = [
    {"label": " Counts", "value": "count"},
    {"label": " Recall", "value": "recall"},
]


def _cell_from_click(click_data: dict | None, name_to_idx: dict[str, int]) -> tuple[int, int] | None:
    """Resolve a confusion-matrix cell click to ``(pred_idx, true_idx)``.

    Heatmap ``clickData`` does not reliably include ``customdata`` the way scatter traces do, so
    prefer it when present but fall back to mapping the predicted (``x``) and true (``y``) axis label
    names back to their class indices. Returns ``None`` for an empty or unrecognised click.
    """
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    custom = point.get("customdata")
    if custom is not None and len(custom) == 2:
        return int(custom[0]), int(custom[1])
    if point.get("x") in name_to_idx and point.get("y") in name_to_idx:
        return name_to_idx[point["x"]], name_to_idx[point["y"]]
    return None


class ErrorAnalysisComponent(WebappComponent):
    """Confusion matrix + per-cell word importance, for auditing where the model is wrong."""

    id = "error-analysis"
    name = "Error Analysis"
    scope = "global"
    requires = frozenset({CAP_GROUND_TRUTH})

    def __init__(self, theme: NlpTheme = DEFAULT_NLP_THEME) -> None:
        self._theme = theme
        # Populated by layout(); register_callbacks() runs after layout() has built the tab (see
        # NlpWebApp._build_layout / _register_callbacks), so these are always set by then.
        self._cm: np.ndarray | None = None
        self._cm_true_idx: np.ndarray | None = None
        self._cm_pred_idx: np.ndarray | None = None

    def layout(self, explanation, engine=None) -> html.Div:
        """Build the matrix + per-cell word-importance panel, caching the matrix for callbacks."""
        label_names = explanation.label_names or [str(i) for i in range(explanation.n_classes)]
        idx_of = explanation.label_to_idx
        y_true, y_pred = explanation.y_true, explanation.y_pred
        assert y_true is not None and y_pred is not None  # noqa: S101 - gated by CAP_GROUND_TRUTH
        # The per-sample class indices stay local: the cell-click handler needs them to recover
        # *which* samples sit in a clicked cell. The matrix itself is a plain derivation of the
        # artifact, so it is read from there rather than recomputed here.
        self._cm = explanation.confusion_matrix()
        self._cm_true_idx = np.array([idx_of.get(str(v), -1) for v in y_true.tolist()])
        self._cm_pred_idx = np.array([idx_of.get(str(v), -1) for v in y_pred.tolist()])

        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H6(
                                "Confusion Matrix — click a cell to inspect that error group",
                                className="fw-bold mb-0",
                            ),
                            width="auto",
                            className="align-self-center",
                        ),
                        dbc.Col(
                            dcc.RadioItems(
                                id="cm-normalize",
                                options=_NORMALIZE_OPTIONS,
                                value="count",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "12px"},
                            ),
                            width="auto",
                            className="align-self-center ms-auto",
                        ),
                    ],
                    className="align-items-center mb-2",
                    style={"flex": "0 0 auto"},
                ),
                dcc.Graph(
                    id="confusion-matrix-graph",
                    figure=plot_confusion_matrix(self._cm, label_names, title="", width=None, height=None),
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "360px", "flex": "0 0 auto"},
                ),
                html.Small(
                    id="error-cell-caption",
                    children="Click a cell to see the words behind those errors.",
                    className="text-muted d-block my-2",
                    style={"flex": "0 0 auto"},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="error-pred-importance",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "320px"},
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="error-true-importance",
                                config={"displayModeBar": False, "responsive": True},
                                style={"height": "320px"},
                            ),
                            width=6,
                        ),
                    ],
                    className="g-2",
                    style={"flex": "0 0 auto"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column", "overflowY": "auto"},
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the normalize toggle, cell click/clear, and the two per-cell word charts."""
        error_cell_store = stores["error_cell"]
        clear_btn_id = stores["error_cell_clear"]

        cm = self._cm
        pred_idx_arr = self._cm_pred_idx
        true_idx_arr = self._cm_true_idx
        assert cm is not None and pred_idx_arr is not None and true_idx_arr is not None  # noqa: S101 - layout() runs first
        label_names = explanation.label_names or [str(i) for i in range(cm.shape[0])]
        name_to_idx = {name: i for i, name in enumerate(label_names)}

        @app.callback(
            Output("confusion-matrix-graph", "figure"),
            Input("cm-normalize", "value"),
        )
        def update_confusion_matrix(normalize):
            fig = plot_confusion_matrix(
                cm,
                label_names,
                normalize="true" if normalize == "recall" else None,
                title="",  # the tab header already labels this panel
            )
            # Let the container height drive size (this panel is only half-column tall).
            fig.layout.width = None
            fig.layout.height = None
            return fig

        # Cell click / clear → the shared error-cell selection. A single owner (dispatching on the
        # trigger) keeps this the only writer, so no allow_duplicate coordination is needed.
        @app.callback(
            Output(error_cell_store, "data"),
            Output("confusion-matrix-graph", "clickData"),
            Input("confusion-matrix-graph", "clickData"),
            Input(clear_btn_id, "n_clicks"),
            prevent_initial_call=True,
        )
        def set_error_cell(click_data, _clear_clicks):
            trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
            if clear_btn_id in trigger:
                return None, None
            cell = _cell_from_click(click_data, name_to_idx)
            if cell is None:
                raise PreventUpdate
            pred_i, true_i = cell
            indices = np.where((pred_idx_arr == pred_i) & (true_idx_arr == true_i))[0].tolist()
            # Reset clickData so re-clicking the *same* cell registers as a change and re-fires.
            return {"pred": pred_i, "true": true_i, "indices": indices}, None

        @app.callback(
            Output(clear_btn_id, "style"),
            Input(error_cell_store, "data"),
        )
        def toggle_error_cell_clear_button(error_cell):
            visible = {"display": "inline", "fontSize": "0.8em"}
            hidden = {"display": "none", "fontSize": "0.8em"}
            return visible if error_cell else hidden

        # Per-cell word importance: words driving the (wrong) predicted class vs. the true class.
        @app.callback(
            Output("error-pred-importance", "figure"),
            Output("error-true-importance", "figure"),
            Output("error-cell-caption", "children"),
            Input(error_cell_store, "data"),
        )
        def update_error_word_charts(error_cell):
            if not error_cell:
                empty = empty_word_figure("Click a confusion-matrix cell to see the words behind those errors.")
                return empty, empty, "Click a cell to see the words behind those errors."
            pred_i, true_i, indices = error_cell["pred"], error_cell["true"], error_cell["indices"]
            pred_name, true_name = label_names[pred_i], label_names[true_i]
            if not indices:
                empty = empty_word_figure("No samples for this (predicted, true) pair.")
                return empty, empty, f"predicted {pred_name} · true {true_name}: 0 samples"
            wi_pred = explanation.word_importance(label_idx=pred_i, n_top=15, sample_indices=indices)
            wi_true = explanation.word_importance(label_idx=true_i, n_top=15, sample_indices=indices)
            # Counts within the clicked cell, not the corpus — the number that matters here is how
            # much evidence sits behind a word *in this error group*, which is often very little.
            counts = explanation.word_counts(sample_indices=indices)["n_occurrences"]
            # show_values=False: these two sit side by side in half a column each, where outside
            # value labels would cost horizontal room the word labels need more. The magnitudes
            # move to the hover instead, alongside the counts.
            fig_pred = plot_word_importance(
                wi_pred,
                title=f"Words toward predicted: {pred_name}",
                width=None,
                height=None,
                show_values=False,
                counts=counts,
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
            )
            fig_true = plot_word_importance(
                wi_true,
                title=f"Words toward true: {true_name}",
                width=None,
                height=None,
                show_values=False,
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
                counts=counts,
            )
            fig_pred.layout.height = None
            fig_true.layout.height = None
            caption = f"predicted {pred_name} · true {true_name}: {len(indices)} sample(s)"
            if pred_i == true_i:
                caption += " — correct predictions (diagonal)"
            elif len(indices) < 5:
                caption += " — few samples; interpret with caution"
            return fig_pred, fig_true, caption
