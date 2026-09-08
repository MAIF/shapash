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

from shapash.explainer.nlp_explanation import select_label_column
from shapash.webapp.nlp_components.base import WebappComponent, error_mask

_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


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

    def _word_contributions(self, explanation, word: str, label_idx: int) -> np.ndarray:
        """Per-sample sum of SHAP contributions for all tokens matching *word*."""
        n = explanation.n_samples
        result = np.zeros(n)
        word_lower = word.lower()
        for i in range(n):
            tokens = explanation.token_strings[i]
            vals = select_label_column(explanation.values[i], label_idx)
            for j, tok in enumerate(tokens):
                if tok.strip().lower() == word_lower:
                    result[i] += vals[j]
        return result

    @staticmethod
    def _emphasize_errors(
        idx_arr: np.ndarray,
        base_opacity: float,
        base_size: float,
        error_mask: np.ndarray | None,
    ) -> tuple[list[float] | float, list[float] | float]:
        """Per-point opacity/size that pops model errors and shadows everything else.

        Keeps the caller's coloring untouched — only opacity and marker size change —
        so points stay grouped/colored however the "Color by" dropdown already draws them.
        Returns scalars (the unmodified base values) when ``error_mask`` is unavailable.
        """
        if error_mask is None or len(idx_arr) == 0:
            return base_opacity, base_size
        is_error = error_mask[idx_arr]
        opacity = np.where(is_error, max(base_opacity, 0.9), 0.12).tolist()
        size = np.where(is_error, base_size + 3, max(base_size - 2, 3)).tolist()
        return opacity, size

    def _build_scatter_fig(
        self,
        explanation,
        color_by: str,
        words: list[str] | None = None,
        label_idx: int = 0,
        errors_only: bool = False,
    ) -> go.Figure:
        """2-D scatter coloured by prediction, ground-truth label, or word SHAP contribution.

        One trace per class for label-based coloring so the legend works correctly
        and Plotly's box/lasso select dims unselected points across all traces.
        Word-contribution mode uses a single diverging-colorscale trace.
        ``customdata`` always stores the original sample index. When ``errors_only`` is
        set, misclassified points are emphasized (larger, opaque) and the rest are
        shadowed (small, faint) without altering their color.
        """
        n = explanation.n_samples
        exp_texts = explanation.texts
        assert exp_texts is not None  # noqa: S101 - the webapp requires an already-compiled explainer
        texts_short = [(t[:120] + "…") if len(t) > 120 else t for t in exp_texts]
        xy = self._scatter_xy
        err_mask = error_mask(explanation) if errors_only else None

        if color_by == "word_contribution" and words:
            contributions = np.sum([self._word_contributions(explanation, w, label_idx) for w in words], axis=0)
            max_abs = float(np.abs(contributions).max()) or 1.0
            present_mask = np.where(contributions != 0.0)[0]
            absent_mask = np.where(contributions == 0.0)[0]
            colorbar_title = " + ".join(f'"{w}"' for w in words) if len(words) <= 3 else f"{len(words)} words"

            fig = go.Figure()
            # Both layers are ALWAYS added (even when a mask is empty) so the trace structure stays
            # constant across word additions/removals. With a stable `uirevision`, a changing WebGL
            # (Scattergl) trace count leaves ghost/"shadow" points from the previous render — keeping
            # exactly two traces avoids that.
            absent_opacity, absent_size = self._emphasize_errors(absent_mask, 0.35, 5, err_mask)
            present_opacity, present_size = self._emphasize_errors(present_mask, 0.9, 9, err_mask)

            # Gray context layer — absent points (no selected word contributes to them).
            fig.add_trace(
                go.Scattergl(
                    x=xy[absent_mask, 0].tolist(),
                    y=xy[absent_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(color="#b0b0b0", size=absent_size, opacity=absent_opacity),
                    customdata=absent_mask.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in absent_mask],
                    hovertemplate="%{text}<extra>absent</extra>",
                    showlegend=False,
                )
            )
            # Colored overlay — samples where at least one selected word contributes.
            fig.add_trace(
                go.Scattergl(
                    x=xy[present_mask, 0].tolist(),
                    y=xy[present_mask, 1].tolist(),
                    mode="markers",
                    marker=dict(
                        color=contributions[present_mask].tolist(),
                        colorscale="RdBu",
                        cmin=-max_abs,
                        cmax=max_abs,
                        size=present_size,
                        opacity=present_opacity,
                        colorbar=dict(
                            title=dict(text=colorbar_title, side="right"),
                            thickness=12,
                            tickformat=".2f",
                        ),
                    ),
                    customdata=present_mask.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in present_mask],
                    hovertemplate="<b>SHAP: %{marker.color:.3f}</b><br>%{text}<extra></extra>",
                    showlegend=False,
                )
            )
            fig.update_layout(
                dragmode="select",
                uirevision="scatter",
                xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
                yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
                plot_bgcolor="#f9f9f9",
                paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
                autosize=True,
                showlegend=False,
            )
            return fig

        if color_by == "ground_truth" and explanation.y_true is not None:
            labels = [str(label) for label in explanation.y_true.tolist()]
        elif explanation.y_pred is not None:
            labels = [str(label) for label in explanation.y_pred.tolist()]
        else:
            labels = [""] * n

        label_names = explanation.label_names or sorted(set(labels))

        fig = go.Figure()
        for i, name in enumerate(label_names):
            mask = [j for j, lbl in enumerate(labels) if lbl == name]
            if not mask:
                continue
            mask_arr = np.array(mask)
            opacity, size = self._emphasize_errors(mask_arr, 0.75, 7, err_mask)
            fig.add_trace(
                go.Scattergl(
                    x=xy[mask_arr, 0].tolist(),
                    y=xy[mask_arr, 1].tolist(),
                    mode="markers",
                    marker=dict(color=_PALETTE[i % len(_PALETTE)], size=size, opacity=opacity),
                    customdata=mask_arr.reshape(-1, 1).tolist(),
                    text=[texts_short[j] for j in mask],
                    hovertemplate=f"<b>{name}</b><br>%{{text}}<extra></extra>",
                    name=name,
                )
            )

        fig.update_layout(
            dragmode="select",
            uirevision="scatter",
            xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
            yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
            plot_bgcolor="#f9f9f9",
            paper_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            # No fixed height — the responsive Graph stretches it to fill the card.
            autosize=True,
            legend=dict(itemsizing="constant", orientation="v", title_text=""),
            showlegend=True,
        )
        return fig
