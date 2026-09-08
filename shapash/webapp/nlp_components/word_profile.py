"""Word Profile panel: one chosen word, aggregated across the corpus, for every class at once.

The global Word Importance chart answers "which words matter for this class" — it fixes a class,
ranks the vocabulary by mean contribution, and shows the top K. Three questions it cannot answer
fall out of that shape, and this panel exists for them:

* **Any word, not just the top K.** A word you have a hypothesis about ("does *not* actually flip
  anything?") is usually nowhere near the top of a ranking, so it is simply not on the chart.
* **All classes at once.** In multi-class, a word's meaning to the model *is* the shape across
  classes — pulls toward ``anger``, pushes away from ``joy``. Reading that off the top-K chart
  means flipping the class selector k times and remembering k pictures.
* **Which aggregation.** The top-K chart is hard-wired to the mean, which is the right default and
  the wrong statistic for two common cases: a frequent mild word that moves the corpus more than
  any single strong one (use ``Sum``), and a word that pushes hard in *both* directions and
  therefore averages to ~0 while being highly influential (use ``Mean |·|`` and compare against the
  signed mean — a large gap between them *is* the context-dependence signal).

Then it drills back down: the ranked sample table shows where the aggregate came from, and each
row's *Inspect* makes that sample the current datapoint, so the Sentence/Waterfall panels show the
word in its actual context. This is the loop the aggregate is only useful as an entry point to —
"this word averages +0.3 toward anger" is a claim you should be able to spot-check in one click.

The panel honours the app's global sample selection (scatter box/lasso, confusion-matrix cell,
Model Errors switch) through the same :func:`~shapash.webapp.nlp_components.base.compose_selection`
the shell's own word-importance chart uses, which is what makes "how does this word behave *on the
errors*" a two-click question.

Data-only: ``requires`` is empty, so it mounts on a loaded snapshot with no live model. The Inspect
button costs no re-explanation — every listed sample is already in the compiled batch.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.explainer.nlp_explanation import (
    WORD_AGGREGATIONS,
    aggregate_word_contributions,
    rank_word_samples,
)
from shapash.plots.plot_word_importance import empty_word_figure
from shapash.plots.plot_word_profile import plot_word_profile
from shapash.style.style_utils import DEFAULT_NLP_THEME, NlpTheme
from shapash.webapp.nlp_components.base import WebappComponent, compose_selection, error_positions
from shapash.webapp.nlp_components.datapoint import pack_datapoint

_CARD_STYLE = {"height": "100%", "display": "flex", "flexDirection": "column", "overflowY": "auto"}
_INLINE = {"display": "flex", "alignItems": "center", "gap": "0.4rem"}

# Only the two mean forms. A sum over one word is that word's mean rescaled by its occurrence
# count — identical across classes, so identical bar shape — and carries information only when
# words of differing frequency are compared, which is the Word Importance ranking's job (its
# "Rank by: Total" mode). See aggregate_word_contributions for the derivation.
_OFFERED_AGGS = ("mean", "mean_abs")
_DEFAULT_AGG = "mean"
_SORT_OPTIONS: list[dcc.RadioItems.Options] = [
    {"label": " A→Z", "value": "alpha"},
    {"label": " Frequency", "value": "frequency"},
]
_DEFAULT_ORDER = "strongest"
_DEFAULT_LIMIT = 10
_TEXT_PREVIEW_CHARS = 140

_ORDER_OPTIONS: list[dcc.Dropdown.Options] = [
    {"label": "Strongest (either way)", "value": "strongest"},
    {"label": "Most positive", "value": "most"},
    {"label": "Most negative", "value": "least"},
]
_LIMIT_OPTIONS: list[dcc.Dropdown.Options] = [{"label": str(n), "value": n} for n in (5, 10, 20, 50)]


def _truncate(text: str) -> str:
    return text if len(text) <= _TEXT_PREVIEW_CHARS else text[: _TEXT_PREVIEW_CHARS - 1] + "…"


def _samples_table(ranked, explanation, component_id: str, show_truth: bool):
    """Ranked samples with their contribution, occurrence count, prediction and an Inspect button."""
    head_cells = [html.Th("Contribution"), html.Th("×", title="Occurrences of the word in this sample")]
    head_cells.append(html.Th("Predicted"))
    if show_truth:
        head_cells.append(html.Th("True"))
    head_cells.extend([html.Th("Text"), html.Th("")])

    y_pred, y_true = explanation.y_pred, explanation.y_true
    rows = []
    for i, record in enumerate(ranked.itertuples(index=False)):
        pos = int(record.sample)
        value = float(record.contribution)
        cells = [
            html.Td(
                f"{value:+.4f}",
                className="fw-bold " + ("text-primary" if value >= 0 else "text-danger"),
            ),
            html.Td(str(int(record.n_occurrences)), className="text-muted"),
            html.Td(str(y_pred.iloc[pos]) if y_pred is not None else "—", style={"fontSize": "0.85em"}),
        ]
        if show_truth:
            cells.append(html.Td(str(y_true.iloc[pos]), style={"fontSize": "0.85em"}))
        cells.append(html.Td(_truncate(str(explanation.texts.iloc[pos])), style={"fontSize": "0.85em"}))
        cells.append(
            html.Td(
                dbc.Button(
                    "Inspect",
                    id={"type": f"{component_id}-inspect", "index": i},
                    color="link",
                    size="sm",
                    className="p-0",
                    title="Show this sample's token contributions on the right",
                )
            )
        )
        rows.append(html.Tr(cells))
    return dbc.Table(
        [html.Thead(html.Tr(head_cells)), html.Tbody(rows)],
        bordered=False,
        hover=True,
        size="sm",
        className="mb-0",
    )


class WordProfileComponent(WebappComponent):
    """Per-class aggregated contributions of one selected word, with drill-down to its samples."""

    id = "word-profile"
    name = "Word Profile"
    scope = "global"
    # Data-only: reads the artifact and nothing else, so it also mounts on a loaded snapshot.
    requires = frozenset()

    def __init__(self, theme: NlpTheme = DEFAULT_NLP_THEME) -> None:
        self._theme = theme
        # Both option orderings, built once in layout() and swapped by a callback: the counts cost
        # a full pass over the corpus, and the sort toggle must not pay it on every click.
        self._options: dict[str, list] = {"alpha": [], "frequency": []}

    def layout(self, explanation, engine=None) -> html.Div:
        """Build the panel, seeded with the corpus's single most important word so it opens full."""
        counts = explanation.word_counts()
        # The count rides in the option *label* while the value stays the bare word, so it is
        # visible in either sort order (frequency is what tells you whether an aggregate is worth
        # trusting) and every existing consumer of the value is unaffected.
        by_frequency = [{"label": f"{w} ({n})", "value": w} for w, n in counts["n_occurrences"].items()]
        self._options = {
            "frequency": by_frequency,
            "alpha": sorted(by_frequency, key=lambda o: o["value"]),
        }
        word_options = self._options["alpha"]
        # Seed with the top word for class 0 rather than an empty panel: the first thing a user needs
        # is an example of what this view *is*, and "the single most important word" is the least
        # arbitrary pick available without asking them. The same frequency floor the Word Importance
        # panel defaults to, so the seed is never a word seen once.
        top = explanation.word_importance(label_idx=0, n_top=1, min_occurrences=2)
        default_word = str(top.index[0]) if len(top) else (word_options[0]["value"] if word_options else None)

        label_names = explanation.label_names or [str(i) for i in range(explanation.n_classes)]
        class_options: list[dcc.Dropdown.Options] = [{"label": n, "value": i} for i, n in enumerate(label_names)]
        agg_options: list[dcc.Dropdown.Options] = [
            {"label": WORD_AGGREGATIONS[key][0], "value": key} for key in _OFFERED_AGGS
        ]
        # With one output column there is nothing to rank *by* — the ranking class is forced to 0 and
        # the control is hidden rather than shown as a single-entry dropdown.
        single_class = explanation.n_classes == 1

        return html.Div(
            [
                html.Small(
                    "Pick any word in the corpus to see how it contributes to every class at once. "
                    "Compare the signed mean against Mean |·|: a large gap means the word cuts both "
                    "ways depending on context. Click a bar to rank the samples below by that class.",
                    className="text-muted d-block mb-2",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Word", className="small fw-bold mb-0"),
                                dcc.Dropdown(
                                    id=f"{self.id}-select",
                                    options=word_options,
                                    value=default_word,
                                    clearable=False,
                                    placeholder="Search a word…",
                                    style={"minWidth": "220px", "fontSize": "0.9em"},
                                ),
                            ],
                            style={**_INLINE, "flex": "1 1 240px"},
                        ),
                        html.Div(
                            dcc.RadioItems(
                                id=f"{self.id}-sort",
                                options=_SORT_OPTIONS,
                                value="alpha",
                                inline=True,
                                inputStyle={"marginRight": "4px"},
                                labelStyle={"marginRight": "10px"},
                                className="small",
                            ),
                            title="Order the word list alphabetically or by how often each word occurs",
                        ),
                        html.Div(
                            [
                                html.Label("Aggregate", className="small fw-bold mb-0"),
                                dcc.Dropdown(
                                    id=f"{self.id}-agg",
                                    options=agg_options,
                                    value=_DEFAULT_AGG,
                                    clearable=False,
                                    style={"minWidth": "130px", "fontSize": "0.9em"},
                                ),
                            ],
                            style=_INLINE,
                        ),
                    ],
                    className="d-flex flex-wrap align-items-center gap-3 mb-2",
                    style={"flex": "0 0 auto"},
                ),
                html.Small(id=f"{self.id}-caption", className="text-muted d-block mb-1"),
                dcc.Graph(
                    id=f"{self.id}-graph",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "280px", "flex": "0 0 auto"},
                ),
                html.Hr(className="my-2"),
                html.Div(
                    [
                        html.Span("Samples where it contributed", className="small fw-bold"),
                        html.Div(
                            [
                                html.Label("to", className="small text-muted mb-0"),
                                dcc.Dropdown(
                                    id=f"{self.id}-class",
                                    options=class_options,
                                    value=0,
                                    clearable=False,
                                    style={"minWidth": "130px", "fontSize": "0.9em"},
                                ),
                            ],
                            style={**_INLINE, "display": "none" if single_class else "flex"},
                        ),
                        dcc.Dropdown(
                            id=f"{self.id}-order",
                            options=_ORDER_OPTIONS,
                            value=_DEFAULT_ORDER,
                            clearable=False,
                            style={"minWidth": "180px", "fontSize": "0.9em"},
                        ),
                        html.Div(
                            [
                                html.Label("Show", className="small text-muted mb-0"),
                                dcc.Dropdown(
                                    id=f"{self.id}-limit",
                                    options=_LIMIT_OPTIONS,
                                    value=_DEFAULT_LIMIT,
                                    clearable=False,
                                    style={"minWidth": "80px", "fontSize": "0.9em"},
                                ),
                            ],
                            style=_INLINE,
                        ),
                    ],
                    className="d-flex flex-wrap align-items-center gap-3 mb-2",
                    style={"flex": "0 0 auto"},
                ),
                html.Div(id=f"{self.id}-results"),
                # Positional indices of the displayed rows, so an Inspect click resolves to a sample.
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the profile (word × aggregation × global selection) and each row's Inspect."""
        current_store = stores["current"]
        selection_store = stores["selection"]
        error_cell_store = stores["error_cell"]
        errors_only_switch = stores["errors_only"]
        word_click_store = stores["word_click"]
        label_names = explanation.label_names or [str(i) for i in range(explanation.n_classes)]
        show_truth = explanation.y_true is not None

        @app.callback(
            Output(f"{self.id}-graph", "figure"),
            Output(f"{self.id}-caption", "children"),
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(f"{self.id}-select", "value"),
            Input(f"{self.id}-agg", "value"),
            Input(f"{self.id}-class", "value"),
            Input(f"{self.id}-order", "value"),
            Input(f"{self.id}-limit", "value"),
            Input(selection_store, "data"),
            Input(error_cell_store, "data"),
            Input(errors_only_switch, "value"),
        )
        def update_profile(word, agg, class_idx, order, limit, selected_indices, error_cell, errors_only):
            if not word:
                return empty_word_figure("Pick a word to profile."), "", None, []

            cell_indices = error_cell.get("indices") if error_cell else None
            errors = error_positions(explanation) if errors_only else None
            indices = compose_selection(selected_indices, cell_indices, errors)
            scope = f" · scoped to {len(indices)} selected sample(s)" if indices is not None else ""

            occurrences = explanation.word_occurrences(word, sample_indices=indices)
            if occurrences.empty:
                where = "in the selected samples" if indices is not None else "in this corpus"
                return (
                    empty_word_figure(f"'{word}' does not occur {where}."),
                    f"'{word}' — no occurrences{scope}",
                    None,
                    [],
                )

            agg = agg or _DEFAULT_AGG
            agg_label, use_abs, reducer = WORD_AGGREGATIONS[agg]
            stats = aggregate_word_contributions(occurrences, agg=agg)

            spread = None
            if reducer == "mean":
                values = occurrences["contribution"].abs() if use_abs else occurrences["contribution"]
                # ddof=0: the spread of the occurrences actually seen, not an estimate of a population
                # — and it renders as 0 rather than NaN when a word occurs once.
                spread = values.groupby(occurrences["class_idx"]).std(ddof=0).sort_index()

            fig = plot_word_profile(
                stats,
                label_names=explanation.label_names,
                spread=spread,
                x_title=f"{agg_label} contribution",
                title="",  # the caption below carries the word and the counts
                width=None,
                height=None,
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
            )
            fig.layout.height = None  # let the CSS container height take over

            n_occ = len(occurrences) // max(explanation.n_classes, 1)
            n_samples = int(occurrences["sample"].nunique())
            total = len(indices) if indices is not None else len(explanation)
            share = f" ({n_samples / total:.0%})" if total else ""
            caption = (
                f"'{word}' — {agg_label} over {n_occ} occurrence(s) in {n_samples} of {total} sample(s){share}{scope}"
            )
            if reducer == "mean":
                caption += ". Error bars: ± std across occurrences."

            ranked = rank_word_samples(
                occurrences,
                class_idx=int(class_idx or 0),
                order=order or _DEFAULT_ORDER,
                n_top=int(limit or _DEFAULT_LIMIT),
            )
            table = html.Div(
                [
                    html.Small(
                        f"Contribution to {label_names[int(class_idx or 0)]}, summed within each sample.",
                        className="text-muted d-block mb-1",
                    ),
                    _samples_table(ranked, explanation, self.id, show_truth),
                ]
            )
            return fig, caption, table, [int(s) for s in ranked["sample"]]

        @app.callback(
            Output(f"{self.id}-select", "options"),
            Input(f"{self.id}-sort", "value"),
        )
        def reorder_word_list(sort):
            # Options only — the selected value is untouched, so reordering never moves the chart.
            return self._options.get(sort or "alpha", self._options["alpha"])

        # Clicking a class's bar ranks the samples below by that class. The two halves of the panel
        # otherwise drift apart: you read "pulls hard toward anger" off the top and then have to
        # find anger again in a separate dropdown before the table below is about the same thing.
        @app.callback(
            Output(f"{self.id}-class", "value"),
            Output(f"{self.id}-graph", "clickData"),
            Input(f"{self.id}-graph", "clickData"),
            prevent_initial_call=True,
        )
        def rank_by_clicked_class(click_data):
            if not click_data or not click_data.get("points"):
                raise PreventUpdate
            custom = click_data["points"][0].get("customdata")
            if not custom:
                raise PreventUpdate
            # Reset clickData so re-clicking the *same* bar counts as a change and re-fires; plotly
            # does not re-emit an unchanged value.
            return int(custom[0]), None

        # A word bar clicked in the Word Importance panel writes the shell's word-click store; follow
        # it so switching to this tab lands on the word the user just clicked. Only ever a *default*
        # — the dropdown stays freely editable, and this never fires on load (which would overwrite
        # the seeded top word with an empty filter).
        @app.callback(
            Output(f"{self.id}-select", "value"),
            Input(word_click_store, "data"),
            prevent_initial_call=True,
        )
        def follow_clicked_word(word_filter):
            # The store holds a bare string without a scatter and a list of words with one (the
            # scatter's multi-select is the source of truth there); take the last word either way.
            words = word_filter if isinstance(word_filter, list) else ([word_filter] if word_filter else [])
            if not words:
                raise PreventUpdate
            return str(words[-1])

        # Inspect: the sample is already in the compiled batch, so it packs straight from the stored
        # contributions — no explain_text round-trip. Mirrors the app shell's set_current_from_row.
        @app.callback(
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-inspect", "index": ALL}, "n_clicks"),
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
            base_values = explanation.base_values
            y_pred = explanation.y_pred
            return pack_datapoint(
                text=str(explanation.texts.iloc[pos]),
                orig_idx=pos,
                tokens=explanation.token_strings[pos],
                values=explanation.values[pos],
                base_values=(base_values[pos] if base_values is not None else None),
                label=(str(y_pred.iloc[pos]) if y_pred is not None else None),
            )
