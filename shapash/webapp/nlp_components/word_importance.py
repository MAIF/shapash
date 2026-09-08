"""Word Importance panel: mean/total SHAP contribution per unique word, for one class or all of them.

The corpus-level counterpart to the Sentence Highlight panel — "which words matter" instead of "why
this one prediction". Honours the app's global sample selection (scatter box/lasso, confusion-matrix
cell, Model Errors switch) through the shared ``stores["selection"]``/``stores["error_cell"]``/
``stores["errors_only"]`` stores, exactly as :class:`~shapash.webapp.nlp_components.word_profile.WordProfileComponent`
does — both call the same :func:`~shapash.webapp.nlp_components.base.compose_selection`.

A clicked bar is this panel's one cross-panel *output*: it always writes the shell's shared
``stores["word_click"]`` store, regardless of whether an embeddings scatter is also mounted — the
scatter, when present, is what reads/mirrors that store rather than the other way around (see
:mod:`~shapash.webapp.nlp_components.scatter`). This panel also publishes its class selector's value
to ``stores["active_class"]``, which is how the scatter's "Word contribution" color mode gets a class
index without reading this panel's DOM id directly.

Data-only: ``requires`` is empty, so it mounts on a loaded snapshot with no live model, the same as
``WordProfileComponent``.
"""

from __future__ import annotations

import dash
from dash import Input, Output, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.plots.plot_word_importance import empty_word_figure, plot_word_importance, word_importance_axis_title
from shapash.style.style_utils import DEFAULT_NLP_THEME, NlpTheme
from shapash.webapp.nlp_components.base import WebappComponent, compose_selection, error_positions

# The frequency floor the global word ranking starts at. 1 (no filter) is the status quo and is
# actively misleading: on the emotion demo 58% of the vocabulary occurs exactly once, so an
# unfiltered |mean| ranking is a list of single attributions. 2 is the principled minimum — a mean
# over one observation is not a mean — and is the least surprising default; 3 is where the chart
# visibly settles into repeated, stable words.
_DEFAULT_MIN_OCCURRENCES = 2

# Bounds for the Top-K box. The browser enforces these on the spinner arrows but not on typed
# input, so the callback clamps rather than trusting them.
_MIN_TOPK, _MAX_TOPK, _DEFAULT_TOPK = 1, 50, 20
# Idle delay (seconds) before a number box commits its value. ``debounce=True`` would commit only on
# Enter or blur, which leaves the spinner arrows apparently dead: clicking one keeps focus in the
# box, so nothing reaches the server until the user clicks away. A numeric debounce commits after a
# pause instead, so arrows respond while a typed "20" still does not recompute at "2".
_INPUT_DEBOUNCE_S = 0.4

# Class-dropdown value meaning "collapse the ranking across every class" — a string so it can never
# collide with a real (integer) class index.
_ALL_CLASSES = "all"

# Shared by the layout and the callback that greys the sign options out under "All classes", so the
# two cannot drift apart.
_SIGN_FILTER_OPTIONS: list[dcc.RadioItems.Options] = [
    {"label": " All", "value": "all"},
    {"label": " Positive", "value": "positive"},
    {"label": " Negative", "value": "negative"},
]


class WordImportanceComponent(WebappComponent):
    """Global word-importance ranking, with class/top-K/sign/exclude controls."""

    id = "word-importance"
    name = "Word Importance"
    scope = "global"
    requires = frozenset()

    def __init__(self, theme: NlpTheme = DEFAULT_NLP_THEME) -> None:
        self._theme = theme

    def layout(self, explanation, engine=None) -> html.Div:
        """Build the controls + graph. DOM ids are the pre-extraction literals (no ``self.id`` prefix)
        so existing layout tests keep matching — these ids were never shared with any other panel.
        """
        label_names = explanation.label_names or [str(i) for i in range(explanation.n_classes)]
        word_options: list[dcc.Dropdown.Options] = [{"label": w, "value": w} for w in explanation.vocabulary()]

        # "All classes" first: it is the overview a reader without a class in mind wants, and it is
        # the only entry whose value is not an index (see _ALL_CLASSES).
        global_class_options: list[dcc.Dropdown.Options] = [{"label": "All classes", "value": _ALL_CLASSES}]
        global_class_options += [{"label": name, "value": i} for i, name in enumerate(label_names)]
        # Ordering only: both statistics are signed and are ranked on |value|, so the sign filter
        # and the bars' red/blue colouring mean the same thing in either mode.
        rank_by_options: list[dcc.Dropdown.Options] = [
            {"label": "Mean |·|", "value": "mean"},
            {"label": "Total |·|", "value": "sum"},
        ]

        return html.Div(
            [
                # One flex row split into two clusters — "what to rank" (left) and "what to keep"
                # (right) — rather than six controls in an unbroken run: grouping reads faster, and
                # a Bootstrap 50/50 column split was tried first but broke (the left cluster's own
                # natural width exceeds half the panel at ordinary sizes, so the fixed-width column
                # either overlapped the right cluster or wrapped mid-cluster). space-between instead
                # pushes the right cluster as far right as the two clusters' actual widths allow —
                # around the midpoint in practice, and never overlapping or force-wrapping either
                # cluster — with flex-wrap kept as a fallback for genuinely narrow panels.
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Class", className="fw-bold small mb-0"),
                                        dcc.Dropdown(
                                            id="global-class-selector",
                                            options=global_class_options,
                                            value=0,
                                            clearable=False,
                                            style={"width": "140px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Rank by",
                                            className="fw-bold small mb-0",
                                            title=(
                                                "Mean: the average pull wherever the word "
                                                "appears. Total: its whole pull on the "
                                                "corpus, so frequent words outrank rare "
                                                "strong ones. The two disagree often."
                                            ),
                                        ),
                                        dcc.Dropdown(
                                            id="rank-by",
                                            options=rank_by_options,
                                            value="mean",
                                            clearable=False,
                                            style={"width": "95px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Min occur.",
                                            className="fw-bold small mb-0",
                                            title=(
                                                "Hide words seen fewer than this many times "
                                                "in the current selection. A mean over one "
                                                "occurrence is not a mean, and most of a "
                                                "corpus is words seen once."
                                            ),
                                        ),
                                        dcc.Input(
                                            id="min-occurrences",
                                            type="number",
                                            min=1,
                                            step=1,
                                            value=_DEFAULT_MIN_OCCURRENCES,
                                            debounce=_INPUT_DEBOUNCE_S,
                                            className="form-control form-control-sm",
                                            style={"width": "80px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Top-K words",
                                            className="fw-bold small mb-0",
                                            title=f"How many words to chart, {_MIN_TOPK}–{_MAX_TOPK}",
                                        ),
                                        dcc.Input(
                                            id="topk-input",
                                            type="number",
                                            min=_MIN_TOPK,
                                            max=_MAX_TOPK,
                                            step=1,
                                            value=_DEFAULT_TOPK,
                                            debounce=_INPUT_DEBOUNCE_S,
                                            className="form-control form-control-sm",
                                            style={"width": "90px"},
                                        ),
                                    ]
                                ),
                            ],
                            className="d-flex align-items-start flex-wrap gap-2",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Contributions", className="fw-bold small mb-0"),
                                        dcc.RadioItems(
                                            id="sign-filter",
                                            options=_SIGN_FILTER_OPTIONS,
                                            value="all",
                                            inline=True,
                                            inputStyle={"marginRight": "4px"},
                                            labelStyle={"marginRight": "12px"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Label(
                                            "Exclude words",
                                            className="fw-bold small mb-0",
                                            style={"whiteSpace": "nowrap"},
                                        ),
                                        dcc.Dropdown(
                                            id="word-filter",
                                            options=word_options,
                                            value=[],
                                            multi=True,
                                            placeholder="Exclude…",
                                            style={"fontSize": "0.85em"},
                                        ),
                                    ],
                                    style={"flex": "1 1 auto", "minWidth": "130px"},
                                ),
                            ],
                            className="d-flex align-items-start flex-wrap gap-2",
                        ),
                    ],
                    className="d-flex align-items-start flex-wrap mb-1",
                    style={"flex": "0 0 auto", "justifyContent": "space-between", "gap": "0.5rem"},
                ),
                # The graph keeps the height plot_word_importance computed for it (30px per word)
                # and this wrapper scrolls. Letting the chart flex to the panel instead — which is
                # what it used to do — gives 50 words 8px each, at which point plotly drops the
                # word labels and the bars become unidentifiable.
                # "responsive" must stay off here: it makes plotly track the *container's* current
                # size instead of the figure's own layout.height, which silently defeats the fixed
                # per-row height above and squeezes every word back into whatever space the flex
                # panel happens to have — the exact illegible-labels bug this height is built to
                # avoid. Width still adapts once, at mount, from the container.
                html.Div(
                    dcc.Graph(
                        id="global-importance-graph",
                        config={"displayModeBar": False},
                        style={"width": "100%"},
                    ),
                    style={"flex": "1 1 auto", "minHeight": "0", "overflowY": "auto"},
                ),
            ],
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the ranking, the sign-filter gate, and the bar-click → shared word/class stores."""
        selection_store = stores["selection"]
        error_cell_store = stores["error_cell"]
        errors_only_switch = stores["errors_only"]
        word_click_store = stores["word_click"]
        word_click_clear_btn = stores["word_click_clear"]
        active_class_store = stores["active_class"]

        @app.callback(
            Output("global-importance-graph", "figure"),
            [
                Input("global-class-selector", "value"),
                Input("topk-input", "value"),
                Input("sign-filter", "value"),
                Input("word-filter", "value"),
                Input("rank-by", "value"),
                Input("min-occurrences", "value"),
                Input(selection_store, "data"),
                Input(error_cell_store, "data"),
                Input(errors_only_switch, "value"),
            ],
        )
        def update_global_importance(
            label_idx,
            topk,
            sign_filter,
            exclude_words_list,
            rank_by,
            min_occurrences,
            selected_indices,
            error_cell,
            errors_only,
        ):
            if label_idx is None:
                raise PreventUpdate
            across_classes = label_idx == _ALL_CLASSES
            rank_by = rank_by or "mean"
            # Typed input bypasses the box's own min/max, and a cleared box arrives as None.
            n_top = max(_MIN_TOPK, min(_MAX_TOPK, int(topk))) if topk is not None else _DEFAULT_TOPK
            # A cleared number input arrives as None; 1 is "no floor", which is what an empty box
            # should mean rather than reverting to the default the user just deleted.
            floor = max(1, int(min_occurrences)) if min_occurrences else 1
            cell_indices = error_cell.get("indices") if error_cell else None
            errors = error_positions(explanation) if (errors_only and error_positions(explanation)) else None
            effective_indices = compose_selection(selected_indices, cell_indices, errors)
            # Across classes the bars are magnitudes (max over classes of |statistic|), so a sign
            # filter has nothing left to select on: it is greyed out in that mode and forced to
            # "all" here as well, so a value left over from a single-class view cannot silently
            # empty the chart.
            sign = "all" if across_classes else (sign_filter or "all")
            word_imp = explanation.word_importance(
                label_idx=None if across_classes else int(label_idx),
                n_top=n_top,
                filter_sign=sign,
                # Punctuation is its own unit since word segmentation splits on word/non-word
                # boundaries; corpus-wide its mean contribution averages to ~0, so it is noise here.
                # Per-instance punctuation contributions stay visible in the Sentence Highlight panel.
                filter_punctuation=True,
                exclude_words=set(exclude_words_list or []) or None,
                sample_indices=effective_indices,
                rank_by=rank_by,
                min_occurrences=floor,
            )
            n_scope = len(effective_indices) if effective_indices is not None else len(explanation)
            if word_imp.empty:
                # Name the filter that actually emptied it, since the fix differs: the sign filter
                # (checked first — it is applied last), then the frequency floor, then everything
                # else (an exclusion list or an empty selection).
                if sign in ("positive", "negative"):
                    reason = f"No {sign} word passes these filters."
                elif floor > 1:
                    reason = f"No word occurs at least {floor} time(s) in these {n_scope} sample(s)."
                else:
                    reason = f"No word passes these filters in these {n_scope} sample(s)."
                return empty_word_figure(reason)

            # Same filters and same sample scope as the ranking above, so the count on a bar's
            # hover is the count that bar's aggregate was computed over — and the count the
            # min-occurrences floor was applied to.
            counts = explanation.word_counts(filter_punctuation=True, sample_indices=effective_indices)
            fig = plot_word_importance(
                word_imp,
                # No title: the tab is already labelled "Word Importance" and the class/floor are
                # both visible in the filter row right above the chart — a repeated title band was
                # just eating vertical space this panel needs for word rows (see the graph wrapper
                # below, which scrolls when it runs out).
                title=None,
                # A mean, a total and a cross-class magnitude are different quantities on an
                # identical-looking chart, so the axis has to say which one is drawn. Derived from
                # what word_importance stamped on its result rather than re-deduced here.
                x_title=word_importance_axis_title(str(word_imp.name)),
                width=None,
                height=None,
                counts=counts["n_occurrences"],
                color_positive=self._theme.xpl_positive,
                color_negative=self._theme.xpl_negative,
            )
            # Height deliberately left as plot_word_importance computed it — see the graph's
            # wrapper in the layout.
            return fig

        # Under "All classes" the ranking is a magnitude, so Positive/Negative would silently
        # return nothing. Grey them out and pull the selection back to "All" rather than leaving a
        # live control that cannot do anything.
        @app.callback(
            Output("sign-filter", "options"),
            Output("sign-filter", "value"),
            Input("global-class-selector", "value"),
        )
        def gate_sign_filter(label_idx):
            if label_idx != _ALL_CLASSES:
                return _SIGN_FILTER_OPTIONS, dash.no_update
            greyed = [{**opt, "disabled": opt["value"] != "all"} for opt in _SIGN_FILTER_OPTIONS]
            return greyed, "all"

        # A word bar clicked here is the one cross-panel *output* of this component: it always
        # writes the shared word-click store, whether or not an embeddings scatter is mounted — the
        # scatter (when present) mirrors this store rather than being the thing this panel branches
        # on. Resetting clickData lets the SAME bar be re-clicked (plotly does not re-fire an
        # unchanged clickData).
        @app.callback(
            Output(word_click_store, "data"),
            Output("global-importance-graph", "clickData"),
            Input("global-importance-graph", "clickData"),
            Input(word_click_clear_btn, "n_clicks"),
            prevent_initial_call=True,
        )
        def update_word_click_filter(click_data, _clear_clicks):
            trigger = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
            if word_click_clear_btn in trigger:
                return None, None
            if not click_data or not click_data.get("points"):
                raise PreventUpdate
            return [click_data["points"][0]["y"]], None

        @app.callback(
            Output(word_click_clear_btn, "style"),
            Input(word_click_store, "data"),
        )
        def toggle_word_clear_button(word_filter):
            if word_filter:
                return {"display": "inline", "fontSize": "0.8em"}
            return {"display": "none", "fontSize": "0.8em"}

        # Publishes the active class so ScatterComponent's word-contribution mode can read it
        # without referencing this component's DOM id directly (cross-component data only flows
        # through `stores`).
        @app.callback(
            Output(active_class_store, "data"),
            Input("global-class-selector", "value"),
        )
        def publish_active_class(label_idx):
            return label_idx if isinstance(label_idx, int) else 0
