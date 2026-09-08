"""Word-level importance plot for NLP explanations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from plotly import graph_objs as go

from shapash.style.style_utils import DEFAULT_NLP_THEME

# Vertical room per word. The y axis is categorical and every category gets a tick (dtick=1
# below), so this is what decides whether those ticks have room to render at all: squeeze 50 words
# into a 400px panel and each row gets 8px, at which point plotly drops labels rather than overlap
# them and the chart silently stops naming its own bars. Callers that want the chart to fill a
# fixed panel should scroll it, not override the height.
_ROW_HEIGHT = 30
_CHROME_HEIGHT = 120
# Room the title band itself takes out of _CHROME_HEIGHT — reclaimed (from both the auto-height
# budget and the top margin) when a caller has nowhere to put a title and skips it.
_TITLE_HEIGHT = 40
_MIN_HEIGHT = 400
# Headroom for the value labels, as a fraction of the largest bar — "outside" text is drawn past
# the bar end and would otherwise be clipped at the plot edge on the longest bar.
_VALUE_LABEL_PAD = 0.18


# X-axis wording per statistic, keyed by the ``.name`` ``NlpExplanation.word_importance`` stamps on
# its result. A mean, a total and a cross-class magnitude are three different quantities drawn as
# identical-looking bars, and the axis is the only place a reader can tell which one they have.
_AXIS_TITLES = {
    "mean": "Mean SHAP contribution",
    "sum": "Total SHAP contribution",
    "mean_across_classes": "Largest |mean SHAP| across classes",
    "sum_across_classes": "Largest |total SHAP| across classes",
}


def word_importance_axis_title(name: str | None) -> str:
    """X-axis label for a ``word_importance`` result, from the ``.name`` it carries.

    Shared by every caller so the wording cannot drift between the library plot and the webapp
    panel drawing the same numbers. Falls back to the neutral "SHAP contribution" for a Series
    that did not come from ``word_importance``.
    """
    return _AXIS_TITLES.get(name or "", "SHAP contribution")


def empty_word_figure(message: str) -> go.Figure:
    """Blank axes carrying a centred hint, for when a word chart has nothing to draw.

    An empty bar chart and a chart of all-zero bars look alike, and both look like a bug. Every
    word-level panel has a legitimate empty state — no confusion-matrix cell picked yet, a word
    absent from the selection, a frequency threshold that excludes the whole vocabulary — so the
    reason is drawn on the figure rather than left to the reader.

    Parameters
    ----------
    message : str
        Why there is nothing to show.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color="#888888"), xref="paper", yref="paper")
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def plot_word_importance(
    word_imp: pd.Series,
    title: str | None = "Word importance",
    x_title: str = "Mean SHAP contribution",
    width: int = 900,
    height: int | None = None,
    show_values: bool = True,
    counts: pd.Series | Mapping[str, int] | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> go.Figure:
    """Horizontal bar chart of aggregated token-level SHAP contributions per word.

    Words are displayed in the order given (highest at top). Positive contributions are shown
    in *color_positive*, negative in *color_negative* — which holds whichever statistic produced
    them, since :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.word_importance` ranks by
    absolute value but returns signed numbers.

    Parameters
    ----------
    word_imp : pd.Series
        Word → aggregated SHAP contribution, already sorted by ``|value|`` descending
        (as returned by ``NlpExplanation.word_importance``).
    title : str, optional
        Figure title. Pass ``None`` or ``""`` to omit it and reclaim the title band's vertical
        space — worth it when a caller already names the chart elsewhere (a tab label, a panel
        header) and would otherwise be paying for the same words twice.
    x_title : str
        X-axis label. Name the statistic actually plotted — a mean and a total are different
        quantities on the same-looking chart, and the axis is where a reader checks which.
    width : int
        Figure width in pixels.
    height : int, optional
        Figure height in pixels. Defaults to ``max(400, 30 * n_words + chrome)``, where ``chrome``
        is ``120`` with a title or ``80`` without one — enough room for every word label to
        render. Override it only for a chart that is scrolled rather than squeezed; forcing a
        short height on a long ranking makes plotly drop the labels.
    show_values : bool
        Print each contribution at the end of its bar. Keeps the magnitude readable without
        tracing back to the axis, at the cost of horizontal room — pass ``False`` in a narrow
        panel.
    counts : pd.Series or Mapping[str, int], optional
        Word → number of occurrences the aggregate was computed over, added to each bar's hover.
        It is the missing half of the ranking: a mean of ``+0.90`` over one occurrence and the
        same mean over two hundred are indistinguishable on the bars themselves. Pass the
        ``n_occurrences`` column of
        :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.word_counts` computed with the
        *same* filters and sample scope as the ranking. A mapping that does not cover every
        plotted word is ignored outright rather than shown partially.
    color_positive, color_negative : str
        Bar colors for a non-negative / negative contribution. Default to the ``"default"``
        palette's ``nlp_xpl_positive``/``nlp_xpl_negative`` in ``shapash/style/colors.json`` — see
        :class:`~shapash.webapp.nlp_app.NlpWebApp`'s ``palette_name``/``colors_dict`` to theme
        every NLP chart at once instead of overriding this one call.

    Returns
    -------
    go.Figure

    Examples
    --------
    >>> imp = explanation.word_importance(label_idx=1)
    >>> fig = plot_word_importance(imp, title="Word importance — joy")
    >>> fig.show()
    """
    words = [str(w) for w in word_imp.index]
    values = [float(v) for v in word_imp.to_numpy()]
    colors = [color_positive if v >= 0 else color_negative for v in values]
    labels = [f"{v:+.3f}" for v in values] if show_values else None
    occurrences = _aligned_counts(words, counts)

    # Name the statistic in the hover too: the axis title is the only thing distinguishing a mean
    # from a total, and it is off-screen once the chart is long enough to scroll.
    hover = f"<b>%{{y}}</b><br>{x_title}: %{{x:.4f}}"
    if occurrences is not None:
        hover += "<br>Occurrences: %{customdata[0]:,}"
    hover += "<extra></extra>"

    # Plotly renders y-axis bottom-to-top; reverse so highest-importance word appears at top.
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=words[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=labels[::-1] if labels is not None else None,
            textposition="outside" if show_values else "none",
            textfont=dict(size=10, color="#666666"),
            # Outside labels sit past the bar end, which is beyond the axis range for the longest
            # bar; without this they are cut off rather than drawn.
            cliponaxis=False,
            customdata=[[n] for n in occurrences][::-1] if occurrences is not None else None,
            hovertemplate=hover,
        )
    )

    chrome = _CHROME_HEIGHT if title else _CHROME_HEIGHT - _TITLE_HEIGHT
    fig.update_layout(
        title=dict(text=title, x=0.5) if title else None,
        xaxis_title=x_title,
        width=width,
        height=height or max(_MIN_HEIGHT, _ROW_HEIGHT * len(words) + chrome),
        plot_bgcolor="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            gridcolor="#eeeeee",
            range=_padded_range(values) if show_values else None,
        ),
        yaxis=dict(
            automargin=True,
            # One tick per word, always. On a categorical axis plotly thins ticks when they do not
            # fit, so a long ranking loses labels for the very bars a reader is trying to identify
            # — and it does it silently. Forcing dtick makes a cramped chart look cramped instead.
            dtick=1,
            tickfont=dict(size=12, color="#222222"),
        ),
        # Slightly thicker bars than plotly's default, so a word label lines up against a bar
        # rather than against whitespace.
        bargap=0.25,
        margin=dict(l=20, r=20, t=60 if title else 20, b=40),
    )

    return fig


def _aligned_counts(words: list[str], counts: pd.Series | Mapping[str, int] | None) -> list[int] | None:
    """Occurrence counts in ``words`` order, or ``None`` when there is nothing trustworthy to show.

    An incomplete mapping is rejected as a whole rather than filled with zeros: a bar labelled
    "0 occurrences" that in fact has forty is worse than a bar with no count at all, and the only
    way the mapping can be incomplete is a caller computing it under different filters — a
    mismatch that would misreport *every* count, not just the missing ones.
    """
    if counts is None:
        return None
    lookup = counts.to_dict() if isinstance(counts, pd.Series) else dict(counts)
    aligned = [lookup.get(w) for w in words]
    if any(n is None for n in aligned):
        return None
    return [int(n) for n in aligned]  # type: ignore[arg-type]


def _padded_range(values: list[float]) -> list[float] | None:
    """X range with headroom for the outside value labels, or ``None`` to let plotly autoscale."""
    if not values:
        return None
    lo, hi = min(0.0, *values), max(0.0, *values)
    span = max(abs(lo), abs(hi))
    if span == 0:
        return None
    pad = _VALUE_LABEL_PAD * span
    return [lo - pad, hi + pad]
