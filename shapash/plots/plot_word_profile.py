"""Per-class profile of a single word's aggregated contributions."""

from __future__ import annotations

import pandas as pd
from plotly import graph_objs as go

from shapash.style.style_utils import DEFAULT_NLP_THEME


def plot_word_profile(
    stats: pd.Series,
    label_names: list[str] | None = None,
    spread: pd.Series | None = None,
    x_title: str = "Aggregated contribution",
    title: str = "Word profile",
    width: int | None = 640,
    height: int | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> go.Figure:
    """Horizontal bar chart of one word's aggregated contribution to each class.

    The transpose of :func:`~shapash.plots.plot_word_importance.plot_word_importance`: that one
    fixes a class and ranks words, this one fixes a word and ranks nothing — it shows every class
    at once, so a word that pushes toward one class and away from another reads as a single shape
    instead of requiring the reader to flip a class selector back and forth.

    Parameters
    ----------
    stats : pd.Series
        ``class_idx`` → aggregated contribution, as returned by
        :func:`~shapash.explainer.nlp_explanation.aggregate_word_contributions`.
    label_names : list[str], optional
        Class names in model-output order, used for the y-axis. Falls back to the class indices.
    spread : pd.Series, optional
        Same index as ``stats``, drawn as symmetric error bars — the standard deviation across
        occurrences, when the aggregation is a mean. Omit for sums, where a spread across
        occurrences says nothing about the total.
    x_title : str
        X-axis label. The caller names the aggregation here (e.g. ``"Mean contribution"``), since
        the bars mean something different for each one.
    title : str
        Figure title.
    width, height : int, optional
        Figure size in pixels. ``height`` defaults to a size that fits the number of classes.
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
    >>> occ = explanation.word_occurrences("terrible")
    >>> stats = aggregate_word_contributions(occ, agg="mean")
    >>> plot_word_profile(stats, label_names=explanation.label_names).show()
    """
    class_idx = list(stats.index)
    names = [label_names[i] if label_names is not None and i < len(label_names) else str(i) for i in class_idx]
    values = [float(v) for v in stats.to_numpy()]
    colors = [color_positive if v >= 0 else color_negative for v in values]

    errors: list[float] | None = None
    if spread is not None and len(spread):
        # Reindexed on the bars actually drawn, so a mismatched spread cannot silently shift the
        # error bars onto the wrong classes.
        aligned = spread.reindex(stats.index).fillna(0.0)
        errors = [float(v) for v in aligned.to_numpy()]

    # Plotly renders the y-axis bottom-to-top; reverse so class 0 sits at the top, matching the
    # order every class selector in the app lists them in.
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=names[::-1],
            orientation="h",
            marker_color=colors[::-1],
            # The class index travels on the bar so a click handler resolves it directly. Matching
            # the y-axis label back to a class would be ambiguous the moment two classes share a
            # display name, and reversed here alongside every other per-bar array.
            customdata=[[int(i)] for i in class_idx][::-1],
            error_x=(
                dict(type="data", array=errors[::-1], visible=True, color="#666666", thickness=1)
                if errors is not None
                else None
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=x_title,
        width=width,
        height=height or max(220, 46 * len(names) + 120),
        plot_bgcolor="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(automargin=True),
        margin=dict(l=20, r=20, t=60, b=40),
        bargap=0.45,
    )

    return fig
