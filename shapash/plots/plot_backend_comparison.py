"""Side-by-side views of several NLP backends' contributions on the same text.

Every function here takes the *already aligned* data — one list of reference word units and, per
backend, one value per unit (``NaN`` where that backend attributed nothing) — so alignment and
normalisation stay in :mod:`shapash.compute.token_alignment` and these stay pure renderers.
Reach them through :meth:`~shapash.explainer.nlp_plotter.NlpPlotter.compare`.

Three display options, each answering a slightly different question:

- :func:`plot_backend_heatmap` — tokens × backends grid: *where* along the sentence do the
  methods agree? The most compact view, and the only one that stays readable past ~4 backends.
- :func:`plot_backend_bars` — grouped bars per token: *how much* do they differ on a given word?
- :func:`plot_backend_highlight` — one highlighted copy of the sentence per backend: the familiar
  text-highlight read, stacked so the eye can scan down a column.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from dash import html
from plotly import graph_objs as go

from shapash.plots.plot_sentence_highlight import _parse_rgb, _shap_style
from shapash.style.style_utils import DEFAULT_NLP_THEME

# Backend identity colors, assigned in this fixed order (never cycled) — the reference categorical
# palette's first slots, validated for colour-vision-deficiency separation. Only the grouped bar
# view needs identity colors; the other two encode the backend by row.
BACKEND_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

_NEUTRAL = "rgb(242, 242, 242)"
_MUTED_INK = "#8a8a8a"
_GRID = "#eeeeee"


def _check(tokens: Sequence[str], values_by_backend: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not values_by_backend:
        raise ValueError("values_by_backend is empty — pass at least one backend.")
    out = {}
    for name, values in values_by_backend.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape != (len(tokens),):
            raise ValueError(f"values for {name!r} have shape {arr.shape}, expected ({len(tokens)},).")
        out[name] = arr
    return out


def _opaque(color: str) -> str:
    r, g, b = _parse_rgb(color)
    return f"rgb({r},{g},{b})"


def plot_backend_heatmap(
    tokens: Sequence[str],
    values_by_backend: Mapping[str, np.ndarray],
    title: str = "Backend comparison",
    subtitle: str | None = None,
    colorbar_title: str = "Contribution",
    width: int | None = None,
    height: int | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> go.Figure:
    """Tokens × backends grid, one diverging-colored cell per (backend, word).

    Parameters
    ----------
    tokens : sequence of str
        Reference word units, in sentence order (columns, left to right).
    values_by_backend : mapping of str to np.ndarray
        Backend label → 1-D aligned contributions, one per token (rows, top to bottom in mapping
        order). ``NaN`` marks a unit the backend did not attribute; it is drawn as a ``·``.
    title, subtitle : str
        Figure title, and an optional smaller line under it (e.g. agreement scores).
    colorbar_title : str
        Colorbar label — say whether the values are normalised.
    width, height : int, optional
        Figure size in pixels. Default to a width that gives each token ~46px and a height that
        gives each backend ~46px.
    color_positive, color_negative : str
        Poles of the diverging scale, around a neutral gray at zero. Same defaults as every other
        NLP chart.

    Returns
    -------
    go.Figure
    """
    values_by_backend = _check(tokens, values_by_backend)
    names = list(values_by_backend)
    z = np.vstack([values_by_backend[n] for n in names])
    finite = z[np.isfinite(z)]
    zmax = float(np.abs(finite).max()) if finite.size and np.abs(finite).max() > 0 else 1.0
    x = np.arange(len(tokens))

    hover_tokens = np.tile(np.asarray(tokens, dtype=object), (len(names), 1))
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=names,
            customdata=hover_tokens,
            zmin=-zmax,
            zmax=zmax,
            zmid=0,
            colorscale=[[0.0, _opaque(color_negative)], [0.5, _NEUTRAL], [1.0, _opaque(color_positive)]],
            xgap=2,
            ygap=2,
            hoverongaps=False,
            hovertemplate="<b>%{customdata}</b><br>%{y}: %{z:+.3f}<extra></extra>",
            colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=12, outlinewidth=0),
        )
    )

    # Unattributed cells: an explicit mark rather than a blank, so "not attributed" never reads as
    # "attributed zero".
    rows, cols = np.where(~np.isfinite(z))
    if rows.size:
        fig.add_trace(
            go.Scatter(
                x=cols,
                y=[names[r] for r in rows],
                mode="text",
                text=["·"] * rows.size,
                textfont=dict(color=_MUTED_INK, size=16),
                customdata=[tokens[int(c)] for c in cols],
                hovertemplate="<b>%{customdata}</b><br>%{y}: not attributed<extra></extra>",
                showlegend=False,
            )
        )

    title_text = title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        width=width or max(600, 46 * len(tokens) + 220),
        height=height or 46 * len(names) + 190,
        plot_bgcolor="white",
        xaxis=dict(
            tickmode="array",
            tickvals=x,
            ticktext=list(tokens),
            tickangle=-45,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(autorange="reversed", showgrid=False, automargin=True),
        margin=dict(l=20, r=20, t=80, b=40),
    )
    return fig


def plot_backend_bars(
    tokens: Sequence[str],
    values_by_backend: Mapping[str, np.ndarray],
    title: str = "Backend comparison",
    subtitle: str | None = None,
    xaxis_title: str = "Contribution",
    width: int = 900,
    height: int | None = None,
    colors: Sequence[str] = BACKEND_COLORS,
) -> go.Figure:
    """Grouped horizontal bars: one group per word (sentence order, top to bottom), one bar per backend.

    Parameters
    ----------
    tokens : sequence of str
        Reference word units, in sentence order.
    values_by_backend : mapping of str to np.ndarray
        Backend label → 1-D aligned contributions, one per token. ``NaN`` units get no bar.
    title, subtitle : str
        Figure title, and an optional smaller line under it.
    xaxis_title : str
        Value-axis label — say whether the values are normalised.
    width, height : int, optional
        Figure size in pixels. ``height`` defaults to give each bar ~12px.
    colors : sequence of str
        Backend identity colors, assigned in order. More backends than colors raises: a generated
        extra hue would not be distinguishable, so use :func:`plot_backend_heatmap` instead.

    Returns
    -------
    go.Figure
    """
    values_by_backend = _check(tokens, values_by_backend)
    if len(values_by_backend) > len(colors):
        raise ValueError(
            f"{len(values_by_backend)} backends but only {len(colors)} identity colors — "
            "use the heatmap view for this many."
        )
    y = np.arange(len(tokens))
    fig = go.Figure()
    for color, (name, values) in zip(colors, values_by_backend.items(), strict=False):
        fig.add_trace(
            go.Bar(
                x=values,
                y=y,
                name=name,
                orientation="h",
                marker=dict(color=color, line=dict(color="white", width=1)),
                customdata=list(tokens),
                hovertemplate="<b>%{customdata}</b><br>" + name + ": %{x:+.3f}<extra></extra>",
            )
        )

    title_text = title if subtitle is None else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        width=width,
        height=height or max(400, 12 * len(tokens) * len(values_by_backend) + 40 * len(tokens) // 2 + 160),
        plot_bgcolor="white",
        xaxis=dict(title=xaxis_title, zeroline=True, zerolinecolor="#333333", zerolinewidth=1, gridcolor=_GRID),
        yaxis=dict(tickmode="array", tickvals=y, ticktext=list(tokens), autorange="reversed", automargin=True),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.0, yanchor="bottom", traceorder="normal"),
        margin=dict(l=20, r=20, t=100, b=40),
    )
    return fig


def plot_backend_highlight(
    tokens: Sequence[str],
    values_by_backend: Mapping[str, np.ndarray],
    subtitle: str | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> html.Div:
    """One highlighted copy of the sentence per backend, stacked.

    Each row is colored on its own magnitude scale (its largest unit is fully saturated), since
    backends' raw values are in different units. Every row renders the same reference units with
    the same padding, so a word sits at the same horizontal position in every row as long as the
    rows wrap at the same width.

    Parameters
    ----------
    tokens : sequence of str
        Reference word units, in sentence order.
    values_by_backend : mapping of str to np.ndarray
        Backend label → 1-D aligned contributions. ``NaN`` units are drawn struck-through in gray.
    subtitle : str, optional
        A line shown under the legend (e.g. agreement scores).
    color_positive, color_negative : str
        ``rgb(...)``/``rgba(...)`` strings, as for
        :func:`~shapash.plots.plot_sentence_highlight.plot_sentence_highlight`.

    Returns
    -------
    html.Div
    """
    values_by_backend = _check(tokens, values_by_backend)
    pos_rgb, neg_rgb = _parse_rgb(color_positive), _parse_rgb(color_negative)
    span_base = {"padding": "3px 5px", "borderRadius": "3px", "margin": "2px 1px", "display": "inline-block"}
    unattributed = {**span_base, "color": "#aaaaaa", "textDecoration": "line-through"}

    rows = []
    for name, values in values_by_backend.items():
        finite = values[np.isfinite(values)]
        max_abs = float(np.abs(finite).max()) if finite.size and np.abs(finite).max() > 0 else 1.0
        spans = []
        for tok, val in zip(tokens, values, strict=True):
            if not np.isfinite(val):
                spans.append(html.Span(f"{tok} ", style=unattributed, title=f"{tok}: not attributed"))
            else:
                style = {**span_base, **_shap_style(float(val), max_abs, pos_rgb, neg_rgb)}
                spans.append(html.Span(f"{tok} ", style=style, title=f"{tok}: {float(val):+.4f}"))
        rows.append(
            html.Div(
                [
                    html.Div(name, style={"fontSize": "0.8em", "fontWeight": "600", "color": "#444"}),
                    html.Div(spans, style={"lineHeight": "2.2", "fontSize": "1.0em"}),
                ],
                style={"padding": "8px 12px", "borderBottom": "1px solid #eeeeee"},
            )
        )

    legend = html.Div(
        [
            html.Span("■ ", style={"color": color_positive}),
            html.Span("positive  ", style={"fontSize": "0.78em", "color": "#555"}),
            html.Span("■ ", style={"color": color_negative}),
            html.Span("negative  ", style={"fontSize": "0.78em", "color": "#555"}),
            html.Span(
                "struck-through", style={"fontSize": "0.78em", "color": "#aaa", "textDecoration": "line-through"}
            ),
            html.Span(" not attributed", style={"fontSize": "0.78em", "color": "#555"}),
        ],
        style={"marginBottom": "4px"},
    )
    header = [legend]
    if subtitle:
        header.append(html.Div(subtitle, style={"fontSize": "0.78em", "color": "#555", "marginBottom": "6px"}))
    return html.Div(
        [
            *header,
            html.Div(rows, style={"backgroundColor": "#fafafa", "border": "1px solid #eeeeee", "borderRadius": "4px"}),
        ]
    )
