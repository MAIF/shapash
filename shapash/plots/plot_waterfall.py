"""Waterfall chart of per-token SHAP contributions for NLP explanations.

Tokens whose absolute contribution falls below a configurable threshold are
collapsed into a single "other (N tokens)" bar so the chart stays readable even
for long sequences.  Tokens are sorted by absolute contribution descending
(standard SHAP convention), giving the most important tokens top billing.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from plotly import graph_objs as go

from shapash.style.style_utils import DEFAULT_NLP_THEME

_SPECIAL_RE = re.compile(r"^\[.*\]$|^##|^\s*$")


def plot_waterfall(
    tokens: list[str],
    values: np.ndarray,
    base_value: float | None = None,
    min_pct: float = 0.10,
    filter_special: bool = True,
    title: str = "Token contributions",
    width: int | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> go.Figure:
    """Horizontal waterfall chart decomposing a single prediction into token SHAP contributions.

    Each token appears as a horizontal bar segment.  Positive contributions
    extend right (*color_positive*), negative extend left (*color_negative*).  Connectors show the
    cumulative path from the baseline to the final prediction.

    Tokens with ``|value| < min_pct * max_abs`` are aggregated into a single
    "other (N tokens)" bar to reduce visual clutter.

    Parameters
    ----------
    tokens : list[str]
        Token strings in sentence order.
    values : np.ndarray
        1-D array of per-token SHAP contributions (same length as ``tokens``).
    base_value : float, optional
        Model baseline for this sample.  Shown as the first "absolute" bar when
        provided.
    min_pct : float
        Fraction of the maximum absolute contribution below which tokens are
        grouped into the "other" bar.  Range [0, 1].  Default 0.10 (10 %).
    filter_special : bool
        If ``True``, skip empty strings, ``[CLS]``/``[SEP]``-style tokens, and
        ``##subword`` prefixes before computing the threshold.
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels.  Defaults to Plotly responsive behaviour.
    color_positive, color_negative : str
        Bar colors for a non-negative / negative contribution. Default to the ``"default"``
        palette's ``nlp_xpl_positive``/``nlp_xpl_negative`` in ``shapash/style/colors.json`` — see
        :class:`~shapash.webapp.nlp_app.NlpWebApp`'s ``palette_name``/``colors_dict`` to theme
        every NLP chart at once instead of overriding this one call.

    Returns
    -------
    go.Figure
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")

    # ── Filter special tokens ───────────────────────────────────────────
    pairs = [
        (tok, float(val))
        for tok, val in zip(tokens, values, strict=False)
        if not (filter_special and _SPECIAL_RE.match(tok.strip()))
    ]

    if not pairs:
        return go.Figure()

    raw_toks = [p[0] for p in pairs]
    all_vals_f = [p[1] for p in pairs]

    # Give each occurrence of a repeated token a unique label so Plotly
    # renders them as separate bars instead of merging them on the same row.
    tok_counts = Counter(raw_toks)
    occurrence: dict[str, int] = {}
    all_toks: list[str] = []
    for tok in raw_toks:
        if tok_counts[tok] > 1:
            occurrence[tok] = occurrence.get(tok, 0) + 1
            all_toks.append(f"{tok} [{occurrence[tok]}]")
        else:
            all_toks.append(tok)

    max_abs = max(abs(v) for v in all_vals_f) or 1.0
    threshold = float(min_pct) * max_abs

    significant = [(t, v) for t, v in zip(all_toks, all_vals_f, strict=False) if abs(v) >= threshold]
    small = [(t, v) for t, v in zip(all_toks, all_vals_f, strict=False) if abs(v) < threshold]

    # Sort significant by |value| descending (SHAP convention)
    significant.sort(key=lambda p: abs(p[1]), reverse=True)

    # ── Build waterfall arrays ──────────────────────────────────────────
    y_labels: list[str] = []
    x_vals: list[float] = []
    measures: list[str] = []

    if base_value is not None:
        y_labels.append("Base")
        x_vals.append(float(base_value))
        measures.append("absolute")

    for tok, val in significant:
        y_labels.append(tok)
        x_vals.append(val)
        measures.append("relative")

    if small:
        n_small = len(small)
        sum_small = sum(v for _, v in small)
        y_labels.append(f"other ({n_small} token{'s' if n_small != 1 else ''})")
        x_vals.append(sum_small)
        measures.append("relative")

    running_total = (float(base_value) if base_value is not None else 0.0) + sum(all_vals_f)
    y_labels.append("Total")
    x_vals.append(0)  # value is ignored for measure="total"; Plotly computes it
    measures.append("total")

    # Per-bar text labels
    texts: list[str] = []
    for val, measure in zip(x_vals, measures, strict=False):
        texts.append(f"{running_total:+.3f}" if measure == "total" else f"{val:+.3f}")

    n_bars = len(y_labels)

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=y_labels,
            x=x_vals,
            measure=measures,
            text=texts,
            textposition="outside",
            increasing=dict(marker=dict(color=color_positive)),
            decreasing=dict(marker=dict(color=color_negative)),
            totals=dict(marker=dict(color="#7f7f7f")),
            connector=dict(line=dict(color="#dddddd", width=1, dash="dot")),
            cliponaxis=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="SHAP contribution",
        height=max(300, 36 * n_bars + 100),
        width=width,
        plot_bgcolor="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#888888",
            zerolinewidth=1.5,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(automargin=True),
        margin=dict(l=20, r=80, t=50, b=40),
        showlegend=False,
    )

    return fig
