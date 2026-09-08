"""Inline sentence highlight for NLP token-level SHAP contributions.

Renders a sentence as a sequence of coloured ``html.Span`` elements where each
token's background intensity is proportional to its SHAP contribution magnitude.
Positive/negative contributions use ``NlpTheme.xpl_positive``/``xpl_negative`` (the same colors
``plot_word_importance`` and ``plot_waterfall`` use), interpolated from white by magnitude.
"""

from __future__ import annotations

import re

import numpy as np
from dash import html

from shapash.style.style_utils import DEFAULT_NLP_THEME

_SPECIAL_RE = re.compile(r"^\[.*\]$|^##|^\s*$")


def _parse_rgb(color: str) -> tuple[int, int, int]:
    """Parse a ``rgb(r, g, b)``/``rgba(r, g, b, a)`` string (the ``colors.json`` format) to ints."""
    nums = color.replace("rgba", "").replace("rgb", "").replace("(", "").replace(")", "").split(",")
    r, g, b = (int(float(n)) for n in nums[:3])
    return r, g, b


def _shap_color(val: float, max_abs: float, pos_rgb: tuple[int, int, int], neg_rgb: tuple[int, int, int]) -> str:
    """Interpolate from white toward the sign-appropriate colour."""
    if max_abs == 0 or val == 0:
        return "transparent"
    t = min(abs(val) / max_abs, 1.0)
    r_base, g_base, b_base = pos_rgb if val >= 0 else neg_rgb
    r = int(255 + t * (r_base - 255))
    g = int(255 + t * (g_base - 255))
    b = int(255 + t * (b_base - 255))
    return f"rgb({r},{g},{b})"


def plot_sentence_highlight(
    tokens: list[str],
    values: np.ndarray,
    base_value: float | None = None,
    title: str = "Sentence Highlight",
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> html.Div:
    """Inline sentence with background-coloured spans proportional to SHAP contribution.

    Each token is rendered as an inline block whose background colour intensity
    is proportional to its contribution magnitude.  Special tokens
    (``[CLS]``, ``[SEP]``, ``##subwords``, empty strings) are shown in muted
    grey.  Hover over any token to see its exact SHAP value.

    Parameters
    ----------
    tokens : list[str]
        Token strings in sentence order (same length as ``values``).
    values : np.ndarray
        1-D array of per-token SHAP contributions.
    base_value : float, optional
        Model baseline for this sample.  When provided, a summary line shows
        ``base + Σ contributions = total``.
    title : str
        Unused (callers render their own ``html.H6`` header); kept for API
        symmetry with the other plot functions.
    color_positive, color_negative : str
        Span background for a non-negative / negative contribution, interpolated from white by
        magnitude. Must be a ``rgb(r, g, b)``/``rgba(r, g, b, a)`` string (the ``colors.json``
        format). Default to the ``"default"`` palette's ``nlp_xpl_positive``/``nlp_xpl_negative``
        — see :class:`~shapash.webapp.nlp_app.NlpWebApp`'s ``palette_name``/``colors_dict`` to
        theme every NLP chart at once instead of overriding this one call.

    Returns
    -------
    html.Div
        A Dash component ready to be placed in a layout or returned from a
        callback targeting ``"children"``.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")

    max_abs = float(np.abs(values).max()) if len(values) > 0 else 1.0
    if max_abs == 0:
        max_abs = 1.0

    pos_rgb, neg_rgb = _parse_rgb(color_positive), _parse_rgb(color_negative)

    _span_base = {
        "padding": "3px 5px",
        "borderRadius": "3px",
        "margin": "2px 1px",
        "display": "inline-block",
        "cursor": "default",
        "lineHeight": "2.2",
    }
    _span_special = {
        **_span_base,
        "backgroundColor": "#eeeeee",
        "color": "#aaaaaa",
        "fontSize": "0.8em",
    }

    spans: list = []
    for tok, val in zip(tokens, values, strict=False):
        is_special = bool(_SPECIAL_RE.match(tok.strip()))
        tooltip = f"{tok}: {float(val):.4f}"
        if is_special:
            spans.append(html.Span(tok + " ", style=_span_special, title=tooltip))
        else:
            spans.append(
                html.Span(
                    tok + " ",
                    style={**_span_base, "backgroundColor": _shap_color(float(val), max_abs, pos_rgb, neg_rgb)},
                    title=tooltip,
                )
            )

    total_shap = float(np.sum(values))
    if base_value is not None:
        total = base_value + total_shap
        summary = html.Div(
            [
                html.Span(f"Base: {base_value:.3f}", style={"color": "#777", "marginRight": "14px"}),
                html.Span(f"Σ contributions: {total_shap:+.3f}", style={"color": "#444", "marginRight": "14px"}),
                html.Span(f"Total: {total:.3f}", style={"fontWeight": "bold", "color": "#111"}),
            ],
            style={"marginTop": "8px", "fontSize": "0.82em"},
        )
    else:
        summary = html.Div(
            html.Span(f"Σ contributions: {total_shap:+.3f}", style={"color": "#444"}),
            style={"marginTop": "8px", "fontSize": "0.82em"},
        )

    legend = html.Div(
        [
            html.Span("■ ", style={"color": color_positive}),
            html.Span("positive  ", style={"fontSize": "0.78em", "color": "#555"}),
            html.Span("■ ", style={"color": color_negative}),
            html.Span("negative", style={"fontSize": "0.78em", "color": "#555"}),
        ],
        style={"marginBottom": "6px"},
    )

    return html.Div(
        [
            legend,
            html.Div(
                spans,
                style={
                    "lineHeight": "2.5",
                    "fontSize": "1.05em",
                    "padding": "12px",
                    "backgroundColor": "#fafafa",
                    "borderRadius": "4px",
                    "border": "1px solid #eeeeee",
                    "minHeight": "80px",
                    "overflowY": "auto",
                    "maxHeight": "300px",
                },
            ),
            summary,
        ]
    )
