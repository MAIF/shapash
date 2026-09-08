"""Token-level contribution plot for NLP explanations."""

from __future__ import annotations

import numpy as np
from plotly import graph_objs as go

from shapash.style.style_utils import DEFAULT_NLP_THEME


def plot_token_highlight(
    tokens: list[str],
    values: np.ndarray,
    title: str = "Token contributions",
    max_tokens: int | None = None,
    width: int = 900,
    height: int | None = None,
    color_positive: str = DEFAULT_NLP_THEME.xpl_positive,
    color_negative: str = DEFAULT_NLP_THEME.xpl_negative,
) -> go.Figure:
    """Horizontal bar chart of token-level SHAP contributions for one sample.

    Positive contributions are shown in *color_positive*, negative in *color_negative*.
    Tokens are displayed in sentence order (top to bottom). When ``max_tokens``
    is set, only the highest-magnitude tokens are kept but their relative order
    in the sentence is preserved.

    Parameters
    ----------
    tokens : list[str]
        Token strings for one sample.
    values : np.ndarray
        1-D contribution array of shape ``(n_tokens,)`` for a single class.
    title : str
        Figure title.
    max_tokens : int, optional
        If provided, keep only the ``max_tokens`` tokens with the largest
        absolute contribution, in their original sentence order.
    width : int
        Figure width in pixels.
    height : int, optional
        Figure height in pixels. Defaults to ``max(400, 30 * n_tokens + 120)``.
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
    >>> fig = plot_token_highlight(["I", "love", "this"], np.array([0.1, 0.5, 0.3]))
    >>> fig.show()
    """
    tokens = list(tokens)
    values = np.asarray(values, dtype=float)

    if max_tokens is not None and max_tokens < len(tokens):
        top_idx = np.argsort(np.abs(values))[::-1][:max_tokens]
        keep = np.sort(top_idx)
        tokens = [tokens[i] for i in keep]
        values = values[keep]

    colors = [color_positive if v >= 0 else color_negative for v in values]

    # Plotly renders y-axis bottom-to-top; reverse so sentence reads top-to-bottom.
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=tokens[::-1],
            orientation="h",
            marker_color=colors[::-1],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Contribution",
        width=width,
        height=height or max(400, 30 * len(tokens) + 120),
        plot_bgcolor="white",
        xaxis=dict(
            zeroline=True,
            zerolinecolor="#333333",
            zerolinewidth=1,
            gridcolor="#eeeeee",
        ),
        yaxis=dict(automargin=True),
        margin=dict(l=20, r=20, t=60, b=40),
    )

    return fig
