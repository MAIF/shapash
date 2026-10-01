"""Confusion-matrix heatmap for NLP error analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from plotly import graph_objs as go

from shapash.style.style_utils import DEFAULT_NLP_THEME

# Faint neutral ramp for the diagonal. The correct predictions are context here, not the subject:
# they keep a readable count but stay out of the colour competition, so the whole chromatic range
# is free for the off-diagonal cells this matrix exists to surface (and to have clicked).
_DIAGONAL_COLORSCALE = [[0.0, "rgb(244, 245, 246)"], [1.0, "rgb(214, 217, 220)"]]

# How hard to ease the pale end of the error colourscale (see _emphasis_colorscale). 2.0 puts a
# cell at ~30% of the worst confusion pair already in the orange band.
_EMPHASIS_GAMMA = 2.0


def _emphasis_colorscale(colors: Sequence[str], gamma: float = _EMPHASIS_GAMMA) -> list[list]:
    """Place *colors* on a ``pos = (i / (n - 1)) ** gamma`` curve instead of evenly.

    Error counts are heavy-tailed — one confusion pair usually dwarfs the rest — so an evenly
    spaced scale spends most of its range on values that never occur and renders every small
    cell in the palest, near-invisible stop. Squeezing the pale stops towards zero means a cell
    holding a handful of errors is already visibly coloured, while the worst pair still reads as
    the darkest.
    """
    stops = list(colors)
    if len(stops) == 1:
        return [[0.0, stops[0]], [1.0, stops[0]]]
    last = len(stops) - 1
    return [[(i / last) ** gamma, color] for i, color in enumerate(stops)]


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    normalize: str | None = None,
    title: str = "Confusion matrix",
    width: int | None = None,
    height: int | None = None,
    colorscale: Sequence[str] | str | None = None,
    emphasize_off_diagonal: bool = True,
) -> go.Figure:
    """Heatmap of a confusion matrix with click-identifiable cells.

    Rows are the **true** class, columns are the **predicted** class (the standard
    scikit-learn orientation). Each cell carries ``customdata = [pred_idx, true_idx]``
    so a Dash ``clickData`` handler can recover which (predicted, true) pair was clicked.

    Parameters
    ----------
    cm : numpy.ndarray
        Square confusion-matrix counts of shape ``(n_classes, n_classes)`` with
        ``cm[true, pred]`` giving the number of samples whose true class is ``true``
        and predicted class is ``pred``.
    labels : list of str
        Class display names, ordered to match the rows/columns of ``cm``.
    normalize : {None, "true"}, optional
        If ``"true"``, divide each row by its sum so cells show recall (the fraction
        of each true class routed to every predicted class). If ``None`` (default),
        show raw counts.
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels. Defaults to a size scaled to the number of classes.
    height : int, optional
        Figure height in pixels. Defaults to a size scaled to the number of classes.
    colorscale : sequence of str or str, optional
        Plotly heatmap colorscale for the error cells. Defaults to the active palette's
        ``confusion_matrix_colorscale`` (yellow → orange → red under the default palette),
        the same family the Feature Contribution plot draws from.
    emphasize_off_diagonal : bool, optional
        Colour the errors on their own scale (default). The diagonal normally holds most of the
        mass, so on a shared scale it takes the whole colour range and flattens every error cell
        to the palest shade — the opposite of what this matrix is read for. With this on, the
        diagonal is drawn in faint grey and the off-diagonal cells are scaled to the *worst
        confusion pair*, so darker always means "more errors, more worth clicking". The counts
        themselves are unchanged; only the colour mapping is. Set ``False`` for a classic
        single-scale matrix where the diagonal is coloured too.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    counts = np.asarray(cm, dtype=float)
    n = counts.shape[0]
    if colorscale is None:
        colorscale = DEFAULT_NLP_THEME.confusion_scale

    if normalize == "true":
        row_sums = counts.sum(axis=1, keepdims=True)
        # Guard against empty true-classes (row sum 0) producing NaNs.
        z = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
        text = np.array([[f"{v:.0%}" for v in row] for row in z])
        hover_val = "%{z:.1%}"
        colorbar_title = "recall"
        colorbar_tickformat = ".0%"  # match the cell text, which is a percentage too
    else:
        z = counts
        text = np.array([[f"{int(v)}" for v in row] for row in counts])
        hover_val = "%{z:.0f}"
        colorbar_title = "count"
        colorbar_tickformat = None

    # customdata[true][pred] = [pred_idx, true_idx] — the order a click handler reads.
    # Plain nested Python lists, not a numpy array: Plotly>=6 serializes an ndarray customdata
    # as a binary blob ({"dtype", "bdata", "shape"}) instead of a per-point JSON list, so a real
    # click in the browser loses the cell indices entirely (see plot_contribution.py for the
    # same fix and tests/integration_tests/test_webapp_click_updates_local_explanation.py for
    # the incident this class of bug caused).
    customdata_arr = np.empty((n, n, 2), dtype=int)
    for i in range(n):
        for j in range(n):
            customdata_arr[i, j] = (j, i)
    customdata = customdata_arr.tolist()

    hovertemplate = "true: %{y}<br>predicted: %{x}<br>" + hover_val + "<extra></extra>"
    common = dict(
        x=labels,
        y=labels,
        customdata=customdata,
        texttemplate="%{text}",
        hovertemplate=hovertemplate,
        # Hairline gaps turn the matrix into distinct tiles, which reads as "clickable".
        xgap=2,
        ygap=2,
    )

    fig = go.Figure()
    if emphasize_off_diagonal and n > 1:
        diagonal_mask = np.eye(n, dtype=bool)
        # NaN renders transparent, so the two traces interleave into one matrix with no overlap.
        diag_z = np.where(diagonal_mask, z, np.nan)
        off_z = np.where(diagonal_mask, np.nan, z)
        off_max = np.nanmax(off_z)
        if not np.isfinite(off_max) or off_max <= 0:
            off_max = 1.0  # a matrix with no errors at all: avoid a degenerate zmin == zmax scale
        fig.add_trace(
            go.Heatmap(
                z=diag_z,
                text=np.where(diagonal_mask, text, ""),
                name="correct",
                colorscale=_DIAGONAL_COLORSCALE,
                showscale=False,
                textfont=dict(color="rgb(110, 113, 116)"),
                hoverongaps=False,
                **common,
            )
        )
        fig.add_trace(
            go.Heatmap(
                z=off_z,
                text=np.where(diagonal_mask, "", text),
                name="errors",
                colorscale=_emphasis_colorscale(colorscale) if not isinstance(colorscale, str) else colorscale,
                zmin=0,
                zmax=off_max,
                colorbar=dict(title=f"errors<br>({colorbar_title})", tickformat=colorbar_tickformat),
                textfont=dict(color="rgb(50, 50, 50)"),
                hoverongaps=False,
                **common,
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                z=z,
                text=text,
                name="cells",
                colorscale=list(colorscale) if not isinstance(colorscale, str) else colorscale,
                colorbar=dict(title=colorbar_title, tickformat=colorbar_tickformat),
                **common,
            )
        )

    size = max(360, 90 * n + 120)
    # The x-axis sits on top, so keep room for its labels even without a title. With a title,
    # pin it above that axis title (rather than at Plotly's default position, which drifts down
    # into the same band) and give the axis title a standoff so the two never share a line.
    top_margin = 115 if title else 40
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98, yanchor="top"),
        xaxis_title=dict(text="Predicted class", standoff=25),
        yaxis_title="True class",
        width=width or size,
        height=height or size,
        plot_bgcolor="white",
        showlegend=False,
        # Put the first true class at the top so the diagonal reads top-left to bottom-right.
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(automargin=True, side="top"),
        margin=dict(l=20, r=20, t=top_margin, b=20),
    )

    return fig
