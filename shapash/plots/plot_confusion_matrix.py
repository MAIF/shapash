"""Confusion-matrix heatmap for NLP error analysis."""

from __future__ import annotations

import numpy as np
from plotly import graph_objs as go

# Plotly's own continuous "Blues" scale, independent of the app's active NLP palette: the
# discrete, hand-picked stops in colors.json's "blues"/"default" palettes band visibly on a
# heatmap, and under the yellow "default" palette low off-diagonal counts (the errors this
# matrix exists to surface) rendered as near-invisible pale yellow. Plotly's built-in scale reads
# cleanly regardless of which palette the rest of the app is themed with.
_DEFAULT_COLORSCALE = "Blues"


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    normalize: str | None = None,
    title: str = "Confusion matrix",
    width: int | None = None,
    height: int | None = None,
    colorscale: list[str] | str = _DEFAULT_COLORSCALE,
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
    colorscale : list of str or str
        Plotly heatmap colorscale. Defaults to Plotly's built-in ``"Blues"``, independent of the
        app's active NLP palette — the errors this matrix exists to surface need to stay readable
        whichever palette the rest of the app is themed with.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    counts = np.asarray(cm, dtype=float)
    n = counts.shape[0]

    if normalize == "true":
        row_sums = counts.sum(axis=1, keepdims=True)
        # Guard against empty true-classes (row sum 0) producing NaNs.
        z = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
        text = np.array([[f"{v:.0%}" for v in row] for row in z])
        hover_val = "%{z:.1%}"
        colorbar_title = "recall"
    else:
        z = counts
        text = np.array([[f"{int(v)}" for v in row] for row in counts])
        hover_val = "%{z:.0f}"
        colorbar_title = "count"

    # customdata[true][pred] = [pred_idx, true_idx] — the order a click handler reads.
    customdata = np.empty((n, n, 2), dtype=int)
    for i in range(n):
        for j in range(n):
            customdata[i, j] = (j, i)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            customdata=customdata,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title),
            hovertemplate=("true: %{y}<br>predicted: %{x}<br>" + hover_val + "<extra></extra>"),
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
        # Put the first true class at the top so the diagonal reads top-left to bottom-right.
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(automargin=True, side="top"),
        margin=dict(l=20, r=20, t=top_margin, b=20),
    )

    return fig
