"""Estimated label-noise matrix heatmap.

Sibling of :mod:`~shapash.plots.plot_confusion_matrix`, and deliberately *not* a mode of it. The two
render the same shape but answer different questions and need different axes, hover text and number
formatting:

* a confusion matrix is **true vs. predicted**, in counts — how the model behaves;
* a noise matrix is **given vs. estimated-true**, as a fraction of the whole corpus — how the labels
  behave.

Folding the second into the first would mean an axis reading "Predicted class" over a column that
means "estimated true class", which is exactly the sort of quietly-wrong label the panel exists to
find.
"""

from __future__ import annotations

import numpy as np
from plotly import graph_objs as go

_COLORSCALE = "Blues"


def plot_noise_matrix(
    noise_matrix: np.ndarray,
    labels: list[str],
    mask_diagonal: bool = True,
    title: str = "Estimated label noise",
    width: int | None = None,
    height: int | None = None,
) -> go.Figure:
    """Heatmap of an estimated label-noise joint distribution.

    Parameters
    ----------
    noise_matrix : numpy.ndarray
        Joint distribution of shape ``(n_classes, n_classes)`` summing to 1, where
        ``noise_matrix[i, j]`` is the estimated fraction of the corpus labelled ``i`` but truly
        ``j`` — :attr:`~shapash.compute.diagnostics.label_noise.LabelNoiseReport.noise_matrix`.
    labels : list of str
        Class display names, ordered to match the rows/columns.
    mask_diagonal : bool, optional
        Blank the diagonal (default). The correctly-labelled mass is normally well over 90% of the
        joint, so leaving it in flattens every off-diagonal cell to the same near-white shade and
        hides the only part of the matrix anyone reads it for. The diagonal is not information the
        caller loses — it is ``1 - noise_rate``.
    title : str
        Figure title.
    width : int, optional
        Figure width in pixels. Defaults to a size scaled to the number of classes.
    height : int, optional
        Figure height in pixels. Defaults to a size scaled to the number of classes.

    Returns
    -------
    plotly.graph_objs.Figure
    """
    joint = np.asarray(noise_matrix, dtype=float).copy()
    n = joint.shape[0]
    if mask_diagonal:
        # NaN rather than 0 so the cells render blank instead of as a "genuinely zero noise" colour.
        np.fill_diagonal(joint, np.nan)

    text = np.array([[("" if np.isnan(v) else f"{v:.1%}") for v in row] for row in joint])

    fig = go.Figure(
        go.Heatmap(
            z=joint,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale=_COLORSCALE,
            colorbar=dict(title="% of corpus", tickformat=".1%"),
            hovertemplate=("labelled: %{y}<br>probably: %{x}<br>%{z:.2%} of corpus<extra></extra>"),
        )
    )

    size = max(360, 90 * n + 120)
    top_margin = 80 if title else 40
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Estimated true class",
        yaxis_title="Given label",
        width=width or size,
        height=height or size,
        plot_bgcolor="white",
        # First class at the top so the (masked) diagonal reads top-left to bottom-right, matching
        # the confusion matrix next door.
        yaxis=dict(autorange="reversed", automargin=True),
        xaxis=dict(automargin=True, side="top"),
        margin=dict(l=20, r=20, t=top_margin, b=20),
    )
    return fig
