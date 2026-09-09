"""2-D scatter of samples (e.g. embeddings projected to a plane), for the NLP explanation.

Coloring has two independent modes that do not share a colorbar/legend, hence the two branches
below rather than one code path parameterized by color: *categorical* groups points into one
trace per class (predicted or ground truth) so Plotly's legend and box/lasso selection work
correctly across every trace, while *word contribution* is a single diverging-colorscale overlay
plus a gray "absent" layer, since the quantity being colored (a signed SHAP sum) has nothing to do
with class membership.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objs as go

# Qualitative palette for the categorical mode — one color per class, cycling if there are more
# classes than colors. Independent of the signed positive/negative theme in
# ``shapash.style.style_utils`` (``DEFAULT_NLP_THEME.xpl_positive``/``xpl_negative``), which colors a
# *contribution*, not a *class*.
DEFAULT_CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

_LAYOUT_COMMON = dict(
    dragmode="select",
    uirevision="scatter",
    xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
    yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#e5e5e5", zeroline=False, title=""),
    plot_bgcolor="#f9f9f9",
    paper_bgcolor="white",
    margin=dict(l=10, r=10, t=10, b=10),
    autosize=True,
)


def _emphasize_errors(
    idx_arr: np.ndarray,
    base_opacity: float,
    base_size: float,
    error_mask: np.ndarray | None,
) -> tuple[list[float] | float, list[float] | float]:
    """Per-point opacity/size that pops model errors and shadows everything else.

    Keeps the caller's coloring untouched — only opacity and marker size change — so points stay
    grouped/colored however the categorical/word-contribution branch already draws them. Returns
    scalars (the unmodified base values) when ``error_mask`` is unavailable.
    """
    if error_mask is None or len(idx_arr) == 0:
        return base_opacity, base_size
    is_error = error_mask[idx_arr]
    opacity = np.where(is_error, max(base_opacity, 0.9), 0.12).tolist()
    size = np.where(is_error, base_size + 3, max(base_size - 2, 3)).tolist()
    return opacity, size


def _short_texts(texts: Sequence[str], max_len: int = 120) -> list[str]:
    return [(t[:max_len] + "…") if len(t) > max_len else t for t in texts]


def plot_scatter(
    xy: np.ndarray,
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    label_names: Sequence[str] | None = None,
    contributions: np.ndarray | None = None,
    colorbar_title: str | None = None,
    error_mask: np.ndarray | None = None,
    palette: Sequence[str] = DEFAULT_CATEGORICAL_PALETTE,
    use_webgl: bool = True,
) -> go.Figure:
    """2-D scatter, colored either by class or by a word's SHAP contribution.

    Exactly one of ``contributions`` or ``labels`` decides the mode. ``customdata`` on every trace
    always carries the point's original position in ``xy``/``texts``, so a caller wiring
    ``clickData``/``selectedData`` back to sample indices does not need to know which mode drew
    the figure.

    Parameters
    ----------
    xy : np.ndarray, shape (n_samples, 2)
        2-D coordinates (e.g. a UMAP/PaCMAP/PCA projection of embeddings).
    texts : Sequence[str]
        One text per sample, shown (truncated to 120 chars) in the hover.
    labels : Sequence[str], optional
        Categorical mode: one class label per sample (e.g. predictions or ground truth).
        Mutually exclusive with ``contributions``.
    label_names : Sequence[str], optional
        Categorical mode: the categories to draw, in legend order — not necessarily
        ``set(labels)``, so a caller can pass a fixed class ordering (and colors stay stable) even
        when a class has zero points in this batch. Required when ``labels`` is given.
    contributions : np.ndarray, shape (n_samples,), optional
        Word-contribution mode: signed SHAP sum per sample. Points with ``0`` are drawn as a gray
        "absent" context layer; the rest as a diverging-colorscale overlay. Mutually exclusive
        with ``labels``.
    colorbar_title : str, optional
        Word-contribution mode: label for the colorbar (e.g. the word(s) being colored by).
    error_mask : np.ndarray of bool, shape (n_samples,), optional
        ``True`` where the model got the sample wrong. When given, error points are drawn larger
        and opaque while the rest are shrunk and faded — coloring is untouched either way.
    palette : Sequence[str]
        Categorical mode: colors assigned to ``label_names`` in order, cycling if there are more
        classes than colors. Defaults to :data:`DEFAULT_CATEGORICAL_PALETTE`.
    use_webgl : bool
        Draw with ``Scattergl`` (default) for smooth interaction up to several thousand points, as
        the Dash webapp needs. Some notebook front-ends (e.g. VSCode's notebook renderer over
        certain remote/SSH or sandboxed setups) can't get a WebGL context and raise "WebGL is not
        supported by your browser" — pass ``False`` there to fall back to the plain SVG ``Scatter``.

    Returns
    -------
    go.Figure

    Raises
    ------
    ValueError
        If neither or both of ``contributions``/``labels`` are given, or ``labels`` is given
        without ``label_names``.
    """
    if (contributions is None) == (labels is None):
        raise ValueError("Pass exactly one of `contributions` (word mode) or `labels` (categorical mode).")

    texts_short = _short_texts(texts)
    scatter_cls = go.Scattergl if use_webgl else go.Scatter

    if contributions is not None:
        return _plot_word_contribution(xy, texts_short, contributions, colorbar_title, error_mask, scatter_cls)
    if labels is None or label_names is None:
        raise ValueError("`label_names` is required alongside `labels`.")
    return _plot_categorical(xy, texts_short, list(labels), list(label_names), error_mask, palette, scatter_cls)


def _plot_word_contribution(
    xy: np.ndarray,
    texts_short: list[str],
    contributions: np.ndarray,
    colorbar_title: str | None,
    error_mask: np.ndarray | None,
    scatter_cls: type[go.Scatter] | type[go.Scattergl],
) -> go.Figure:
    max_abs = float(np.abs(contributions).max()) or 1.0
    present_mask = np.where(contributions != 0.0)[0]
    absent_mask = np.where(contributions == 0.0)[0]

    fig = go.Figure()
    # Both layers are ALWAYS added (even when a mask is empty) so the trace structure stays constant
    # across word additions/removals. With a stable `uirevision`, a changing trace count leaves
    # ghost/"shadow" points from the previous render — keeping exactly two traces avoids that.
    absent_opacity, absent_size = _emphasize_errors(absent_mask, 0.35, 5, error_mask)
    present_opacity, present_size = _emphasize_errors(present_mask, 0.9, 9, error_mask)

    fig.add_trace(
        scatter_cls(
            x=xy[absent_mask, 0].tolist(),
            y=xy[absent_mask, 1].tolist(),
            mode="markers",
            marker=dict(color="#b0b0b0", size=absent_size, opacity=absent_opacity),
            customdata=absent_mask.reshape(-1, 1).tolist(),
            text=[texts_short[j] for j in absent_mask],
            hovertemplate="%{text}<extra>absent</extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        scatter_cls(
            x=xy[present_mask, 0].tolist(),
            y=xy[present_mask, 1].tolist(),
            mode="markers",
            marker=dict(
                color=contributions[present_mask].tolist(),
                colorscale="RdBu",
                cmin=-max_abs,
                cmax=max_abs,
                size=present_size,
                opacity=present_opacity,
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right"),
                    thickness=12,
                    tickformat=".2f",
                ),
            ),
            customdata=present_mask.reshape(-1, 1).tolist(),
            text=[texts_short[j] for j in present_mask],
            hovertemplate="<b>SHAP: %{marker.color:.3f}</b><br>%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(**_LAYOUT_COMMON, showlegend=False)
    return fig


def _plot_categorical(
    xy: np.ndarray,
    texts_short: list[str],
    labels: list[str],
    label_names: list[str],
    error_mask: np.ndarray | None,
    palette: Sequence[str],
    scatter_cls: type[go.Scatter] | type[go.Scattergl],
) -> go.Figure:
    fig = go.Figure()
    for i, name in enumerate(label_names):
        mask = [j for j, lbl in enumerate(labels) if lbl == name]
        if not mask:
            continue
        mask_arr = np.array(mask)
        opacity, size = _emphasize_errors(mask_arr, 0.75, 7, error_mask)
        fig.add_trace(
            scatter_cls(
                x=xy[mask_arr, 0].tolist(),
                y=xy[mask_arr, 1].tolist(),
                mode="markers",
                marker=dict(color=palette[i % len(palette)], size=size, opacity=opacity),
                customdata=mask_arr.reshape(-1, 1).tolist(),
                text=[texts_short[j] for j in mask],
                hovertemplate=f"<b>{name}</b><br>%{{text}}<extra></extra>",
                name=name,
            )
        )
    fig.update_layout(
        **_LAYOUT_COMMON,
        legend=dict(itemsizing="constant", orientation="v", title_text=""),
        showlegend=True,
    )
    return fig
