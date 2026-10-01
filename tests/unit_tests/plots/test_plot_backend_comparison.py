"""Unit tests for the cross-backend comparison renderers."""

import numpy as np
import pytest
from dash import html
from plotly import graph_objs as go

from shapash.plots.plot_backend_comparison import (
    BACKEND_COLORS,
    plot_backend_bars,
    plot_backend_heatmap,
    plot_backend_highlight,
)

TOKENS = ["I", "love", "it", "!"]
VALUES = {"shap": np.array([0.1, 0.9, -0.2, 0.05]), "lime": np.array([0.0, 1.0, -0.1, np.nan])}


def _texts(component) -> list[str]:
    """Every string child under a Dash component, depth first."""
    children = getattr(component, "children", None)
    if isinstance(children, str):
        return [children]
    if children is None:
        return []
    if not isinstance(children, list):
        children = [children]
    return [s for child in children for s in _texts(child)]


class TestHeatmap:
    def test_one_row_per_backend_one_column_per_token(self):
        fig = plot_backend_heatmap(TOKENS, VALUES)
        heat = fig.data[0]
        assert isinstance(heat, go.Heatmap)
        assert list(heat.y) == ["shap", "lime"]
        assert np.asarray(heat.z).shape == (2, 4)
        assert list(fig.layout.xaxis.ticktext) == TOKENS

    def test_scale_is_symmetric_around_zero(self):
        heat = plot_backend_heatmap(TOKENS, VALUES).data[0]
        assert heat.zmin == -heat.zmax == -1.0

    def test_unattributed_cells_are_marked(self):
        fig = plot_backend_heatmap(TOKENS, VALUES)
        marks = fig.data[1]
        assert list(marks.x) == [3] and list(marks.y) == ["lime"]

    def test_no_marker_trace_when_everything_is_attributed(self):
        assert len(plot_backend_heatmap(TOKENS, {"shap": VALUES["shap"]}).data) == 1

    def test_repeated_tokens_keep_their_own_column(self):
        fig = plot_backend_heatmap(["a", "a"], {"x": np.array([1.0, -1.0])})
        assert list(fig.layout.xaxis.tickvals) == [0, 1]

    def test_subtitle(self):
        assert "ρ=0.5" in plot_backend_heatmap(TOKENS, VALUES, subtitle="ρ=0.5").layout.title.text

    def test_rejects_misaligned_values(self):
        with pytest.raises(ValueError, match="expected"):
            plot_backend_heatmap(TOKENS, {"shap": np.ones(3)})

    def test_rejects_no_backend(self):
        with pytest.raises(ValueError, match="empty"):
            plot_backend_heatmap(TOKENS, {})


class TestBars:
    def test_one_trace_per_backend_in_fixed_color_order(self):
        fig = plot_backend_bars(TOKENS, VALUES)
        assert [t.name for t in fig.data] == ["shap", "lime"]
        assert [t.marker.color for t in fig.data] == list(BACKEND_COLORS[:2])
        assert fig.layout.barmode == "group"

    def test_refuses_more_backends_than_colors(self):
        many = {str(i): np.zeros(len(TOKENS)) for i in range(3)}
        with pytest.raises(ValueError, match="heatmap"):
            plot_backend_bars(TOKENS, many, colors=BACKEND_COLORS[:2])


class TestHighlight:
    def test_one_row_per_backend_with_every_token(self):
        div = plot_backend_highlight(TOKENS, VALUES, subtitle="agreement")
        assert isinstance(div, html.Div)
        texts = _texts(div)
        assert "shap" in texts and "lime" in texts and "agreement" in texts
        assert texts.count("love ") == 2

    def test_unattributed_token_is_struck_through(self):
        div = plot_backend_highlight(TOKENS, VALUES)
        rows = div.children[-1].children
        lime_spans = rows[1].children[1].children
        assert lime_spans[3].style["textDecoration"] == "line-through"
        assert lime_spans[3].title == "!: not attributed"
