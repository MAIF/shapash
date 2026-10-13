"""
Unit tests for shapash.plots.plot_contribution.

Regression coverage for the incident where clicking a point on the Feature
Contribution plot did not update the Local Explanation panel. Root cause:
customdata built from a numpy array is serialized by Plotly>=6 into a binary
blob ({"dtype", "bdata", "shape"}) instead of a plain per-point JSON list.
Plotly.js cannot read a per-point value out of that binary form, so
`clickData.points[0].customdata` is silently missing in the browser and the
webapp callback falls back to "nothing selected". See plot_scatter and
_add_violin_and_scatter, which now call `.tolist()` before handing customdata
to a trace.
"""

import json
import unittest

import numpy as np
import pandas as pd
import plotly.io as pio

from shapash.plots.plot_contribution import (
    plot_interactions_scatter,
    plot_interactions_violin,
    plot_scatter,
    plot_violin,
)
from shapash.style.style_utils import define_style, get_palette


class TestPlotContributionCustomdata(unittest.TestCase):
    def setUp(self):
        self.style_dict = define_style(get_palette("default"))

    def _assert_all_customdata_are_plain_lists(self, fig):
        raw = pio.to_json(fig)
        traces = json.loads(raw)["data"]
        found_customdata = False
        for trace in traces:
            customdata = trace.get("customdata")
            if customdata is None:
                continue
            found_customdata = True
            self.assertIsInstance(
                customdata,
                list,
                msg=(
                    f"customdata was serialized as {type(customdata)} instead of a plain "
                    "list. Plotly.js cannot recover a per-point value from a binary-encoded "
                    "array, so clicking a point loses its row index in the browser."
                ),
            )
        self.assertTrue(found_customdata, "No trace with customdata found in the figure")

    def test_plot_scatter_customdata_is_json_list(self):
        """Numeric feature contribution plot (plot_scatter) must keep customdata as a plain list."""
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        feature_values = pd.DataFrame({"num_feat": rng.normal(size=50)}, index=index)
        contributions = pd.DataFrame({"contribution": rng.normal(size=50)}, index=index)

        fig = plot_scatter(
            feature_values=feature_values,
            contributions=contributions,
            feature_name="num_feat",
            case="regression",
            style_dict=self.style_dict,
        )
        self._assert_all_customdata_are_plain_lists(fig)

    def test_plot_violin_customdata_is_json_list(self):
        """Categorical feature contribution plot (plot_violin) must keep customdata as a plain list."""
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        feature_values = pd.DataFrame({"cat_feat": rng.integers(0, 4, size=50)}, index=index)
        contributions = pd.DataFrame({"contribution": rng.normal(size=50)}, index=index)

        fig = plot_violin(
            feature_values=feature_values,
            contributions=contributions,
            feature_name="cat_feat",
            case="regression",
            style_dict=self.style_dict,
        )
        self._assert_all_customdata_are_plain_lists(fig)

    def test_plot_interactions_scatter_customdata_is_json_list(self):
        """Interaction scatter plot must keep customdata as a plain list."""
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        x_values = pd.DataFrame({"feat_x": rng.normal(size=50)}, index=index)
        y_values = pd.DataFrame({"interaction": rng.normal(size=50)}, index=index)
        col_values = pd.DataFrame({"feat_color": rng.normal(size=50)}, index=index)

        fig = plot_interactions_scatter(
            x_name="feat_x",
            y_name="interaction",
            col_name="feat_color",
            x_values=x_values,
            y_values=y_values,
            col_values=col_values,
            col_scale=self.style_dict["interactions_col_scale"],
            style_dict=self.style_dict,
        )
        self._assert_all_customdata_are_plain_lists(fig)

    def test_plot_interactions_violin_customdata_is_json_list(self):
        """Interaction violin overlay scatter must keep customdata as a plain list."""
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        x_values = pd.DataFrame({"feat_x": rng.integers(0, 4, size=50)}, index=index)
        y_values = pd.DataFrame({"interaction": rng.normal(size=50)}, index=index)
        col_values = pd.DataFrame({"feat_color": rng.normal(size=50)}, index=index)

        fig = plot_interactions_violin(
            x_name="feat_x",
            y_name="interaction",
            col_name="feat_color",
            x_values=x_values,
            y_values=y_values,
            col_values=col_values,
            col_scale=self.style_dict["interactions_col_scale"],
            style_dict=self.style_dict,
        )
        self._assert_all_customdata_are_plain_lists(fig)


class TestPlotViolinGrid(unittest.TestCase):
    """
    Regression coverage for the incident where violin contribution plots lost
    their horizontal background grid. Root cause: plot_violin draws on a
    secondary y-axis (yaxis2, "overlaying" the hidden primary yaxis used for
    the background distribution histogram). Plotly.js v2/v3 (bundled with
    plotly<6) drew grid lines for an overlaying axis by falling back to the
    anchor axis's schema default (showgrid=True); Plotly.js v4 (bundled with
    plotly>=7) does not, so the grid silently disappeared with no code change
    on our side. Fix: explicitly keep the anchor yaxis "visible" (it draws the
    grid) while hiding its ticks/line/title, since it is otherwise only a
    layout helper axis with no data of its own.
    """

    def setUp(self):
        self.style_dict = define_style(get_palette("default"))

    def test_violin_yaxis_grid_is_enabled_and_no_title_leaks(self):
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        feature_values = pd.DataFrame({"cat_feat": rng.integers(0, 4, size=50)}, index=index)
        contributions = pd.DataFrame({"contribution": rng.normal(size=50)}, index=index)

        fig = plot_violin(
            feature_values=feature_values,
            contributions=contributions,
            feature_name="cat_feat",
            case="regression",
            style_dict=self.style_dict,
        )

        self.assertTrue(
            fig.layout.yaxis.showgrid,
            "yaxis.showgrid must be True - this is the axis that actually draws the "
            "violin plot's horizontal grid lines, even though it carries no visible "
            "ticks or labels of its own.",
        )
        self.assertTrue(fig.layout.yaxis.visible, "yaxis must stay visible=True, otherwise its grid is suppressed too")
        self.assertFalse(
            fig.layout.yaxis.showticklabels, "the hidden helper yaxis must not show tick labels"
        )
        self.assertIsNone(
            fig.layout.yaxis.title.text,
            "the hidden helper yaxis must not carry a 'Contribution' title - that belongs on yaxis2",
        )

    def test_interactions_violin_yaxis_grid_is_enabled_and_no_title_leaks(self):
        rng = np.random.default_rng(0)
        index = pd.RangeIndex(50)
        x_values = pd.DataFrame({"feat_x": rng.integers(0, 4, size=50)}, index=index)
        y_values = pd.DataFrame({"interaction": rng.normal(size=50)}, index=index)
        col_values = pd.DataFrame({"feat_color": rng.normal(size=50)}, index=index)

        fig = plot_interactions_violin(
            x_name="feat_x",
            y_name="interaction",
            col_name="feat_color",
            x_values=x_values,
            y_values=y_values,
            col_values=col_values,
            col_scale=self.style_dict["interactions_col_scale"],
            style_dict=self.style_dict,
        )

        self.assertTrue(fig.layout.yaxis.showgrid)
        self.assertTrue(fig.layout.yaxis.visible)
        self.assertFalse(fig.layout.yaxis.showticklabels)
        self.assertIsNone(fig.layout.yaxis.title.text)


if __name__ == "__main__":
    unittest.main()
