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

import unittest

import numpy as np
import pandas as pd
import plotly.io as pio

from shapash.plots.plot_contribution import plot_scatter, plot_violin
from shapash.style.style_utils import define_style, get_palette


class TestPlotContributionCustomdata(unittest.TestCase):
    def setUp(self):
        self.style_dict = define_style(get_palette("default"))

    def _assert_all_customdata_are_plain_lists(self, fig):
        raw = pio.to_json(fig)
        import json

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


if __name__ == "__main__":
    unittest.main()
