"""
Tests for the webapp callbacks that show individuals while the dataset is filtered.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp


class TestFilteredSubset(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 20
        x_df = pd.DataFrame({"feat": rng.normal(size=n), "other_feat": rng.normal(size=n)})
        y = pd.Series(np.arange(n) % 2, index=x_df.index, name="target")
        model = LogisticRegression().fit(x_df, y)
        contributions = pd.DataFrame(rng.normal(size=(n, 2)), columns=x_df.columns, index=x_df.index)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=x_df, contributions=contributions, y_pred=pd.DataFrame({"pred": model.predict(x_df)}))
        self.smart_app = SmartApp(xpl.explainer)
        records = self.smart_app.round_dataframe.to_dict("records")
        self.filtered_data = records[:5]
        self.filtered_out_index = records[-1]["_index_"]

    def _callback(self, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in self.smart_app.app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def _contribution_plot_indexes(self, feature, data):
        update_feature_selector = self._callback("feature_selector.figure")
        figure = update_feature_selector(feature, data, 1, None, 1000, 10)
        return {point[1] for trace in figure.data if trace.customdata is not None for point in trace.customdata}

    @staticmethod
    def _bar_click(curve_number):
        return {"points": [{"curveNumber": curve_number, "label": "feat", "pointIndex": 0}]}

    def test_contribution_plot_shows_filtered_rows_without_bar_click(self):
        indexes = self._contribution_plot_indexes(None, self.filtered_data)
        self.assertEqual(indexes, {row["_index_"] for row in self.filtered_data})

    def test_contribution_plot_shows_filtered_rows_on_subset_bar(self):
        indexes = self._contribution_plot_indexes(self._bar_click(0), self.filtered_data)
        self.assertEqual(indexes, {row["_index_"] for row in self.filtered_data})

    def test_contribution_plot_shows_whole_sample_on_global_bar(self):
        indexes = self._contribution_plot_indexes(self._bar_click(1), self.filtered_data)
        self.assertEqual(indexes, set(self.smart_app.list_index))

    def test_local_plot_shows_filtered_out_individual(self):
        update_detail_feature = self._callback("detail_feature.figure")
        figure = update_detail_feature(None, 5, [1], [1], None, 1, 1, False, None, self.filtered_out_index)
        self.assertTrue(figure.data)

    def test_local_plot_is_empty_for_unknown_index(self):
        update_detail_feature = self._callback("detail_feature.figure")
        figure = update_detail_feature(None, 5, [1], [1], None, 1, 1, False, None, -1)
        self.assertFalse(figure.data)

    def test_id_card_shows_filtered_out_individual(self):
        update_id_card = self._callback("id_card_title_contrib.children")
        style, children, _ = update_id_card(1, 1, "feature_name", True, self.filtered_out_index)
        self.assertEqual(style["display"], "flex")
        self.assertTrue(children)

    def test_id_card_is_hidden_for_unknown_index(self):
        update_id_card = self._callback("id_card_title_contrib.children")
        style, children, _ = update_id_card(1, 1, "feature_name", True, -1)
        self.assertEqual(style["display"], "none")
        self.assertEqual(children, [])


if __name__ == "__main__":
    unittest.main()
