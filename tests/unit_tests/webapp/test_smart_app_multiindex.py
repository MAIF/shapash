"""
SmartApp callbacks with a MultiIndex dataset.

Dash serializes a tuple as a JSON list, and pandas cannot look a row up from a list, so before
string row keys were introduced, clicking a row of the datatable made the local explanation, the
identity card and the datatable highlight fail with ``InvalidIndexError``. The data sent to these
callbacks goes through a JSON round trip first, as it does in the browser.
"""

import json
import unittest

import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
from sklearn.ensemble import RandomForestRegressor

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp


def _through_browser(value):
    return json.loads(json.dumps(value, cls=PlotlyJSONEncoder))


class TestSmartAppMultiIndex(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        index = pd.MultiIndex.from_product(
            [["FR", "DE"], ["Paris", "Berlin"], range(5)],
            names=["country", "city", "n"],
        )
        x_df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"], index=index)
        y = pd.Series(x_df["a"] * 2 + x_df["b"], index=index, name="y")
        model = RandomForestRegressor(n_estimators=3, random_state=0).fit(x_df, y)

        self.xpl = SmartExplainer(model=model)
        self.xpl.compile(x=x_df)
        self.smart_app = SmartApp(self.xpl.explainer)
        self.client = self.smart_app.server.test_client()
        self.data = _through_browser(self.smart_app.round_dataframe.to_dict("records"))

    def _callback(self, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in self.smart_app.app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def _post(self, output, outputs, inputs, state, changed):
        payload = {"output": output, "outputs": outputs, "inputs": inputs, "state": state, "changedPropIds": changed}
        response = self.client.post("/_dash-update-component", json=payload)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True)[-500:])
        return response.get_json()["response"]

    def _update_index_id(self, changed, value):
        outputs = [{"id": "index_id", "property": "value"}, {"id": "index_id", "property": "n_submit"}]
        trigger = changed.split(".")[0]
        inputs = [
            {
                "id": "feature_selector",
                "property": "clickData",
                "value": value if trigger == "feature_selector" else None,
            },
            {
                "id": "prediction_picking",
                "property": "clickData",
                "value": value if trigger == "prediction_picking" else None,
            },
            {"id": "clusters", "property": "clickData", "value": value if trigger == "clusters" else None},
            {"id": "dataset", "property": "active_cell", "value": value if trigger == "dataset" else None},
            {"id": "apply_filter", "property": "n_clicks", "value": None},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"index": ["ALL"], "type": "del_dropdown_button"}, "property": "n_clicks", "value": []},
        ]
        state = [
            {"id": "dataset", "property": "data", "value": self.data},
            {"id": "dataset", "property": "derived_viewport_data", "value": None},
            {"id": "index_id", "property": "value", "value": None},
            {"id": "index_id", "property": "n_submit", "value": None},
        ]
        resp = self._post("..index_id.value...index_id.n_submit..", outputs, inputs, state, [changed])
        return resp["index_id"]["value"]

    def test_index_column_holds_readable_string_keys(self):
        first_key = self.smart_app.round_dataframe.index[0]

        self.assertEqual(self.data[0]["_index_"], " | ".join(str(level) for level in first_key))

    def test_clicking_a_datatable_row_selects_its_key(self):
        selected = self._update_index_id("dataset.active_cell", {"row": 2, "column": 0})

        self.assertEqual(selected, self.data[2]["_index_"])

    def test_clicking_a_plot_point_selects_its_key(self):
        key = self.smart_app.round_dataframe.index[4]
        clicks = {
            "feature_selector.clickData": [0.5, key],
            "prediction_picking.clickData": key,
            "clusters.clickData": key,
        }
        for changed, customdata in clicks.items():
            with self.subTest(changed):
                click = _through_browser({"points": [{"customdata": customdata}]})

                self.assertEqual(self._update_index_id(changed, click), self.data[4]["_index_"])

    def test_local_explanation_for_the_selected_row(self):
        update_detail_feature = self._callback("detail_feature.figure")
        selected = self.data[2]["_index_"]

        figure = update_detail_feature(None, None, [1], [1], None, None, 1, False, None, selected, self.data)
        expected = self.xpl.plot.local_plot(index=self.smart_app.round_dataframe.index[2], show_masked=True)

        self.assertEqual([trace.x for trace in figure.data], [trace.x for trace in expected.data])

    def test_identity_card_for_the_selected_row(self):
        update_id_card = self._callback("id_card_title_contrib.children")
        selected = self.data[2]["_index_"]

        style, children, _ = update_id_card(1, None, "feature_name", True, self.data, selected)

        self.assertEqual(style["display"], "flex")
        self.assertTrue(children)

    def test_datatable_highlights_the_selected_row(self):
        properties = ["style_data_conditional", "style_filter_conditional", "style_header_conditional"]
        properties.append("style_cell_conditional")
        outputs = [{"id": "dataset", "property": prop} for prop in properties]
        inputs = [{"id": "validation", "property": "n_clicks", "value": 1}]
        state = [
            {"id": "dataset", "property": "data", "value": self.data},
            {"id": "dataset", "property": "derived_viewport_data", "value": None},
            {"id": "index_id", "property": "value", "value": self.data[2]["_index_"]},
        ]
        output = ".." + "...".join(f"dataset.{prop}" for prop in properties) + ".."

        resp = self._post(output, outputs, inputs, state, ["validation.n_clicks"])

        highlighted = resp["dataset"]["style_data_conditional"]
        self.assertIn({"if": {"row_index": 2}, "backgroundColor": self.smart_app.color[0]}, highlighted)

    def test_feature_importance_for_a_datatable_subset(self):
        outputs = [
            {"id": "global_feature_importance", "property": "figure"},
            {"id": "global_feature_importance", "property": "clickData"},
            {"id": "clickdata-store", "property": "data"},
            {"id": "selected-clickdata-store", "property": "data"},
            {"id": "page_feature_importance", "property": "children"},
            {"id": "total_pages", "property": "children"},
            {"id": "page_viewer_feature_importance", "property": "style"},
            {"id": "goback_feature_importance", "property": "style"},
        ]
        inputs = [
            {"id": "select_label", "property": "value", "value": None},
            {"id": "dataset", "property": "data", "value": self.data[:5]},
            {"id": "prediction_picking", "property": "selectedData", "value": None},
            {"id": "clusters", "property": "selectedData", "value": None},
            {"id": "apply_filter", "property": "n_clicks", "value": None},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"type": "del_dropdown_button", "index": ["ALL"]}, "property": "n_clicks", "value": []},
            {"id": "card_global_feature_importance", "property": "n_clicks", "value": None},
            {"id": "bool_groups", "property": "on", "value": False},
            {"id": "ember_global_feature_importance", "property": "n_clicks", "value": None},
            {"id": "page_left", "property": "n_clicks", "value": None},
            {"id": "page_right", "property": "n_clicks", "value": None},
            {"id": "goback_feature_importance", "property": "n_clicks", "value": None},
        ]
        state = [
            {"id": "global_feature_importance", "property": "clickData", "value": None},
            {"id": "global_feature_importance", "property": "selectedData", "value": None},
            {"id": "features", "property": "value", "value": 3},
            {"id": "clickdata-store", "property": "data", "value": None},
            {"id": "selected-clickdata-store", "property": "data", "value": None},
            {"id": "page_feature_importance", "property": "children", "value": 1},
        ]
        output = (
            "..global_feature_importance.figure...global_feature_importance.clickData...clickdata-store.data"
            "...selected-clickdata-store.data...page_feature_importance.children...total_pages.children"
            "...page_viewer_feature_importance.style...goback_feature_importance.style.."
        )
        resp = self._post(output, outputs, inputs, state, ["dataset.data"])

        # One bar trace for the whole dataset and one for the subset.
        self.assertEqual(len(resp["global_feature_importance"]["figure"]["data"]), 2)


if __name__ == "__main__":
    unittest.main()
