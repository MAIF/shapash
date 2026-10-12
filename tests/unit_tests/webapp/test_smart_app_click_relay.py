"""
Dash callback tests for click relay, filtering, and feature-importance navigation
in shapash.webapp.smart_app.

Regression coverage for the second half of the incident where clicking a point
on the Feature Contribution plot did not update the Local Explanation panel:
`update_index_id` and `click_validation` relayed the selected row through
`index_id.n_submit` / `validation.n_clicks` using constant sentinel values
(``return selected, True`` / ``return 1``). A Dash `Input` only re-triggers a
downstream callback when its value actually *changes*, so a correctly-read
click only refreshed the Local Explanation once per session - every click
after the first was silently dropped. Both callbacks must now return a
counter that increments on every trigger.

These tests drive callbacks directly through the app's Flask test client,
hitting the real `/_dash-update-component` route (so `dash.callback_context`
is set up exactly as in production) without starting a live WSGI server - unlike
tests/integration_tests/test_webapp_click_updates_local_explanation.py, which
exercises the full click -> index -> refresh chain (including the sibling
customdata-serialization fix) against a real running server.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp


class TestSmartAppCallbacks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 20
        x_df = pd.DataFrame({"feat": rng.normal(size=n), "other_feat": rng.normal(size=n)})
        y_pred = pd.DataFrame({"pred": rng.normal(size=n)})
        model = DecisionTreeRegressor(max_depth=2).fit(x_df, y_pred)
        contributions = pd.DataFrame(rng.normal(size=(n, 2)), columns=x_df.columns, index=x_df.index)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=x_df, contributions=contributions, y_pred=y_pred)

        self.smart_app = SmartApp(xpl.explainer)
        self.client = self.smart_app.server.test_client()

    def _post(self, output, outputs, inputs, state, changed):
        payload = {"output": output, "outputs": outputs, "inputs": inputs, "state": state, "changedPropIds": changed}
        response = self.client.post("/_dash-update-component", json=payload)
        return response.get_json()["response"]

    def _click_feature_point(self, row_index, current_n_submit):
        outputs = [{"id": "index_id", "property": "value"}, {"id": "index_id", "property": "n_submit"}]
        inputs = [
            {
                "id": "feature_selector",
                "property": "clickData",
                "value": {"points": [{"customdata": [0.0, row_index]}]},
            },
            {"id": "prediction_picking", "property": "clickData", "value": None},
            {"id": "clusters", "property": "clickData", "value": None},
            {"id": "dataset", "property": "active_cell", "value": None},
            {"id": "apply_filter", "property": "n_clicks", "value": None},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"index": ["ALL"], "type": "del_dropdown_button"}, "property": "n_clicks", "value": []},
        ]
        state = [
            {"id": "dataset", "property": "data", "value": []},
            {"id": "dataset", "property": "derived_viewport_data", "value": None},
            {"id": "index_id", "property": "value", "value": None},
            {"id": "index_id", "property": "n_submit", "value": current_n_submit},
        ]
        resp = self._post(
            "..index_id.value...index_id.n_submit..", outputs, inputs, state, ["feature_selector.clickData"]
        )
        return resp["index_id"]["value"], resp["index_id"]["n_submit"]

    def _submit_validation(self, n_submit, current_n_clicks):
        outputs = {"id": "validation", "property": "n_clicks"}
        inputs = [{"id": "index_id", "property": "n_submit", "value": n_submit}]
        state = [{"id": "validation", "property": "n_clicks", "value": current_n_clicks}]
        resp = self._post("validation.n_clicks", outputs, inputs, state, ["index_id.n_submit"])
        return resp["validation"]["n_clicks"]

    def test_update_index_id_n_submit_increments_on_every_click(self):
        _, n_submit_1 = self._click_feature_point(row_index=3, current_n_submit=None)
        _, n_submit_2 = self._click_feature_point(row_index=7, current_n_submit=n_submit_1)
        _, n_submit_3 = self._click_feature_point(row_index=11, current_n_submit=n_submit_2)

        self.assertEqual([n_submit_1, n_submit_2, n_submit_3], [1, 2, 3])

    def test_update_index_id_selects_the_clicked_row_on_every_click(self):
        index_1, n_submit_1 = self._click_feature_point(row_index=3, current_n_submit=None)
        index_2, _ = self._click_feature_point(row_index=7, current_n_submit=n_submit_1)

        self.assertEqual([index_1, index_2], [3, 7])

    def test_click_validation_n_clicks_increments_on_every_relayed_submit(self):
        n_clicks_1 = self._submit_validation(n_submit=1, current_n_clicks=None)
        n_clicks_2 = self._submit_validation(n_submit=2, current_n_clicks=n_clicks_1)
        n_clicks_3 = self._submit_validation(n_submit=3, current_n_clicks=n_clicks_2)

        self.assertEqual([n_clicks_1, n_clicks_2, n_clicks_3], [1, 2, 3])

    def test_reset_filter_button_follows_filter_changes(self):
        outputs = {"id": "reset_dropdown_button", "property": "disabled"}
        inputs = [
            {"id": "add_dropdown_button", "property": "n_clicks", "value": 1},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"type": "del_dropdown_button", "index": ["ALL"]}, "property": "n_clicks", "value": [None]},
        ]
        for changed, expected in [
            ("add_dropdown_button.n_clicks", False),
            ("reset_dropdown_button.n_clicks", True),
            ('{"index":0,"type":"del_dropdown_button"}.n_clicks', False),
        ]:
            result = self._post("reset_dropdown_button.disabled", outputs, inputs, [], [changed])
            self.assertIs(result["reset_dropdown_button"]["disabled"], expected)

    def test_filter_controls_can_be_added_removed_and_reset(self):
        outputs = {"id": "dropdowns_container", "property": "children"}
        inputs = [
            {"id": "add_dropdown_button", "property": "n_clicks", "value": 1},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"type": "del_dropdown_button", "index": ["ALL"]}, "property": "n_clicks", "value": [None]},
        ]
        initial = [{"props": {"id": {"type": "bloc_div", "index": 0}, "children": []}, "type": "Div"}]
        state = [
            {"id": "dropdowns_container", "property": "children", "value": initial},
            {"id": "name", "property": "value", "value": None},
        ]
        added = self._post("dropdowns_container.children", outputs, inputs, state, ["add_dropdown_button.n_clicks"])
        children = added["dropdowns_container"]["children"]
        self.assertEqual(len(children), 2)
        self.assertEqual(children[1]["props"]["id"], {"type": "bloc_div", "index": 1})

        state[0]["value"] = children
        removed = self._post(
            "dropdowns_container.children",
            outputs,
            inputs,
            state,
            ['{"index":1,"type":"del_dropdown_button"}.n_clicks'],
        )
        self.assertEqual(removed["dropdowns_container"]["children"], initial)

        reset = self._post("dropdowns_container.children", outputs, inputs, state, ["reset_dropdown_button.n_clicks"])
        self.assertEqual(reset["dropdowns_container"]["children"][0]["props"]["id"]["index"], 0)

    def test_datatable_applies_and_resets_numeric_filter(self):
        outputs = [
            {"id": "dataset", "property": "data"},
            {"id": "dataset", "property": "tooltip_data"},
            {"id": "dataset", "property": "columns"},
            {"id": "filtered_subset_info", "property": "children"},
            {"id": "filtered_subset_info", "property": "color"},
        ]
        inputs = [
            {"id": "prediction_picking", "property": "selectedData", "value": None},
            {"id": "clusters", "property": "selectedData", "value": None},
            {"id": "modal", "property": "is_open", "value": False},
            {"id": "apply_filter", "property": "n_clicks", "value": 1},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"type": "del_dropdown_button", "index": ["ALL"]}, "property": "n_clicks", "value": [None]},
        ]
        state = [
            {"id": "rows", "property": "value", "value": 20},
            {"id": "name", "property": "value", "value": None},
            {"id": {"type": "var_dropdown", "index": ["ALL"]}, "property": "value", "value": ["feat"]},
            {"id": {"type": "var_dropdown", "index": ["ALL"]}, "property": "id", "value": [{"index": 1}]},
            {"id": {"type": "dynamic-str", "index": ["ALL"]}, "property": "value", "value": []},
            {"id": {"type": "dynamic-str", "index": ["ALL"]}, "property": "id", "value": []},
            {"id": {"type": "dynamic-bool", "index": ["ALL"]}, "property": "value", "value": []},
            {"id": {"type": "dynamic-bool", "index": ["ALL"]}, "property": "id", "value": []},
            {"id": {"type": "dynamic-date", "index": ["ALL"]}, "property": "start_date", "value": []},
            {"id": {"type": "dynamic-date", "index": ["ALL"]}, "property": "end_date", "value": []},
            {"id": {"type": "dynamic-date", "index": ["ALL"]}, "property": "id", "value": []},
            {"id": {"type": "lower", "index": ["ALL"]}, "property": "value", "value": [0]},
            {"id": {"type": "lower", "index": ["ALL"]}, "property": "id", "value": [{"index": 1}]},
            {"id": {"type": "upper", "index": ["ALL"]}, "property": "value", "value": [100]},
        ]
        output = "..dataset.data...dataset.tooltip_data...dataset.columns...filtered_subset_info.children...filtered_subset_info.color.."
        filtered = self._post(output, outputs, inputs, state, ["apply_filter.n_clicks"])
        filtered_rows = filtered["dataset"]["data"]
        self.assertTrue(filtered_rows)
        self.assertTrue(all(row["feat"] >= 0 for row in filtered_rows))
        self.assertLess(len(filtered_rows), len(self.smart_app.round_dataframe))
        self.assertIn(f"Subset length: {len(filtered_rows)}", filtered["filtered_subset_info"]["children"])

        reset = self._post(output, outputs, inputs, state, ["reset_dropdown_button.n_clicks"])
        self.assertEqual(len(reset["dataset"]["data"]), len(self.smart_app.round_dataframe))
        self.assertIsNone(reset["filtered_subset_info"]["children"])

    def test_feature_importance_updates_for_dataset_and_page_navigation(self):
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
            {"id": "dataset", "property": "data", "value": self.smart_app.round_dataframe.to_dict("records")},
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
            {"id": "features", "property": "value", "value": 1},
            {"id": "clickdata-store", "property": "data", "value": None},
            {"id": "selected-clickdata-store", "property": "data", "value": None},
            {"id": "page_feature_importance", "property": "children", "value": 1},
        ]
        output = (
            "..global_feature_importance.figure...global_feature_importance.clickData...clickdata-store.data"
            "...selected-clickdata-store.data...page_feature_importance.children...total_pages.children"
            "...page_viewer_feature_importance.style...goback_feature_importance.style.."
        )
        initial = self._post(output, outputs, inputs, state, ["dataset.data"])
        self.assertTrue(initial["global_feature_importance"]["figure"]["data"])
        self.assertEqual(initial["page_feature_importance"]["children"], 1)

        next(item for item in inputs if item["id"] == "page_right")["value"] = 1
        next_page = self._post(output, outputs, inputs, state, ["page_right.n_clicks"])
        self.assertEqual(next_page["page_feature_importance"]["children"], 2)


if __name__ == "__main__":
    unittest.main()
