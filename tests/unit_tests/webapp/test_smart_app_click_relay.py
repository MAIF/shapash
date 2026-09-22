"""
Fast, isolated unit tests for the index_id/validation click-relay callbacks in
shapash.webapp.smart_app.

Regression coverage for the second half of the incident where clicking a point
on the Feature Contribution plot did not update the Local Explanation panel:
`update_index_id` and `click_validation` relayed the selected row through
`index_id.n_submit` / `validation.n_clicks` using constant sentinel values
(``return selected, True`` / ``return 1``). A Dash `Input` only re-triggers a
downstream callback when its value actually *changes*, so a correctly-read
click only refreshed the Local Explanation once per session - every click
after the first was silently dropped. Both callbacks must now return a
counter that increments on every trigger.

These tests drive the two callbacks directly through the app's Flask test
client, hitting the real `/_dash-update-component` route (so `dash.
callback_context` is set up exactly as in production) without starting a
live WSGI server - unlike
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


class TestSmartAppClickRelay(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 20
        x_df = pd.DataFrame({"feat": rng.normal(size=n)})
        y_pred = pd.DataFrame({"pred": rng.normal(size=n)})
        model = DecisionTreeRegressor(max_depth=2).fit(x_df, y_pred)
        contributions = pd.DataFrame(rng.normal(size=(n, 1)), columns=x_df.columns, index=x_df.index)

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


if __name__ == "__main__":
    unittest.main()
