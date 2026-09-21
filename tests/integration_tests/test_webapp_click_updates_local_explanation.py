"""
Integration test reproducing the "clicking a point on the Feature Contribution
plot does not update the Local Explanation panel" incident, against a real,
running Shapash webapp (actual WSGI server, actual Dash callback dispatch).

Two independent defects combined to cause it:
  1. `customdata` built from a numpy array is serialized by Plotly>=6 as a
     binary blob instead of a plain per-point JSON list, so the browser's
     click event loses the row index entirely (fixed by `.tolist()` in
     shapash/plots/plot_contribution.py). See also
     tests/unit_tests/plots/test_plot_contribution.py for a fast, server-less
     regression test of this half.
  2. The click -> index -> refresh callback chain in shapash/webapp/smart_app.py
     relayed the selection through `index_id.n_submit` / `validation.n_clicks`
     using constant sentinel values, so a Dash `Input` never changed after the
     first click and the chain went dead for every click after that.

This test drives the three chained callbacks exactly as the browser does (HTTP
POST to /_dash-update-component) for two different features - one numeric, one
categorical, matching the original bug report - and asserts the Local
Explanation figure actually changes on every click, not just the first.
"""

import json
import socket
import time
import unittest

import numpy as np
import pandas as pd
import requests
from sklearn.tree import DecisionTreeRegressor

from shapash import SmartExplainer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _customdata_for_index(figure_json, target_index):
    """
    Finds a real (value, index) customdata pair for a specific row index, reading
    it out of the figure JSON exactly as the server sent it over the wire (i.e.
    after Dash/Plotly serialization) - not off the live Python Figure object.
    This is what actually reproduces the incident: if customdata was serialized
    as a binary blob ({"dtype", "bdata", "shape"}) instead of a plain list, no
    trace below has an iterable, index-matching customdata and this raises.
    """
    for trace in figure_json["data"]:
        customdata = trace.get("customdata")
        if customdata is None:
            continue
        assert isinstance(customdata, list), (
            f"customdata was serialized as {type(customdata)} instead of a plain list "
            f"(got: {customdata!r}); Plotly.js cannot recover a per-point value out of "
            "a binary-encoded array, so a real click loses its row index in the browser."
        )
        for point in customdata:
            if int(point[1]) == target_index:
                return list(point)
    raise AssertionError(f"No customdata point found for index {target_index}")


class TestWebappClickUpdatesLocalExplanation(unittest.TestCase):
    """Reproduces the exact user-reported incident against a live webapp instance."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n = 60
        x_df = pd.DataFrame(
            {
                "numeric_feat": rng.normal(size=n),
                "categorical_feat": rng.integers(0, 4, size=n),
            }
        )
        y_pred = pd.DataFrame({"pred": rng.normal(size=n)})
        model = DecisionTreeRegressor(max_depth=3).fit(x_df, y_pred)
        contributions = pd.DataFrame(rng.normal(size=(n, 2)), columns=x_df.columns, index=x_df.index)

        cls.xpl = SmartExplainer(model=model)
        cls.xpl.compile(x=x_df, contributions=contributions, y_pred=y_pred)

        cls.port = _free_port()
        cls.app_thread = cls.xpl.run_app(port=cls.port, host="127.0.0.1")
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls._wait_until_up()

        layout = requests.get(f"{cls.base_url}/_dash-layout", timeout=5).json()
        dataset_component = cls._find_component(layout, "dataset")
        cls.dataset_data = dataset_component["props"]["data"]
        cls.points_setting = cls._find_component(layout, "points")["props"].get("value", 200)
        cls.violin_setting = cls._find_component(layout, "violin")["props"].get("value", 10)

    @classmethod
    def tearDownClass(cls):
        cls.app_thread.kill()

    @classmethod
    def _wait_until_up(cls, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                requests.get(cls.base_url, timeout=1)
                return
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        raise RuntimeError("Webapp did not start in time")

    @staticmethod
    def _find_component(node, component_id):
        if isinstance(node, dict):
            if node.get("props", {}).get("id") == component_id:
                return node
            for value in node.get("props", {}).values():
                found = TestWebappClickUpdatesLocalExplanation._find_component(value, component_id)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = TestWebappClickUpdatesLocalExplanation._find_component(item, component_id)
                if found is not None:
                    return found
        return None

    def _post(self, output, outputs, inputs, state, changed):
        payload = {"output": output, "outputs": outputs, "inputs": inputs, "state": state, "changedPropIds": changed}
        r = requests.post(f"{self.base_url}/_dash-update-component", json=payload, timeout=5)
        r.raise_for_status()
        return json.loads(r.text)["response"]

    def _fetch_feature_selector_figure(self, feature_name):
        """
        Fetches the real Feature Contribution figure for a given feature, over
        HTTP, via the actual update_feature_selector callback - i.e. exactly the
        JSON a real browser would receive for that plot.
        """
        outputs = {"id": "feature_selector", "property": "figure"}
        inputs = [
            {
                "id": "global_feature_importance",
                "property": "clickData",
                "value": {"points": [{"label": feature_name, "curveNumber": 1}]},
            },
            {"id": "dataset", "property": "data", "value": self.dataset_data},
            {"id": "select_label", "property": "value", "value": None},
            {"id": "ember_feature_selector", "property": "n_clicks", "value": None},
        ]
        state = [
            {"id": "points", "property": "value", "value": self.points_setting},
            {"id": "violin", "property": "value", "value": self.violin_setting},
            {"id": "global_feature_importance", "property": "figure", "value": {"data": []}},
        ]
        resp = self._post(
            "feature_selector.figure", outputs, inputs, state, ["global_feature_importance.clickData"]
        )
        return resp["feature_selector"]["figure"]

    def _click_feature_point(self, customdata_point, current_n_submit):
        """Simulates a browser click on the Feature Contribution plot: update_index_id callback."""
        outputs = [{"id": "index_id", "property": "value"}, {"id": "index_id", "property": "n_submit"}]
        inputs = [
            {
                "id": "feature_selector",
                "property": "clickData",
                "value": {"points": [{"customdata": customdata_point}]},
            },
            {"id": "prediction_picking", "property": "clickData", "value": None},
            {"id": "clusters", "property": "clickData", "value": None},
            {"id": "dataset", "property": "active_cell", "value": None},
            {"id": "apply_filter", "property": "n_clicks", "value": None},
            {"id": "reset_dropdown_button", "property": "n_clicks", "value": None},
            {"id": {"index": ["ALL"], "type": "del_dropdown_button"}, "property": "n_clicks", "value": []},
        ]
        state = [
            {"id": "dataset", "property": "data", "value": self.dataset_data},
            {"id": "dataset", "property": "derived_viewport_data", "value": None},
            {"id": "index_id", "property": "value", "value": None},
            {"id": "index_id", "property": "n_submit", "value": current_n_submit},
        ]
        resp = self._post(
            "..index_id.value...index_id.n_submit..", outputs, inputs, state, ["feature_selector.clickData"]
        )
        return resp["index_id"]["value"], resp["index_id"]["n_submit"]

    def _submit_validation(self, n_submit, current_n_clicks):
        """Simulates the index_id.n_submit -> validation.n_clicks relay: click_validation callback."""
        outputs = {"id": "validation", "property": "n_clicks"}
        inputs = [{"id": "index_id", "property": "n_submit", "value": n_submit}]
        state = [{"id": "validation", "property": "n_clicks", "value": current_n_clicks}]
        resp = self._post("validation.n_clicks", outputs, inputs, state, ["index_id.n_submit"])
        return resp["validation"]["n_clicks"]

    def _refresh_local_explanation(self, n_clicks, index_value):
        """Simulates the validation.n_clicks -> Local Explanation refresh: update_detail_feature callback."""
        outputs = {"id": "detail_feature", "property": "figure"}
        inputs = [
            {"id": "threshold_id", "property": "value", "value": 0},
            {"id": "max_contrib_id", "property": "value", "value": 20},
            {"id": "check_id_positive", "property": "value", "value": True},
            {"id": "check_id_negative", "property": "value", "value": True},
            {"id": "masked_contrib_id", "property": "value", "value": []},
            {"id": "select_label", "property": "value", "value": None},
            {"id": "validation", "property": "n_clicks", "value": n_clicks},
            {"id": "bool_groups", "property": "on", "value": False},
            {"id": "ember_detail_feature", "property": "n_clicks", "value": None},
        ]
        state = [
            {"id": "index_id", "property": "value", "value": index_value},
            {"id": "dataset", "property": "data", "value": self.dataset_data},
        ]
        resp = self._post("detail_feature.figure", outputs, inputs, state, ["validation.n_clicks"])
        return resp["detail_feature"]["figure"]

    def test_two_consecutive_clicks_on_different_features_both_update_local_explanation(self):
        """
        Reproduces the reported incident: click a point on a numeric feature's
        Feature Contribution plot, then a point on a categorical feature's -
        the Local Explanation panel must refresh to the newly selected row
        both times, not just on the first click.
        """
        numeric_figure = self._fetch_feature_selector_figure("numeric_feat")
        categorical_figure = self._fetch_feature_selector_figure("categorical_feat")
        first_point = _customdata_for_index(numeric_figure, target_index=0)
        second_point = _customdata_for_index(categorical_figure, target_index=1)

        n_submit = None
        n_clicks = None
        selected_indices = []
        detail_figures = []
        for point in (first_point, second_point):
            index_value, n_submit = self._click_feature_point(point, n_submit)
            n_clicks = self._submit_validation(n_submit, n_clicks)
            detail_figure = self._refresh_local_explanation(n_clicks, index_value)

            selected_indices.append(index_value)
            detail_figures.append(detail_figure)

        self.assertEqual(selected_indices, [0, 1], "Each click must select its own row, in order")
        self.assertTrue(detail_figures[0]["data"], "First click produced an empty Local Explanation")
        self.assertTrue(detail_figures[1]["data"], "Second click produced an empty Local Explanation")
        self.assertNotEqual(
            detail_figures[0],
            detail_figures[1],
            "Local Explanation did not change between two clicks on different rows "
            "-- the panel is stuck, reproducing the reported incident",
        )


if __name__ == "__main__":
    unittest.main()
