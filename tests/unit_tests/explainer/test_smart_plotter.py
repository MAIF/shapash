"""
Unit test smart plotter
"""

import unittest
from unittest.mock import patch

import category_encoders as ce
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapash import SmartExplainer
from shapash.backend import ShapBackend
from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState
from shapash.plots.plot_bar_chart import plot_bar_chart
from shapash.plots.plot_feature_importance import _plot_features_import
from shapash.plots.plot_line_comparison import plot_line_comparison
from shapash.style.style_utils import get_palette
from shapash.utils.check import check_model
from shapash.utils.sampling import subset_sampling


class TestSmartPlotter(unittest.TestCase):
    """
    Unit test Smart plotter class
    check the different plots available
    Parameters
    ----------
    unittest : [type]
        [description]
    """

    def predict_proba(self, arg1, X):
        """
        predict_proba method
        """
        classes = [1, 3]
        proba = 1 / len(classes)
        vect = [proba] * len(classes)
        matrx = np.repeat(np.array([vect]), X.shape[0], axis=0)

        return matrx

    def predict(self, arg1, arg2):
        """
        predict method
        """
        matrx = np.array([1, 3])
        return matrx

    def setUp(self):
        """
        SetUp
        """
        self.x_init = pd.DataFrame(
            data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        self.contrib0 = pd.DataFrame(
            data=np.array([[-3.4, 0.78], [1.2, 3.6]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        self.contrib1 = pd.DataFrame(
            data=np.array([[-0.3, 0.89], [4.7, 0.6]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        self.contrib_sorted = pd.DataFrame(
            data=np.array([[-3.4, 0.78], [3.6, 1.2]]),
            columns=["contrib_0", "contrib_1"],
            index=["person_A", "person_B"],
        )
        self.x_sorted = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]),
            columns=["feature_0", "feature_1"],
            index=["person_A", "person_B"],
        )
        self.var_dict = pd.DataFrame(
            data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=["person_A", "person_B"]
        )
        self.mask = pd.DataFrame(
            data=np.array([[True, False], [True, True]]),
            columns=["feature_0", "feature_1"],
            index=["person_A", "person_B"],
        )
        self.features_compacity = {"features_needed": [1, 1], "distance_reached": np.array([0.12, 0.16])}
        encoder = ce.OrdinalEncoder(cols=["X1"], handle_unknown="None").fit(self.x_init)
        model = CatBoostClassifier().fit(encoder.transform(self.x_init), [0, 1])
        self.model = model
        # Declare explainer object
        self.feature_dictionary = {"X1": "Education", "X2": "Age"}
        self.smart_explainer = SmartExplainer(model, features_dict=self.feature_dictionary, preprocessing=encoder)
        self.smart_explainer.data = dict()
        self.smart_explainer.data["contrib_sorted"] = self.contrib_sorted
        self.smart_explainer.data["x_sorted"] = self.x_sorted
        self.smart_explainer.data["var_dict"] = self.var_dict
        self.smart_explainer.x_encoded = encoder.transform(self.x_init)
        self.smart_explainer.x_init = self.x_init
        self.smart_explainer.postprocessing_modifications = False
        self.smart_explainer.backend = ShapBackend(model=model)
        self.smart_explainer.backend.state = MultiDecorator(SmartState())
        self.smart_explainer.explain_data = None
        # self.smart_explainer.x_contrib_plot = self.x_contrib_plot
        self.smart_explainer.columns_dict = {i: col for i, col in enumerate(self.smart_explainer.x_init.columns)}
        self.smart_explainer.inv_columns_dict = {v: k for k, v in self.smart_explainer.columns_dict.items()}
        self.smart_explainer.mask = self.mask
        self.smart_explainer.contributions = [self.contrib0, self.contrib1]
        self.smart_explainer.features_imp = None
        self.smart_explainer.model = model
        self.smart_explainer._case, self.smart_explainer._classes = check_model(model)
        self.smart_explainer.state = MultiDecorator(SmartState())
        self.smart_explainer.y_pred = None
        self.smart_explainer.proba_values = None
        self.smart_explainer.features_desc = dict(self.x_init.nunique())
        self.smart_explainer.features_compacity = self.features_compacity
        self.smart_explainer.inv_features_dict = {v: k for k, v in self.smart_explainer.features_dict.items()}

    def test_define_style_attributes(self):
        # clear style attributes
        del self.smart_explainer.plot._style_dict

        colors_dict = get_palette("default")
        self.smart_explainer.plot.define_style_attributes(colors_dict=colors_dict)

        assert hasattr(self.smart_explainer.plot, "_style_dict")
        assert len(list(self.smart_explainer.plot._style_dict.keys())) > 0

    @patch("shapash.explainer.smart_explainer.SmartExplainer.filter")
    @patch("shapash.explainer.smart_explainer.SmartExplainer._local_pred")
    def test_local_plot_1(self, local_pred, filter):
        """
        Unit test Local plot 1
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 12.88
        filter.return_value = None
        self.smart_explainer._case = "regression"
        output = self.smart_explainer.plot.local_plot(index="person_B")
        output_data = output.data

        feature_values = ["<b>Age :</b><br />27", "<b>Education :</b><br />Master"]
        contributions = [3.6, 1.2]

        bars = []
        bars.append(go.Bar(x=[contributions[1]], y=[feature_values[1]], orientation="h"))
        bars.append(go.Bar(x=[contributions[0]], y=[feature_values[0]], orientation="h"))
        expected_output = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))

        for part in list(zip(output_data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    @patch("shapash.explainer.smart_plotter.select_lines")
    def test_local_plot_2(self, select_lines):
        """
        Unit test local plot 2
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        select_lines.return_value = [0, 1]
        self.smart_explainer._case = "regression"

        with self.assertRaises(ValueError):
            condition = ""
            output = self.smart_explainer.plot.local_plot(query=condition)
            expected_output = go.Figure(
                data=go.Bar(x=[], y=[], orientation="h"), layout=go.Layout(yaxis=dict(type="category"))
            )
            assert output == expected_output

    @patch("shapash.explainer.smart_plotter.select_lines")
    def test_local_plot_3(self, select_lines):
        """
        Unit test local plot 3
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        select_lines.return_value = []
        condition = ""
        output = self.smart_explainer.plot.local_plot(query=condition)
        expected_output = go.Figure()
        assert output.data == expected_output.data
        assert (
            output.layout.annotations[0].text == "Select a valid single sample to display<br />Local Explanation plot."
        )

    @patch("shapash.explainer.smart_explainer.SmartExplainer.filter")
    @patch("shapash.explainer.smart_plotter.select_lines")
    @patch("shapash.explainer.smart_explainer.SmartExplainer._local_pred")
    def test_local_plot_4(self, local_pred, select_lines, filter):
        """
        Unit test local plot 4
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        filter.return_value = None
        local_pred.return_value = 0.58
        select_lines.return_value = ["B"]
        index = ["A", "B"]
        x_init = pd.DataFrame(data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index)
        contrib_sorted1 = pd.DataFrame(
            data=np.array([[-3.4, 0.78], [3.6, 1.2]]), columns=["contrib_0", "contrib_1"], index=index
        )
        contrib_sorted2 = pd.DataFrame(
            data=np.array([[-0.4, 0.78], [0.6, 0.2]]), columns=["contrib_0", "contrib_1"], index=index
        )
        x_sorted1 = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]), columns=["feature_0", "feature_1"], index=index
        )
        x_sorted2 = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]), columns=["feature_0", "feature_1"], index=index
        )
        var_dict1 = pd.DataFrame(data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=index)
        var_dict2 = pd.DataFrame(data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=index)
        mask1 = pd.DataFrame(
            data=np.array([[True, False], [True, True]]), columns=["feature_0", "feature_1"], index=index
        )
        mask2 = pd.DataFrame(
            data=np.array([[True, True], [True, True]]), columns=["feature_0", "feature_1"], index=index
        )
        feature_dictionary = {"X1": "Education", "X2": "Age"}
        smart_explainer_mi = SmartExplainer(model=self.model, features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["contrib_sorted"] = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["x_sorted"] = [x_sorted1, x_sorted2]
        smart_explainer_mi.data["var_dict"] = [var_dict1, var_dict2]
        smart_explainer_mi.x_init = x_init
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_init.columns)}
        smart_explainer_mi.mask = [mask1, mask2]
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]
        smart_explainer_mi.inv_features_dict = {}
        smart_explainer_mi.state = MultiDecorator(SmartState())
        condition = "index == 'B'"
        output = smart_explainer_mi.plot.local_plot(query=condition)
        feature_values = ["<b>Age :</b><br />27", "<b>Education :</b><br />Master"]
        contributions = [0.6, 0.2]
        bars = []
        bars.append(go.Bar(x=[contributions[1]], y=[feature_values[1]], orientation="h"))
        bars.append(go.Bar(x=[contributions[0]], y=[feature_values[0]], orientation="h"))
        expected_output = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y
        tit = "Local Explanation - Id: <b>B</b><br><sup>Response: <b>1</b> - Proba: <b>0.5800</b></sup>"
        assert output.layout.title.text == tit

    @patch("shapash.explainer.smart_explainer.SmartExplainer.filter")
    @patch("shapash.explainer.smart_plotter.select_lines")
    @patch("shapash.explainer.smart_explainer.SmartExplainer._local_pred")
    def test_local_plot_5(self, local_pred, select_lines, filter):
        """
        Unit test local plot 5
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 0.58
        select_lines.return_value = ["B"]
        filter.return_value = None
        index = ["A", "B"]
        x_init = pd.DataFrame(data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index)
        contrib_sorted1 = pd.DataFrame(
            data=np.array([[-3.4, 0.78], [3.6, 1.2]]), columns=["contrib_0", "contrib_1"], index=index
        )
        contrib_sorted2 = pd.DataFrame(
            data=np.array([[-0.4, 0.78], [0.6, 0.2]]), columns=["contrib_0", "contrib_1"], index=index
        )
        x_sorted1 = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]), columns=["feature_0", "feature_1"], index=index
        )
        x_sorted2 = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]), columns=["feature_0", "feature_1"], index=index
        )
        var_dict1 = pd.DataFrame(data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=index)
        var_dict2 = pd.DataFrame(data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=index)
        mask1 = pd.DataFrame(
            data=np.array([[True, False], [True, True]]), columns=["feature_0", "feature_1"], index=index
        )
        mask2 = pd.DataFrame(
            data=np.array([[False, True], [False, True]]), columns=["feature_0", "feature_1"], index=index
        )
        mask_contrib1 = pd.DataFrame(
            data=np.array([[0.0, 0.78], [0.0, 1.20]]), columns=["masked_neg", "masked_pos"], index=index
        )
        mask_contrib2 = pd.DataFrame(
            data=np.array([[0.0, 0.78], [0.0, 0.20]]), columns=["masked_neg", "masked_pos"], index=index
        )

        feature_dictionary = {"X1": "Education", "X2": "Age"}
        smart_explainer_mi = SmartExplainer(model=self.model, features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["contrib_sorted"] = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["x_sorted"] = [x_sorted1, x_sorted2]
        smart_explainer_mi.data["var_dict"] = [var_dict1, var_dict2]
        smart_explainer_mi.x_init = x_init
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_init.columns)}
        smart_explainer_mi.mask = [mask1, mask2]
        smart_explainer_mi.masked_contributions = [mask_contrib1, mask_contrib2]
        smart_explainer_mi.inv_features_dict = {}
        smart_explainer_mi.mask_params = {
            "features_to_hide": None,
            "threshold": None,
            "positive": None,
            "max_contrib": 1,
        }
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]

        smart_explainer_mi.state = MultiDecorator(SmartState())
        condition = "index == 'B'"
        output = smart_explainer_mi.plot.local_plot(query=condition)
        feature_values = ["<i>Hidden Positive Contributions</i>", "<b>Education :</b><br />Master"]
        contributions = [0.2, 0.2]
        bars = []
        for elem in list(zip(feature_values, contributions)):
            bars.append(go.Bar(x=[elem[1]], y=[elem[0]], orientation="h"))
        expected_output = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))

        assert len(expected_output.data) == len(output.data)
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y
        tit = "Local Explanation - Id: <b>B</b><br><sup>Response: <b>1</b> - Proba: <b>0.5800</b></sup>"
        assert output.layout.title.text == tit

        output2 = smart_explainer_mi.plot.local_plot(query=condition, show_masked=False)
        assert len(output2.data) == 1
        assert expected_output.data[-1].x == output2.data[0].x
        smart_explainer_mi.mask_params = {
            "features_to_hide": None,
            "threshold": None,
            "positive": True,
            "max_contrib": 1,
        }
        output3 = smart_explainer_mi.plot.local_plot(row_num=1)
        assert len(output3.data) == 2
        assert expected_output.data[-1].x == output3.data[-1].x
        assert expected_output.data[-2].x == output3.data[-2].x

    @patch("shapash.explainer.smart_explainer.SmartExplainer.filter")
    @patch("shapash.explainer.smart_plotter.select_lines")
    @patch("shapash.explainer.smart_explainer.SmartExplainer._local_pred")
    def test_local_plot_groups_features(self, local_pred, select_lines, filter):
        """
        Unit test local plot 6 for groups of features
        """
        local_pred.return_value = 0.58
        select_lines.return_value = [10]
        filter.return_value = None
        index = [3, 10]
        x_init = pd.DataFrame(
            {"X1": {3: 1, 10: 5}, "X2": {3: 42, 10: 4}, "X3": {3: 9, 10: 1}, "X4": {3: 2008, 10: 2008}}
        )
        x_init_groups = pd.DataFrame(
            {"X2": {3: 42, 10: 4}, "X4": {3: 2008, 10: 2008}, "group1": {3: -2361.80078125, 10: 2361.80078125}}
        )
        contrib_sorted1 = pd.DataFrame(
            {
                "contribution_0": {3: 0.15, 10: 0.22},
                "contribution_1": {3: -0.13, 10: -0.12},
                "contribution_2": {3: -0.03, 10: 0.02},
                "contribution_3": {3: 0.0, 10: 0.01},
            }
        )
        contrib_sorted2 = pd.DataFrame(
            {
                "contribution_0": {3: -0.15, 10: -0.22},
                "contribution_1": {3: 0.13, 10: 0.12},
                "contribution_2": {3: 0.03, 10: -0.02},
                "contribution_3": {3: -0.0, 10: -0.01},
            }
        )
        x_sorted1 = pd.DataFrame(
            {
                "feature_0": {3: 9, 10: 5},
                "feature_1": {3: 1, 10: 1},
                "feature_2": {3: 2008, 10: 2008},
                "feature_3": {3: 42, 10: 4},
            }
        )
        x_sorted2 = pd.DataFrame(
            {
                "feature_0": {3: 9, 10: 5},
                "feature_1": {3: 1, 10: 1},
                "feature_2": {3: 2008, 10: 2008},
                "feature_3": {3: 42, 10: 4},
            }
        )
        var_dict1 = pd.DataFrame(
            {
                "feature_0": {3: 2, 10: 0},
                "feature_1": {3: 0, 10: 2},
                "feature_2": {3: 3, 10: 3},
                "feature_3": {3: 1, 10: 1},
            }
        )
        var_dict2 = pd.DataFrame(
            {
                "feature_0": {3: 2, 10: 0},
                "feature_1": {3: 0, 10: 2},
                "feature_2": {3: 3, 10: 3},
                "feature_3": {3: 1, 10: 1},
            }
        )

        contrib_groups_sorted1 = pd.DataFrame(
            {
                "contribution_0": {3: 0.03, 10: 0.09},
                "contribution_1": {3: -0.03, 10: 0.02},
                "contribution_2": {3: 0.0, 10: 0.01},
            }
        )
        contrib_groups_sorted2 = pd.DataFrame(
            {
                "contribution_0": {3: -0.03, 10: -0.09},
                "contribution_1": {3: 0.03, 10: -0.02},
                "contribution_2": {3: -0.0, 10: -0.01},
            }
        )
        x_groups_sorted1 = pd.DataFrame(
            {
                "feature_0": {3: -2361.8, 10: 2361.8},
                "feature_1": {3: 2008.0, 10: 2008.0},
                "feature_2": {3: 42.0, 10: 4.0},
            }
        )
        x_groups_sorted2 = pd.DataFrame(
            {
                "feature_0": {3: -2361.8, 10: 2361.8},
                "feature_1": {3: 2008.0, 10: 2008.0},
                "feature_2": {3: 42.0, 10: 4.0},
            }
        )
        groups_var_dict1 = pd.DataFrame(
            {"feature_0": {3: 2, 10: 2}, "feature_1": {3: 1, 10: 1}, "feature_2": {3: 0, 10: 0}}
        )
        groups_var_dict2 = pd.DataFrame(
            {"feature_0": {3: 2, 10: 2}, "feature_1": {3: 1, 10: 1}, "feature_2": {3: 0, 10: 0}}
        )

        mask1 = pd.DataFrame(
            data=np.array([[True, True, False], [True, True, False]]),
            columns=["contrib_0", "contrib_1", "contrib_2"],
            index=index,
        )
        mask2 = pd.DataFrame(
            data=np.array([[True, True, False], [True, True, False]]),
            columns=["contrib_0", "contrib_1", "contrib_2"],
            index=index,
        )

        mask_contrib1 = pd.DataFrame(
            data=np.array([[-0.002, 0.0], [-0.024, 0.0]]), columns=["masked_neg", "masked_pos"], index=index
        )

        mask_contrib2 = pd.DataFrame(
            data=np.array([[0.0, 0.002], [0.0, 0.024]]), columns=["masked_neg", "masked_pos"], index=index
        )

        feature_dictionary = {
            "X1": "X1_label",
            "X2": "X2_label",
            "X3": "X3_label",
            "X4": "X4_label",
            "group1": "group1_label",
        }
        smart_explainer_mi = SmartExplainer(model=self.model, features_dict=feature_dictionary)
        smart_explainer_mi.features_groups = {"group1": ["X1", "X3"]}
        smart_explainer_mi.inv_features_dict = {
            "X1_label": "X1",
            "X2_label": "X2",
            "X3_label": "X3",
            "X4_label": "X4",
            "group1_label": "group1",
        }
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["contrib_sorted"] = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data["x_sorted"] = [x_sorted1, x_sorted2]
        smart_explainer_mi.data["var_dict"] = [var_dict1, var_dict2]

        smart_explainer_mi.data_groups = dict()
        smart_explainer_mi.data_groups["contrib_sorted"] = [contrib_groups_sorted1, contrib_groups_sorted2]
        smart_explainer_mi.data_groups["x_sorted"] = [x_groups_sorted1, x_groups_sorted2]
        smart_explainer_mi.data_groups["var_dict"] = [groups_var_dict1, groups_var_dict2]

        smart_explainer_mi.x_init = x_init
        smart_explainer_mi.x_init_groups = x_init_groups

        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_init.columns)}
        smart_explainer_mi.mask = [mask1, mask2]
        smart_explainer_mi.masked_contributions = [mask_contrib1, mask_contrib2]
        smart_explainer_mi.mask_params = {
            "features_to_hide": None,
            "threshold": None,
            "positive": None,
            "max_contrib": 2,
        }
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]

        smart_explainer_mi.state = MultiDecorator(SmartState())

        output_fig = smart_explainer_mi.plot.local_plot(row_num=1)

        assert len(output_fig.data) == 3

    @patch("shapash.explainer.smart_explainer.SmartExplainer.filter")
    @patch("shapash.explainer.smart_plotter.select_lines")
    @patch("shapash.explainer.smart_explainer.SmartExplainer._local_pred")
    def test_local_plot_multi_index(self, local_pred, select_lines, filter):
        """
        Unit test local plot multi index
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 12.78
        select_lines.return_value = [("C", "A")]
        filter.return_value = None

        index = pd.MultiIndex.from_tuples([("A", "A"), ("C", "A")], names=("col1", "col2"))

        x_init_multi_index = pd.DataFrame(
            data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index
        )

        contrib_sorted_multi_index = pd.DataFrame(
            data=np.array([[-3.4, 0.78], [3.6, 1.2]]), columns=["contrib_0", "contrib_1"], index=index
        )

        x_sorted_multi_index = pd.DataFrame(
            data=np.array([["PhD", 34], [27, "Master"]]), columns=["feature_0", "feature_1"], index=index
        )

        var_dict_multi_index = pd.DataFrame(
            data=np.array([[0, 1], [1, 0]]), columns=["feature_0", "feature_1"], index=index
        )
        mask_multi_index = pd.DataFrame(
            data=np.array([[True, False], [True, True]]), columns=["feature_0", "feature_1"], index=index
        )

        feature_dictionary = {"X1": "Education", "X2": "Age"}

        smart_explainer_mi = SmartExplainer(model=self.model, features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = contrib_sorted_multi_index
        smart_explainer_mi.data["contrib_sorted"] = contrib_sorted_multi_index
        smart_explainer_mi.data["x_sorted"] = x_sorted_multi_index
        smart_explainer_mi.data["var_dict"] = var_dict_multi_index
        smart_explainer_mi.x_init = x_init_multi_index
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_init.columns)}
        smart_explainer_mi.mask = mask_multi_index
        smart_explainer_mi._case = "regression"
        smart_explainer_mi.inv_features_dict = {}
        smart_explainer_mi.state = SmartState()
        smart_explainer_mi.y_pred = None

        condition = "index == 'person_B'"

        output = smart_explainer_mi.plot.local_plot(query=condition)

        feature_values = ["<b>Age :</b><br />27", "<b>Education :</b><br />Master"]
        contributions = [3.6, 1.2]

        bars = []
        bars.append(go.Bar(x=[contributions[1]], y=[feature_values[1]], orientation="h"))
        bars.append(go.Bar(x=[contributions[0]], y=[feature_values[0]], orientation="h"))
        expected_output = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_get_selection(self):
        """
        Unit test get selection
        """
        line = ["person_A"]
        output = self.smart_explainer.plot._get_selection(line, self.var_dict, self.x_sorted, self.contrib_sorted)
        expected_output = np.array([0, 1]), np.array(["PhD", 34]), np.array([-3.4, 0.78])
        assert len(output) == 3
        assert np.array_equal(output[0], expected_output[0])
        assert np.array_equal(output[1], expected_output[1])
        assert np.array_equal(output[2], expected_output[2])

    def test_apply_mask_one_line(self):
        """
        Unit test apply mask one line
        """
        line = ["person_A"]
        var_dict = np.array([0, 1])
        x_sorted = np.array(["PhD", 34])
        contrib_sorted = np.array([-3.4, 0.78])
        output = self.smart_explainer.plot._apply_mask_one_line(line, var_dict, x_sorted, contrib_sorted)
        expected_output = np.array([0]), np.array(["PhD"]), np.array([-3.4])
        assert len(output) == 3
        assert np.array_equal(output[0], expected_output[0])
        assert np.array_equal(output[1], expected_output[1])
        assert np.array_equal(output[2], expected_output[2])

    def test_check_masked_contributions_1(self):
        """
        Unit test check masked contributions 1
        """
        line = ["person_A"]
        var_dict = ["X1", "X2"]
        x_val = ["PhD", 34]
        contrib = [-3.4, 0.78]
        var_dict, x_val, contrib = self.smart_explainer.plot._check_masked_contributions(line, var_dict, x_val, contrib)
        expected_var_dict = ["X1", "X2"]
        expected_x_val = ["PhD", 34]
        expected_contrib = [-3.4, 0.78]
        self.assertListEqual(var_dict, expected_var_dict)
        self.assertListEqual(x_val, expected_x_val)
        self.assertListEqual(contrib, expected_contrib)

    def test_check_masked_contributions_2(self):
        """
        Unit test check masked contributions 2
        """
        line = ["person_A"]
        var_dict = ["X1", "X2"]
        x_val = ["PhD", 34]
        contrib = [-3.4, 0.78]
        self.smart_explainer.masked_contributions = pd.DataFrame(
            data=[[0.0, 2.5], [0.0, 1.6]], columns=["masked_neg", "masked_pos"], index=["person_A", "person_B"]
        )
        var_dict, x_val, contrib = self.smart_explainer.plot._check_masked_contributions(line, var_dict, x_val, contrib)
        expected_var_dict = ["X1", "X2", "Hidden Positive Contributions"]
        expected_x_val = ["PhD", 34, ""]
        expected_contrib = [-3.4, 0.78, 2.5]
        self.assertListEqual(var_dict, expected_var_dict)
        self.assertListEqual(x_val, expected_x_val)
        self.assertListEqual(contrib, expected_contrib)

    def test_plot_bar_chart_1(self):
        """
        Unit test plot bar chart 1
        """
        var_dict = ["X1", "X2"]
        x_val = ["PhD", 34]
        contributions = [-3.4, 0.78]
        bars = []
        for num, elem in enumerate(var_dict):
            bars.append(
                go.Bar(x=[contributions[num]], y=["<b>{} :</b><br />{}".format(elem, x_val[num])], orientation="h")
            )
        expected_output_fig = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))
        self.smart_explainer._case = "regression"
        fig_output = plot_bar_chart("ind", var_dict, x_val, contributions, self.smart_explainer.plot._style_dict)
        for part in list(zip(fig_output.data, expected_output_fig.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_plot_bar_chart_2(self):
        """
        Unit test plot bar chart 2
        """
        var_dict = ["X1", "X2", "Hidden Positive Contributions"]
        x_val = ["PhD", 34, ""]
        order = [3, 1, 2]
        contributions = [-3.4, 0.78, 2.5]
        ylabel = ["<b>X1 :</b><br />PhD", "<b>X2 :</b><br />34", "<i>Hidden Positive Contributions</i>"]
        self.smart_explainer.masked_contributions = pd.DataFrame()
        bars = []
        comblist = list(zip(order, contributions, ylabel))
        comblist.sort(reverse=True)
        for elem in comblist:
            bars.append(go.Bar(x=[elem[1]], y=[elem[2]], orientation="h"))
        expected_output_fig = go.Figure(data=bars, layout=go.Layout(yaxis=dict(type="category")))

        self.smart_explainer._case = "regression"
        fig_output = plot_bar_chart("ind", var_dict, x_val, contributions, self.smart_explainer.plot._style_dict)
        for part in list(zip(fig_output.data, expected_output_fig.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_contribution_plot_1(self):
        """
        Classification
        """
        col = "X1"
        output = self.smart_explainer.plot.contribution_plot(col, violin_maxf=0, proba=False)
        feature_values = self.smart_explainer.x_init[col].sort_values()
        contributions = self.smart_explainer.contributions[-1][col].loc[feature_values.index]
        expected_output = go.Scatter(
            x=feature_values,
            y=contributions,
            mode="markers",
            hovertext=[f"Id: {x}<br />" for x in feature_values.index],
        )
        assert np.array_equal(output.data[-1].x, expected_output.x)
        assert np.array_equal(output.data[-1].y, expected_output.y)
        assert len(np.unique(output.data[-1].marker.color)) == 1
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_2(self):
        """
        Regression
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.contributions = self.contrib1
        xpl._case = "regression"
        xpl.state = SmartState()
        output = xpl.plot.contribution_plot(col, violin_maxf=0)
        feature_values = xpl.x_init[col].sort_values()
        contributions = xpl.contributions[col].loc[feature_values.index]
        expected_output = go.Scatter(
            x=feature_values,
            y=contributions,
            mode="markers",
            hovertext=[f"Id: {x}" for x in feature_values.index],
        )

        assert np.array_equal(output.data[-1].x, expected_output.x)
        assert np.array_equal(output.data[-1].y, expected_output.y)
        assert np.array_equal(output.data[-1].hovertext, expected_output.hovertext)
        assert len(np.unique(output.data[-1].marker.color)) == 1
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_3(self):
        """
        Color Plot classification
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.y_pred = pd.DataFrame([0, 1], columns=["pred"], index=xpl.x_init.index)
        xpl._classes = [0, 1]
        output = xpl.plot.contribution_plot(col, violin_maxf=0, proba=False)
        feature_values = xpl.x_init[col].sort_values()
        contributions = xpl.contributions[-1][col].loc[feature_values.index]
        expected_output = go.Scatter(
            x=feature_values,
            y=contributions,
            mode="markers",
            hovertext=[
                f"Id: {x}<br />Predict: {y}"
                for x, y in zip(feature_values.index, xpl.y_pred.loc[feature_values.index].iloc[:, 0].tolist())
            ],
        )

        assert np.array_equal(output.data[-1].x, expected_output.x)
        assert np.array_equal(output.data[-1].y, expected_output.y)
        assert np.array_equal(output.data[-1].hovertext, expected_output.hovertext)
        assert len(np.unique(output.data[-1].marker.color)) == 2
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_4(self):
        """
        Regression Color Plot
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.contributions = self.contrib1
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.y_pred = pd.DataFrame([0.46989877093, 12.749302948], columns=["pred"], index=xpl.x_init.index)
        xpl.plot._tuning_round_digit()
        output = xpl.plot.contribution_plot(col, violin_maxf=0)
        feature_values = xpl.x_init[col].sort_values()
        contributions = xpl.contributions[col].loc[feature_values.index]
        expected_output = go.Scatter(
            x=feature_values,
            y=contributions,
            mode="markers",
            hovertext=[
                f"Id: {x}<br />Predict: {round(y,3)}"
                for x, y in zip(feature_values.index, xpl.y_pred.loc[feature_values.index].iloc[:, 0].tolist())
            ],
        )

        assert np.array_equal(output.data[-1].x, expected_output.x)
        assert np.array_equal(output.data[-1].y, expected_output.y)
        assert len(np.unique(output.data[-1].marker.color)) >= 2
        assert np.array_equal(output.data[-1].hovertext, expected_output.hovertext)
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_5(self):
        """
        Regression Color Plot with pred
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.contributions = pd.concat([self.contrib1] * 10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.y_pred = pd.concat([pd.DataFrame([0.46989877093, 12.749302948])] * 10, ignore_index=True)
        xpl.plot._tuning_round_digit()
        output = xpl.plot.contribution_plot(col)
        new_index = xpl.x_init[col].sort_values().index
        np_hv = np.array(
            [
                f"Id: {x}<br />Predict: {round(y,2)}"
                for x, y in zip(new_index, xpl.y_pred.loc[new_index].iloc[:, 0].tolist())
            ]
        )
        output_hovertext = np.concatenate((output.data[2].hovertext, output.data[5].hovertext), axis=0)
        nb_marker_volor = len(np.unique(output.data[2].marker.color)) + len(np.unique(output.data[5].marker.color))

        assert len(output.data) == 7
        assert output.data[0].type == "bar"
        assert output.data[1].type == "violin"
        assert output.data[2].type == "scatter"
        assert output.data[3].type == "bar"
        assert output.data[4].type == "violin"
        assert output.data[5].type == "scatter"
        assert output.data[6].type == "scatter"
        assert nb_marker_volor >= 2
        assert np.array_equal(output_hovertext, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_6(self):
        """
        Regression without pred
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.contributions = pd.concat([self.contrib1] * 10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        output = xpl.plot.contribution_plot(col)
        np_hv = [f"Id: {x}" for x in xpl.x_init.index]
        np_hv.sort()
        annot_list = []
        for data_plot in output.data:
            if data_plot.type == "scatter":
                annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 6
        for elem in output.data:
            assert (elem.type == "violin") or (elem.type == "bar") or (elem.type == "scatter")
        assert output.data[-3].marker.color == output.data[-6].marker.color
        self.assertListEqual(annot_list, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_7(self):
        """
        Classification without pred
        """
        col = "X1"
        xpl = self.smart_explainer
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        np_hv = [f"Id: {x}" for x in xpl.x_init.index]
        np_hv.sort()
        output = xpl.plot.contribution_plot(col, proba=False)
        annot_list = []
        for data_plot in output.data:
            if data_plot.type == "scatter":
                annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 6
        for elem in output.data:
            assert (elem.type == "violin") or (elem.type == "bar") or (elem.type == "scatter")
        assert output.data[-3].marker.color == output.data[-6].marker.color
        self.assertListEqual(annot_list, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_8(self):
        """
        Classification with pred
        """
        col = "X1"
        xpl = self.smart_explainer
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.x_init.index = [i for i in range(xpl.x_init.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.contributions[0].index = xpl.x_init.index
        xpl.contributions[1].index = xpl.x_init.index
        xpl.y_pred = pd.DataFrame(
            [3, 1, 1, 3, 3, 3, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 3], columns=["pred"], index=xpl.x_init.index
        )
        model = lambda: None
        model.classes_ = np.array([0, 1])
        xpl.model = model
        np_hv = [f"Id: {x}<br />Predict: {y}" for x, y in zip(xpl.x_init.index, xpl.y_pred.iloc[:, 0].tolist())]
        np_hv.sort()
        output = xpl.plot.contribution_plot(col, proba=False)
        annot_list = []
        for data_plot in output.data:
            if data_plot.type == "scatter":
                annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 10
        for elem in output.data:
            assert (elem.type == "violin") or (elem.type == "bar") or (elem.type == "scatter")
        assert output.data[1].side == "negative"
        assert output.data[3].side == "positive"
        assert output.data[-1].line.color == output.data[-3].line.color
        assert output.data[-1].line.color != output.data[-2].line.color
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_9(self):
        """
        Classification with pred and sampling
        """
        col = "X1"
        xpl = self.smart_explainer
        xpl.x_init = pd.concat([xpl.x_init] * 20, ignore_index=True)
        xpl.x_init.index = [i for i in range(xpl.x_init.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 20, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 20, ignore_index=True)
        xpl.contributions[0].index = xpl.x_init.index
        xpl.contributions[1].index = xpl.x_init.index
        xpl.y_pred = pd.DataFrame([3, 1, 1, 3, 3] * 8, columns=["pred"], index=xpl.x_init.index)
        model = lambda: None
        model.classes_ = np.array([0, 1])
        xpl.model = model
        output = xpl.plot.contribution_plot(col, max_points=39, proba=False)
        assert len(output.data) == 10
        for elem in output.data:
            assert (elem.type == "violin") or (elem.type == "bar") or (elem.type == "scatter")
        assert output.data[1].side == "negative"
        assert output.data[3].side == "positive"
        assert output.data[-1].line.color == output.data[-3].line.color
        assert output.data[-1].line.color != output.data[-2].line.color
        assert output.layout.xaxis.title.text == xpl.features_dict[col]
        total_row = 0
        for data in output.data:
            if data.type == "scatter":
                total_row = total_row + data.x.shape[0]
        assert total_row == 39
        expected_title = "<b>Education</b> - Feature Contribution<br><sup>Response: <b>1</b> - Length of smart Subset: 39 (98%)</sup>"
        assert output.layout.title["text"] == expected_title

    def test_contribution_plot_10(self):
        """
        Regression with pred and subset
        """
        col = "X2"
        xpl = self.smart_explainer
        xpl.x_init = pd.concat([xpl.x_init] * 4, ignore_index=True)
        xpl.x_init.index = [i for i in range(xpl.x_init.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions = pd.concat([self.contrib1] * 4, ignore_index=True)
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.y_pred = pd.DataFrame([0.46989877093, 12.749302948] * 4, columns=["pred"], index=xpl.x_init.index)
        subset = [1, 2, 6, 7]
        output = xpl.plot.contribution_plot(col, selection=subset, violin_maxf=0)
        feature_values = xpl.x_init[col].loc[subset].sort_values()
        contributions = xpl.contributions[col].loc[feature_values.index]
        expected_output = go.Scatter(
            x=feature_values,
            y=contributions,
            mode="markers",
            hovertext=[
                f"Id: {x}<br />Predict: {y:.2f}"
                for x, y in zip(xpl.x_init.loc[subset].index, xpl.y_pred.loc[subset].iloc[:, 0].tolist())
            ],
        )

        assert np.array_equal(output.data[1].x, expected_output.x)
        assert np.array_equal(output.data[1].y, expected_output.y)
        assert len(np.unique(output.data[1].marker.color)) >= 2
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]
        expected_title = "<b>Age</b> - Feature Contribution<br><sup>Length of user-defined Subset: 4 (50%)</sup>"
        assert output.layout.title["text"] == expected_title

    def test_contribution_plot_11(self):
        """
        classification with proba
        """
        col = "X1"
        xpl = self.smart_explainer
        xpl.proba_values = pd.DataFrame(
            data=np.array([[0.4, 0.6], [0.3, 0.7]]), columns=["class_1", "class_2"], index=xpl.x_encoded.index.values
        )
        output = self.smart_explainer.plot.contribution_plot(col)
        assert str(type(output.data[-1])) == "<class 'plotly.graph_objs._scatter.Scatter'>"
        self.assertListEqual(list(output.data[-1]["marker"]["color"]), [0.7, 0.6])
        self.assertListEqual(list(output.data[1]["y"].round(2)), [4.7])

    def test_contribution_plot_12(self):
        """
        contribution plot with groups of features for classification case
        """
        x_encoded = pd.DataFrame(
            data=np.array([[0, 34], [1, 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        xpl = self.smart_explainer
        xpl.inv_features_dict = {}
        col = "group1"
        xpl.x_encoded = x_encoded
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.x_encoded = pd.concat([xpl.x_encoded] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.preprocessing = None
        # Creates a group of features named group1
        xpl.features_groups = {"group1": ["X1", "X2"]}
        xpl.contributions_groups = xpl.state.compute_grouped_contributions(xpl.contributions, xpl.features_groups)
        xpl.features_imp_groups = None
        xpl._update_features_dict_with_groups(features_groups=xpl.features_groups)

        output = xpl.plot.contribution_plot(col, proba=False)

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"
        self.setUp()

    def test_contribution_plot_13(self):
        """
        contribution plot with groups of features for regression case
        """
        x_encoded = pd.DataFrame(
            data=np.array([[0, 34], [1, 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        xpl = self.smart_explainer
        xpl.inv_features_dict = {}
        col = "group1"
        xpl.x_encoded = x_encoded
        xpl.contributions = pd.concat([self.contrib1] * 10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.backend.state = SmartState()
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.x_encoded = pd.concat([xpl.x_encoded] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.preprocessing = None
        # Creates a group of features named group1
        xpl.features_groups = {"group1": ["X1", "X2"]}
        xpl.contributions_groups = xpl.state.compute_grouped_contributions(xpl.contributions, xpl.features_groups)
        xpl.features_imp_groups = None
        xpl._update_features_dict_with_groups(features_groups=xpl.features_groups)

        output = xpl.plot.contribution_plot(col, proba=False)

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"
        self.setUp()

    def test_contribution_plot_14(self):
        """
        contribution plot with groups of features for classification case and subset
        """
        x_encoded = pd.DataFrame(
            data=np.array([[0, 34], [1, 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        xpl = self.smart_explainer
        xpl.inv_features_dict = {}
        col = "group1"
        xpl.x_encoded = x_encoded
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.x_encoded = pd.concat([xpl.x_encoded] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.preprocessing = None
        # Creates a group of features named group1
        xpl.features_groups = {"group1": ["X1", "X2"]}
        xpl.contributions_groups = xpl.state.compute_grouped_contributions(xpl.contributions, xpl.features_groups)
        xpl.features_imp_groups = None
        xpl._update_features_dict_with_groups(features_groups=xpl.features_groups)

        subset = list(range(10))
        output = xpl.plot.contribution_plot(col, proba=False, selection=subset)

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"
        assert len(output.data[1].x) == 10
        self.setUp()

    def test_contribution_plot_15(self):
        """
        contribution plot with groups of features for regression case with subset
        """
        x_encoded = pd.DataFrame(
            data=np.array([[0, 34], [1, 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        xpl = self.smart_explainer
        xpl.inv_features_dict = {}
        col = "group1"
        xpl.x_encoded = x_encoded
        xpl.contributions = pd.concat([self.contrib1] * 10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = SmartState()
        xpl.backend.state = SmartState()
        xpl.x_init = pd.concat([xpl.x_init] * 10, ignore_index=True)
        xpl.x_encoded = pd.concat([xpl.x_encoded] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.preprocessing = None
        # Creates a group of features named group1
        xpl.features_groups = {"group1": ["X1", "X2"]}
        xpl.contributions_groups = xpl.state.compute_grouped_contributions(xpl.contributions, xpl.features_groups)
        xpl.features_imp_groups = None
        xpl._update_features_dict_with_groups(features_groups=xpl.features_groups)

        subset = list(range(10))
        output = xpl.plot.contribution_plot(col, proba=False, selection=subset)

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"
        assert len(output.data[1].x) == 10
        self.setUp()

    def test_plot_features_import_1(self):
        """
        Unit test plot features import 1
        """
        xpl = self.smart_explainer
        serie1 = pd.Series([0.131, 0.51], index=["col1", "col2"])
        output = _plot_features_import(serie1, xpl.plot._style_dict, {})
        data = go.Bar(x=serie1, y=serie1.index, name="Global", orientation="h")

        expected_output = go.Figure(data=data)
        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation

    def test_plot_features_import_2(self):
        """
        Unit test plot features import 2
        """
        xpl = self.smart_explainer
        serie1 = pd.Series([0.131, 0.51], index=["col1", "col2"])
        serie2 = pd.Series([0.33, 0.11], index=["col1", "col2"])
        output = _plot_features_import(serie1, xpl.plot._style_dict, {}, feature_imp2=serie2)
        data1 = go.Bar(x=serie1, y=serie1.index, name="Global", orientation="h")
        data2 = go.Bar(x=serie2, y=serie2.index, name="Subset", orientation="h")
        expected_output = go.Figure(data=[data2, data1])
        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_features_importance_1(self):
        """
        Unit test features importance 1
        """
        xpl = self.smart_explainer
        xpl.explain_data = None
        output = xpl.plot.features_importance(selection=["person_A", "person_B"], zoom=True)

        data1 = go.Bar(x=np.array([0.2296, 0.7704]), y=np.array(["Age", "Education"]), name="Subset", orientation="h")

        data2 = go.Bar(x=np.array([0.2296, 0.7704]), y=np.array(["Age", "Education"]), name="Global", orientation="h")

        expected_output = go.Figure(data=[data1, data2])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_features_importance_cumulative_1(self):
        """
        Unit test features importance cumulative 1
        """
        xpl = self.smart_explainer
        xpl.explain_data = None
        output = xpl.plot.features_importance(mode="cumulative", selection=["person_A", "person_B"], zoom=True)

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"

    def test_features_importance_local_1(self):
        """
        Unit test features importance local 1
        """
        xpl = self.smart_explainer
        xpl.explain_data = None
        output = xpl.plot.features_importance(mode="global-local", selection=["person_A", "person_B"], zoom=True)

        assert len(output.data) == 3
        assert output.data[0].type == "bar"
        assert output.data[1].type == "bar"
        assert output.data[2].type == "bar"

    def test_features_importance_2(self):
        """
        Unit test features importance 2
        """
        xpl = self.smart_explainer
        # regression
        xpl.contributions = self.contrib1
        xpl.backend.state = SmartState()
        xpl.explain_data = None
        xpl._case = "regression"
        xpl.state = SmartState()
        output = xpl.plot.features_importance(selection=["person_A", "person_B"])

        data1 = go.Bar(x=np.array([0.2296, 0.7704]), y=np.array(["Age", "Education"]), name="Subset", orientation="h")

        data2 = go.Bar(x=np.array([0.2296, 0.7704]), y=np.array(["Age", "Education"]), name="Global", orientation="h")

        expected_output = go.Figure(data=[data1, data2])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_features_importance_cumulative_2(self):
        """
        Unit test features importance cumulative 2
        """
        xpl = self.smart_explainer
        # regression
        xpl.contributions = self.contrib1
        xpl.backend.state = SmartState()
        xpl.explain_data = None
        xpl._case = "regression"
        xpl.state = SmartState()
        output = xpl.plot.features_importance(mode="cumulative", selection=["person_A", "person_B"])

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"

    def test_features_importance_local_2(self):
        """
        Unit test features importance local 2
        """
        xpl = self.smart_explainer
        # regression
        xpl.contributions = self.contrib1
        xpl.backend.state = SmartState()
        xpl.explain_data = None
        xpl._case = "regression"
        xpl.state = SmartState()
        output = xpl.plot.features_importance(mode="global-local", selection=["person_A", "person_B"])

        assert len(output.data) == 3
        assert output.data[0].type == "bar"
        assert output.data[1].type == "bar"
        assert output.data[2].type == "bar"

    def test_features_importance_3(self):
        """
        Unit test features importance for groups of features
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer.state = MultiDecorator(SmartState())
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )

        output = smart_explainer.plot.features_importance()

        data1 = go.Bar(x=np.array([0.1682, 0.8318]), y=np.array(["X3", "<b>group0"]), name="Global", orientation="h")
        expected_output = go.Figure(data=[data1])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation

    def test_features_importance_cumulative_3(self):
        """
        Unit test features importance cumulative for groups of features
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer.state = MultiDecorator(SmartState())
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )

        output = smart_explainer.plot.features_importance(mode="cumulative")

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"

    def test_features_importance_local_3(self):
        """
        Unit test features importance local for groups of features
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer.state = MultiDecorator(SmartState())
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )
        smart_explainer.features_imp_groups_local_lev1 = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups, norm=3
        )
        smart_explainer.features_imp_groups_local_lev2 = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups, norm=7
        )

        output = smart_explainer.plot.features_importance(mode="global-local")

        assert len(output.data) == 3
        assert output.data[0].type == "bar"
        assert output.data[1].type == "bar"
        assert output.data[2].type == "bar"

    def test_features_importance_4(self):
        """
        Unit test features importance for groups of features when displaying a group
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.state = smart_explainer.backend.state
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )

        output = smart_explainer.plot.features_importance(group_name="group0")

        data1 = go.Bar(x=np.array([0.4179, 0.4389]), y=np.array(["X2", "X1"]), name="Global", orientation="h")
        expected_output = go.Figure(data=[data1])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation

    def test_features_importance_cumulative_4(self):
        """
        Unit test features importance cumulative for groups of features when displaying a group
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.state = smart_explainer.backend.state
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )

        output = smart_explainer.plot.features_importance(mode="cumulative", group_name="group0")

        assert len(output.data) == 2
        assert output.data[0].type == "scatter"
        assert output.data[1].type == "scatter"

    def test_features_importance_local_4(self):
        """
        Unit test features importance local for groups of features when displaying a group
        """
        x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 1], ["Master", 27, 0]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        contrib = pd.DataFrame(
            data=np.array([[-3.4, 0.78, 1.2], [1.2, 3.6, -0.3]]),
            columns=["X1", "X2", "X3"],
            index=["person_A", "person_B"],
        )

        smart_explainer = SmartExplainer(model=self.model)
        smart_explainer.x_encoded = x_init
        smart_explainer.x_init = x_init
        smart_explainer.postprocessing_modifications = False
        smart_explainer.features_imp_groups = None
        smart_explainer.features_imp = None
        smart_explainer.features_groups = {"group0": ["X1", "X2"]}
        smart_explainer.contributions = [contrib, -contrib]
        smart_explainer.features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.inv_features_dict = {"X1": "X1", "X2": "X2", "X3": "X3", "group0": "group0"}
        smart_explainer.model = self.smart_explainer.model
        smart_explainer.backend = ShapBackend(model=self.smart_explainer.model)
        smart_explainer.backend.state = MultiDecorator(SmartState())
        smart_explainer.explain_data = None
        smart_explainer._case, smart_explainer._classes = check_model(self.smart_explainer.model)
        smart_explainer.state = smart_explainer.backend.state
        smart_explainer.contributions_groups = smart_explainer.state.compute_grouped_contributions(
            smart_explainer.contributions, smart_explainer.features_groups
        )
        smart_explainer.features_imp_groups = smart_explainer.state.compute_features_import(
            smart_explainer.contributions_groups
        )

        output = smart_explainer.plot.features_importance(mode="global-local", group_name="group0")

        assert len(output.data) == 3
        assert output.data[0].type == "bar"
        assert output.data[1].type == "bar"
        assert output.data[2].type == "bar"

    def test_local_pred_1(self):
        xpl = self.smart_explainer
        xpl.proba_values = pd.DataFrame(
            data=np.array([[0.4, 0.6], [0.3, 0.7]]), columns=["class_1", "class_2"], index=xpl.x_encoded.index.values
        )
        output = xpl._local_pred("person_A", label=0)
        assert isinstance(output, float)

    def test_plot_line_comparison_1(self):
        """
        Unit test 1 for plot_line_comparison
        """
        xpl = self.smart_explainer
        index = ["person_A", "person_B"]
        data = pd.DataFrame(data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index)
        features_dict = {"X1": "X1", "X2": "X2"}
        xpl.inv_features_dict = {"X1": "X1", "X2": "X2"}
        colors = ["rgba(244, 192, 0, 1.0)", "rgba(74, 99, 138, 0.7)"]
        var_dict = ["X1", "X2"]
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        title = "Compare plot - index : <b>person_A</b> ; <b>person_B</b>"
        predictions = [data.loc[id] for id in index]

        fig = list()
        for i in range(2):
            fig.append(
                go.Scatter(
                    x=[contributions[0][i], contributions[1][i]],
                    y=["<b>" + feat + "</b>" for feat in var_dict],
                    mode="lines+markers",
                    name=f"Id: <b>{index[i]}</b>",
                    hovertext=[
                        f"Id: <b>{index[i]}</b><br /><b>X1</b> <br />Contribution: {contributions[0][i]:.4f} <br />"
                        + f"Value: {data.iloc[i,0]}",
                        f"Id: <b>{index[i]}</b><br /><b>X2</b> <br />Contribution: {contributions[1][i]:.4f} <br />"
                        + f"Value: {data.iloc[i,1]}",
                    ],
                    marker={"color": colors[i]},
                )
            )
        expected_output = go.Figure(data=fig)
        output = plot_line_comparison(
            ["person_A", "person_B"],
            var_dict,
            contributions,
            style_dict=xpl.plot._style_dict,
            predictions=predictions,
            dict_features=features_dict,
        )

        for i in range(2):
            assert np.array_equal(output.data[i]["x"], expected_output.data[i]["x"])
            assert np.array_equal(output.data[i]["y"], expected_output.data[i]["y"])
            assert output.data[i].name == expected_output.data[i].name
            assert output.data[i].hovertext == expected_output.data[i].hovertext
            assert output.data[i].marker == expected_output.data[i].marker
        assert title == output.layout.title.text

    def test_plot_line_comparison_2(self):
        """
        Unit test 2 for plot_line_comparison
        """
        xpl = self.smart_explainer
        index = ["person_A", "person_B"]
        data = pd.DataFrame(data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index)
        xpl.inv_features_dict = {"X1": "X1", "X2": "X2"}
        var_dict = ["X1", "X2"]
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        subtitle = "This is a good test"
        title = (
            "Compare plot - index : <b>person_A</b> ; <b>person_B</b>"
            + "<span style='font-size: 12px;'><br />This is a good test</span>"
        )
        predictions = [data.loc[id] for id in index]

        output = plot_line_comparison(
            index,
            var_dict,
            contributions,
            style_dict=xpl.plot._style_dict,
            subtitle=subtitle,
            predictions=predictions,
            dict_features=xpl.inv_features_dict,
        )

        assert title == output.layout.title.text

    def test_compare_plot_1(self):
        """
        Unit test 1 for compare_plot
        """
        xpl = self.smart_explainer
        xpl.contributions = pd.DataFrame(
            data=[[-3.4, 0.78], [1.2, 3.6]], index=["person_A", "person_B"], columns=["X1", "X2"]
        )
        xpl.inv_features_dict = {"Education": "X1", "Age": "X2"}
        xpl._case = "regression"
        output = xpl.plot.compare_plot(row_num=[1], show_predict=False)
        title = "Compare plot - index : <b>person_B</b><span style='font-size: 12px;'><br /></span>"
        data = [
            go.Scatter(
                x=[1.2, 3.6],
                y=["<b>Education</b>", "<b>Age</b>"],
                name="Id: <b>person_B</b>",
                hovertext=[
                    "Id: <b>person_B</b><br /><b>Education</b> <br />Contribution: 1.2000 <br />Value: Master",
                    "Id: <b>person_B</b><br /><b>Age</b> <br />Contribution: 3.6000 <br />Value: 27",
                ],
            )
        ]
        expected_output = go.Figure(data=data)
        assert np.array_equal(expected_output.data[0].x, output.data[0].x)
        assert np.array_equal(expected_output.data[0].y, output.data[0].y)
        assert np.array_equal(expected_output.data[0].hovertext, output.data[0].hovertext)
        assert expected_output.data[0].name == output.data[0].name
        assert title == output.layout.title.text

    def test_compare_plot_2(self):
        """
        Unit test 2 for compare_plot
        """
        xpl = self.smart_explainer
        xpl.inv_features_dict = {"Education": "X1", "Age": "X2"}
        index = ["person_A", "person_B"]
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        xpl.contributions = pd.DataFrame(data=contributions, index=index, columns=["X1", "X2"])
        data = np.array([["PhD", 34], ["Master", 27]])
        xpl._case = "regression"
        output = xpl.plot.compare_plot(index=index, show_predict=True)
        title_and_subtitle = (
            "Compare plot - index : <b>person_A</b> ;"
            " <b>person_B</b><span style='font-size: 12px;'><br />"
            "Predictions: person_A: <b>0</b> ; person_B: <b>1</b></span>"
        )
        fig = list()
        for i in range(2):
            fig.append(
                go.Scatter(
                    x=contributions[i][::-1],
                    y=["<b>Age</b>", "<b>Education</b>"],
                    name=f"Id: <b>{index[i]}</b>",
                    hovertext=[
                        f"Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {contributions[i][1]:.4f}"
                        f" <br />Value: {data[i][1]}",
                        f"Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {contributions[i][0]:.4f}"
                        f" <br />Value: {data[i][0]}",
                    ],
                )
            )

        expected_output = go.Figure(data=fig)
        for i in range(2):
            assert np.array_equal(expected_output.data[i].x, output.data[i].x)
            assert np.array_equal(expected_output.data[i].y, output.data[i].y)
            assert np.array_equal(expected_output.data[i].hovertext, output.data[i].hovertext)
            assert expected_output.data[i].name == output.data[i].name
        assert title_and_subtitle == output.layout.title.text

    def test_compare_plot_3(self):
        """
        Unit test 3 for compare_plot classification
        """
        index = ["A", "B"]
        x_init = pd.DataFrame(data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=index)
        contributions1 = pd.DataFrame(data=np.array([[-3.4, 0.78], [1.2, 3.6]]), columns=["X1", "X2"], index=index)
        contributions2 = pd.DataFrame(data=np.array([[-0.4, 0.78], [0.2, 0.6]]), columns=["X1", "X2"], index=index)
        feature_dictionary = {"X1": "Education", "X2": "Age"}
        smart_explainer_mi = SmartExplainer(model=self.model, features_dict=feature_dictionary)
        smart_explainer_mi.contributions = [
            pd.DataFrame(data=contributions1, index=index, columns=["X1", "X2"]),
            pd.DataFrame(data=contributions2, index=index, columns=["X1", "X2"]),
        ]
        smart_explainer_mi.inv_features_dict = {"Education": "X1", "Age": "X2"}
        smart_explainer_mi.data = dict()
        smart_explainer_mi.x_init = x_init
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_init.columns)}
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]
        smart_explainer_mi.model = "predict_proba"

        output_label0 = smart_explainer_mi.plot.compare_plot(index=["A", "B"], label=0, show_predict=False)
        output_label1 = smart_explainer_mi.plot.compare_plot(index=["A", "B"], show_predict=False)

        title_0 = "Compare plot - index : <b>A</b> ; <b>B</b><span style='font-size: 12px;'><br /></span>"
        title_1 = "Compare plot - index : <b>A</b> ; <b>B</b><span style='font-size: 12px;'><br /></span>"

        fig_0 = list()
        x0 = contributions1.to_numpy()
        x0.sort(axis=1)
        for i in range(2):
            fig_0.append(
                go.Scatter(
                    x=x0[i][::-1],
                    y=["<b>Age</b>", "<b>Education</b>"],
                    name=f"Id: <b>{index[i]}</b>",
                    hovertext=[
                        f"Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {x0[i][1]:.4f}"
                        f" <br />Value: {x_init.to_numpy()[i][1]}",
                        f"Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {x0[i][0]:.4f}"
                        f" <br />Value: {x_init.to_numpy()[i][0]}",
                    ],
                )
            )

        fig_1 = list()
        x1 = contributions2.to_numpy()
        x1.sort(axis=1)
        for i in range(2):
            fig_1.append(
                go.Scatter(
                    x=x1[i][::-1],
                    y=["<b>Age</b>", "<b>Education</b>"],
                    name=f"Id: <b>{index[i]}</b>",
                    hovertext=[
                        f"Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {x1[i][1]:.4f}"
                        f" <br />Value: {x_init.to_numpy()[i][1]}",
                        f"Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {x1[i][0]:.4f}"
                        f" <br />Value: {x_init.to_numpy()[i][0]}",
                    ],
                )
            )

        expected_output_0 = go.Figure(data=fig_0)
        expected_output_1 = go.Figure(data=fig_1)

        assert title_0 == output_label0.layout.title.text
        assert title_1 == output_label1.layout.title.text
        for i in range(2):
            assert np.array_equal(output_label1.data[i].x, expected_output_1.data[i].x)
            assert np.array_equal(output_label1.data[i].y, expected_output_1.data[i].y)
            assert np.array_equal(output_label1.data[i].hovertext, expected_output_1.data[i].hovertext)

            assert np.array_equal(output_label0.data[i].x, expected_output_0.data[i].x)
            assert np.array_equal(output_label0.data[i].y, expected_output_0.data[i].y)
            assert np.array_equal(output_label0.data[i].hovertext, expected_output_0.data[i].hovertext)

    def test_interactions_plot_1(self):
        """
        Unit test 1 for test interaction plot : scatter plot for categorical x categorical features
        """

        col1 = "X1"
        col2 = "X2"

        interaction_values = np.array([[[0.1, -0.7], [-0.6, 0.3]], [[0.2, -0.1], [-0.2, 0.1]]])
        self.smart_explainer.interaction_values = interaction_values
        self.smart_explainer.x_interaction = self.smart_explainer.x_encoded

        output = self.smart_explainer.plot.interactions_plot(col1, col2, violin_maxf=0)

        expected_output = px.scatter(
            x=self.x_init[col1], y=self.smart_explainer.interaction_values[:, 0, 1], color=self.x_init[col1]
        )

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y * 2)
        assert output.data[0].showlegend is True
        assert len(output.data) == 2

    def test_interactions_plot_2(self):
        """
        Unit test 1 for test interaction plot : scatter plot for categorical x numeric features
        """
        col1 = "X1"
        col2 = "X2"
        smart_explainer = self.smart_explainer
        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        smart_explainer.x_encoded["X2"] = smart_explainer.x_encoded["X2"].astype(float)

        interaction_values = np.array([[[0.1, -0.7], [-0.7, 0.3]], [[0.2, -0.1], [-0.1, 0.1]]])

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.interactions_plot(col1, col2, violin_maxf=0)

        assert np.array_equal(output.data[0].x, ["PhD", "Master"])
        assert np.array_equal(output.data[0].y, [-1.4, -0.2])
        assert np.array_equal(output.data[0].marker.color, [34.0, 27.0])
        assert len(output.data) == 1

        self.setUp()

    def test_interactions_plot_3(self):
        """
        Unit test 1 for test interaction plot : scatter plot for numeric x categorical features
        """
        col1 = "X1"
        col2 = "X2"
        smart_explainer = self.smart_explainer
        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        smart_explainer.x_encoded["X2"] = smart_explainer.x_encoded["X2"].astype(float)

        interaction_values = np.array([[[0.1, -0.7], [-0.7, 0.3]], [[0.2, -0.1], [-0.1, 0.1]]])

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.interactions_plot(col2, col1, violin_maxf=0)

        assert np.array_equal(output.data[0].x, [34.0])
        assert np.array_equal(output.data[0].y, [-1.4])
        assert output.data[0].name == "PhD"

        assert np.array_equal(output.data[1].x, [27.0])
        assert np.array_equal(output.data[1].y, [-0.2])
        assert output.data[1].name == "Master"

        assert len(output.data) == 2

        self.setUp()

    def test_interactions_plot_4(self):
        """
        Unit test 1 for test interaction plot : scatter plot for numeric x numeric features
        """
        col1 = "X1"
        col2 = "X2"
        smart_explainer = self.smart_explainer

        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([[520, 34], [12800, 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        smart_explainer.x_encoded["X1"] = smart_explainer.x_encoded["X1"].astype(float)
        smart_explainer.x_encoded["X2"] = smart_explainer.x_encoded["X2"].astype(float)

        interaction_values = np.array([[[0.1, -0.7], [-0.7, 0.3]], [[0.2, -0.1], [-0.1, 0.1]]])

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.interactions_plot(col1, col2, violin_maxf=0)

        assert np.array_equal(output.data[0].x, [520, 12800])
        assert np.array_equal(output.data[0].y, [-1.4, -0.2])
        assert np.array_equal(output.data[0].marker.color, [34.0, 27.0])

        assert len(output.data) == 1

        self.setUp()

    def test_interactions_plot_5(self):
        """
        Unit test 1 for test interaction plot : violin plot for categorical x numeric features
        """
        col1 = "X1"
        col2 = "X2"
        smart_explainer = self.smart_explainer
        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([["PhD", 34], ["Master", 27]]), columns=["X1", "X2"], index=["person_A", "person_B"]
        )
        smart_explainer.x_encoded["X2"] = smart_explainer.x_encoded["X2"].astype(float)

        interaction_values = np.array([[[0.1, -0.7], [-0.7, 0.3]], [[0.2, -0.1], [-0.1, 0.1]]])

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.interactions_plot(col1, col2)

        assert len(output.data) == 3

        assert output.data[0].type == "violin"
        assert output.data[1].type == "violin"
        assert output.data[2].type == "scatter"

        assert np.array_equal(output.data[2].x, ["PhD", "Master"])
        assert np.array_equal(output.data[2].y, [-1.4, -0.2])
        assert np.array_equal(output.data[2].marker.color, [34.0, 27.0])

        self.setUp()

    def test_top_interactions_plot_1(self):
        """
        Test top interactions plot with scatter plots only
        """
        smart_explainer = self.smart_explainer
        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 16, 0.2, 12], ["Master", 27, -10, 0.65, 18]]),
            columns=["X1", "X2", "X3", "X4", "X5"],
            index=["person_A", "person_B"],
        ).astype({"X1": str, "X2": float, "X3": float, "X4": float, "X5": float})

        smart_explainer.features_desc = dict(smart_explainer.x_init.nunique())
        smart_explainer.columns_dict = {i: col for i, col in enumerate(smart_explainer.x_init.columns)}

        interaction_values = np.array(
            [
                [
                    [0.1, -0.7, 0.01, -0.9, 0.6],
                    [-0.1, 0.8, 0.02, 0.7, -0.5],
                    [0.2, 0.5, 0.04, -0.88, 0.7],
                    [0.15, 0.6, -0.2, 0.5, 0.3],
                ],
                [
                    [0.2, -0.1, 0.2, 0.8, 0.55],
                    [-0.2, 0.6, 0.02, -0.67, -0.6],
                    [0.1, -0.5, 0.05, 1, 0.5],
                    [0.3, 0.6, 0.02, -0.9, 0.4],
                ],
            ]
        )

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.top_interactions_plot(nb_top_interactions=5, violin_maxf=0)

        assert len(output.layout.updatemenus[0].buttons) == 5
        assert isinstance(output.layout.updatemenus[0].buttons[0].args[0]["visible"], list)
        assert len(output.layout.updatemenus[0].buttons[0].args[0]["visible"]) >= 5
        assert True in output.layout.updatemenus[0].buttons[0].args[0]["visible"]

        self.setUp()

    def test_top_interactions_plot_2(self):
        """
        Test top interactions plot with violin and scatter plots
        """
        smart_explainer = self.smart_explainer
        smart_explainer.x_encoded = smart_explainer.x_init = pd.DataFrame(
            data=np.array([["PhD", 34, 16, 0.2, 12], ["Master", 27, -10, 0.65, 18]]),
            columns=["X1", "X2", "X3", "X4", "X5"],
            index=["person_A", "person_B"],
        ).astype({"X1": str, "X2": float, "X3": float, "X4": float, "X5": float})

        smart_explainer.features_desc = dict(smart_explainer.x_init.nunique())
        smart_explainer.columns_dict = {i: col for i, col in enumerate(smart_explainer.x_init.columns)}

        interaction_values = np.array(
            [
                [
                    [0.1, -0.7, 0.01, -0.9, 0.6],
                    [-0.1, 0.8, 0.02, 0.7, -0.5],
                    [0.2, 0.5, 0.04, -0.88, 0.7],
                    [0.15, 0.6, -0.2, 0.5, 0.3],
                ],
                [
                    [0.2, -0.1, 0.2, 0.8, 0.55],
                    [-0.2, 0.6, 0.02, -0.67, -0.6],
                    [0.1, -0.5, 0.05, 1, 0.5],
                    [0.3, 0.6, 0.02, -0.9, 0.4],
                ],
            ]
        )

        smart_explainer.interaction_values = interaction_values
        smart_explainer.x_interaction = smart_explainer.x_encoded

        output = smart_explainer.plot.top_interactions_plot(nb_top_interactions=4)

        assert len(output.layout.updatemenus[0].buttons) == 4
        assert isinstance(output.layout.updatemenus[0].buttons[0].args[0]["visible"], list)
        assert len(output.layout.updatemenus[0].buttons[0].args[0]["visible"]) >= 4
        assert True in output.layout.updatemenus[0].buttons[0].args[0]["visible"]

        self.setUp()

    def test_correlations_1(self):
        """
        Test correlations plot 1
        """
        smart_explainer = self.smart_explainer

        smart_explainer.x_init = pd.DataFrame(
            {"A": [8, 90, 10, 110], "B": [4.3, 7.4, 10.2, 15.7], "C": ["C8", "C8", "C9", "C9"], "D": [1, -3, -5, -10]},
            index=[8, 9, 10, 11],
        )

        output = smart_explainer.plot.correlations_plot(max_features=3)

        assert len(output.data) == 1
        assert len(output.data[0].x) == 3
        assert len(output.data[0].y) == 3
        assert output.data[0].z.shape == (3, 3)

    def test_correlations_2(self):
        """
        Test correlations plot 2
        """
        smart_explainer = self.smart_explainer

        df = pd.DataFrame(
            {"A": [8, 90, 10, 110], "B": [4.3, 7.4, 10.2, 15.7], "C": ["C8", "C8", "C9", "C9"], "D": [1, -3, -5, -10]},
            index=[8, 9, 10, 11],
        )

        output = smart_explainer.plot.correlations_plot(df, max_features=3, facet_col="C")

        assert len(output.data) == 2
        assert len(output.data[0].x) == 3
        assert len(output.data[0].y) == 3
        assert output.data[0].z.shape == (3, 3)

    def test_stability_plot_1(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        output = xpl.plot.stability_plot(distribution="none")

        assert len(output.data[0].x) == X.shape[1]
        assert len(output.data[0].y) == X.shape[1]
        assert np.array(list(output.data[0].x)).dtype == "float"
        assert np.array(list(output.data[0].y)).dtype == "float"

    def test_stability_plot_2(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        selection = list(range(40))
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        for max_features in [2, 5]:
            output = xpl.plot.stability_plot(selection=selection, distribution="boxplot", max_features=max_features)

            actual_shape = sum([1 if output.data[i].type == "box" else 0 for i in range(len(output.data))])
            expected_shape = X.shape[1] if X.shape[1] < max_features else max_features

            assert actual_shape == expected_shape
            assert len(output.data[0].x) == len(selection)
            assert np.array(list(output.data[0].x)).dtype == "float"

    def test_stability_plot_3(self):
        np.random.seed(79)
        df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        for max_features in [2, 5]:
            output = xpl.plot.stability_plot(distribution="boxplot", max_features=max_features)

            actual_shape = sum([1 if output.data[i].type == "box" else 0 for i in range(len(output.data))])
            expected_shape = X.shape[1] if X.shape[1] < max_features else max_features

            assert actual_shape == expected_shape
            assert len(output.data[0].x) == 50
            assert np.array(list(output.data[0].x)).dtype == "float"

    def test_stability_plot_4(self):
        np.random.seed(79)
        df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        selection = list(range(6))
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        for max_features in [2, 5]:
            output = xpl.plot.stability_plot(selection=selection, distribution="violin", max_features=max_features)

            actual_shape = sum([1 if output.data[i].type == "violin" else 0 for i in range(len(output.data))])
            expected_shape = X.shape[1] if X.shape[1] < max_features else max_features

            assert actual_shape == expected_shape
            assert len(output.data[0].x) == len(selection)
            assert np.array(list(output.data[0].x)).dtype == "float"

    def test_stability_plot_5(self):
        np.random.seed(79)
        df = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        for max_features in [2, 5]:
            output = xpl.plot.stability_plot(distribution="violin", max_features=max_features)

            actual_shape = sum([1 if output.data[i].type == "violin" else 0 for i in range(len(output.data))])
            expected_shape = X.shape[1] if X.shape[1] < max_features else max_features

            assert actual_shape == expected_shape
            assert len(output.data[0].x) == 50
            assert np.array(list(output.data[0].x)).dtype == "float"

    def test_local_neighbors_plot(self):
        df = pd.DataFrame(np.random.randint(0, 100, size=(15, 4)), columns=list("ABCD"))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X, y)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=X)

        for max_features in [2, 5]:
            output = xpl.plot.local_neighbors_plot(index=1, max_features=max_features)
            actual_shape = len(output.data[0].x)
            expected_shape = X.shape[1] if X.shape[1] < max_features else max_features

            assert actual_shape == expected_shape
            assert np.array(list(output.data[0].x)).dtype == "float"

    @patch("shapash.explainer.smart_explainer.SmartExplainer.compute_features_compacity")
    def test_compacity_plot(self, compute_features_compacity):

        compute_features_compacity.return_value = None
        selection = ["person_A", "person_B"]
        approx = 0.9
        nb_features = 5

        output = self.smart_explainer.plot.compacity_plot(selection=selection, approx=approx, nb_features=nb_features)

        assert len(output.data[0].x) == len(selection)
        assert len(output.data[1].x) == len(selection)
        assert f"at least {approx*100:.0f}%" in output.data[0].hovertemplate
        assert f"Top {nb_features} features" in output.data[1].hovertemplate

        """
        Regression
        """
        df_train = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        df_test = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X_train, y_train)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        output = xpl.plot.scatter_plot_prediction()
        actual_shape = len(output.data[1].x)
        expected_shape = X_test.shape[0]
        assert actual_shape == expected_shape
        assert np.array(list(output.data[1].x)).dtype == "int64"
        assert np.array(list(output.data[1].y)).dtype == "float64"
        assert output.data[1].type == "scatter"
        assert "True Values" in output.data[1].hovertext[0]

    def test_scatter_plot_prediction_2(self):
        """
        Regression
        """
        df_train = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        df_test = pd.DataFrame(np.random.randint(0, 100, size=(50, 4)), columns=list("ABCD"))
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]
        model = DecisionTreeRegressor().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        selection = list(range(20))
        output = xpl.plot.scatter_plot_prediction(selection=selection)
        actual_shape = len(output.data[1].x)
        expected_shape = len(selection)
        assert actual_shape == expected_shape
        assert np.array(list(output.data[1].x)).dtype == "int64"
        assert np.array(list(output.data[1].y)).dtype == "float64"
        assert output.data[1].type == "scatter"
        assert "True Values" in output.data[1].hovertext[0]

    def test_scatter_plot_prediction_3(self):
        """
        Classification
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(50, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)))
        X_test = pd.DataFrame(np.random.randint(0, 100, size=(50, 3)), columns=list("ABC"))
        y_test = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        output = xpl.plot.scatter_plot_prediction()
        actual_shape = len(output.data[0].x)
        expected_shape = X_test.shape[0]
        assert actual_shape == expected_shape
        assert np.array(list(output.data[0].x)).dtype == "int64"
        assert np.array(list(output.data[0].y)).dtype == "float64"
        assert output.data[0].type == "violin"
        assert output.data[1].type == "scatter"
        assert output.data[2].type == "scatter"
        assert f"True Values" in output.data[1].hovertext[0]

    def test_scatter_plot_prediction_4(self):
        """
        Classification
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(50, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)))
        X_test = pd.DataFrame(np.random.randint(0, 100, size=(50, 3)), columns=list("ABC"))
        y_test = pd.DataFrame(np.random.randint(0, 2, size=(50, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        selection = list(range(20))
        output = xpl.plot.scatter_plot_prediction(selection=selection)
        actual_shape = len(output.data[0].x)
        expected_shape = len(selection)

        assert actual_shape == expected_shape
        assert np.array(list(output.data[0].x)).dtype == "int64"
        assert np.array(list(output.data[0].y)).dtype == "float64"
        assert output.data[0].type == "violin"
        assert output.data[1].type == "scatter"
        assert output.data[2].type == "scatter"
        assert f"True Values" in output.data[1].hovertext[0]

    def test_scatter_plot_prediction_5(self):
        """
        Classification Multiclass
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(100, 1)))
        X_test = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
        y_test = pd.DataFrame(np.random.randint(0, 3, size=(100, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        output = xpl.plot.scatter_plot_prediction()
        actual_shape = len(output.data[0].x)
        expected_shape = X_test.shape[0]

        assert actual_shape == expected_shape
        assert np.array(list(output.data[0].x)).dtype == "int64"
        assert np.array(list(output.data[0].y)).dtype == "float64"
        assert output.data[0].type == "violin"
        assert output.data[1].type == "scatter"
        assert output.data[2].type == "scatter"
        assert f"True Values" in output.data[1].hovertext[0]

    def test_scatter_plot_prediction_6(self):
        """
        Classification Multiclass
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(100, 1)))
        X_test = pd.DataFrame(np.random.randint(0, 100, size=(100, 3)), columns=list("ABC"))
        y_test = pd.DataFrame(np.random.randint(0, 3, size=(100, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_test, y_target=y_test)

        selection = list(range(20))
        output = xpl.plot.scatter_plot_prediction(selection=selection)
        actual_shape = len(output.data[0].x)
        expected_shape = len(selection)

        assert actual_shape == expected_shape
        assert np.array(list(output.data[0].x)).dtype == "int64"
        assert np.array(list(output.data[0].y)).dtype == "float64"
        assert output.data[0].type == "violin"
        assert output.data[1].type == "scatter"
        assert output.data[2].type == "scatter"
        assert f"True Values" in output.data[1].hovertext[0]

    def test_subset_sampling_1(self):
        """
        test _subset_sampling
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(30, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(30, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_train, y_target=y_train)
        list_ind, addnote = subset_sampling(df=xpl.x_init, max_points=10)
        assert len(list_ind) == 10
        assert addnote == "Length of random Subset: 10 (33%)"

    def test_subset_sampling_2(self):
        """
        test _subset_sampling
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(30, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(30, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_train, y_target=y_train)
        list_ind, addnote = subset_sampling(df=xpl.x_init, max_points=50)
        assert len(list_ind) == 30
        assert addnote is None

    def test_subset_sampling_3(self):
        """
        test _subset_sampling
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(30, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(30, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_train, y_target=y_train)
        selection = list(range(10, 20))
        list_ind, addnote = subset_sampling(df=xpl.x_init, selection=selection)
        assert len(list_ind) == 10
        assert addnote == "Length of user-defined Subset: 10 (33%)"
        assert list_ind == selection

    def test_subset_sampling_4(self):
        """
        test _subset_sampling
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(30, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(30, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_train, y_target=y_train)
        selection = list(range(10, 20))
        list_ind, addnote = subset_sampling(df=xpl.x_init, selection=selection, max_points=50)
        assert len(list_ind) == 10
        assert addnote == "Length of user-defined Subset: 10 (33%)"
        assert list_ind == selection

    def test_subset_sampling_5(self):
        """
        test _subset_sampling
        """
        X_train = pd.DataFrame(np.random.randint(0, 100, size=(30, 3)), columns=list("ABC"))
        y_train = pd.DataFrame(np.random.randint(0, 3, size=(30, 1)))
        model = DecisionTreeClassifier().fit(X_train, y_train)
        xpl = SmartExplainer(model=model)
        xpl.compile(x=X_train, y_target=y_train)
        selection = np.array(list(range(10, 20)))
        with self.assertRaises(ValueError):
            list_ind, addnote = subset_sampling(df=xpl.x_init, selection=selection, max_points=50)
