import copy
import unittest

import numpy as np
import pandas as pd
from dash import dcc, html
from sklearn.tree import DecisionTreeClassifier

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp
from shapash.webapp.utils.callbacks import (
    create_dropdown_feature_filter,
    create_filter_modalities_selection,
    create_id_card_data,
    create_id_card_layout,
    get_feature_contributions_sign_to_show,
    get_feature_filter_options,
    get_feature_from_clicked_data,
    get_feature_from_features_groups,
    get_figure_zoom,
    get_id_card_contrib,
    get_id_card_features,
    get_indexes_from_datatable,
    handle_page_navigation,
    select_data_from_bool_filters,
    select_data_from_date_filters,
    select_data_from_numeric_filters,
    select_data_from_prediction_picking,
    select_data_from_str_filters,
    update_click_data_on_subset_changes,
    update_features_to_display,
)


class TestCallbacks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        data = {
            "column1": [1, 2, 3, 4, 5],
            "column2": ["a", "b", "c", "d", "e"],
            "column3": [1.1, 3.3, 2.2, 4.4, 5.5],
            "column4": [True, False, True, False, False],
            "column5": pd.date_range("2023-01-01", periods=5),
        }

        df = pd.DataFrame(data)
        self.df = df

        dataframe_x = df[["column1", "column3"]].copy()
        y_target = pd.DataFrame(data=np.array([0, 0, 0, 1, 1]), columns=["pred"])
        model = DecisionTreeClassifier().fit(dataframe_x, y_target)
        features_dict = {"column3": "Useless col"}
        additional_data = df[["column2", "column4", "column5"]].copy()
        additional_features_dict = {"column2": "Additional col"}
        self.xpl = SmartExplainer(model=model, features_dict=features_dict)
        self.xpl.compile(
            x=dataframe_x,
            y_pred=y_target,
            y_target=y_target,
            additional_data=additional_data,
            additional_features_dict=additional_features_dict,
        )
        self.smart_app = SmartApp(self.xpl)

        self.click_data = {
            "points": [
                {
                    "curveNumber": 0,
                    "pointNumber": 3,
                    "pointIndex": 3,
                    "x": 0.4649,
                    "y": "Sex",
                    "label": "Sex",
                    "value": 0.4649,
                    "customdata": "Sex",
                    "marker.color": "rgba(244, 192, 0, 1.0)",
                    "bbox": {"x0": 717.3, "x1": 717.3, "y0": 82.97, "y1": 130.78},
                }
            ]
        }
        self.special_cols = ["_index_", "_predict_", "_target_"]

        super().__init__(*args, **kwargs)

    def test_default_init_data(self):
        expected_result = pd.DataFrame(
            {
                "_index_": [0, 1, 2, 3, 4],
                "_predict_": [0, 0, 0, 1, 1],
                "_target_": [0, 0, 0, 1, 1],
                "column1": [1, 2, 3, 4, 5],
                "column3": [1.1, 3.3, 2.2, 4.4, 5.5],
                "_column2": ["a", "b", "c", "d", "e"],
                "_column4": [True, False, True, False, False],
                "_column5": pd.date_range("2023-01-01", periods=5),
            },
        )
        self.smart_app.init_data()
        pd.testing.assert_frame_equal(expected_result, self.smart_app.round_dataframe)

    def test_limited_rows_init_data(self):
        self.smart_app.init_data(3)
        assert len(self.smart_app.round_dataframe) == 3

    def test_select_data_from_prediction_picking(self):
        selected_data = {"points": [{"customdata": 0}, {"customdata": 2}]}
        expected_result = pd.DataFrame(
            {
                "column1": [1, 3],
                "column2": ["a", "c"],
                "column3": [1.1, 2.2],
                "column4": [True, True],
                "column5": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")],
            },
            index=[0, 2],
        )
        result = select_data_from_prediction_picking(self.df, selected_data)
        pd.testing.assert_frame_equal(expected_result, result)

    def test_select_data_from_str_filters(self):
        round_dataframe = self.df
        id_feature = [{"type": "var_dropdown", "index": 1}]
        feature_id = [id_feature[i]["index"] for i in range(len(id_feature))]
        id_str_modality = [{"type": "dynamic-str", "index": 1}]
        val_feature = ["column2"]
        val_str_modality = [["a", "c"]]

        expected_result = pd.DataFrame(
            {
                "column1": [1, 3],
                "column2": ["a", "c"],
                "column3": [1.1, 2.2],
                "column4": [True, True],
                "column5": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")],
            },
            index=[0, 2],
        )
        result = select_data_from_str_filters(
            round_dataframe,
            feature_id,
            id_str_modality,
            val_feature,
            val_str_modality,
        )
        pd.testing.assert_frame_equal(expected_result, result)

    def test_select_data_from_bool_filters(self):
        round_dataframe = self.df
        id_feature = [{"type": "var_dropdown", "index": 2}]
        feature_id = [id_feature[i]["index"] for i in range(len(id_feature))]
        id_bool_modality = [{"type": "dynamic-bool", "index": 2}]
        val_feature = ["column4"]
        val_bool_modality = [True]

        expected_result = pd.DataFrame(
            {
                "column1": [1, 3],
                "column2": ["a", "c"],
                "column3": [1.1, 2.2],
                "column4": [True, True],
                "column5": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")],
            },
            index=[0, 2],
        )
        result = select_data_from_bool_filters(
            round_dataframe,
            feature_id,
            id_bool_modality,
            val_feature,
            val_bool_modality,
        )
        pd.testing.assert_frame_equal(expected_result, result)

    def test_select_data_from_date_filters(self):
        round_dataframe = self.df
        id_feature = [{"type": "var_dropdown", "index": 1}]
        feature_id = [id_feature[i]["index"] for i in range(len(id_feature))]
        id_date = [{"type": "dynamic-date", "index": 1}]
        val_feature = ["column5"]
        start_date = [pd.Timestamp("2023-01-01")]
        end_date = [pd.Timestamp("2023-01-03")]

        expected_result = pd.DataFrame(
            {
                "column1": [1, 2, 3],
                "column2": ["a", "b", "c"],
                "column3": [1.1, 3.3, 2.2],
                "column4": [True, False, True],
                "column5": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
            },
            index=[0, 1, 2],
        )
        result = select_data_from_date_filters(
            round_dataframe,
            feature_id,
            id_date,
            val_feature,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)

    def test_select_data_from_numeric_filters(self):
        round_dataframe = self.df
        id_feature = [{"type": "var_dropdown", "index": 1}, {"type": "var_dropdown", "index": 2}]
        feature_id = [id_feature[i]["index"] for i in range(len(id_feature))]
        id_lower_modality = [{"type": "lower", "index": 1}, {"type": "lower", "index": 2}]
        val_feature = ["column1", "column3"]
        val_lower_modality = [0, 0]
        val_upper_modality = [3, 3]

        expected_result = pd.DataFrame(
            {
                "column1": [1, 3],
                "column2": ["a", "c"],
                "column3": [1.1, 2.2],
                "column4": [True, True],
                "column5": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")],
            },
            index=[0, 2],
        )
        result = select_data_from_numeric_filters(
            round_dataframe,
            feature_id,
            id_lower_modality,
            val_feature,
            val_lower_modality,
            val_upper_modality,
        )
        pd.testing.assert_frame_equal(expected_result, result)

    def test_get_feature_from_clicked_data(self):
        feature = get_feature_from_clicked_data(self.click_data)
        assert feature == "Sex"

    def test_get_feature_from_features_groups(self):
        features_groups = {"A": ["column1", "column3"]}
        feature = get_feature_from_features_groups("column3", features_groups)
        assert feature == "A"

        feature = get_feature_from_features_groups("A", features_groups)
        assert feature == "A"

        feature = get_feature_from_features_groups("column2", features_groups)
        assert feature == "column2"

    def test_get_indexes_from_datatable(self):
        data = self.smart_app.components["table"]["dataset"].data
        subset = get_indexes_from_datatable(data)
        assert subset == [0, 1, 2, 3, 4]

    def test_get_indexes_from_datatable_no_subset(self):
        data = self.smart_app.components["table"]["dataset"].data
        subset = get_indexes_from_datatable(data, [0, 1, 2, 3, 4])
        assert subset == None

    def test_get_indexes_from_datatable_empty(self):
        subset = get_indexes_from_datatable([], [0, 1, 2, 3, 4])
        assert subset == None

    def test_update_click_data_on_subset_changes(self):
        click_data = {
            "points": [
                {
                    "curveNumber": 1,
                    "pointNumber": 3,
                    "pointIndex": 3,
                    "x": 0.4649,
                    "y": "Sex",
                    "label": "Sex",
                    "value": 0.4649,
                    "customdata": "Sex",
                    "marker.color": "rgba(244, 192, 0, 1.0)",
                    "bbox": {"x0": 717.3, "x1": 717.3, "y0": 82.97, "y1": 130.78},
                }
            ]
        }
        click_data = update_click_data_on_subset_changes(click_data)
        assert click_data == self.click_data

    def test_get_figure_zoom(self):
        zoom_active = get_figure_zoom(None)
        assert zoom_active == False

        zoom_active = get_figure_zoom(1)
        assert zoom_active == True

        zoom_active = get_figure_zoom(4)
        assert zoom_active == False

    def test_get_feature_contributions_sign_to_show(self):
        sign = get_feature_contributions_sign_to_show([1], [1])
        assert sign == None

        sign = get_feature_contributions_sign_to_show([1], [])
        assert sign == True

        sign = get_feature_contributions_sign_to_show([], [])
        assert sign == None

        sign = get_feature_contributions_sign_to_show([], [1])
        assert sign == False

    def test_update_features_to_display(self):
        value, max, marks = update_features_to_display(20, 40, 22)
        assert value == 20
        assert max == 20
        assert marks == {"1": "1", "5": "5", "10": "10", "15": "15", "20": "20"}

        value, max, marks = update_features_to_display(7, 40, 6)
        assert value == 6
        assert max == 7
        assert marks == {"1": "1", "7": "7"}

    def test_get_id_card_features(self):
        data = self.smart_app.components["table"]["dataset"].data
        features_dict = copy.deepcopy(self.xpl.features_dict)
        features_dict.update(self.xpl.additional_features_dict)
        selected_row = get_id_card_features(data, 3, self.special_cols, features_dict)
        expected_result = pd.DataFrame(
            {
                "feature_value": [3, 1, 1, 4, 4.4, "d", False, pd.Timestamp("2023-01-04")],
                "feature_name": [
                    "_index_",
                    "_predict_",
                    "_target_",
                    "column1",
                    "Useless col",
                    "_Additional col",
                    "_column4",
                    "_column5",
                ],
            },
            index=["_index_", "_predict_", "_target_", "column1", "column3", "_column2", "_column4", "_column5"],
        )
        pd.testing.assert_frame_equal(selected_row, expected_result)

    def test_get_id_card_contrib(self):
        data = self.xpl.data
        selected_contrib = get_id_card_contrib(data, 3, self.xpl.features_dict, self.xpl.columns_dict, 0)
        assert set(selected_contrib["feature_name"]) == {"Useless col", "column1"}
        assert selected_contrib.columns.tolist() == ["feature_name", "feature_contrib"]

    def test_create_id_card_data(self):
        selected_row = pd.DataFrame(
            {
                "feature_value": [3, 1, 1, 4, 4.4, "d", False, pd.Timestamp("2023-01-04")],
                "feature_name": [
                    "_index_",
                    "_predict_",
                    "_target_",
                    "column1",
                    "Useless col",
                    "_Additional col",
                    "_column4",
                    "_column5",
                ],
            },
            index=["_index_", "_predict_", "_target_", "column1", "column3", "_column2", "_column4", "_column5"],
        )

        selected_contrib = pd.DataFrame(
            {
                "feature_name": ["column1", "Useless col"],
                "feature_contrib": [-0.6, 0],
            }
        )

        selected_data = create_id_card_data(
            selected_row, selected_contrib, "feature_name", True, self.special_cols, self.xpl.additional_features_dict
        )
        expected_result = pd.DataFrame(
            {
                "feature_value": [3, 1, 1, 4.4, 4, "d", False, pd.Timestamp("2023-01-04")],
                "feature_name": [
                    "_index_",
                    "_predict_",
                    "_target_",
                    "Useless col",
                    "column1",
                    "_Additional col",
                    "_column4",
                    "_column5",
                ],
                "feature_contrib": [np.nan, np.nan, np.nan, 0.0, -0.6, np.nan, np.nan, np.nan],
            },
            index=["_index_", "_predict_", "_target_", "column3", "column1", "_column2", "_column4", "_column5"],
        )
        pd.testing.assert_frame_equal(selected_data, expected_result)

    def test_create_id_card_layout(self):
        selected_data = pd.DataFrame(
            {
                "feature_value": [3, 1, 1, 4.4, 4, "d"],
                "feature_name": ["_index_", "_predict_", "_target_", "Useless col", "column1", "_Additional col"],
                "feature_contrib": [np.nan, np.nan, np.nan, 0.0, -0.6, np.nan],
            },
            index=["_index_", "_predict_", "_target_", "column3", "column1", "_column2"],
        )
        children = create_id_card_layout(selected_data, self.xpl.additional_features_dict)
        assert len(children) == 6

    def test_get_feature_filter_options(self):
        features_dict = copy.deepcopy(self.xpl.features_dict)
        features_dict.update(self.xpl.additional_features_dict)
        options = get_feature_filter_options(self.smart_app.dataframe, features_dict, self.special_cols)
        assert [option["label"] for option in options] == [
            "_index_",
            "_predict_",
            "_target_",
            "Useless col",
            "_Additional col",
            "_column4",
            "_column5",
            "column1",
        ]

    def test_create_filter_modalities_selection(self):
        new_element = create_filter_modalities_selection(
            "column3", {"type": "var_dropdown", "index": 1}, self.smart_app.round_dataframe
        )
        assert type(new_element.children[0]) == dcc.Input

        new_element = create_filter_modalities_selection(
            "_column2", {"type": "var_dropdown", "index": 1}, self.smart_app.round_dataframe
        )
        assert type(new_element.children) == dcc.Dropdown

    def test_handle_page_navigation_1(self):
        page, selected_feature = handle_page_navigation(
            triggered_input="page_left.n_clicks", page="3", selected_feature="column1"
        )
        assert page == 2
        assert selected_feature == None

    def test_handle_page_navigation_2(self):
        page, selected_feature = handle_page_navigation(
            triggered_input="page_right.n_clicks", page="3", selected_feature="column1"
        )
        assert page == 4
        assert selected_feature == None

    def test_handle_page_navigation_3(self):
        page, selected_feature = handle_page_navigation(
            triggered_input="bool_groups.on", page="3", selected_feature="column1"
        )
        assert page == 1
        assert selected_feature == None

    def test_handle_page_navigation_4(self):
        page, selected_feature = handle_page_navigation(triggered_input="erreur", page="3", selected_feature="column1")
        assert page == 3
        assert selected_feature == "column1"

    def test_create_dropdown_feature_filter(self):
        dropdown = create_dropdown_feature_filter(1, [])
        assert type(dropdown) == html.Div
