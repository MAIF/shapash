import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp
from shapash.webapp.utils.callbacks import select_data_from_prediction_picking, select_data_from_filters


class TestCallbacks(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        data = {
            'column1': [1, 2, 3, 4, 5],
            'column2': ['a', 'b', 'c', 'd', 'e'],
            'column3': [1.1, 3.3, 2.2, 4.4, 5.5],
            'column4': [True, False, True, False, False],
            'column5': pd.date_range('2023-01-01', periods=5),
        }

        df = pd.DataFrame(data)
        self.df = df

        dataframe_x = df[['column1','column3']].copy()
        y_target = pd.DataFrame(data=np.array([1, 2, 3, 4, 5]), columns=['pred'])
        model = DecisionTreeClassifier().fit(dataframe_x, y_target)
        features_dict = {'column3': 'Useless col'}
        additional_data = df[['column2']].copy()
        additional_features_dict = {'column2': 'Additional col'}
        self.xpl = SmartExplainer(model=model, features_dict=features_dict)
        self.xpl.compile(
            x=dataframe_x, 
            y_pred=y_target,
            y_target=y_target,
            additional_data=additional_data, 
            additional_features_dict=additional_features_dict
        )
        self.smart_app = SmartApp(self.xpl)
        super(TestCallbacks, self).__init__(*args, **kwargs)

    def test_default_init_data(self):
        expected_result = pd.DataFrame(
            {
                '_index_': [0, 1, 2, 3, 4],
                '_predict_': [1, 2, 3, 4, 5],
                '_target_': [1, 2, 3, 4, 5],
                'column1': [1, 2, 3, 4, 5],
                'column3': [1.1, 3.3, 2.2, 4.4, 5.5],
                '_column2': ['a', 'b', 'c', 'd', 'e'],
            },
        )
        self.smart_app.init_data()
        pd.testing.assert_frame_equal(expected_result, self.smart_app.round_dataframe)
    
    def test_limited_rows_init_data(self):
        self.smart_app.init_data(3)
        assert len(self.smart_app.round_dataframe)==3

    def test_select_data_from_prediction_picking(self):
        selected_data = {"points": [{"customdata":0}, {"customdata":2}]}
        expected_result = pd.DataFrame(
            {
                'column1': [1, 3],
                'column2': ['a', 'c'],
                'column3': [1.1, 2.2],
                'column4': [True, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 2]
        )
        result = select_data_from_prediction_picking(self.df, selected_data)
        pd.testing.assert_frame_equal(expected_result, result)
    
    def test_select_data_from_filters_string(self):
        round_dataframe = self.df
        id_feature = [{'type': 'var_dropdown', 'index': 1}]
        id_str_modality = [{'type': 'dynamic-str', 'index': 1}]
        id_bool_modality = []
        id_lower_modality = []
        id_date = []
        val_feature = ['column2'] 
        val_str_modality = [['a', 'c']]
        val_bool_modality = []
        val_lower_modality = []
        val_upper_modality = []
        start_date = []
        end_date = []

        expected_result = pd.DataFrame(
            {
                'column1': [1, 3],
                'column2': ['a', 'c'],
                'column3': [1.1, 2.2],
                'column4': [True, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 2]
        )
        result = select_data_from_filters(
            round_dataframe,
            id_feature, 
            id_str_modality, 
            id_bool_modality, 
            id_lower_modality, 
            id_date, 
            val_feature, 
            val_str_modality,
            val_bool_modality,
            val_lower_modality,
            val_upper_modality,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)
    
    def test_select_data_from_filters_bool(self):
        round_dataframe = self.df
        id_feature = [{'type': 'var_dropdown', 'index': 2}]
        id_str_modality = []
        id_bool_modality = [{'type': 'dynamic-bool', 'index': 2}]
        id_lower_modality = []
        id_date = []
        val_feature = ['column4']
        val_str_modality = []
        val_bool_modality = [True]
        val_lower_modality = []
        val_upper_modality = []
        start_date = []
        end_date = []

        expected_result = pd.DataFrame(
            {
                'column1': [1, 3],
                'column2': ['a', 'c'],
                'column3': [1.1, 2.2],
                'column4': [True, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 2]
        )
        result = select_data_from_filters(
            round_dataframe,
            id_feature, 
            id_str_modality, 
            id_bool_modality, 
            id_lower_modality, 
            id_date, 
            val_feature, 
            val_str_modality,
            val_bool_modality,
            val_lower_modality,
            val_upper_modality,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)
    
    def test_select_data_from_filters_date(self):
        round_dataframe = self.df
        id_feature = [{'type': 'var_dropdown', 'index': 1}]
        id_str_modality = []
        id_bool_modality = []
        id_lower_modality = []
        id_date = [{'type': 'dynamic-date', 'index': 1}]
        val_feature = ['column5']
        val_str_modality = []
        val_bool_modality = []
        val_lower_modality = []
        val_upper_modality = []
        start_date = [pd.Timestamp('2023-01-01')]
        end_date = [pd.Timestamp('2023-01-03')]

        expected_result = pd.DataFrame(
            {
                'column1': [1, 2, 3],
                'column2': ['a', 'b', 'c'],
                'column3': [1.1, 3.3, 2.2],
                'column4': [True, False, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 1, 2]
        )
        result = select_data_from_filters(
            round_dataframe,
            id_feature, 
            id_str_modality, 
            id_bool_modality, 
            id_lower_modality, 
            id_date, 
            val_feature, 
            val_str_modality,
            val_bool_modality,
            val_lower_modality,
            val_upper_modality,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)
    
    def test_select_data_from_filters_numeric(self):
        round_dataframe = self.df
        id_feature = [{'type': 'var_dropdown', 'index': 1}, {'type': 'var_dropdown', 'index': 2}]
        id_str_modality = []
        id_bool_modality = []
        id_lower_modality = [{'type': 'lower', 'index': 1}, {'type': 'lower', 'index': 2}]
        id_date = []
        val_feature = ['column1', 'column3']
        val_str_modality = []
        val_bool_modality = []
        val_lower_modality = [0, 0]
        val_upper_modality = [3, 3]
        start_date = []
        end_date = []

        expected_result = pd.DataFrame(
            {
                'column1': [1, 3],
                'column2': ['a', 'c'],
                'column3': [1.1, 2.2],
                'column4': [True, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 2]
        )
        result = select_data_from_filters(
            round_dataframe,
            id_feature, 
            id_str_modality, 
            id_bool_modality, 
            id_lower_modality, 
            id_date, 
            val_feature, 
            val_str_modality,
            val_bool_modality,
            val_lower_modality,
            val_upper_modality,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)
    
    def test_select_data_from_filters_multi_types(self):
        round_dataframe = self.df
        id_feature = [{'type': 'var_dropdown', 'index': 1}, {'type': 'var_dropdown', 'index': 2}]
        id_str_modality = [{'type': 'dynamic-str', 'index': 2}]
        id_bool_modality = []
        id_lower_modality = [{'type': 'lower', 'index': 1}]
        id_date = []
        val_feature = ['column1', 'column2']
        val_str_modality = [['a', 'c', 'd', 'e']]
        val_bool_modality = []
        val_lower_modality = [0]
        val_upper_modality = [3]
        start_date = []
        end_date = []

        expected_result = pd.DataFrame(
            {
                'column1': [1, 3],
                'column2': ['a', 'c'],
                'column3': [1.1, 2.2],
                'column4': [True, True],
                'column5': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-03')],
            },
            index=[0, 2]
        )
        result = select_data_from_filters(
            round_dataframe,
            id_feature, 
            id_str_modality, 
            id_bool_modality, 
            id_lower_modality, 
            id_date, 
            val_feature, 
            val_str_modality,
            val_bool_modality,
            val_lower_modality,
            val_upper_modality,
            start_date,
            end_date,
        )
        pd.testing.assert_frame_equal(expected_result, result)