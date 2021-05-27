"""
Unit test for smart explainer
"""
import unittest
from unittest.mock import patch, Mock
import os
from os import path
from pathlib import Path
import types
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.linear_model import LinearRegression
from shapash.explainer.smart_explainer import SmartExplainer
from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_state import SmartState
import category_encoders as ce
import shap

def init_sme_to_pickle_test():
    """
    Init sme to pickle test
    TODO: Docstring
    Returns
    -------
    [type]
        [description]
    """
    current = Path(path.abspath(__file__)).parent.parent.parent
    pkl_file = path.join(current, 'data/xpl.pkl')
    xpl = SmartExplainer()
    contributions = pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]])
    y_pred = pd.DataFrame(data=np.array([1, 2]), columns=['pred'])
    dataframe_x = pd.DataFrame([[1, 2, 3], [1, 2, 3]])
    xpl.compile(contributions=contributions, x=dataframe_x, y_pred=y_pred, model=LinearRegression())
    xpl.filter(max_contrib=2)
    return pkl_file, xpl

class TestSmartExplainer(unittest.TestCase):
    """
    Unit test smart explainer
    TODO: Docstring
    """
    def test_init(self):
        """
        test init smart explainer
        """
        xpl = SmartExplainer()
        assert hasattr(xpl, 'plot')

    def assertRaisesWithMessage(self, msg, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            self.assertFail()
        except Exception as inst:
            self.assertEqual(inst.args[0]['message'], msg)

    @patch('shapash.explainer.smart_explainer.SmartState')
    def test_choose_state_1(self, mock_smart_state):
        """
        Unit test choose state 1
        Parameters
        ----------
        mock_smart_state : [type]
            [description]
        """
        xpl = SmartExplainer()
        xpl.choose_state('contributions')
        mock_smart_state.assert_called()

    @patch('shapash.explainer.smart_explainer.MultiDecorator')
    def test_choose_state_2(self, mock_multi_decorator):
        """
        Unit test choose state 2
        Parameters
        ----------
        mock_multi_decorator : [type]
            [description]
        """
        xpl = SmartExplainer()
        xpl.choose_state([1, 2, 3])
        mock_multi_decorator.assert_called()

    def test_validate_contributions_1(self):
        """
        Unit test validate contributions 1
        """
        xpl = SmartExplainer()
        contributions = [
            np.array([[2, 1], [8, 4]]),
            np.array([[5, 5], [0, 0]])
        ]
        model = Mock()
        model._classes = np.array([1, 3])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        xpl.model = model
        xpl._case = "classification"
        xpl._classes = list(model._classes)
        xpl.state = xpl.choose_state(contributions)
        xpl.x_init = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        expected_output = [
            pd.DataFrame(
                [[2, 1], [8, 4]],
                columns=['Col1', 'Col2'],
                index=['Id1', 'Id2']
            ),
            pd.DataFrame(
                [[5, 5], [0, 0]],
                columns=['Col1', 'Col2'],
                index=['Id1', 'Id2']
            )
        ]
        output = xpl.validate_contributions(contributions)
        assert len(expected_output) == len(output)
        test_list = [pd.testing.assert_frame_equal(e, m) for e, m in zip(expected_output, output)]
        assert all(x is None for x in test_list)

    def test_apply_preprocessing_1(self):
        """
        Unit test apply preprocessing 1
        """
        xpl = SmartExplainer()
        contributions = [1, 2, 3]
        output = xpl.apply_preprocessing(contributions)
        expected = contributions
        self.assertListEqual(output, expected)

    def test_apply_preprocessing_2(self):
        """
        Unit test apply preprocessing 2
        """
        xpl = SmartExplainer()
        xpl.state = Mock()
        preprocessing = Mock()
        contributions = [1, 2, 3]
        xpl.apply_preprocessing(contributions, preprocessing)
        xpl.state.inverse_transform_contributions.assert_called()

    def test_modify_postprocessing_1(self):
        """
        Unit test modify postprocessing 1
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        xpl.features_dict = {'Col1': 'Column1', 'Col2': 'Column2'}
        xpl.columns_dict = {0: 'Col1', 1:'Col2'}
        xpl.inv_features_dict = {'Column1': 'Col1', 'Column2': 'Col2'}
        postprocessing = {0: {'type' : 'suffix', 'rule':' t'},
            'Column2': {'type' : 'prefix', 'rule' : 'test'}}

        expected_output = {
            'Col1': {'type' : 'suffix', 'rule':' t'},
            'Col2': {'type' : 'prefix', 'rule' : 'test'}
        }
        output = xpl.modify_postprocessing(postprocessing)
        assert output == expected_output

    def test_modify_postprocessing_2(self):
        """
        Unit test modify postprocessing 2
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        xpl.features_dict = {'Col1': 'Column1', 'Col2': 'Column2'}
        xpl.columns_dict = {0: 'Col1', 1: 'Col2'}
        xpl.inv_features_dict = {'Column1': 'Col1', 'Column2': 'Col2'}
        postprocessing = {'Error': {'type': 'suffix', 'rule': ' t'}}
        with self.assertRaises(ValueError):
            xpl.modify_postprocessing(postprocessing)

    def test_check_postprocessing_1(self):
        """
        Unit test check_postprocessing
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        xpl.features_dict = {'Col1': 'Column1', 'Col2': 'Column2'}
        xpl.columns_dict = {0: 'Col1', 1: 'Col2'}
        xpl.inv_features_dict = {'Column1': 'Col1', 'Column2': 'Col2'}
        postprocessing1 = {0: {'Error': 'suffix', 'rule': ' t'}}
        postprocessing2 = {0: {'type': 'Error', 'rule': ' t'}}
        postprocessing3 = {0: {'type': 'suffix', 'Error': ' t'}}
        postprocessing4 = {0: {'type': 'suffix', 'rule': ' '}}
        postprocessing5 = {0: {'type': 'case', 'rule': 'lower'}}
        postprocessing6 = {0: {'type': 'case', 'rule': 'Error'}}
        with self.assertRaises(ValueError):
            xpl.check_postprocessing(postprocessing1)
            xpl.check_postprocessing(postprocessing2)
            xpl.check_postprocessing(postprocessing3)
            xpl.check_postprocessing(postprocessing4)
            xpl.check_postprocessing(postprocessing5)
            xpl.check_postprocessing(postprocessing6)

    def test_apply_postprocessing_1(self):
        """
        Unit test apply_postprocessing 1
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        xpl.features_dict = {'Col1': 'Column1', 'Col2': 'Column2'}
        xpl.columns_dict = {0: 'Col1', 1: 'Col2'}
        xpl.inv_features_dict = {'Column1': 'Col1', 'Column2': 'Col2'}
        assert np.array_equal(xpl.x_pred, xpl.apply_postprocessing())

    def test_apply_postprocessing_2(self):
        """
        Unit test apply_postprocessing 2
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[1, 2],
             [3, 4]],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        xpl.features_dict = {'Col1': 'Column1', 'Col2': 'Column2'}
        xpl.columns_dict = {0: 'Col1', 1: 'Col2'}
        xpl.inv_features_dict = {'Column1': 'Col1', 'Column2': 'Col2'}
        postprocessing = {'Col1': {'type': 'suffix', 'rule': ' t'},
                          'Col2': {'type': 'prefix', 'rule': 'test'}}
        expected_output = pd.DataFrame(
            data=[['1 t', 'test2'],
                  ['3 t', 'test4']],
            columns=['Col1', 'Col2'],
            index=['Id1', 'Id2']
        )
        output = xpl.apply_postprocessing(postprocessing)
        assert np.array_equal(output, expected_output)

    def test_check_contributions_1(self):
        """
        Unit test check contributions 1
        """
        xpl = SmartExplainer()
        xpl.contributions, xpl.x_pred = Mock(), Mock()
        xpl.state = Mock()
        xpl.check_contributions()
        xpl.state.check_contributions.assert_called_with(xpl.contributions, xpl.x_pred)

    def test_check_contributions_2(self):
        """
        Unit test check contributions 2
        """
        xpl = SmartExplainer()
        xpl.contributions, xpl.x_pred = Mock(), Mock()
        mock_state = Mock()
        mock_state.check_contributions.return_value = False
        xpl.state = mock_state
        with self.assertRaises(ValueError):
            xpl.check_contributions()

    def test_check_label_dict_1(self):
        """
        Unit test check label dict 1
        """
        xpl = SmartExplainer(label_dict={1: 'Yes', 0: 'No'})
        xpl._classes = [0, 1]
        xpl._case = 'classification'
        xpl.check_label_dict()

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        xpl = SmartExplainer()
        xpl._case = 'regression'
        xpl.check_label_dict()

    def test_check_features_dict_1(self):
        """
        Unit test check features dict 1
        """
        xpl = SmartExplainer(features_dict={'Age': 'Age (Years Old)'})
        xpl.columns_dict = {0: 'Age', 1: 'Education', 2: 'Sex'}
        xpl.check_features_dict()
        assert xpl.features_dict['Age'] == 'Age (Years Old)'
        assert xpl.features_dict['Education'] == 'Education'

    @patch('shapash.explainer.smart_explainer.SmartExplainer.choose_state')
    @patch('shapash.explainer.smart_explainer.SmartExplainer.apply_preprocessing')
    def test_compile_0(self, mock_apply_preprocessing, mock_choose_state):
        """
        Unit test compile
        Parameters
        ----------
        mock_apply_preprocessing : [type]
            [description]
        mock_choose_state : [type]
            [description]
        """
        xpl = SmartExplainer()
        mock_state = Mock()
        mock_choose_state.return_value = mock_state
        model = lambda: None
        model.predict = types.MethodType(self.predict, model)
        mock_state.rank_contributions.return_value = 1, 2, 3
        contributions = pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]])
        mock_state.validate_contributions.return_value = contributions
        mock_apply_preprocessing.return_value = contributions
        x_pred = pd.DataFrame([[1, 2, 3], [1, 2, 3]])
        xpl.compile(x=x_pred, model=model, contributions=contributions)
        assert hasattr(xpl, 'state')
        assert xpl.state == mock_state
        assert hasattr(xpl, 'x_pred')
        pd.testing.assert_frame_equal(xpl.x_pred, x_pred)
        assert hasattr(xpl, 'contributions')
        pd.testing.assert_frame_equal(xpl.contributions, contributions)
        mock_choose_state.assert_called()
        mock_state.validate_contributions.assert_called()
        mock_apply_preprocessing.assert_called()
        mock_state.rank_contributions.assert_called()
        assert xpl._case == "regression"

    def test_compile_1(self):
        """
        Unit test compile 1
        checking compile method without model
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        xpl = SmartExplainer()
        xpl.compile(model=clf, x=df[['x1', 'x2']])
        assert xpl._case == "classification"
        self.assertListEqual(xpl._classes, [0, 1])

    def test_compile_2(self):
        """
        Unit test compile 2
        checking new attributes added to the compile method
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        output = df[["x1", "x2"]].copy()
        output["x2"] = ["single", "married", "single", "divorced", "married"]
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])

        postprocessing_1 = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        postprocessing_2 = {
            "family_situation": {
                "type": "transcoding",
                "rule": {"S": "single", "M": "married", "D": "divorced"}}}

        xpl_postprocessing1 = SmartExplainer()
        xpl_postprocessing2 = SmartExplainer(features_dict={"x1": "age",
                                                            "x2": "family_situation"}
                                             )
        xpl_postprocessing3 = SmartExplainer()

        xpl_postprocessing1.compile(model=clf,
                                    x=df_encoded[['x1', 'x2']],
                                    preprocessing=encoder_fitted,
                                    postprocessing=postprocessing_1)
        xpl_postprocessing2.compile(model=clf,
                                    x=df_encoded[['x1', 'x2']],
                                    preprocessing=encoder_fitted,
                                    postprocessing=postprocessing_2)
        xpl_postprocessing3.compile(model=clf,
                                    x=df_encoded[['x1', 'x2']],
                                    preprocessing=None,
                                    postprocessing=None)

        assert hasattr(xpl_postprocessing1, "preprocessing")
        assert hasattr(xpl_postprocessing1, "postprocessing")
        assert hasattr(xpl_postprocessing2, "preprocessing")
        assert hasattr(xpl_postprocessing2, "postprocessing")
        assert hasattr(xpl_postprocessing3, "preprocessing")
        assert hasattr(xpl_postprocessing3, "postprocessing")
        pd.testing.assert_frame_equal(xpl_postprocessing1.x_pred, output)
        pd.testing.assert_frame_equal(xpl_postprocessing2.x_pred, output)
        assert xpl_postprocessing1.preprocessing == encoder_fitted
        assert xpl_postprocessing2.preprocessing == encoder_fitted
        assert xpl_postprocessing1.postprocessing == postprocessing_1
        assert xpl_postprocessing2.postprocessing == postprocessing_1

    def test_compile_3(self):
        """
        Unit test compile 3
        checking compile method without model
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)

        contrib = pd.DataFrame(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )

        xpl = SmartExplainer()
        with self.assertRaises(ValueError):
            xpl.compile(model=clf, x=df[['x1', 'x2']], explainer=clf_explainer, contributions=contrib)

    def test_filter_0(self):
        """
        Unit test filter 0
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter()
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_not_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_not_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    @patch('shapash.explainer.smart_explainer.SmartExplainer.check_features_name')
    def test_filter_1(self, mock_check_features_name):
        """
        Unit test filter 1
        Parameters
        ----------
        mock_check_features_name : [type]
            [description]
        """
        xpl = SmartExplainer()
        mock_check_features_name.return_value = [1, 2]
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter(features_to_hide=['X1', 'X2'])
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_not_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_not_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_2(self):
        """
        Unit test filter 2
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter(threshold=0.1)
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_called()
        mock_state.sign_contributions.assert_not_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_not_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_3(self):
        """
        Unit test filter 3
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter(positive=True)
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_not_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_4(self):
        """
        Unit test filter 4
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter(max_contrib=10)
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_not_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_5(self):
        """
        Unit test filter 5
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter(positive=True, max_contrib=10)
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_6(self):
        """
        Unit test filter 6
        """
        xpl = SmartExplainer()
        mock_data = {'var_dict': 1, 'contrib_sorted': 2, 'x_sorted': 3}
        xpl.data = mock_data
        mock_state = Mock()
        xpl.state = mock_state
        xpl.filter()
        mock_state.init_mask.assert_called()
        mock_state.hide_contributions.assert_not_called()
        mock_state.cap_contributions.assert_not_called()
        mock_state.sign_contributions.assert_not_called()
        mock_state.combine_masks.assert_called()
        mock_state.cutoff_contributions.assert_not_called()
        assert hasattr(xpl, 'mask')
        mock_state.compute_masked_contributions.assert_called()
        assert hasattr(xpl, 'masked_contributions')

    def test_filter_7(self):
        """
        Unit test filter 7
        """
        xpl = SmartExplainer()
        contributions = [
            pd.DataFrame(
                data=[[0.5, 0.4, 0.3], [0.9, 0.8, 0.7]],
                columns=['Col1', 'Col2', 'Col3']
            ),
            pd.DataFrame(
                data=[[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]],
                columns=['Col1', 'Col2', 'Col3']
            )
        ]
        xpl.data = {'var_dict': 1, 'contrib_sorted': contributions, 'x_sorted': 3}
        xpl.state = MultiDecorator(SmartState())
        xpl.filter(threshold=0.5, max_contrib=2)
        expected_mask = [
            pd.DataFrame(
                data=[[True, False, False], [True, True, False]],
                columns=['contrib_1', 'contrib_2', 'contrib_3']
            ),
            pd.DataFrame(
                data=[[False, False, False], [True, True, False]],
                columns=['contrib_1', 'contrib_2', 'contrib_3']
            )
        ]
        assert len(expected_mask) == len(xpl.mask)
        test_list = [pd.testing.assert_frame_equal(e, m) for e, m in zip(expected_mask, xpl.mask)]
        assert all(x is None for x in test_list)
        expected_masked_contributions = [
            pd.DataFrame(
                data=[[0.0, 0.7], [0.0, 0.7]],
                columns=['masked_neg', 'masked_pos']
            ),
            pd.DataFrame(
                data=[[0.0, 0.6], [0.0, 0.4]],
                columns=['masked_neg', 'masked_pos']
            )
        ]
        assert len(expected_masked_contributions) == len(xpl.masked_contributions)
        test_list = [pd.testing.assert_frame_equal(e, m) for e, m in
                     zip(expected_masked_contributions, xpl.masked_contributions)]
        assert all(x is None for x in test_list)
        expected_param_dict = {
            'features_to_hide': None,
            'threshold': 0.5,
            'positive': None,
            'max_contrib': 2
        }
        self.assertDictEqual(expected_param_dict, xpl.mask_params)

    def test_check_label_name_1(self):
        """
        Unit test check label name 1
        """
        label_dict = {1: 'Age', 2: 'Education'}
        xpl = SmartExplainer(label_dict=label_dict)
        xpl.inv_label_dict = {v: k for k, v in xpl.label_dict.items()}
        xpl._classes = [1, 2]
        entry = 'Age'
        expected_num = 0
        expected_code = 1
        expected_value = 'Age'
        label_num, label_code, label_value = xpl.check_label_name(entry, 'value')
        assert expected_num == label_num
        assert expected_code == label_code
        assert expected_value == label_value

    def test_check_label_name_2(self):
        """
        Unit test check label name 2
        """
        xpl = SmartExplainer(label_dict = None)
        xpl._classes = [1, 2]
        entry = 1
        expected_num = 0
        expected_code = 1
        expected_value = 1
        label_num, label_code, label_value = xpl.check_label_name(entry, 'code')
        assert expected_num == label_num
        assert expected_code == label_code
        assert expected_value == label_value

    def test_check_label_name_3(self):
        """
        Unit test check label name 3
        """
        label_dict = {1: 'Age', 2: 'Education'}
        xpl = SmartExplainer(label_dict=label_dict)
        xpl.inv_label_dict = {v: k for k, v in xpl.label_dict.items()}
        xpl._classes = [1, 2]
        entry = 0
        expected_num = 0
        expected_code = 1
        expected_value = 'Age'
        label_num, label_code, label_value = xpl.check_label_name(entry, 'num')
        assert expected_num == label_num
        assert expected_code == label_code
        assert expected_value == label_value

    def test_check_label_name_4(self):
        """
        Unit test check label name 4
        """
        xpl = SmartExplainer()
        label = 0
        origin = 'error'
        expected_msg = "Origin must be 'num', 'code' or 'value'."
        self.assertRaisesWithMessage(expected_msg, xpl.check_label_name, **{'label': label, 'origin': origin})

    def test_check_label_name_5(self):
        """
        Unit test check label name 5
        """
        label_dict = {1: 'Age', 2: 'Education'}
        xpl = SmartExplainer(label_dict=label_dict)
        xpl.inv_label_dict = {v: k for k, v in xpl.label_dict.items()}
        xpl._classes = [1, 2]
        label = 'Absent'
        expected_msg = f"Label (Absent) not found for origin (value)"
        origin = 'value'
        self.assertRaisesWithMessage(expected_msg, xpl.check_label_name, **{'label': label, 'origin': origin})

    def test_check_features_name_1(self):
        """
        Unit test check features name 1
        """
        xpl = SmartExplainer()
        xpl.features_dict = {'tech_0': 'domain_0', 'tech_1': 'domain_1', 'tech_2': 'domain_2'}
        xpl.inv_features_dict = {v: k for k, v in xpl.features_dict.items()}
        xpl.columns_dict = {0: 'tech_0', 1: 'tech_1', 2: 'tech_2'}
        xpl.inv_columns_dict = {v: k for k, v in xpl.columns_dict.items()}
        feature_list_1 = ['domain_0', 'tech_1']
        feature_list_2 = ['domain_0', 0]
        self.assertRaises(ValueError, xpl.check_features_name, feature_list_1)
        self.assertRaises(ValueError, xpl.check_features_name, feature_list_2)

    def test_check_features_name_2(self):
        """
        Unit test check features name 2
        """
        xpl = SmartExplainer()
        xpl.features_dict = {'tech_0': 'domain_0', 'tech_1': 'domain_1', 'tech_2': 'domain_2'}
        xpl.inv_features_dict = {v: k for k, v in xpl.features_dict.items()}
        xpl.columns_dict = {0: 'tech_0', 1: 'tech_1', 2: 'tech_2'}
        xpl.inv_columns_dict = {v: k for k, v in xpl.columns_dict.items()}
        feature_list = ['domain_0', 'domain_2']
        output = xpl.check_features_name(feature_list)
        expected_output = [0, 2]
        np.testing.assert_array_equal(output, expected_output)

    def test_check_features_name_3(self):
        """
        Unit test check features name 3
        """
        xpl = SmartExplainer()
        xpl.columns_dict = {0: 'tech_0', 1: 'tech_1', 2: 'tech_2'}
        xpl.inv_columns_dict = {v: k for k, v in xpl.columns_dict.items()}
        feature_list = ['tech_2']
        output = xpl.check_features_name(feature_list)
        expected_output = [2]
        np.testing.assert_array_equal(output, expected_output)

    def test_check_features_name_4(self):
        """
        Unit test check features name 4
        """
        xpl = SmartExplainer()
        xpl.columns_dict = None
        xpl.features_dict = None
        feature_list = [1, 2, 4]
        output = xpl.check_features_name(feature_list)
        expected_output = feature_list
        np.testing.assert_array_equal(output, expected_output)

    def test_save_1(self):
        """
        Unit test save 1
        """
        pkl_file, xpl = init_sme_to_pickle_test()
        xpl.save(pkl_file)
        assert path.exists(pkl_file)
        os.remove(pkl_file)

    def test_load_1(self):
        """
        Unit test load 1
        """
        temp, xpl = init_sme_to_pickle_test()
        xpl2 = SmartExplainer()
        current = Path(path.abspath(__file__)).parent.parent.parent
        pkl_file = path.join(current, 'data/xpl_to_load.pkl')
        xpl2.load(pkl_file)

        attrib_xpl = [element for element in xpl.__dict__.keys()]
        attrib_xpl2 = [element for element in xpl2.__dict__.keys()]

        assert all(attrib in attrib_xpl2 for attrib in attrib_xpl)
        assert all(attrib2 in attrib_xpl for attrib2 in attrib_xpl2)

    def test_save_load(self):
        """
        Test save + load methods
        """
        pkl_file, xpl = init_sme_to_pickle_test()
        xpl.save(pkl_file)
        xpl2 = SmartExplainer()
        xpl2.load(pkl_file)

        attrib_xpl = [element for element in xpl.__dict__.keys()]
        attrib_xpl2 = [element for element in xpl2.__dict__.keys()]

        assert all(attrib in attrib_xpl2 for attrib in attrib_xpl)
        assert all(attrib2 in attrib_xpl for attrib2 in attrib_xpl2)
        os.remove(pkl_file)

    def test_check_y_pred_1(self):
        """
        Unit test check y pred
        """
        xpl = SmartExplainer()
        xpl.y_pred = None
        xpl.x_pred = None
        xpl.check_y_pred()

    def test_check_y_pred_2(self):
        """
        Unit test check y pred 2
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        xpl.y_pred = pd.DataFrame(
            data=np.array(['1', 0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            xpl.check_y_pred(xpl.y_pred)

    def test_check_y_pred_3(self):
        """
        Unit test check y pred 3
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        xpl.y_pred = pd.DataFrame(
            data=np.array([0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            xpl.check_y_pred(xpl.y_pred)

    def test_check_y_pred_4(self):
        """
        Unit test check y pred 4
        """
        xpl = SmartExplainer()
        xpl.y_pred = [0, 1]
        self.assertRaises(AttributeError, xpl.check_y_pred)

    def test_check_y_pred_5(self):
        """
        Unit test check y pred 5
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        xpl.y_pred = pd.Series(
            data=np.array(['0'])
        )
        with self.assertRaises(ValueError):
            xpl.check_y_pred(xpl.y_pred)

    def test_predict_1(self):
        """
        Test predict method 1
        """
        xpl = SmartExplainer()
        X = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y_true = pd.DataFrame(data=np.array([1, 2, 3]), columns=['pred'])
        y_false = pd.DataFrame(data=np.array([1, 2, 4]), columns=['pred'])
        model = LinearRegression().fit(X, y_true)

        xpl.compile(x=X, y_pred=y_false, model=model)
        xpl.predict()  # y_false should be replaced by predictions which are equal to y_true

        pd.testing.assert_frame_equal(xpl.y_pred, y_true, check_dtype=False)

    def test_predict_2(self):
        """
        Test predict method 2
        """
        xpl = SmartExplainer()
        X = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y_true = pd.DataFrame(data=np.array([1, 2, 3]), columns=['pred'])
        model = LinearRegression().fit(X, y_true)

        xpl.compile(x=X, model=model)
        xpl.predict()

        pd.testing.assert_frame_equal(xpl.y_pred, y_true, check_dtype=False)

    def test_check_model_1(self):
        """
        Unit test check model 1
        """
        model = lambda: None
        model.predict = types.MethodType(self.predict, model)
        xpl = SmartExplainer()
        xpl.model = model
        xpl._case, xpl._classes = xpl.check_model()
        assert xpl._case == 'regression'
        assert xpl._classes is None

    def test_check_model_2(self):
        """
        Unit test check model 2
        """
        xpl = SmartExplainer()
        df1 = pd.DataFrame([1, 2])
        df2 = pd.DataFrame([3, 4])
        xpl.contributions = [df1, df2]
        xpl.state = xpl.choose_state(xpl.contributions)
        model = lambda: None
        model._classes = np.array([1, 2])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        xpl.model = model
        xpl._case, xpl._classes = xpl.check_model()
        assert xpl._case == 'classification'
        self.assertListEqual(xpl._classes, [1, 2])

    def test_check_features_desc_1(self):
        """
        Unit test check features desc 1
        """
        xpl = SmartExplainer()
        xpl.x_pred = pd.DataFrame(
            [[0.12, 0, 13, 1],
             [0.13, 1, 14, 1],
             [0.14, 1, 15, 1],
             [0.15, np.NaN, 13, 1]],
            columns=['col1', 'col2', 'col3', 'col4']
        )
        expected = {
            'col1' : 4,
            'col2' : 2,
            'col3' : 3,
            'col4' : 1
        }
        assert xpl.check_features_desc() == expected

    @patch('shapash.explainer.smart_explainer.SmartExplainer.check_y_pred')
    def test_add_1(self, mock_check_y_pred):
        """
        Unit test add 1
        Parameters
        ----------
        mock_check_y_pred : [type]
            [description]
        """
        xpl = SmartExplainer()
        dataframe_yp = pd.DataFrame([1, 3, 1], columns=['pred'], index=[0, 1, 2])
        mock_y_pred = Mock(return_value=dataframe_yp)
        mock_check_y_pred.return_value = mock_y_pred()
        xpl.x_pred = dataframe_yp
        xpl.add(y_pred=dataframe_yp)
        expected = SmartExplainer()
        expected.y_pred = dataframe_yp
        assert not pd.testing.assert_frame_equal(xpl.y_pred, expected.y_pred)
        mock_check_y_pred.assert_called()

    def test_add_2(self):
        """
        Unit test add 2
        """
        xpl = SmartExplainer()
        xpl._classes = [0, 1]
        xpl._case = "classification"
        xpl.add(label_dict={0: 'Zero', 1: 'One'})
        assert xpl.label_dict[0] == 'Zero'
        assert xpl.label_dict[1] == 'One'

    def test_add_3(self):
        """
        Unit test add 3
        """
        xpl = SmartExplainer()
        xpl.columns_dict = {0: 'Age', 1: 'Education', 2: 'Sex'}
        xpl.add(features_dict={'Age': 'Age (Years Old)'})
        assert xpl.features_dict['Age'] == 'Age (Years Old)'
        assert xpl.features_dict['Education'] == 'Education'

    def test_to_pandas_1(self):
        """
        Unit test to pandas 1
        """
        xpl = SmartExplainer()
        xpl.state = SmartState()
        data = {}
        data['contrib_sorted'] = pd.DataFrame(
            [[0.32230754, 0.1550689, 0.10183475, 0.05471339],
             [-0.58547512, -0.37050409, -0.07249285, 0.00171975],
             [-0.48666675, 0.25507156, -0.16968889, 0.0757443]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        data['var_dict'] = pd.DataFrame(
            [[1, 0, 2, 3],
             [1, 0, 3, 2],
             [1, 0, 2, 3]],
            columns=['feature_0', 'feature_1', 'feature_2', 'feature_3'],
            index=[0, 1, 2]
        )
        data['x_sorted'] = pd.DataFrame(
            [[1., 3., 22., 1.],
             [2., 1., 2., 38.],
             [2., 3., 26., 1.]],
            columns=['feature_0', 'feature_1', 'feature_2', 'feature_3'],
            index=[0, 1, 2]
        )
        xpl.data = data
        xpl.columns_dict = {0: 'Pclass', 1: 'Sex', 2: 'Age', 3: 'Embarked'}
        xpl.features_dict = {'Pclass': 'Pclass', 'Sex': 'Sex', 'Age': 'Age', 'Embarked': 'Embarked'}
        xpl.x = pd.DataFrame(
            [[3., 1., 22., 1.],
             [1., 2., 38., 2.],
             [3., 2., 26., 1.]],
            columns=['Pclass', 'Sex', 'Age', 'Embarked'],
            index=[0, 1, 2]
        )
        xpl.x_pred = xpl.x
        xpl.contributions = data['contrib_sorted']
        xpl.y_pred = pd.DataFrame([1, 2, 3], columns=['pred'], index=[0, 1, 2])
        model = lambda : None
        model.predict = types.MethodType(self.predict, model)
        xpl.model = model
        xpl._case, xpl._classes = xpl.check_model()
        xpl.state = xpl.choose_state(xpl.contributions)
        output = xpl.to_pandas(max_contrib=2)
        expected = pd.DataFrame(
            [[1, 'Sex', 1.0, 0.32230754, 'Pclass', 3.0, 0.1550689],
             [2, 'Sex', 2.0, -0.58547512, 'Pclass', 1.0, -0.37050409],
             [3, 'Sex', 2.0, -0.48666675, 'Pclass', 3.0, 0.25507156]],
            columns=['pred',
                     'feature_1',
                     'value_1',
                     'contribution_1',
                     'feature_2',
                     'value_2',
                     'contribution_2'],
            index=[0, 1, 2],
            dtype=object
        )
        expected['pred'] = expected['pred'].astype(int)
        assert not pd.testing.assert_frame_equal(expected, output)

    def predict_proba(self, arg1, arg2):
        """
        predict_proba method
        """
        matrx = np.array(
            [[0.2, 0.8],
             [0.3, 0.7],
             [0.4, 0.6]]
        )
        return matrx

    def predict(self, arg1, arg2):
        """
        predict method
        """
        matrx = np.array(
            [12, 3, 7]
        )
        return matrx

    def test_to_pandas_2(self):
        """
        Unit test to_pandas :
        test to_pandas method in classification case with
        predict_proba output and column_dict attribute
        """
        xpl = SmartExplainer()
        contrib = pd.DataFrame(
            [[0.32230754, 0.1550689, 0.10183475, 0.05471339],
             [-0.58547512, -0.37050409, -0.07249285, 0.00171975],
             [-0.48666675, 0.25507156, -0.16968889, 0.0757443]],
            index=[0, 1, 2]
        )
        model = lambda: None
        model._classes = np.array([1, 3])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        x = pd.DataFrame(
            [[3., 1., 22., 1.],
             [1., 2., 38., 2.],
             [3., 2., 26., 1.]],
            index=[0, 1, 2]
        )
        pred = pd.DataFrame([3, 1, 1], columns=['pred'], index=[0, 1, 2])
        xpl.compile(contributions=contrib,x=x,model=model,y_pred=pred)
        xpl.columns_dict = {0: 'Pclass', 1: 'Sex', 2: 'Age', 3: 'Embarked'}
        xpl.features_dict = {'Pclass': 'Pclass', 'Sex': 'Sex', 'Age': 'Age', 'Embarked': 'Embarked'}
        output = xpl.to_pandas(max_contrib=3, positive=True, proba=True)
        expected = pd.DataFrame(
            [[3, 0.8, 'Pclass', 3.0, 0.32230754, 'Sex', 1.0, 0.1550689, 'Age', 22.0, 0.10183475],
            [1, 0.3, 'Pclass', 1.0, 0.58547512, 'Sex', 2.0, 0.37050409, 'Age', 38.0, 0.07249285],
            [1, 0.4, 'Pclass', 3.0, 0.48666675, 'Age', 26.0, 0.16968889, np.nan, np.nan, np.nan]],
            columns=['pred', 'proba', 'feature_1', 'value_1', 'contribution_1', 'feature_2',
                     'value_2', 'contribution_2', 'feature_3', 'value_3', 'contribution_3'],
            index=[0, 1, 2],
            dtype=object
        )
        expected['pred'] = expected['pred'].astype(int)
        expected['proba'] = expected['proba'].astype(float)
        assert not pd.testing.assert_frame_equal(expected, output)

    def test_compute_features_import_1(self):
        """
        Unit test compute_features_import 1
        Checking regression case
        """
        xpl = SmartExplainer()
        contributions = pd.DataFrame(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        xpl.features_imp = None
        xpl.contributions = contributions
        xpl.state = xpl.choose_state(contributions)
        xpl._case = 'regression'
        xpl.compute_features_import()
        expected = contributions.abs().sum().sort_values(ascending=True)
        expected = expected / expected.sum()
        assert expected.equals(xpl.features_imp)

    def test_compute_features_import_2(self):
        """
        Unit test compute_features_import 2
        Checking classification case
        """
        xpl = SmartExplainer()
        contrib1 = pd.DataFrame(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        contrib2 = pd.DataFrame(
            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        contributions = [contrib1, contrib2]
        xpl.features_imp = None
        xpl.contributions = contributions
        xpl.state = xpl.choose_state(contributions)
        xpl._case = "classification"
        xpl.compute_features_import()
        expect1 = contrib1.abs().sum().sort_values(ascending=True)
        expect1 = expect1 / expect1.sum()
        expect2 = contrib2.abs().sum().sort_values(ascending=True)
        expect2 = expect2 / expect2.sum()
        assert expect1.equals(xpl.features_imp[0])
        assert expect2.equals(xpl.features_imp[1])

    def test_adapt_contributions_1(self):
        """
        Unit test 1 adapt_contributions
        Classification with one contribution pd.DataFrame
        """
        xpl = SmartExplainer()
        contrib = pd.DataFrame(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        model = Mock()
        model._classes = np.array([1, 3])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        xpl.model = model
        xpl._case, xpl._classes = xpl.check_model()
        output = xpl.adapt_contributions(contrib)
        assert isinstance(output, list)
        assert len(output) == 2

    def test_adapt_contributions_2(self):
        """
        Unit test 1 adapt_contributions
        Classification with one contribution pd.DataFrame
        """
        xpl = SmartExplainer()
        contrib = pd.DataFrame(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            columns=['contribution_0', 'contribution_1', 'contribution_2', 'contribution_3'],
            index=[0, 1, 2]
        )
        xpl._case = "regression"
        output = xpl.adapt_contributions(contrib)
        pd.testing.assert_frame_equal(contrib, output)

    def test_to_smartpredictor_1(self):
        """
        Unit test 1  to_smartpredictor
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df['y'])

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        xpl = SmartExplainer(features_dict={"x1": "age", "x2": "family_situation"})

        xpl.compile(model=clf,
                    x=df_encoded[['x1', 'x2']],
                    preprocessing=encoder_fitted,
                    postprocessing=postprocessing)
        predictor_1 = xpl.to_smartpredictor()

        xpl.mask_params = {
            'features_to_hide': None,
            'threshold': None,
            'positive': True,
            'max_contrib': 1
        }

        predictor_2 = xpl.to_smartpredictor()

        assert hasattr(predictor_1, 'model')
        assert hasattr(predictor_1, 'explainer')
        assert hasattr(predictor_1, 'features_dict')
        assert hasattr(predictor_1, 'label_dict')
        assert hasattr(predictor_1, '_case')
        assert hasattr(predictor_1, '_classes')
        assert hasattr(predictor_1, 'columns_dict')
        assert hasattr(predictor_1, 'features_types')
        assert hasattr(predictor_1, 'preprocessing')
        assert hasattr(predictor_1, 'postprocessing')
        assert hasattr(predictor_1, 'mask_params')
        assert hasattr(predictor_2, 'mask_params')

        assert predictor_1.model == xpl.model
        assert predictor_1.explainer == xpl.explainer
        assert predictor_1.features_dict == xpl.features_dict
        assert predictor_1.label_dict == xpl.label_dict
        assert predictor_1._case == xpl._case
        assert predictor_1._classes == xpl._classes
        assert predictor_1.columns_dict == xpl.columns_dict
        assert predictor_1.preprocessing == xpl.preprocessing
        assert predictor_1.postprocessing == xpl.postprocessing
        assert all(predictor_1.features_types[feature] == str(xpl.x_pred[feature].dtypes)
                   for feature in xpl.x_pred.columns )

        assert predictor_2.mask_params == xpl.mask_params

    def test_get_interaction_values_1(self):
        df = pd.DataFrame({
            "y": np.random.randint(2, size=50),
            "a": np.random.rand(50),
            "b": np.random.rand(50),
        })

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['a', 'b']], df['y'])

        xpl = SmartExplainer()
        xpl.compile(x=df.drop('y', axis=1), model=clf)

        shap_interaction_values = xpl.get_interaction_values(n_samples_max=10)
        assert shap_interaction_values.shape[0] == 10

        shap_interaction_values = xpl.get_interaction_values()
        assert shap_interaction_values.shape[0] == df.shape[0]


    @patch('shapash.explainer.smart_explainer.SmartApp')
    @patch('shapash.explainer.smart_explainer.CustomThread')
    @patch('shapash.explainer.smart_explainer.get_host_name')
    def test_run_app_1(self, mock_get_host_name, mock_custom_thread, mock_smartapp):
        """
        Test that when y_pred is not given, y_pred is automatically computed.
        """
        xpl = SmartExplainer()

        X = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        contributions = pd.DataFrame([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3], [0.1, -0.2, 0.3]])
        y_true = pd.DataFrame(data=np.array([1, 2, 3]), columns=['pred'])
        model = LinearRegression().fit(X, y_true)

        xpl.compile(contributions=contributions, x=X, model=model)
        xpl.run_app()

        assert xpl.y_pred is not None

    @patch('shapash.report.generation.export_and_save_report')
    @patch('shapash.report.generation.execute_report')
    def test_generate_report(self, mock_execute_report, mock_export_and_save_report):
        """
        Test generate report method
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        xpl = SmartExplainer()
        xpl.compile(model=clf, x=df[['x1', 'x2']])
        xpl.generate_report(output_file='test', project_info_file='test')
        mock_execute_report.assert_called_once()
        mock_export_and_save_report.assert_called_once()
