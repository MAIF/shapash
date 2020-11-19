"""
Unit test smart predictor
"""

import unittest
from shapash.explainer.smart_explainer import SmartPredictor
from shapash.explainer.smart_explainer import SmartExplainer
import pandas as pd
import numpy as np
import catboost as cb
from catboost import Pool
import category_encoders as ce
from unittest.mock import patch
import types
from sklearn.compose import ColumnTransformer
import sklearn.preprocessing as skp
import shap





class TestSmartPredictor(unittest.TestCase):
    """
    Unit test Smart Predictor class
    """
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

    def test_init_1(self):
        """
        Test init smart predictor
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)

        columns_dict = {0:"x1",1:"x2"}
        label_dict = {0:"Yes",1:"No"}

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict={"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']]}

        predictor_1 = SmartPredictor(features_dict, clf,
                 columns_dict, clf_explainer, features_types, label_dict,
                 encoder_fitted,postprocessing)

        mask_params = {
            'features_to_hide': None,
            'threshold': None,
            'positive': True,
            'max_contrib': 1
        }

        predictor_2 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing,
                                     mask_params)

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

        assert predictor_1.model == clf
        assert predictor_1.explainer == clf_explainer
        assert predictor_1.features_dict == features_dict
        assert predictor_1.label_dict == label_dict
        assert predictor_1._case == "classification"
        assert predictor_1._classes == [0,1]
        assert predictor_1.columns_dict == columns_dict
        assert predictor_1.preprocessing == encoder_fitted
        assert predictor_1.postprocessing == postprocessing

        assert predictor_2.mask_params == mask_params

    def add_input_1(self):
        """
        Test add_input method from smart predictor
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df.columns}

        ypred = pd.DataFrame({"y": [1, 0], "id": [0, 1]}).set_index("id")
        shap_values = clf.get_feature_importance(Pool(df_encoded), type="ShapValues")

        xpl = SmartExplainer(features_dict, label_dict)
        xpl.compile(contributions=shap_values[:, :-1], x=df_encoded, model=clf, preprocessing=encoder_fitted)

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        predictor_1.add_input(x = df[["x1","x2"]], contributions = shap_values[:,:-1])
        xpl_contrib = xpl.contributions
        predictor_1_contrib = predictor_1.contributions

        assert hasattr(predictor_1, "x")
        assert hasattr(predictor_1, "x_preprocessed")
        assert not hasattr(predictor_1, "ypred")
        assert hasattr(predictor_1, "contributions")
        assert predictor_1.x.shape == predictor_1.x_preprocessed.shape
        assert all(feature in predictor_1.x.columns for feature in predictor_1.x_preprocessed.columns)

        assert isinstance(predictor_1_contrib, list)
        assert len(predictor_1_contrib) == len(xpl_contrib)
        for i, contrib in enumerate(predictor_1_contrib):
            pd.testing.assert_frame_equal(contrib, xpl_contrib[i])
            assert all(contrib.index == xpl_contrib[i].index)
            assert all(contrib.columns == xpl_contrib[i].columns)
            assert all(contrib.dtypes == xpl_contrib[i].dtypes)

        predictor_1.add_input(ypred=ypred)
        assert hasattr(predictor_1, "ypred")
        assert predictor_1.ypred.shape[0] == predictor_1.x.shape[0]
        assert all(predictor_1.ypred.index == predictor_1.x.index)

    @patch('shapash.explainer.smart_predictor.SmartState')
    def test_choose_state_1(self, mock_smart_state):
        """
        Unit test choose state 1
        Parameters
        ----------
        mock_smart_state : [type]
            [description]
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']]}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)
        predictor_1.choose_state('contributions')
        mock_smart_state.assert_called()

    @patch('shapash.explainer.smart_predictor.MultiDecorator')
    def test_choose_state_2(self, mock_multi_decorator):
        """
        Unit test choose state 2
        Parameters
        ----------
        mock_multi_decorator : [type]
            [description]
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']]}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)
        predictor_1.choose_state('contributions')
        predictor_1.choose_state([1, 2, 3])
        mock_multi_decorator.assert_called()

    def test_validate_contributions_1(self):
        """
        Unit test validate contributions 1
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df[['x1', 'x2']]}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        contributions = [
            np.array([[2, 1], [8, 4]]),
            np.array([[5, 5], [0, 0]])
        ]
        predictor_1.state = predictor_1.choose_state(contributions)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = pd.DataFrame(
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
        output = predictor_1.validate_contributions(contributions)
        assert len(expected_output) == len(output)
        test_list = [pd.testing.assert_frame_equal(e, m) for e, m in zip(expected_output, output)]
        assert all(x is None for x in test_list)

    def test_check_contributions(self):
        """
        Unit test check_contributions 1
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 100, df.shape[0])
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"x1": "age", "x2": "weight"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        shap_values = clf.get_feature_importance(Pool(df_encoded), type="ShapValues")

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict)

        predictor_1.add_input(x=df[["x1", "x2"]], contributions=shap_values[:, :-1], ypred=df["y"])

        adapt_contrib = predictor_1.adapt_contributions(shap_values[:, :-1])
        predictor_1.state = predictor_1.choose_state(adapt_contrib)
        contributions = predictor_1.validate_contributions(adapt_contrib)
        predictor_1.check_contributions(contributions)

        with self.assertRaises(ValueError):
            predictor_1.check_contributions(shap_values[:, :-1])

    def test_check_model_1(self):
        """
        Unit test check model 1
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
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        model = lambda: None
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)

        predictor_1.model = model
        _case, _classes = predictor_1.check_model()
        assert _case == 'regression'
        assert _classes is None

    def test_check_model_2(self):
        """
        Unit test check model 2
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
        clf_explainer = shap.TreeExplainer(clf)
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     encoder_fitted, postprocessing)

        model = lambda: None
        model._classes = np.array([1, 2])
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)

        predictor_1.model = model

        _case, _classes = predictor_1.check_model()
        assert _case == 'classification'
        self.assertListEqual(_classes, [1, 2])

    def test_check_preprocessing_1(self):
        """
        Test check preprocessing on multiple preprocessing
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        features_dict = None
        columns_dict = {i:features for i,features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal_all = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                            'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal_all  = enc_ordinal_all.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)


        y = pd.DataFrame(data=[0, 1, 0, 0], columns=['y'])

        enc_onehot = ce.OneHotEncoder(cols=['Onehot1', 'Onehot2']).fit(train)
        train_onehot = enc_onehot.transform(train)
        enc_binary = ce.BinaryEncoder(cols=['Binary1', 'Binary2']).fit(train_onehot)
        train_binary = enc_binary.transform(train_onehot)
        enc_ordinal = ce.OrdinalEncoder(cols=['Ordinal1', 'Ordinal2']).fit(train_binary)
        train_ordinal = enc_ordinal.transform(train_binary)
        enc_basen = ce.BaseNEncoder(cols=['BaseN1', 'BaseN2']).fit(train_ordinal)
        train_basen = enc_basen.transform(train_ordinal)
        enc_target = ce.TargetEncoder(cols=['Target1', 'Target2']).fit(train_basen, y)

        input_dict1 = dict()
        input_dict1['col'] = 'Onehot2'
        input_dict1['mapping'] = pd.Series(data=['C', 'D', np.nan], index=['C', 'D', 'missing'])
        input_dict1['data_type'] = 'object'

        input_dict2 = dict()
        input_dict2['col'] = 'Binary2'
        input_dict2['mapping'] = pd.Series(data=['G', 'H', np.nan], index=['G', 'H', 'missing'])
        input_dict2['data_type'] = 'object'

        input_dict = dict()
        input_dict['col'] = 'state'
        input_dict['mapping'] = pd.Series(data=['US', 'FR-1', 'FR-2'], index=['US', 'FR', 'FR'])
        input_dict['data_type'] = 'object'

        input_dict3 = dict()
        input_dict3['col'] = 'Ordinal2'
        input_dict3['mapping'] = pd.Series(data=['K', 'L', np.nan], index=['K', 'L', 'missing'])
        input_dict3['data_type'] = 'object'
        list_dict = [input_dict2, input_dict3]

        y = pd.DataFrame(data=[0, 1], columns=['y'])

        train = pd.DataFrame({'city': ['chicago', 'paris'],
                              'state': ['US', 'FR'],
                              'other': ['A', 'B']})
        enc = ColumnTransformer(
            transformers=[
                ('onehot', skp.OneHotEncoder(), ['city', 'state'])
            ],
            remainder='drop')
        enc.fit(train, y)

        wrong_prepro = skp.OneHotEncoder().fit(train, y)

        predictor_1.preprocessing = [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1,
                                           list_dict]
        predictor_1.check_preprocessing()

        for preprocessing in [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target]:
            predictor_1.preprocessing = preprocessing
            predictor_1.check_preprocessing()

        predictor_1.preprocessing = input_dict2
        predictor_1.check_preprocessing()

        predictor_1.preprocessing = enc
        predictor_1.check_preprocessing()

        predictor_1.preprocessing = None
        predictor_1.check_preprocessing()

        with self.assertRaises(Exception):
            predictor_1.preprocessing = wrong_prepro
            predictor_1.check_preprocessing()

    def test_check_label_dict_1(self):
        """
        Unit test check label dict 1
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = {1: 'Yes', 0: 'No'}

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)

        predictor_1.check_label_dict()

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = None

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)

        predictor_1._case = 'regression'
        predictor_1.check_label_dict()

    def test_check_mask_params(self):
        """
        Unit test check mask params
        """

        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                                  'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal = enc_ordinal.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal, y)
        clf_explainer = shap.TreeExplainer(model)

        wrong_mask_params_1 = list()
        wrong_mask_params_2 = None
        wrong_mask_params_3 = {
            "features_to_hide": None,
            "threshold": None,
            "positive": None
        }
        wright_mask_params = {
            "features_to_hide": None,
            "threshold": None,
            "positive": True,
            "max_contrib": 5
        }
        with self.assertRaises(ValueError):
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_1)
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_2)
            predictor_1 = SmartPredictor(features_dict, model,
                                         columns_dict, clf_explainer, features_types, label_dict,
                                         mask_params=wrong_mask_params_3)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict,
                                     mask_params=wright_mask_params)

    def test_check_ypred_1(self):
        """
        Unit test check y pred
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                              'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal = enc_ordinal.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal, y)
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = train
        y_pred = None
        predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_2(self):
        """
        Unit test check y pred 2
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = None

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)

        y_pred = pd.DataFrame(
            data=np.array(['1', 0]),
            columns=['Y']
        )
        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = x_pred

        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_ypred_3(self):
        """
        Unit test check y pred 3
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = None

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = x_pred
        y_pred = pd.DataFrame(
            data=np.array([0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_y_pred_4(self):
        """
        Unit test check y pred 4
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = None

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        y_pred = [0, 1]
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_5(self):
        """
        Unit test check y pred 5
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(x_pred.columns)}
        features_types = {features: str(x_pred[features].dtypes) for features in x_pred.columns}
        label_dict = None

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['Col1'] = np.random.randint(1, 123, df.shape[0])
        df['Col2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        model = cb.CatBoostClassifier(n_estimators=1).fit(df[['Col1', 'Col2']], df['y'])
        clf_explainer = shap.TreeExplainer(model)

        predictor_1 = SmartPredictor(features_dict, model,
                                     columns_dict, clf_explainer, features_types, label_dict)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = x_pred

        y_pred = pd.Series(
            data=np.array(['0'])
        )
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_predict_proba_1(self):
        """
        Unit test of predict_proba method.
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostRegressor(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict,
                                     encoder_fitted, postprocessing)

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])
        clf_explainer = shap.TreeExplainer(clf)
        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict,
                                     encoder_fitted, postprocessing)

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(KeyError):
            predictor_1.predict_proba()

    def test_predict_proba_2(self):
        """
        Unit test of predict_proba method.
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(30, 150, df.shape[0])
        df = df.set_index('id')

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"x1": "age", "x2": "weight"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)
        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict)

        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed":None}
        predictor_1.data["x"] = df[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = df[["x1", "x2"]]

        prediction = predictor_1.predict_proba()
        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

        predictor_1.data["ypred"] = pd.DataFrame(df["y"])
        prediction = predictor_1.predict_proba()

        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

    def test_detail_contributions_1(self):
        """
        Unit test of detail_contributions method.
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = ["S", "M", "S", "D", "M"]
        df = df.set_index('id')
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostRegressor(n_estimators=1).fit(df_encoded[['x1', 'x2']], df_encoded['y'])

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}

        postprocessing = {"x2": {
            "type": "transcoding",
            "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}
        clf_explainer = shap.TreeExplainer(clf)
        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict,
                                     encoder_fitted, postprocessing)

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict,
                                     encoder_fitted, postprocessing)
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["x"] = df[["x1", "x2"]]

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

    def test_detail_contributions_2(self):
        """
        Unit test 2 of detail_contributions method.
        """
        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 2 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(30, 150, df.shape[0])
        df = df.set_index('id')

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"x1": "age", "x2": "weight"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        clf = cb.CatBoostRegressor(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict)

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x_preprocessed"] = df[["x1", "x2"]]
        predictor_1.data["x"] = df[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(df["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 1

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])
        clf_explainer = shap.TreeExplainer(clf)

        predictor_1 = SmartPredictor(features_dict, clf,
                                     columns_dict, clf_explainer,
                                     features_types, label_dict)

        df['false_y'] = [2, 2, 1, 1, 1]
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x_preprocessed"] = df[["x1", "x2"]]
        predictor_1.data["x"] = df[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(df["false_y"])

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["ypred"] = pd.DataFrame(df["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 2














