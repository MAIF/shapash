"""
Unit test of Check
"""
import unittest
import pandas as pd
import numpy as np
import category_encoders as ce
from shapash.utils.check import check_preprocessing, check_model, check_label_dict,\
                                check_mask_params, check_ypred, check_contribution_object,\
                                check_model_explainer, check_smartpredictor_length_attributes
from sklearn.compose import ColumnTransformer
import sklearn.preprocessing as skp
import types
import sklearn.ensemble as ske
import sklearn.svm as svm
import sklearn.linear_model as skl
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap




class TestCheck(unittest.TestCase):
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

        check_preprocessing([enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1,
                                           list_dict])
        for preprocessing in [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target]:
            check_preprocessing(preprocessing)

        check_preprocessing(input_dict2)
        check_preprocessing(enc)
        check_preprocessing(None)

        with self.assertRaises(Exception):
            check_preprocessing(wrong_prepro)

    def test_check_model_1(self):
        """
        Unit test check model 1
        """
        model = lambda: None
        model.predict = types.MethodType(self.predict, model)
        _case, _classes = check_model(model)
        assert _case == 'regression'
        assert _classes is None

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

    def test_check_model_2(self):
        """
        Unit test check model 2
        """
        model = lambda: None
        model._classes = np.array([1, 2])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        _case, _classes = check_model(model)
        assert _case == 'classification'
        self.assertListEqual(_classes, [1, 2])

    def test_check_label_dict_1(self):
        """
        Unit test check label dict 1
        """
        label_dict={1: 'Yes', 0: 'No'}
        _classes = [0, 1]
        _case = 'classification'
        check_label_dict(label_dict, _case, _classes)

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        label_dict = {}
        _case = 'regression'
        check_label_dict(label_dict, _case)

    def test_check_mask_params(self):
        """
        Unit test check mask params
        """
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
            check_mask_params(wrong_mask_params_1)
            check_mask_params(wrong_mask_params_2)
            check_mask_params(wrong_mask_params_3)
        check_mask_params(wright_mask_params)

    def test_check_ypred_1(self):
        """
        Unit test check y pred
        """
        y_pred = None
        check_ypred(ypred=y_pred)

    def test_check_ypred_2(self):
        """
        Unit test check y pred 2
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        y_pred = pd.DataFrame(
            data=np.array(['1', 0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            check_ypred(x_pred, y_pred)

    def test_check_ypred_3(self):
        """
        Unit test check y pred 3
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        y_pred = pd.DataFrame(
            data=np.array([0]),
            columns=['Y']
        )
        with self.assertRaises(ValueError):
            check_ypred(x_pred, y_pred)

    def test_check_y_pred_4(self):
        """
        Unit test check y pred 4
        """
        y_pred = [0, 1]
        with self.assertRaises(ValueError):
            check_ypred(ypred=y_pred)

    def test_check_y_pred_5(self):
        """
        Unit test check y pred 5
        """
        x_pred = pd.DataFrame(
            data=np.array([[1, 2], [3, 4]]),
            columns=['Col1', 'Col2']
        )
        y_pred = pd.Series(
            data=np.array(['0'])
        )
        with self.assertRaises(ValueError):
            check_ypred(x_pred, y_pred)

    def test_check_contribution_object_1(self):
        """
        Unit test check_contribution_object 1
        """
        contributions_1 = [
            np.array([[2, 1], [8, 4]]),
            np.array([[5, 5], [0, 0]])
        ]

        contributions_2 = np.array([[2, 1], [8, 4]])
        model = lambda: None
        model._classes = np.array([1, 3])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        _case = "classification"
        _classes = list(model._classes)

        check_contribution_object(_case, _classes, contributions_1)
        assert len(contributions_1) == len(_classes)
        assert isinstance(contributions_1, list)

        check_contribution_object("regression", None, contributions_2)
        assert isinstance(contributions_2, np.ndarray)

        with self.assertRaises(ValueError):
            check_contribution_object(_case, _classes, contributions_2)
            check_mask_params("regression", None, contributions_1)


    def test_check_model_explainer_1(self):
        """
        test check model explainer
        """
        self.modellist = [
            lgb.LGBMRegressor(n_estimators=1), lgb.LGBMClassifier(n_estimators=1),
            xgb.XGBRegressor(n_estimators=1), xgb.XGBRegressor(n_estimators=1),
            svm.SVR(kernel='linear'), svm.SVC(kernel='linear'),
            cb.CatBoostRegressor(n_estimators=1), cb.CatBoostClassifier(n_estimators=1),
            ske.GradientBoostingRegressor(n_estimators=1), ske.GradientBoostingClassifier(n_estimators=1),
            ske.ExtraTreesRegressor(n_estimators=1), ske.ExtraTreesClassifier(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1), ske.RandomForestClassifier(n_estimators=1),
            skl.LogisticRegression(), skl.LinearRegression()
        ]

        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df = df[['x1', 'x2']]
        self.y_df = df['y'].to_frame()

        simple_tree_model = (
            "<class 'sklearn.ensemble._forest.ExtraTreesClassifier'>",
            "<class 'sklearn.ensemble._forest.ExtraTreesRegressor'>",
            "<class 'sklearn.ensemble._forest.RandomForestClassifier'>",
            "<class 'sklearn.ensemble._forest.RandomForestRegressor'>",
            "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>",
            "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>",
            "<class 'lightgbm.sklearn.LGBMClassifier'>",
            "<class 'lightgbm.sklearn.LGBMRegressor'>",
            "<class 'xgboost.sklearn.XGBClassifier'>",
            "<class 'xgboost.sklearn.XGBRegressor'>"
        )

        catboost_model = (
            "<class 'catboost.core.CatBoostClassifier'>",
            "<class 'catboost.core.CatBoostRegressor'>"
        )

        linear_model = (
            "<class 'sklearn.linear_model._logistic.LogisticRegression'>",
            "<class 'sklearn.linear_model._base.LinearRegression'>"
        )

        svm_model = (
            "<class 'sklearn.svm._classes.SVC'>",
            "<class 'sklearn.svm._classes.SVR'>"
        )

        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df, self.y_df)
            if str(type(model)) in simple_tree_model:
                explainer = shap.TreeExplainer(model)

            elif str(type(model)) in catboost_model:
                explainer = shap.TreeExplainer(model)

            elif str(type(model)) in linear_model:
                explainer = shap.LinearExplainer(model, self.x_df)

            elif str(type(model)) in svm_model:
                explainer = shap.KernelExplainer(model.predict, self.x_df)

            check_model_explainer(model, explainer)

    def test_check_model_explainer_2(self):
        """
        test check model explainer 2
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[['x1', 'x2']], df['y'])

        explainer = None
        with self.assertRaises(ValueError):
            check_model_explainer(clf, explainer)

    def test_check_smartpredictor_length_attributes_1(self):
        """
        Test test_check_smartpredictor_length_attributes 1
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

        check_smartpredictor_length_attributes(features_dict, model, columns_dict,
                                               features_types, label_dict)

    def test_check_smartpredictor_length_attributes_2(self):
        """
        Test test_check_smartpredictor_length_attributes 2
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
        label_dict = {1: 'Yes', 0: 'No'}

        enc_ordinal_all = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2', 'Binary1', 'Binary2', 'Ordinal1', 'Ordinal2',
                                            'BaseN1', 'BaseN2', 'Target1', 'Target2', 'other']).fit(train)
        train_ordinal_all  = enc_ordinal_all.transform(train)

        y = pd.DataFrame({'y_class': [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)

        with self.assertRaises(ValueError):
            check_smartpredictor_length_attributes(features_dict, model, columns_dict,
                                                   features_types, label_dict)







