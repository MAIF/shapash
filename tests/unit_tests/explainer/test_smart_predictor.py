"""
Unit test smart predictor
"""

import os
import types
import unittest
from os import path
from pathlib import Path
from unittest.mock import patch

import catboost as cb
import category_encoders as ce
import numpy as np
import pandas as pd
import shap
import sklearn.preprocessing as skp
from catboost import Pool
from sklearn.compose import ColumnTransformer

from shapash import SmartExplainer
from shapash.backend import ShapBackend
from shapash.explainer.multi_decorator import MultiDecorator
from shapash.explainer.smart_predictor import SmartPredictor
from shapash.explainer.smart_state import SmartState


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
    pkl_file = path.join(current, "data/predictor.pkl")
    y_pred = pd.DataFrame(data=np.array([1, 2]), columns=["pred"])
    dataframe_x = pd.DataFrame([[1, 2, 4], [1, 2, 3]])
    clf = cb.CatBoostClassifier(n_estimators=1).fit(dataframe_x, y_pred)
    xpl = SmartExplainer(model=clf, features_dict={})
    xpl.compile(x=dataframe_x, y_pred=y_pred)
    predictor = xpl.to_smartpredictor()
    return pkl_file, predictor


class TestSmartPredictor(unittest.TestCase):
    """
    Unit test Smart Predictor class
    """

    def setUp(self):
        df = pd.DataFrame(range(0, 5), columns=["id"])
        df["y"] = df["id"].apply(lambda x: 1 if x < 2 else 0)
        df["x1"] = np.random.randint(1, 123, df.shape[0])
        df["x2"] = ["S", "M", "S", "D", "M"]
        df = df.set_index("id")
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[["x1", "x2"]], df["y"])
        backend = ShapBackend(model=clf)

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}

        postprocessing = {"x2": {"type": "transcoding", "rule": {"S": "single", "M": "married", "D": "divorced"}}}
        features_dict = {"x1": "age", "x2": "family_situation"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]]}

        self.df_1 = df.copy()
        self.preprocessing_1 = encoder_fitted
        self.df_encoded_1 = df_encoded
        self.clf_1 = clf
        self.backend_1 = backend
        self.columns_dict_1 = columns_dict
        self.label_dict_1 = label_dict
        self.postprocessing_1 = postprocessing
        self.features_dict_1 = features_dict
        self.features_types_1 = features_types

        self.predictor_1 = SmartPredictor(
            features_dict, clf, columns_dict, backend, features_types, label_dict, encoder_fitted, postprocessing
        )

        self.features_groups = {"group1": ["x1", "x2"]}
        self.features_dict_w_groups = {"x1": "age", "x2": "weight", "group1": "group1"}
        self.predictor_1_w_groups = SmartPredictor(
            self.features_dict_w_groups,
            clf,
            columns_dict,
            backend,
            features_types,
            label_dict,
            encoder_fitted,
            postprocessing,
            self.features_groups,
        )
        self.predictor_1.backend.state = SmartState()
        df["x2"] = np.random.randint(1, 100, df.shape[0])
        encoder = ce.OrdinalEncoder(cols=["x2"], handle_unknown="None")
        encoder_fitted = encoder.fit(df[["x1", "x2"]])
        df_encoded = encoder_fitted.transform(df[["x1", "x2"]])

        clf = cb.CatBoostClassifier(n_estimators=1).fit(df[["x1", "x2"]], df["y"])
        backend = ShapBackend(model=clf)
        features_dict = {"x1": "age", "x2": "weight"}
        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        self.df_2 = df.copy()
        self.preprocessing_2 = encoder_fitted
        self.df_encoded_2 = df_encoded
        self.clf_2 = clf
        self.backend_2 = backend
        self.columns_dict_2 = columns_dict
        self.label_dict_2 = label_dict
        self.postprocessing_2 = postprocessing
        self.features_dict_2 = features_dict
        self.features_types_2 = features_types

        self.predictor_2 = SmartPredictor(
            features_dict, clf, columns_dict, backend, features_types, label_dict, encoder_fitted, postprocessing
        )
        self.predictor_2.backend.state = SmartState()

        df["x1"] = [25, 39, 50, 43, 67]
        df["x2"] = [90, 78, 84, 85, 53]

        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "No", 1: "Yes"}
        features_dict = {"x1": "age", "x2": "weight"}

        features_types = {features: str(df[features].dtypes) for features in df[["x1", "x2"]].columns}

        clf = cb.CatBoostRegressor(n_estimators=1).fit(df[["x1", "x2"]], df["y"])
        backend_3 = ShapBackend(model=clf)

        self.df_3 = df.copy()
        self.preprocessing_3 = None
        self.df_encoded_3 = df
        self.clf_3 = clf
        self.backend_3 = backend
        self.columns_dict_3 = columns_dict
        self.label_dict_3 = label_dict
        self.postprocessing_3 = None
        self.features_dict_3 = features_dict
        self.features_types_3 = features_types

        self.predictor_3 = SmartPredictor(features_dict, clf, columns_dict, backend, features_types, label_dict)
        self.predictor_3.backend.state = SmartState()

    def predict_proba(self, arg1, arg2):
        """
        predict_proba method
        """
        matrx = np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        return matrx

    def predict(self, arg1, arg2):
        """
        predict method
        """
        matrx = np.array([12, 3, 7])
        return matrx

    def test_init_1(self):
        """
        Test init smart predictor
        """
        predictor_1 = SmartPredictor(
            self.features_dict_1,
            self.clf_1,
            self.columns_dict_1,
            self.backend_1,
            self.features_types_1,
            self.label_dict_1,
            self.preprocessing_1,
            self.postprocessing_1,
        )

        assert hasattr(predictor_1, "model")
        assert hasattr(predictor_1, "backend")
        assert hasattr(predictor_1, "features_dict")
        assert hasattr(predictor_1, "label_dict")
        assert hasattr(predictor_1, "_case")
        assert hasattr(predictor_1, "_classes")
        assert hasattr(predictor_1, "columns_dict")
        assert hasattr(predictor_1, "features_types")
        assert hasattr(predictor_1, "preprocessing")
        assert hasattr(predictor_1, "postprocessing")
        assert hasattr(predictor_1, "mask_params")
        assert hasattr(predictor_1, "features_groups")

        assert predictor_1.model == self.clf_1
        assert predictor_1.backend == self.backend_1
        assert predictor_1.features_dict == self.features_dict_1
        assert predictor_1.label_dict == self.label_dict_1
        assert predictor_1._case == "classification"
        assert predictor_1._classes == [0, 1]
        assert predictor_1.columns_dict == self.columns_dict_1
        assert predictor_1.preprocessing == self.preprocessing_1
        assert predictor_1.postprocessing == self.postprocessing_1

        mask_params = {"features_to_hide": None, "threshold": None, "positive": True, "max_contrib": 1}
        predictor_1.mask_params = mask_params
        assert predictor_1.mask_params == mask_params

    def test_init_2(self):
        """
        Test init smart predictor with groups of features
        """
        predictor_1 = SmartPredictor(
            self.features_dict_w_groups,
            self.clf_1,
            self.columns_dict_1,
            self.backend_1,
            self.features_types_1,
            self.label_dict_1,
            self.preprocessing_1,
            self.postprocessing_1,
            self.features_groups,
        )

        assert hasattr(predictor_1, "model")
        assert hasattr(predictor_1, "backend")
        assert hasattr(predictor_1, "features_dict")
        assert hasattr(predictor_1, "label_dict")
        assert hasattr(predictor_1, "_case")
        assert hasattr(predictor_1, "_classes")
        assert hasattr(predictor_1, "columns_dict")
        assert hasattr(predictor_1, "features_types")
        assert hasattr(predictor_1, "preprocessing")
        assert hasattr(predictor_1, "postprocessing")
        assert hasattr(predictor_1, "mask_params")
        assert hasattr(predictor_1, "features_groups")

        assert predictor_1.model == self.clf_1
        assert predictor_1.backend == self.backend_1
        assert predictor_1.features_dict == self.features_dict_w_groups
        assert predictor_1.label_dict == self.label_dict_1
        assert predictor_1._case == "classification"
        assert predictor_1._classes == [0, 1]
        assert predictor_1.columns_dict == self.columns_dict_1
        assert predictor_1.preprocessing == self.preprocessing_1
        assert predictor_1.postprocessing == self.postprocessing_1

        mask_params = {"features_to_hide": None, "threshold": None, "positive": True, "max_contrib": 1}
        predictor_1.mask_params = mask_params
        assert predictor_1.mask_params == mask_params

    def test_add_input_1(self):
        """
        Test add_input method from smart predictor
        """
        ypred = self.df_1["y"]
        shap_values = self.clf_1.get_feature_importance(Pool(self.df_encoded_1), type="ShapValues")

        predictor_1 = self.predictor_1
        predictor_1.add_input(x=self.df_1[["x1", "x2"]], contributions=shap_values[:, :-1])
        predictor_1_contrib = predictor_1.data["contributions"]

        assert all(
            attribute in predictor_1.data.keys() for attribute in ["x", "x_preprocessed", "contributions", "ypred"]
        )
        assert predictor_1.data["x"].shape == predictor_1.data["x_preprocessed"].shape
        assert all(feature in predictor_1.data["x"].columns for feature in predictor_1.data["x_preprocessed"].columns)
        assert predictor_1_contrib.shape == predictor_1.data["x"].shape

        predictor_1.add_input(ypred=ypred)

        assert "ypred" in predictor_1.data.keys()
        assert predictor_1.data["ypred"].shape[0] == predictor_1.data["x"].shape[0]
        assert all(predictor_1.data["ypred"].index == predictor_1.data["x"].index)

    def test_add_input_2(self):
        """
        Test add_input method from smart predictor with groups of features
        """
        ypred = self.df_1["y"]
        shap_values = self.clf_1.get_feature_importance(Pool(self.df_encoded_1), type="ShapValues")

        predictor_1 = self.predictor_1_w_groups
        predictor_1.add_input(x=self.df_1[["x1", "x2"]], contributions=shap_values[:, :-1])
        predictor_1_contrib = predictor_1.data["contributions"]

        assert all(
            attribute in predictor_1.data.keys()
            for attribute in ["x", "x_preprocessed", "x_postprocessed", "contributions", "ypred"]
        )

        assert hasattr(predictor_1, "data_groups")
        assert all(
            attribute in predictor_1.data_groups.keys() for attribute in ["x_postprocessed", "contributions", "ypred"]
        )

        assert predictor_1.data["x"].shape == predictor_1.data["x_preprocessed"].shape
        assert all(feature in predictor_1.data["x"].columns for feature in predictor_1.data["x_preprocessed"].columns)
        assert predictor_1_contrib.shape == predictor_1.data["x"].shape

        predictor_1.add_input(ypred=ypred)

        assert "ypred" in predictor_1.data.keys()
        assert predictor_1.data["ypred"].shape[0] == predictor_1.data["x"].shape[0]
        assert all(predictor_1.data["ypred"].index == predictor_1.data["x"].index)

        print(predictor_1.data["contributions"].sum(axis=1).values)

        print(predictor_1.data_groups["contributions"])

        assert all(
            predictor_1.data_groups["contributions"]["group1"].values == predictor_1.data["contributions"].sum(axis=1)
        )

    def test_check_contributions(self):
        """
        Unit test check_shape_contributions 1
        """

        shap_values = self.clf_2.get_feature_importance(Pool(self.df_encoded_2), type="ShapValues")

        predictor_1 = self.predictor_2

        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed": None}
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]
        predictor_1.data["ypred"] = self.df_2["y"]

        adapt_contrib = [
            np.array(
                [
                    [-0.04395604, 0.13186813],
                    [-0.04395604, 0.13186813],
                    [-0.0021978, 0.01318681],
                    [-0.0021978, 0.01318681],
                    [-0.04395604, 0.13186813],
                ]
            ),
            np.array(
                [
                    [0.04395604, -0.13186813],
                    [0.04395604, -0.13186813],
                    [0.0021978, -0.01318681],
                    [0.0021978, -0.01318681],
                    [0.04395604, -0.13186813],
                ]
            ),
        ]
        contributions = list()
        for element in adapt_contrib:
            contributions.append(pd.DataFrame(element, columns=["x1", "x2"]))

        predictor_1.backend.state = MultiDecorator(SmartState())
        predictor_1.check_contributions(contributions)

        with self.assertRaises(ValueError):
            predictor_1.check_contributions(shap_values[:, :-1])

    def test_check_model_1(self):
        """
        Unit test check model 1
        """
        predictor_1 = self.predictor_1

        model = lambda: None
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)

        predictor_1.model = model
        _case, _classes = predictor_1.check_model()
        assert _case == "regression"
        assert _classes is None

    def test_check_model_2(self):
        """
        Unit test check model 2
        """
        predictor_1 = self.predictor_1

        model = lambda: None
        model._classes = np.array([1, 2])
        model.n_features_in_ = 2
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)

        predictor_1.model = model

        _case, _classes = predictor_1.check_model()
        assert _case == "classification"
        self.assertListEqual(_classes, [1, 2])

    @patch("shapash.explainer.smart_predictor.SmartPredictor.check_model")
    @patch("shapash.utils.check.check_preprocessing_options")
    @patch("shapash.utils.check.check_consistency_model_features")
    @patch("shapash.utils.check.check_consistency_model_label")
    def test_check_preprocessing_1(
        self, check_consistency_model_label, check_consistency_model_features, check_preprocessing_options, check_model
    ):
        """
        Test check preprocessing on multiple preprocessing
        """
        train = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A", "B"],
                "Onehot2": ["C", "D", "C", "D"],
                "Binary1": ["E", "F", "E", "F"],
                "Binary2": ["G", "H", "G", "H"],
                "Ordinal1": ["I", "J", "I", "J"],
                "Ordinal2": ["K", "L", "K", "L"],
                "BaseN1": ["M", "N", "M", "N"],
                "BaseN2": ["O", "P", "O", "P"],
                "Target1": ["Q", "R", "Q", "R"],
                "Target2": ["S", "T", "S", "T"],
                "other": ["other", np.nan, "other", "other"],
            }
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal_all = ce.OrdinalEncoder(
            cols=[
                "Onehot1",
                "Onehot2",
                "Binary1",
                "Binary2",
                "Ordinal1",
                "Ordinal2",
                "BaseN1",
                "BaseN2",
                "Target1",
                "Target2",
                "other",
            ]
        ).fit(train)
        train_ordinal_all = enc_ordinal_all.transform(train)

        y = pd.DataFrame({"y_class": [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)
        backend = ShapBackend(model=model)

        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_model.return_value = "classification", [0, 1]

        predictor_1 = SmartPredictor(features_dict, model, columns_dict, backend, features_types, label_dict)

        y = pd.DataFrame(data=[0, 1, 0, 0], columns=["y"])

        enc_onehot = ce.OneHotEncoder(cols=["Onehot1", "Onehot2"]).fit(train)
        train_onehot = enc_onehot.transform(train)
        enc_binary = ce.BinaryEncoder(cols=["Binary1", "Binary2"]).fit(train_onehot)
        train_binary = enc_binary.transform(train_onehot)
        enc_ordinal = ce.OrdinalEncoder(cols=["Ordinal1", "Ordinal2"]).fit(train_binary)
        train_ordinal = enc_ordinal.transform(train_binary)
        enc_basen = ce.BaseNEncoder(cols=["BaseN1", "BaseN2"]).fit(train_ordinal)
        train_basen = enc_basen.transform(train_ordinal)
        enc_target = ce.TargetEncoder(cols=["Target1", "Target2"]).fit(train_basen, y)

        input_dict1 = dict()
        input_dict1["col"] = "Onehot2"
        input_dict1["mapping"] = pd.Series(data=["C", "D", np.nan], index=["C", "D", "missing"])
        input_dict1["data_type"] = "object"

        input_dict2 = dict()
        input_dict2["col"] = "Binary2"
        input_dict2["mapping"] = pd.Series(data=["G", "H", np.nan], index=["G", "H", "missing"])
        input_dict2["data_type"] = "object"

        input_dict = dict()
        input_dict["col"] = "state"
        input_dict["mapping"] = pd.Series(data=["US", "FR-1", "FR-2"], index=["US", "FR", "FR"])
        input_dict["data_type"] = "object"

        input_dict3 = dict()
        input_dict3["col"] = "Ordinal2"
        input_dict3["mapping"] = pd.Series(data=["K", "L", np.nan], index=["K", "L", "missing"])
        input_dict3["data_type"] = "object"
        list_dict = [input_dict2, input_dict3]

        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})
        enc = ColumnTransformer(transformers=[("onehot", skp.OneHotEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)

        wrong_prepro = skp.OneHotEncoder().fit(train, y)

        predictor_1.preprocessing = [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1, list_dict]
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
        predictor_1 = self.predictor_1

        predictor_1.check_label_dict()

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        predictor_1 = self.predictor_1

        predictor_1.label_dict = None
        predictor_1._case = "regression"
        predictor_1.check_label_dict()

    @patch("shapash.explainer.smart_predictor.SmartPredictor.check_model")
    @patch("shapash.utils.check.check_preprocessing_options")
    @patch("shapash.utils.check.check_consistency_model_features")
    @patch("shapash.utils.check.check_consistency_model_label")
    def test_check_mask_params(
        self, check_consistency_model_label, check_consistency_model_features, check_preprocessing_options, check_model
    ):
        """
        Unit test check mask params
        """
        train = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A", "B"],
                "Onehot2": ["C", "D", "C", "D"],
                "Binary1": ["E", "F", "E", "F"],
                "Binary2": ["G", "H", "G", "H"],
                "Ordinal1": ["I", "J", "I", "J"],
                "Ordinal2": ["K", "L", "K", "L"],
                "BaseN1": ["M", "N", "M", "N"],
                "BaseN2": ["O", "P", "O", "P"],
                "Target1": ["Q", "R", "Q", "R"],
                "Target2": ["S", "T", "S", "T"],
                "other": ["other", np.nan, "other", "other"],
            }
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        label_dict = None

        enc_ordinal = ce.OrdinalEncoder(
            cols=[
                "Onehot1",
                "Onehot2",
                "Binary1",
                "Binary2",
                "Ordinal1",
                "Ordinal2",
                "BaseN1",
                "BaseN2",
                "Target1",
                "Target2",
                "other",
            ]
        ).fit(train)
        train_ordinal = enc_ordinal.transform(train)

        y = pd.DataFrame({"y_class": [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal, y)
        backend = ShapBackend(model=model)

        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_model.return_value = "classification", [0, 1]

        wrong_mask_params_1 = list()
        wrong_mask_params_2 = None
        wrong_mask_params_3 = {"features_to_hide": None, "threshold": None, "positive": None}
        wright_mask_params = {"features_to_hide": None, "threshold": None, "positive": True, "max_contrib": 5}
        with self.assertRaises(ValueError):
            predictor_1 = SmartPredictor(
                features_dict, model, columns_dict, backend, features_types, label_dict, mask_params=wrong_mask_params_1
            )
            predictor_1 = SmartPredictor(
                features_dict, model, columns_dict, backend, features_types, label_dict, mask_params=wrong_mask_params_2
            )
            predictor_1 = SmartPredictor(
                features_dict, model, columns_dict, backend, features_types, label_dict, mask_params=wrong_mask_params_3
            )

        predictor_1 = SmartPredictor(
            features_dict, model, columns_dict, backend, features_types, label_dict, mask_params=wright_mask_params
        )

    def test_check_ypred_1(self):
        """
        Unit test check y pred
        """
        predictor_1 = self.predictor_1

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1", "x2"]]
        y_pred = None
        predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_2(self):
        """
        Unit test check y pred 2
        """
        y_pred = pd.DataFrame(data=np.array(["1", 0, 0, 1, 0]), columns=["Y"])
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1

        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_ypred_3(self):
        """
        Unit test check y pred 3
        """
        predictor_1 = self.predictor_1

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1", "x2"]]
        y_pred = pd.DataFrame(data=np.array([0]), columns=["Y"])
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_check_y_pred_4(self):
        """
        Unit test check y pred 4
        """
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        y_pred = [0, 1, 0, 1, 0]
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(ypred=y_pred)

    def test_check_ypred_5(self):
        """
        Unit test check y pred 5
        """
        predictor_1 = self.predictor_1
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = self.df_1[["x1", "x2"]]

        y_pred = pd.Series(data=np.array(["0"]))
        with self.assertRaises(ValueError):
            predictor_1.check_ypred(y_pred)

    def test_predict_proba_1(self):
        """
        Unit test of predict_proba method.
        """
        predictor_1 = self.predictor_1

        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_encoded_1[["x1", "x2"]], self.df_1["y"])
        backend = ShapBackend(model=clf)
        predictor_1.model = clf
        predictor_1.backend = backend
        predictor_1._case = "regression"
        predictor_1._classes = None

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        predictor_1 = self.predictor_1

        with self.assertRaises(AttributeError):
            predictor_1.predict_proba()

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(KeyError):
            predictor_1.predict_proba()

    def test_predict_proba_2(self):
        """
        Unit test of predict_proba method.
        """
        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_2[["x1", "x2"]], self.df_2["y"])
        predictor_1 = self.predictor_2

        predictor_1.model = clf
        predictor_1.backend = ShapBackend(model=clf)
        predictor_1.preprocessing = None

        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed": None}
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]

        prediction = predictor_1.predict_proba()
        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

        predictor_1.data["ypred"] = pd.DataFrame(self.df_2["y"])
        prediction = predictor_1.predict_proba()

        assert prediction.shape[0] == predictor_1.data["x"].shape[0]

    def test_detail_contributions_1(self):
        """
        Unit test of detail_contributions method.
        """
        predictor_1 = self.predictor_1

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["x_preprocessed"] = self.df_1[["x1", "x2"]]

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

    def test_detail_contributions_2(self):
        """
        Unit test 2 of detail_contributions method.
        """
        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_2[["x1", "x2"]], self.df_2["y"])
        predictor_1 = self.predictor_1

        predictor_1.model = clf
        predictor_1.backend = ShapBackend(model=clf)
        predictor_1.preprocessing = None
        predictor_1.backend._case = "regression"
        predictor_1._case = "regression"
        predictor_1._classes = None

        predictor_1.data = {"x": None, "ypred": None, "contributions": None, "x_preprocessed": None}
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]
        predictor_1.data["ypred_init"] = pd.DataFrame(self.df_2["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 1

        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_2[["x1", "x2"]], self.df_2["y"])
        backend = ShapBackend(model=clf)

        predictor_1 = self.predictor_2
        predictor_1.preprocessing = None
        predictor_1.model = clf
        predictor_1.backend = backend
        predictor_1._case = "classification"
        predictor_1._classes = [0, 1]

        false_y = pd.DataFrame({"y_false": [2, 2, 1, 1, 1]})
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x_preprocessed"] = self.df_2[["x1", "x2"]]
        predictor_1.data["x"] = self.df_2[["x1", "x2"]]
        predictor_1.data["ypred_init"] = false_y

        with self.assertRaises(ValueError):
            predictor_1.detail_contributions()

        predictor_1.data["ypred_init"] = pd.DataFrame(self.df_2["y"])

        contributions = predictor_1.detail_contributions()

        assert contributions.shape[0] == predictor_1.data["x"].shape[0]
        assert all(contributions.index == predictor_1.data["x"].index)
        assert contributions.shape[1] == predictor_1.data["x"].shape[1] + 2

    def test_save_1(self):
        """
        Unit test save 1
        """
        pkl_file, predictor = init_sme_to_pickle_test()
        predictor.save(pkl_file)
        assert path.exists(pkl_file)
        os.remove(pkl_file)

    @patch("shapash.explainer.smart_predictor.SmartPredictor.check_model")
    @patch("shapash.utils.check.check_preprocessing_options")
    @patch("shapash.utils.check.check_consistency_model_features")
    @patch("shapash.utils.check.check_consistency_model_label")
    def test_apply_preprocessing_1(
        self, check_consistency_model_label, check_consistency_model_features, check_preprocessing_options, check_model
    ):
        """
        Unit test for apply preprocessing method
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])
        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2]})
        enc = ColumnTransformer(
            transformers=[("power", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="passthrough"
        )
        enc.fit(train, y)
        train_preprocessed = pd.DataFrame(enc.transform(train), index=train.index)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)

        features_types = {features: str(train[features].dtypes) for features in train.columns}
        backend = ShapBackend(model=clf)
        columns_dict = {0: "num1", 1: "num2"}
        label_dict = {0: "Yes", 1: "No"}
        features_dict = {"num1": "city", "num2": "state"}

        check_preprocessing_options.return_value = True
        check_consistency_model_features.return_value = True
        check_consistency_model_label.return_value = True
        check_model.return_value = "classification", [0, 1]

        predictor_1 = SmartPredictor(features_dict, clf, columns_dict, backend, features_types, label_dict, enc)
        predictor_1.data = {"x": None}
        predictor_1.data["x"] = train
        predictor_1.data["x_preprocessed"] = predictor_1.apply_preprocessing()

        output_preprocessed = predictor_1.data["x_preprocessed"]
        assert output_preprocessed.shape == train_preprocessed.shape
        assert [column in clf.feature_names_ for column in output_preprocessed.columns]
        assert all(train.index == output_preprocessed.index)
        assert all(
            [
                str(type_result) == str(train_preprocessed.dtypes[index])
                for index, type_result in enumerate(output_preprocessed.dtypes)
            ]
        )

    def test_summarize_1(self):
        """
        Unit test 1 summarize method
        """
        clf = cb.CatBoostRegressor(n_estimators=1).fit(self.df_3[["x1", "x2"]], self.df_3["y"])
        backend = ShapBackend(model=clf)

        predictor_1 = self.predictor_3
        predictor_1.model = clf
        predictor_1.backend = backend
        predictor_1.data = {
            "x": None,
            "x_preprocessed": None,
            "x_postprocessed": None,
            "ypred": None,
            "contributions": None,
        }

        predictor_1.data["x"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_postprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["ypred"] = self.df_3["y"]
        contribution = pd.DataFrame(
            [[0.0, 0.094286], [0.0, -0.023571], [0.0, -0.023571], [0.0, -0.023571], [0.0, -0.023571]],
            columns=["x1", "x2"],
        )
        predictor_1.data["contributions"] = contribution

        output = predictor_1.summarize()
        print(output)
        expected_output = pd.DataFrame(
            {
                "y": [1, 1, 0, 0, 0],
                "feature_1": ["weight", "weight", "weight", "weight", "weight"],
                "value_1": ["90", "78", "84", "85", "53"],
                "contribution_1": ["0.0942857", "-0.0235714", "-0.0235714", "-0.0235714", "-0.0235714"],
                "feature_2": ["age", "age", "age", "age", "age"],
                "value_2": ["25", "39", "50", "43", "67"],
                "contribution_2": ["0", "0", "0", "0", "0"],
            },
            dtype=object,
        )
        expected_output["y"] = expected_output["y"].astype(int)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert expected_output.shape == output.shape
        assert len(feature_expected) == len(feature_output)
        assert len(value_expected) == len(value_output)
        assert len(contribution_expected) == len(contribution_output)

    def test_summarize_2(self):
        """
        Unit test 2 summarize method
        """
        predictor_1 = self.predictor_3
        predictor_1._case = "classification"
        predictor_1._classes = [0, 1]
        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_3[["x1", "x2"]], self.df_3["y"])
        backend = ShapBackend(model=clf)
        predictor_1.model = clf
        predictor_1.backend = backend

        with self.assertRaises(ValueError):
            predictor_1.summarize()

        predictor_1.data = {
            "x": None,
            "x_preprocessed": None,
            "x_postprocessed": None,
            "ypred": None,
            "contributions": None,
        }

        predictor_1.data["x"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_postprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(
            {"y": ["Yes", "Yes", "No", "No", "No"], "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209]}
        )

        predictor_1.data["contributions"] = pd.DataFrame(
            {"x1": [0, 0, -0, -0, -0], "x2": [0.161538, -0.0403846, 0.0403846, 0.0403846, 0.0403846]}
        )
        output = predictor_1.summarize()

        expected_output = pd.DataFrame(
            {
                "y": ["Yes", "Yes", "No", "No", "No"],
                "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209],
                "feature_1": ["weight", "weight", "weight", "weight", "weight"],
                "value_1": ["90", "78", "84", "85", "53"],
                "contribution_1": ["0.161538", "-0.0403846", "0.0403846", "0.0403846", "0.0403846"],
                "feature_2": ["age", "age", "age", "age", "age"],
                "value_2": ["25", "39", "50", "43", "67"],
                "contribution_2": ["0", "0", "0", "0", "0"],
            },
            dtype=object,
        )
        expected_output["proba"] = expected_output["proba"].astype(float)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert expected_output.shape == output.shape
        assert len(feature_expected) == len(feature_output)
        assert len(value_expected) == len(value_output)
        assert len(contribution_expected) == len(contribution_output)
        assert all(output.columns == expected_output.columns)

    def test_summarize_3(self):
        """
        Unit test 3 summarize method
        """
        predictor_1 = self.predictor_3
        predictor_1.mask_params = {"features_to_hide": None, "threshold": None, "positive": None, "max_contrib": 1}

        predictor_1.data = {
            "x": None,
            "x_preprocessed": None,
            "x_postprocessed": None,
            "ypred": None,
            "contributions": None,
        }

        predictor_1.data["x"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_postprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(
            {"y": ["Yes", "Yes", "No", "No", "No"], "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209]}
        )
        predictor_1.data["contributions"] = pd.DataFrame(
            {"x1": [0, 0, -0, -0, -0], "x2": [0.161538, -0.0403846, 0.0403846, 0.0403846, 0.0403846]}
        )
        output = predictor_1.summarize()

        expected_output = pd.DataFrame(
            {
                "y": ["Yes", "Yes", "No", "No", "No"],
                "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209],
                "feature_1": ["weight", "weight", "weight", "weight", "weight"],
                "value_1": ["90", "78", "84", "85", "53"],
                "contribution_1": ["0.161538", "-0.0403846", "0.0403846", "0.0403846", "0.0403846"],
                "feature_2": ["age", "age", "age", "age", "age"],
                "value_2": ["25", "39", "50", "43", "67"],
                "contribution_2": ["0", "0", "0", "0", "0"],
            },
            dtype=object,
        )
        expected_output["proba"] = expected_output["proba"].astype(float)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert not expected_output.shape == output.shape
        assert not len(feature_expected) == len(feature_output)
        assert not len(value_expected) == len(value_output)
        assert not len(contribution_expected) == len(contribution_output)
        assert not len(output.columns) == len(expected_output.columns)

        predictor_1.mask_params = {"features_to_hide": None, "threshold": None, "positive": None, "max_contrib": None}

    def test_summarize_4(self):
        """
        Unit test 4 summarize method : with groups of features
        """
        predictor_1 = self.predictor_1_w_groups
        predictor_1._case = "classification"
        predictor_1._classes = [0, 1]
        clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df_3[["x1", "x2"]], self.df_3["y"])
        backend = ShapBackend(model=clf)
        predictor_1.model = clf
        predictor_1.backend = backend

        with self.assertRaises(ValueError):
            predictor_1.summarize()

        predictor_1.data = {
            "x": None,
            "x_preprocessed": None,
            "x_postprocessed": None,
            "ypred": None,
            "contributions": None,
        }

        predictor_1.data_groups = {"x_postprocessed": None, "ypred": None, "contributions": None}

        predictor_1.data["x"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_preprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["x_postprocessed"] = self.df_3[["x1", "x2"]]
        predictor_1.data["ypred"] = pd.DataFrame(
            {"y": ["Yes", "Yes", "No", "No", "No"], "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209]}
        )
        predictor_1.data_groups["x_postprocessed"] = self.df_3[["x1"]].rename(columns={"x1": "group1"})
        predictor_1.data_groups["ypred"] = predictor_1.data["ypred"]

        predictor_1.data["contributions"] = pd.DataFrame(
            {"x1": [0, 0, -0, -0, -0], "x2": [0.161538, -0.0403846, 0.0403846, 0.0403846, 0.0403846]}
        )
        predictor_1.data_groups["contributions"] = pd.DataFrame(
            {"group1": [0.161538, -0.0403846, 0.0403846, 0.0403846, 0.0403846]}
        )
        output = predictor_1.summarize()

        expected_output = pd.DataFrame(
            {
                "y": ["Yes", "Yes", "No", "No", "No"],
                "proba": [0.519221, 0.468791, 0.531209, 0.531209, 0.531209],
                "feature_1": ["weight", "weight", "weight", "weight", "weight"],
                "value_1": ["90", "78", "84", "85", "53"],
                "contribution_1": ["0.161538", "-0.0403846", "0.0403846", "0.0403846", "0.0403846"],
            },
            dtype=object,
        )
        expected_output["proba"] = expected_output["proba"].astype(float)

        feature_expected = [column for column in expected_output.columns if column.startswith("feature_")]
        feature_output = [column for column in output.columns if column.startswith("feature_")]

        value_expected = [column for column in expected_output.columns if column.startswith("value_")]
        value_output = [column for column in output.columns if column.startswith("value_")]

        contribution_expected = [column for column in expected_output.columns if column.startswith("contribution_")]
        contribution_output = [column for column in output.columns if column.startswith("contribution_")]

        assert expected_output.shape == output.shape
        assert len(feature_expected) == len(feature_output)
        assert len(value_expected) == len(value_output)
        assert len(contribution_expected) == len(contribution_output)
        assert all(output.columns == expected_output.columns)

    def test_modfiy_mask(self):
        """
        Unit test modify_mask method
        """
        predictor_1 = self.predictor_2

        assert all([value is None for value in predictor_1.mask_params.values()])

        predictor_1.modify_mask(max_contrib=1)

        assert not all([value is None for value in predictor_1.mask_params.values()])
        assert predictor_1.mask_params["max_contrib"] == 1
        assert predictor_1.mask_params["positive"] == None

        predictor_1.modify_mask(max_contrib=2)

    def test_apply_postprocessing_1(self):
        """
        Unit test apply_postprocessing 1
        """
        predictor_1 = self.predictor_3
        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = pd.DataFrame([[1, 2], [3, 4]], columns=["Col1", "Col2"], index=["Id1", "Id2"])
        assert np.array_equal(predictor_1.data["x"], predictor_1.apply_postprocessing())

    def test_apply_postprocessing_2(self):
        """
        Unit test apply_postprocessing 2
        """
        postprocessing = {"x1": {"type": "suffix", "rule": " t"}, "x2": {"type": "prefix", "rule": "test"}}

        predictor_1 = self.predictor_3
        predictor_1.postprocessing = postprocessing

        predictor_1.data = {"x": None, "ypred": None, "contributions": None}
        predictor_1.data["x"] = pd.DataFrame([[1, 2], [3, 4]], columns=["x1", "x2"], index=["Id1", "Id2"])
        expected_output = pd.DataFrame(
            data=[["1 t", "test2"], ["3 t", "test4"]], columns=["x1", "x2"], index=["Id1", "Id2"]
        )
        output = predictor_1.apply_postprocessing()
        assert np.array_equal(output, expected_output)

    def test_convert_dict_dataset(self):
        """
        Unit test convert_dict_dataset
        """
        predictor_1 = self.predictor_1

        x = predictor_1.convert_dict_dataset(x={"x1": 1, "x2": "M"})

        assert all(
            [
                str(x[feature].dtypes) == predictor_1.features_types[feature]
                for feature in predictor_1.features_types.keys()
            ]
        )

        with self.assertRaises(ValueError):
            predictor_1.convert_dict_dataset(x={"x1": "M", "x2": "M"})
            predictor_1.convert_dict_dataset(x={"x1": 1, "x2": "M", "x3": "M"})

    @patch("shapash.explainer.smart_predictor.SmartPredictor.convert_dict_dataset")
    def test_check_dataset_type(self, convert_dict_dataset):
        """
        Unit test check_dataset_type
        """
        convert_dict_dataset.return_value = pd.DataFrame({"x1": [1], "x2": ["M"]})
        predictor_1 = self.predictor_1

        with self.assertRaises(ValueError):
            predictor_1.check_dataset_type(x=1)
            predictor_1.check_dataset_type(x=["x1", "x2"])
            predictor_1.check_dataset_type(x=("x1", "x2"))

        predictor_1.check_dataset_type(x=pd.DataFrame({"x1": [1], "x2": ["M"]}))
        predictor_1.check_dataset_type(x={"x1": 1, "x2": "M"})

    def test_check_dataset_features(self):
        """
        Unit test check_dataset_features
        """
        predictor_1 = self.predictor_1

        with self.assertRaises(AssertionError):
            predictor_1.check_dataset_features(x=pd.DataFrame({"x1": [1], "x2": ["M"], "x3": ["M"]}))

        with self.assertRaises(ValueError):
            predictor_1.check_dataset_features(x=pd.DataFrame({"x1": [1], "x2": [1]}))
            predictor_1.check_dataset_features(x=pd.DataFrame({"x1": ["M"], "x2": ["M"]}))

        x = predictor_1.check_dataset_features(x=pd.DataFrame({"x2": ["M"], "x1": [1]}))
        assert all(
            [
                str(x[feature].dtypes) == predictor_1.features_types[feature]
                for feature in predictor_1.features_types.keys()
            ]
        )

        features_order = []
        for order in range(min(predictor_1.columns_dict.keys()), max(predictor_1.columns_dict.keys()) + 1):
            features_order.append(predictor_1.columns_dict[order])
        assert all(x.columns == features_order)

        predictor_1.check_dataset_features(x=pd.DataFrame({"x1": [1], "x2": ["M"]}))

    def test_to_smartexplainer(self):
        """
        Unit test to_smartexplainer
        """
        predictor_1 = self.predictor_1

        data_x = pd.DataFrame({"x1": [113, 51, 60, 110, 60], "x2": ["S", "M", "S", "D", "M"]})
        data_x["id"] = [0, 1, 2, 3, 4]
        data_x = data_x.set_index("id")
        data_x_preprocessed = self.preprocessing_1.transform(data_x)
        data_ypred_init = pd.DataFrame({"ypred": [1, 0, 0, 0, 0]})
        data_ypred_init.index = data_x.index

        predictor_1.data = {"x": data_x, "ypred_init": data_ypred_init, "x_preprocessed": data_x_preprocessed}

        xpl = predictor_1.to_smartexplainer()

        assert str(type(xpl)) == "<class 'shapash.explainer.smart_explainer.SmartExplainer'>"
        assert xpl.x_encoded.equals(predictor_1.data["x_preprocessed"])
        assert predictor_1.model == xpl.model
        assert predictor_1.backend == xpl.backend
        assert predictor_1.features_dict == xpl.features_dict
        assert predictor_1.label_dict == xpl.label_dict
        assert predictor_1._case == xpl._case
        assert predictor_1._classes == xpl._classes
        assert predictor_1.columns_dict == xpl.columns_dict
        assert predictor_1.preprocessing == xpl.preprocessing
        assert predictor_1.postprocessing == xpl.postprocessing

        ct = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OrdinalEncoder(), ["x2"]),
            ],
            remainder="passthrough",
        )
        ct.fit(data_x)

        predictor_1.preprocessing = ct
        with self.assertRaises(ValueError):
            predictor_1.to_smartexplainer()
