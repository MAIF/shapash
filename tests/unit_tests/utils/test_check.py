"""
Unit test of Check
"""
import types
import unittest

import catboost as cb
import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import sklearn.preprocessing as skp
import sklearn.svm as svm
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from shapash.utils.check import (
    check_additional_data,
    check_consistency_model_features,
    check_consistency_model_label,
    check_contribution_object,
    check_label_dict,
    check_mask_params,
    check_model,
    check_postprocessing,
    check_preprocessing,
    check_preprocessing_options,
    check_y,
)


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.modellist = [
            lgb.LGBMRegressor(n_estimators=1),
            lgb.LGBMClassifier(n_estimators=1),
            xgb.XGBRegressor(n_estimators=1),
            xgb.XGBRegressor(n_estimators=1),
            cb.CatBoostRegressor(n_estimators=1),
            cb.CatBoostClassifier(n_estimators=1),
            ske.GradientBoostingRegressor(n_estimators=1),
            ske.GradientBoostingClassifier(n_estimators=1),
            ske.ExtraTreesRegressor(n_estimators=1),
            ske.ExtraTreesClassifier(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1),
            ske.RandomForestClassifier(n_estimators=1),
            skl.LogisticRegression(),
            skl.LinearRegression(),
            svm.SVR(kernel="linear"),
            svm.SVC(kernel="linear"),
        ]

    def test_check_preprocessing_1(self):
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

        check_preprocessing([enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1, list_dict])
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
        assert _case == "regression"
        assert _classes is None

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

    def test_check_model_2(self):
        """
        Unit test check model 2
        """
        model = lambda: None
        model._classes = np.array([1, 2])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        _case, _classes = check_model(model)
        assert _case == "classification"
        self.assertListEqual(_classes, [1, 2])

    def test_check_label_dict_1(self):
        """
        Unit test check label dict 1
        """
        label_dict = {1: "Yes", 0: "No"}
        _classes = [0, 1]
        _case = "classification"
        check_label_dict(label_dict, _case, _classes)

    def test_check_label_dict_2(self):
        """
        Unit test check label dict 2
        """
        label_dict = {}
        _case = "regression"
        check_label_dict(label_dict, _case)

    def test_check_mask_params(self):
        """
        Unit test check mask params
        """
        wrong_mask_params_1 = list()
        wrong_mask_params_2 = None
        wrong_mask_params_3 = {"features_to_hide": None, "threshold": None, "positive": None}
        wright_mask_params = {"features_to_hide": None, "threshold": None, "positive": True, "max_contrib": 5}
        with self.assertRaises(ValueError):
            check_mask_params(wrong_mask_params_1)
            check_mask_params(wrong_mask_params_2)
            check_mask_params(wrong_mask_params_3)
        check_mask_params(wright_mask_params)

    def test_check_y_1(self):
        """
        Unit test check y
        """
        y_pred = None
        check_y(y=y_pred)

    def test_check_y_2(self):
        """
        Unit test check y 2
        """
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        y_pred = pd.DataFrame(data=np.array(["1", 0]), columns=["Y"])
        with self.assertRaises(ValueError):
            check_y(x_init, y_pred)

    def test_check_y_3(self):
        """
        Unit test check y 3
        """
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        y_pred = pd.DataFrame(data=np.array([0]), columns=["Y"])
        with self.assertRaises(ValueError):
            check_y(x_init, y_pred)

    def test_check_y_4(self):
        """
        Unit test check y 4
        """
        y_pred = [0, 1]
        with self.assertRaises(ValueError):
            check_y(y=y_pred)

    def test_check_y_5(self):
        """
        Unit test check y 5
        """
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        y_pred = pd.Series(data=np.array(["0"]))
        with self.assertRaises(ValueError):
            check_y(x_init, y_pred)

    def test_check_y_6(self):
        """
        Unit test check y 6
        """
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        y_pred = pd.Series(data=np.array(["0"]))
        with pytest.raises(Exception) as exc_info:
            check_y(x_init, y_pred, y_name="y_pred")
        assert str(exc_info.value) == "x and y_pred should have the same index."

    def test_check_contribution_object_1(self):
        """
        Unit test check_contribution_object 1
        """
        contributions_1 = [np.array([[2, 1], [8, 4]]), np.array([[5, 5], [0, 0]])]

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

    def test_check_consistency_model_features_1(self):
        """
        Test check_consistency_model_features 1
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
        mask_params = None

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
        preprocessing = enc_ordinal_all

        y = pd.DataFrame({"y_class": [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)

        check_consistency_model_features(features_dict, model, columns_dict, features_types, mask_params, preprocessing)

    def test_check_consistency_model_features_2(self):
        """
        Test check_consistency_model_features 2
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

        mask_params = {"features_to_hide": "Binary3", "threshold": None, "positive": True, "max_contrib": 5}

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
        preprocessing = enc_ordinal_all

        y = pd.DataFrame({"y_class": [0, 0, 0, 1]})

        model = cb.CatBoostClassifier(n_estimators=1).fit(train_ordinal_all, y)

        with self.assertRaises(ValueError):
            check_consistency_model_features(
                features_dict, model, columns_dict, features_types, mask_params, preprocessing
            )

    def test_check_preprocessing_options_1(self):
        """
        Unit test 1 for check_preprocessing_options
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])
        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})
        enc = ColumnTransformer(
            transformers=[("power", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )
        enc.fit(train, y)

        with self.assertRaises(ValueError):
            check_preprocessing_options(enc)

        enc = ColumnTransformer(
            transformers=[("power", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="passthrough"
        )
        enc.fit(train, y)
        check_preprocessing_options(enc)

    def test_check_consistency_model_features_4(self):
        """
        Test check_consistency_model_features 1
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
        mask_params = None

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
        preprocessing = enc_ordinal_all

        y = pd.DataFrame({"y_class": [0, 0, 0, 1]})

        for model in self.modellist:
            print(type(model))
            model.fit(train_ordinal_all, y)

            check_consistency_model_features(
                features_dict,
                model,
                columns_dict,
                features_types,
                mask_params,
                preprocessing,
                list_preprocessing=[preprocessing],
            )

    def test_check_consistency_model_features_5(self):
        """
        Unit test check_consistency_model_features 5
        """
        train = pd.DataFrame(
            {"city": ["chicago", "paris"], "state": ["US", "FR"], "other": [5, 10]}, index=["index1", "index2"]
        )

        features_dict = None
        columns_dict = {i: features for i, features in enumerate(train.columns)}
        features_types = {features: str(train[features].dtypes) for features in train.columns}
        mask_params = None

        enc = ColumnTransformer(
            transformers=[
                ("Ordinal_ce", ce.OrdinalEncoder(), ["city", "state"]),
                ("Ordinal_skp", skp.OrdinalEncoder(), ["city", "state"]),
            ],
            remainder="passthrough",
        )

        enc_2 = ColumnTransformer(
            transformers=[
                ("Ordinal_ce", ce.OrdinalEncoder(), ["city", "state"]),
                ("Ordinal_skp", skp.OrdinalEncoder(), ["city", "state"]),
            ],
            remainder="drop",
        )

        enc.fit(train)
        train_1 = pd.DataFrame(enc.transform(train), columns=["city_ce", "state_ce", "city_skp", "state_skp", "other"])
        train_1["y"] = np.array([1, 0])

        clf_1 = cb.CatBoostClassifier(n_estimators=1).fit(
            train_1[["city_ce", "state_ce", "city_skp", "state_skp", "other"]], train_1["y"]
        )

        enc_2.fit(train)
        train_2 = pd.DataFrame(enc_2.transform(train), columns=["city_ce", "state_ce", "city_skp", "state_skp"])
        train_2["y"] = np.array([1, 0])

        clf_2 = cb.CatBoostClassifier(n_estimators=1).fit(
            train_2[["city_ce", "state_ce", "city_skp", "state_skp"]], train_2["y"]
        )

        enc_3 = ce.OneHotEncoder(cols=["city", "state"])
        enc_3.fit(train)
        train_3 = enc_3.transform(train)
        train_3["y"] = np.array([1, 0])

        clf_3 = cb.CatBoostClassifier(n_estimators=1).fit(
            train_3[["city_1", "city_2", "state_1", "state_2", "other"]], train_3["y"]
        )

        dict_4 = {"col": "state", "mapping": pd.Series(data=[1, 2], index=["US", "FR"]), "data_type": "object"}

        dict_5 = {"col": "city", "mapping": pd.Series(data=[1, 2], index=["chicago", "paris"]), "data_type": "object"}

        enc_4 = [enc_3, [dict_4]]

        enc_5 = [enc_3, [dict_4, dict_5]]

        check_consistency_model_features(
            features_dict, clf_1, columns_dict, features_types, mask_params, enc, list_preprocessing=[enc]
        )

        check_consistency_model_features(
            features_dict, clf_2, columns_dict, features_types, mask_params, enc_2, list_preprocessing=[enc_2]
        )

        check_consistency_model_features(
            features_dict, clf_3, columns_dict, features_types, mask_params, enc_3, list_preprocessing=[enc_3]
        )

        check_consistency_model_features(
            features_dict, clf_3, columns_dict, features_types, mask_params, enc_4, list_preprocessing=enc_4
        )

        check_consistency_model_features(
            features_dict, clf_3, columns_dict, features_types, mask_params, enc_5, list_preprocessing=enc_5
        )

    def test_check_consistency_model_label_1(self):
        """
        Test check_consistency_model_label 1
        """
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 1: "No"}

        check_consistency_model_label(columns_dict, label_dict)

    def test_check_consistency_model_label_2(self):
        """
        Test check_consistency_model_label 2
        """
        columns_dict = {0: "x1", 1: "x2"}
        label_dict = {0: "Yes", 2: "No"}

        with self.assertRaises(ValueError):
            check_consistency_model_label(columns_dict, label_dict)

    def test_check_postprocessing_1(self):
        """
        Unit test check_consistency_postprocessing
        """
        x = pd.DataFrame([[1, 2], [3, 4]], columns=["Col1", "Col2"], index=["Id1", "Id2"])
        columns_dict = {0: "Col1", 1: "Col2"}
        features_types = {features: str(x[features].dtypes) for features in x.columns}
        postprocessing1 = {0: {"Error": "suffix", "rule": " t"}}
        postprocessing2 = {0: {"type": "Error", "rule": " t"}}
        postprocessing3 = {0: {"type": "suffix", "Error": " t"}}
        postprocessing4 = {0: {"type": "suffix", "rule": " "}}
        postprocessing5 = {0: {"type": "case", "rule": "lower"}}
        postprocessing6 = {0: {"type": "case", "rule": "Error"}}
        with self.assertRaises(ValueError):
            check_postprocessing(features_types, postprocessing1)
            check_postprocessing(features_types, postprocessing2)
            check_postprocessing(features_types, postprocessing3)
            check_postprocessing(features_types, postprocessing4)
            check_postprocessing(features_types, postprocessing5)
            check_postprocessing(features_types, postprocessing6)

    def test_check_preprocessing_options_1(self):
        """
        Unit test check_preprocessing_options 1
        """
        df = pd.DataFrame(range(0, 5), columns=["id"])
        df["y"] = df["id"].apply(lambda x: 1 if x < 2 else 0)
        df["x1"] = np.random.randint(1, 123, df.shape[0])
        df = df.set_index("id")
        df["x2"] = ["S", "M", "S", "D", "M"]
        df["x3"] = np.random.randint(1, 123, df.shape[0])
        df["x4"] = ["S", "M", "S", "D", "M"]

        features_dict = {"x1": "age", "x2": "weight", "x3": "test", "x4": "test2"}
        columns_dict = {0: "x1", 1: "x2", 2: "x3", 3: "x4"}

        encoder = ColumnTransformer(
            transformers=[("onehot_ce_1", ce.OneHotEncoder(), ["x2"]), ("onehot_ce_2", ce.OneHotEncoder(), ["x4"])],
            remainder="drop",
        )
        encoder_fitted = encoder.fit(df[["x1", "x2", "x3", "x4"]])

        encoder_2 = ColumnTransformer(
            transformers=[("onehot_ce_1", ce.OneHotEncoder(), ["x2"]), ("onehot_ce_2", ce.OneHotEncoder(), ["x4"])],
            remainder="passthrough",
        )
        encoder_fitted_2 = encoder_2.fit(df[["x1", "x2", "x3", "x4"]])

        expected_dict = {
            "features_to_drop": ["x1", "x3"],
            "features_dict_op": {"x2": "weight", "x4": "test2"},
            "columns_dict_op": {0: "x2", 1: "x4"},
        }

        expected_dict_2 = None

        drop_option_1 = check_preprocessing_options(columns_dict, features_dict, encoder_fitted, [encoder_fitted])
        drop_option_2 = check_preprocessing_options(columns_dict, features_dict, encoder_fitted_2, [encoder_fitted_2])
        assert drop_option_1 == expected_dict
        assert drop_option_2 == expected_dict_2

    def test_check_additional_data_raises_index(self):
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        additional_data = pd.DataFrame(data=np.array([[5]]), columns=["Col3"])
        with pytest.raises(Exception) as exc_info:
            check_additional_data(x_init, additional_data)
        assert str(exc_info.value) == "x and additional_data should have the same index."

    def test_check_additional_data_raises_type(self):
        x_init = pd.DataFrame(data=np.array([[1, 2], [3, 4]]), columns=["Col1", "Col2"])
        additional_data = pd.Series(data=np.array([5, 6]), name="Col3")
        with pytest.raises(Exception) as exc_info:
            check_additional_data(x_init, additional_data)
        assert str(exc_info.value) == "additional_data must be a pd.Dataframe."
