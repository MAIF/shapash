"""
Unit test of Inverse Transform
"""
import unittest

import catboost as cb
import category_encoders as ce
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import GradientBoostingClassifier

from shapash.utils.transform import apply_preprocessing, get_col_mapping_ce, inverse_transform


class TestInverseTransformCaterogyEncoder(unittest.TestCase):
    def test_inverse_transform_1(self):
        """
        Test no preprocessing
        """
        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"]})
        original = inverse_transform(train)
        pd.testing.assert_frame_equal(original, train)

    def test_inverse_transform_2(self):
        """
        Test multiple preprocessing
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

        test = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A"],
                "Onehot2": ["C", "D", "ZZ"],
                "Binary1": ["E", "F", "F"],
                "Binary2": ["G", "H", "ZZ"],
                "Ordinal1": ["I", "J", "J"],
                "Ordinal2": ["K", "L", "ZZ"],
                "BaseN1": ["M", "N", "N"],
                "BaseN2": ["O", "P", "ZZ"],
                "Target1": ["Q", "R", "R"],
                "Target2": ["S", "T", "ZZ"],
                "other": ["other", "123", np.nan],
            }
        )

        expected = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A"],
                "Onehot2": ["C", "D", "missing"],
                "Binary1": ["E", "F", "F"],
                "Binary2": ["G", "H", "missing"],
                "Ordinal1": ["I", "J", "J"],
                "Ordinal2": ["K", "L", "missing"],
                "BaseN1": ["M", "N", "N"],
                "BaseN2": ["O", "P", np.nan],
                "Target1": ["Q", "R", "R"],
                "Target2": ["S", "T", "NaN"],
                "other": ["other", "123", np.nan],
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

        input_dict3 = dict()
        input_dict3["col"] = "Ordinal2"
        input_dict3["mapping"] = pd.Series(data=["K", "L", np.nan], index=["K", "L", "missing"])
        input_dict3["data_type"] = "object"
        list_dict = [input_dict2, input_dict3]

        result1 = enc_onehot.transform(test)
        result2 = enc_binary.transform(result1)
        result3 = enc_ordinal.transform(result2)
        result4 = enc_basen.transform(result3)
        result5 = enc_target.transform(result4)

        original = inverse_transform(
            result5, [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1, list_dict]
        )

        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_3(self):
        """
        Test target encoding
        """
        train = pd.DataFrame(
            {
                "city": ["chicago", "paris", "paris", "chicago", "chicago"],
                "state": ["US", "FR", "FR", "US", "US"],
                "other": ["A", "A", np.nan, "B", "B"],
            }
        )
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        expected = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1, 0, 1], columns=["y"])

        enc = ce.TargetEncoder(cols=["city", "state"]).fit(train, y)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_4(self):
        """
        Test ordinal encoding
        """
        train = pd.DataFrame({"city": ["chicago", "st louis"]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})
        expected = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OrdinalEncoder(handle_missing="value", handle_unknown="value")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_5(self):
        """
        Test inverse_transform having Nan in train and handle missing value expect returned with nan_Ordinal
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OrdinalEncoder(handle_missing="value", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_6(self):
        """
        test inverse_transform having Nan in train and handle missing return Nan expect returned with nan_Ordinal
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OrdinalEncoder(handle_missing="return_nan", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_7(self):
        """
        test inverse_transform both fields are return Nan with Nan Expect ValueError Ordinal
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})
        enc = ce.OrdinalEncoder(handle_missing="return_nan", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_8(self):
        """
        test inverse_transform having missing and no Uknown expect inversed ordinal
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})
        enc = ce.OrdinalEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_9(self):
        """
        test inverse_transform having handle missing value and handle unknown return Nan expect best inverse ordinal
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", np.nan, "los angeles"]})
        expected = pd.DataFrame({"city": ["chicago", np.nan, np.nan]})
        enc = ce.OrdinalEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_10(self):
        """
        test inverse_transform with multiple ordinal
        """
        data = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["a", "b"]})
        test = pd.DataFrame({"city": [1, 2, 2], "state": [1, 2, 2], "other": ["a", "b", "a"]})
        expected = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["a", "b", "a"]}
        )
        enc = ce.OrdinalEncoder(cols=["city", "state"])
        enc.fit(data)
        original = inverse_transform(test, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inverse_transform_11(self):
        """
        Test binary encoding
        """
        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", np.nan]})

        test = pd.DataFrame(
            {"city": ["chicago", "paris", "monaco"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, "B"]}
        )

        expected = pd.DataFrame(
            {"city": ["chicago", "paris", np.nan], "state": ["US", "FR", "FR"], "other": ["A", np.nan, "B"]}
        )

        enc = ce.BinaryEncoder(cols=["city", "state"]).fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inverse_transform_12(self):
        """
        test inverse_transform having data expecting a returned result
        """
        train = pd.Series(list("abcd")).to_frame("letter")
        enc = ce.BaseNEncoder(base=2)
        result = enc.fit_transform(train)
        inversed_result = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, inversed_result)

    def test_inverse_transform_13(self):
        """
        Test basen encoding
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.BaseNEncoder(handle_missing="value", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_14(self):
        """
        test inverse_transform having Nan in train and handle missing expected a result with Nan
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})

        enc = ce.BaseNEncoder(handle_missing="return_nan", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_15(self):
        """
        test inverse_transform having missing and no unknown
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})

        enc = ce.BaseNEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_16(self):
        """
        test inverse_transform having handle missing value and Unknown
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", np.nan, "los angeles"]})
        expected = pd.DataFrame({"city": ["chicago", np.nan, np.nan]})
        enc = ce.BaseNEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_17(self):
        """
        test inverse_transform with multiple baseN
        """
        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"]})
        test = pd.DataFrame({"city_0": [0, 1], "city_1": [1, 0], "state_0": [0, 1], "state_1": [1, 0]})
        enc = ce.BaseNEncoder(cols=["city", "state"], handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        original = inverse_transform(test, enc)
        pd.testing.assert_frame_equal(original, train)

    def test_inverse_transform_18(self):
        """
        Test Onehot encoding
        """
        encoder = ce.OneHotEncoder(cols=["match", "match_box"], use_cat_names=True)
        value = pd.DataFrame({"match": pd.Series("box_-1"), "match_box": pd.Series(-1)})
        transformed = encoder.fit_transform(value)
        inversed_result = inverse_transform(transformed, encoder)
        pd.testing.assert_frame_equal(value, inversed_result)

    def test_inverse_transform_19(self):
        """
        test inverse_transform having no categories names
        """
        encoder = ce.OneHotEncoder(cols=["match", "match_box"], use_cat_names=False)
        value = pd.DataFrame({"match": pd.Series("box_-1"), "match_box": pd.Series(-1)})
        transformed = encoder.fit_transform(value)
        inversed_result = inverse_transform(transformed, encoder)
        pd.testing.assert_frame_equal(value, inversed_result)

    def test_inverse_transform_20(self):
        """
        test inverse_transform with Nan in training expecting Nan_Onehot returned result
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OneHotEncoder(handle_missing="value", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_21(self):
        """
        test inverse_transform with Nan in training expecting Nan_Onehot returned result
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OneHotEncoder(handle_missing="return_nan", handle_unknown="value")
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_22(self):
        """
        test inverse_transform with Both fields return_nan
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})
        expected = pd.DataFrame({"city": ["chicago", np.nan]})
        enc = ce.OneHotEncoder(handle_missing="return_nan", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inverse_transform_23(self):
        """
        test inverse_transform having missing and No Unknown
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", "los angeles"]})
        enc = ce.OneHotEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_24(self):
        """
        test inverse_transform having handle missing value and Handle Unknown
        """
        train = pd.DataFrame({"city": ["chicago", np.nan]})
        test = pd.DataFrame({"city": ["chicago", np.nan, "los angeles"]})
        expected = pd.DataFrame({"city": ["chicago", np.nan, np.nan]})
        enc = ce.OneHotEncoder(handle_missing="value", handle_unknown="return_nan")
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_25(self):
        """
        Test dict encoding
        """
        data = pd.DataFrame(
            {"city": ["chicago", "paris-1", "paris-2"], "state": ["US", "FR-1", "FR-2"], "other": ["A", "B", np.nan]}
        )

        expected = pd.DataFrame(
            {"city": ["chicago", "paris-1", "paris-2"], "state": ["US", "FR", "FR"], "other": ["A", "B", np.nan]}
        )
        input_dict = dict()
        input_dict["col"] = "state"
        input_dict["mapping"] = pd.Series(data=["US", "FR-1", "FR-2"], index=["US", "FR", "FR"])
        input_dict["data_type"] = "object"
        result = inverse_transform(data, input_dict)
        pd.testing.assert_frame_equal(result, expected)

    def test_inverse_transform_26(self):
        """
        Test multiple dict encoding
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

        test = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A"],
                "Onehot2": ["C", "D", "ZZ"],
                "Binary1": ["E", "F", "F"],
                "Binary2": ["G", "H", "ZZ"],
                "Ordinal1": ["I", "J", "J"],
                "Ordinal2": ["K", "L", "ZZ"],
                "BaseN1": ["M", "N", "N"],
                "BaseN2": ["O", "P", "ZZ"],
                "Target1": ["Q", "R", "R"],
                "Target2": ["S", "T", "ZZ"],
                "other": ["other", "123", np.nan],
            },
            index=["index1", "index2", "index3"],
        )

        expected = pd.DataFrame(
            {
                "Onehot1": ["A", "B", "A"],
                "Onehot2": ["C", "D", "missing"],
                "Binary1": ["E", "F", "F"],
                "Binary2": ["G", "H", "missing"],
                "Ordinal1": ["I", "J", "J"],
                "Ordinal2": ["K", "L", "missing"],
                "BaseN1": ["M", "N", "N"],
                "BaseN2": ["O", "P", np.nan],
                "Target1": ["Q", "R", "R"],
                "Target2": ["S", "T", "NaN"],
                "other": ["other", "123", np.nan],
            },
            index=["index1", "index2", "index3"],
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

        input_dict3 = dict()
        input_dict3["col"] = "Ordinal2"
        input_dict3["mapping"] = pd.Series(data=["K", "L", np.nan], index=["K", "L", "missing"])
        input_dict3["data_type"] = "object"
        list_dict = [input_dict2, input_dict3]

        result1 = enc_onehot.transform(test)
        result2 = enc_binary.transform(result1)
        result3 = enc_ordinal.transform(result2)
        result4 = enc_basen.transform(result3)
        result5 = enc_target.transform(result4)

        original = inverse_transform(
            result5, [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1, list_dict]
        )

        pd.testing.assert_frame_equal(expected, original)

    def test_transform_ce_1(self):
        """
        Unit test for apply preprocessing on OneHotEncoder
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.one_hot.OneHotEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_2(self):
        """
        Unit test for apply preprocessing on OrdinalEncoder
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.ordinal.OrdinalEncoder(cols=["num1", "num2"])
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_3(self):
        """
        Unit test for apply preprocessing on BaseNEncoder
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.basen.BaseNEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_4(self):
        """
        Unit test for apply preprocessing on BinaryEncoder
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.binary.BinaryEncoder(cols=["num1", "num2"])
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_5(self):
        """
        Unit test for apply preprocessing with sklearn model
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.ordinal.OrdinalEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = GradientBoostingClassifier().fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert all(expected.index == result.index)

    def test_transform_ce_6(self):
        """
        Unit test for apply preprocessing with catboost model
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.ordinal.OrdinalEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_7(self):
        """
        Unit test for apply preprocessing with lightgbm model
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.ordinal.OrdinalEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = lightgbm.sklearn.LGBMClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.booster_.feature_name() for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ce_8(self):
        """
        Unit test for apply preprocessing with xgboost model
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ce.ordinal.OrdinalEncoder(cols=["num1", "num2"])

        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = xgboost.sklearn.XGBClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.get_booster().feature_names for column in result.columns]
        assert all(expected.index == result.index)

    def test_get_col_mapping_ce_1(self):
        """
        Test test_get_col_mapping_ce with target encoding
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.TargetEncoder(cols=["city", "state"])
        enc.fit(test, y)

        mapping = get_col_mapping_ce(enc)
        expected_mapping = {"city": ["city"], "state": ["state"]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ce_2(self):
        """
        Test test_get_col_mapping_ce with target OrdinalEncoder
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.OrdinalEncoder(handle_missing="value", handle_unknown="value")
        enc.fit(test, y)

        mapping = get_col_mapping_ce(enc)
        expected_mapping = {"city": ["city"], "state": ["state"], "other": ["other"]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ce_3(self):
        """
        Test test_get_col_mapping_ce with target BinaryEncoder
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "paris"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.BinaryEncoder(cols=["city", "state"])
        enc.fit(test, y)

        mapping = get_col_mapping_ce(enc)
        expected_mapping = {"city": ["city_0", "city_1"], "state": ["state_0", "state_1"]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ce_4(self):
        """
        Test test_get_col_mapping_ce with target BaseNEncoder
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "new york"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.BaseNEncoder(base=2)
        enc.fit(test, y)

        mapping = get_col_mapping_ce(enc)
        expected_mapping = {
            "city": ["city_0", "city_1"],
            "state": ["state_0", "state_1"],
            "other": ["other_0", "other_1"],
        }

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ce_5(self):
        """
        Test test_get_col_mapping_ce with target BaseNEncoder
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["A", np.nan, np.nan]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.OneHotEncoder(cols=["city", "state"], use_cat_names=True)
        enc.fit(test, y)

        mapping = get_col_mapping_ce(enc)
        expected_mapping = {"city": ["city_chicago", "city_paris"], "state": ["state_US", "state_FR"]}

        self.assertDictEqual(mapping, expected_mapping)
