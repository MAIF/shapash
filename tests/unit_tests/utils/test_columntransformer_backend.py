"""
Unit test of Inverse Transform
"""

import unittest

import catboost as cb
import category_encoders as ce
import lightgbm
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing as skp
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

from shapash.utils.columntransformer_backend import (
    get_col_mapping_ct,
    get_feature_names,
    get_list_features_names,
    get_names,
)
from shapash.utils.transform import apply_preprocessing, inverse_transform

# TODO
# StandardScaler return object vs float vs int
# Target encoding return object vs float


class TestInverseTransformColumnsTransformer(unittest.TestCase):
    def test_inv_transform_ct_1(self):
        """
        test inv_transform_ct with multiple encoding and drop option
        """
        train = pd.DataFrame(
            {"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]}, index=["index1", "index2"]
        )

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="drop",
        )
        enc.fit(train)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        expected = pd.DataFrame(
            {
                "onehot_ce_city": ["chicago", "chicago", "paris"],
                "onehot_ce_state": ["US", "FR", "FR"],
                "onehot_skp_city": ["chicago", "chicago", "paris"],
                "onehot_skp_state": ["US", "FR", "FR"],
            },
            index=["index1", "index2", "index3"],
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3_0", "col3_1", "col4_0", "col4_1"]
        result.index = ["index1", "index2", "index3"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_2(self):
        """
        test inv_transform_ct with multiple encoding and passthrough option
        """
        train = pd.DataFrame(
            {"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]}, index=["index1", "index2"]
        )

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        expected = pd.DataFrame(
            {
                "onehot_ce_city": ["chicago", "chicago", "paris"],
                "onehot_ce_state": ["US", "FR", "FR"],
                "onehot_skp_city": ["chicago", "chicago", "paris"],
                "onehot_skp_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            },
            index=["index1", "index2", "index3"],
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3_0", "col3_1", "col4_0", "col4_1", "other"]
        result.index = ["index1", "index2", "index3"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_3(self):
        """
        test inv_transform_ct with multiple encoding and dictionnary
        """
        train = pd.DataFrame(
            {"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]}, index=["index1", "index2"]
        )

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        expected = pd.DataFrame(
            {
                "onehot_ce_city": ["CH", "CH", "PR"],
                "onehot_ce_state": ["US-FR", "US-FR", "US-FR"],
                "onehot_skp_city": ["chicago", "chicago", "paris"],
                "onehot_skp_state": ["US", "FR", "FR"],
                "other": ["A-B", "A-B", "C"],
            },
            index=["index1", "index2", "index3"],
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3_0", "col3_1", "col4_0", "col4_1", "other"]
        result.index = ["index1", "index2", "index3"]

        input_dict1 = dict()
        input_dict1["col"] = "onehot_ce_city"
        input_dict1["mapping"] = pd.Series(data=["chicago", "paris"], index=["CH", "PR"])
        input_dict1["data_type"] = "object"

        input_dict2 = dict()
        input_dict2["col"] = "other"
        input_dict2["mapping"] = pd.Series(data=["A", "B", "C"], index=["A-B", "A-B", "C"])
        input_dict2["data_type"] = "object"

        input_dict3 = dict()
        input_dict3["col"] = "onehot_ce_state"
        input_dict3["mapping"] = pd.Series(data=["US", "FR"], index=["US-FR", "US-FR"])
        input_dict3["data_type"] = "object"
        list_dict = [input_dict2, input_dict3]

        original = inverse_transform(result, [enc, input_dict1, list_dict])
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_4(self):
        """
        test inv_transform_ct with single target category encoders and passthrough option
        """
        y = pd.DataFrame(data=[0, 1, 1, 1], columns=["y"])

        train = pd.DataFrame(
            {
                "city": ["chicago", "paris", "paris", "chicago"],
                "state": ["US", "FR", "FR", "US"],
                "other": ["A", "B", "B", "B"],
            }
        )

        enc = ColumnTransformer(
            transformers=[("target", ce.TargetEncoder(), ["city", "state"])], remainder="passthrough"
        )

        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            data={
                "target_city": ["chicago", "chicago", "paris"],
                "target_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            },
            dtype=object,
        )

        enc.fit(train, y)
        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_5(self):
        """
        test inv_transform_ct with single target category encoders and drop option
        """
        y = pd.DataFrame(data=[0, 1, 0, 0], columns=["y"])

        train = pd.DataFrame(
            {
                "city": ["chicago", "paris", "chicago", "paris"],
                "state": ["US", "FR", "US", "FR"],
                "other": ["A", "B", "A", "B"],
            }
        )

        enc = ColumnTransformer(transformers=[("target", ce.TargetEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            data={"target_city": ["chicago", "chicago", "paris"], "target_state": ["US", "FR", "FR"]}
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_6(self):
        """
        test inv_transform_ct with Ordinal Category Encoder and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("ordinal", ce.OrdinalEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)

        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"ordinal_city": ["chicago", "chicago", "paris"], "ordinal_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_7(self):
        """
        test inv_transform_ct with category Ordinal Encoder and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("ordinal", ce.OrdinalEncoder(), ["city", "state"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {
                "ordinal_city": ["chicago", "chicago", "paris"],
                "ordinal_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_8(self):
        """
        test inv_transform_ct with Binary encoder and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("binary", ce.BinaryEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)

        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"binary_city": ["chicago", "chicago", "paris"], "binary_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_9(self):
        """
        test inv_transform_ct with Binary Encoder and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("binary", ce.BinaryEncoder(), ["city", "state"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {
                "binary_city": ["chicago", "chicago", "paris"],
                "binary_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_10(self):
        """
        test inv_transform_ct with BaseN Encoder and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("basen", ce.BaseNEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"basen_city": ["chicago", "chicago", "paris"], "basen_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_11(self):
        """
        test inv_transform_ct with BaseN Encoder and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("basen", ce.BaseNEncoder(), ["city", "state"])], remainder="passthrough")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {"basen_city": ["chicago", "chicago", "paris"], "basen_state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_12(self):
        """
        test inv_transform_ct with single OneHotEncoder and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("onehot", ce.OneHotEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"onehot_city": ["chicago", "chicago", "paris"], "onehot_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_13(self):
        """
        test inv_transform_ct with OneHotEncoder and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("onehot", ce.OneHotEncoder(), ["city", "state"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {
                "onehot_city": ["chicago", "chicago", "paris"],
                "onehot_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_14(self):
        """
        test inv_transform_ct with OneHotEncoder Sklearn and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("onehot", skp.OneHotEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"onehot_city": ["chicago", "chicago", "paris"], "onehot_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_15(self):
        """
        test inv_transform_ct with OneHotEncoder Sklearn and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("onehot", skp.OneHotEncoder(), ["city", "state"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {
                "onehot_city": ["chicago", "chicago", "paris"],
                "onehot_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_16(self):
        """
        test inv_tranform_ct with ordinal Encoder sklearn and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("ordinal", skp.OrdinalEncoder(), ["city", "state"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame({"ordinal_city": ["chicago", "chicago", "paris"], "ordinal_state": ["US", "FR", "FR"]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_17(self):
        """
        test inv_transform_ct with OrdinalEncoder Sklearn and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"city": ["chicago", "paris"], "state": ["US", "FR"], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("ordinal", skp.OrdinalEncoder(), ["city", "state"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]}
        )

        expected = pd.DataFrame(
            {
                "ordinal_city": ["chicago", "chicago", "paris"],
                "ordinal_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_18(self):
        """
        test inv_transform_ct with Standardscaler Encoder Sklearn and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("std", skp.StandardScaler(), ["num1", "num2"])], remainder="passthrough")
        enc.fit(train, y)
        test = pd.DataFrame(
            {"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]},
        )
        if sk.__version__ >= "1.0.0":
            expected = pd.DataFrame(
                {"std_num1": [0.0, 1.0, 1.0], "std_num2": [0.0, 2.0, 3.0], "other": ["A", "B", "C"]},
            )
        else:
            expected = pd.DataFrame(
                {"std_num1": [0.0, 1.0, 1.0], "std_num2": [0.0, 2.0, 3.0], "other": ["A", "B", "C"]}, dtype=object
            )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_19(self):
        """
        test inv_transform_ct with StandarScaler Encoder Sklearn and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(transformers=[("std", skp.StandardScaler(), ["num1", "num2"])], remainder="drop")
        enc.fit(train, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame({"std_num1": [0.0, 1.0, 1.0], "std_num2": [0.0, 2.0, 3.0]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["col1", "col2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_20(self):
        """
        test inv_transform_ct with QuantileTransformer Encoder Sklearn and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])],
            remainder="passthrough",
        )
        enc.fit(train, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame(
            {"quantile_num1": [0.0, 1.0, 1.0], "quantile_num2": [0.0, 2.0, 2.0], "other": ["A", "B", "C"]}
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["num1", "num2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_21(self):
        """
        test inv_transform_ct with QuandtileTransformer Encoder Sklearn and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )
        enc.fit(train, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame({"quantile_num1": [0.0, 1.0, 1.0], "quantile_num2": [0.0, 2.0, 2.0]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["num1", "num2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_22(self):
        """
        test inv_transform_ct with PowerTransformer Encoder Sklearn and passthrough option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("power", skp.PowerTransformer(), ["num1", "num2"])], remainder="passthrough"
        )
        enc.fit(train, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame(
            {
                "power_num1": [0.0, 1.0, 1.0],
                "power_num2": [0.0, 1.9999999997665876, 3.000000000169985],
                "other": ["A", "B", "C"],
            }
        )

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["num1", "num2", "other"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inv_transform_ct_23(self):
        """
        test inv_transform_ct with PowerTransformer Encoder Sklearn and drop option
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("power", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )
        enc.fit(train, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame({"power_num1": [0.0, 1.0, 1.0], "power_num2": [0.0, 2.0, 2.0]})

        result = pd.DataFrame(enc.transform(test))
        result.columns = ["num1", "num2"]
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_transform_ct_1(self):
        """
        Unit test for apply_preprocessing on ColumnTransformer with drop option and sklearn encoder.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc = ColumnTransformer(
            transformers=[("power", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))

        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)

        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        expected = pd.DataFrame(enc.transform(test))
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)
        assert all([str(type_result) == str(expected.dtypes[index]) for index, type_result in enumerate(result.dtypes)])

    def test_transform_ct_2(self):
        """
        Unit test for apply_preprocessing on ColumnTransformer with passthrough option and category encoder.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"])], remainder="passthrough"
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": [1, 0, 3]})

        expected = pd.DataFrame(enc.transform(test))
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)
        assert all([str(type_result) == str(expected.dtypes[index]) for index, type_result in enumerate(result.dtypes)])

    def test_transform_ct_3(self):
        """
        Unit test for apply_preprocessing on ColumnTransformer with sklearn encoder and category encoder.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"]),
                ("onehot_skp", skp.OneHotEncoder(), ["num1", "num2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ct_4(self):
        """
        Unit test for apply_preprocessing on list of a dict, a list of dict and a ColumnTransformer.
        """
        train = pd.DataFrame(
            {"city": ["CH", "CH", "PR"], "state": ["US-FR", "US-FR", "US-FR"], "other": ["A-B", "A-B", "C"]},
            index=["index1", "index2", "index3"],
        )

        y = pd.DataFrame(data=[0, 1, 0], columns=["y"], index=["index1", "index2", "index3"])

        train_preprocessed = train.copy()
        input_dict1 = dict()
        input_dict1["col"] = "city"
        input_dict1["mapping"] = pd.Series(data=["chicago", "paris"], index=["CH", "PR"])
        input_dict1["data_type"] = "object"

        transform_input_1 = pd.Series(data=input_dict1.get("mapping").values, index=input_dict1.get("mapping").index)
        train_preprocessed[input_dict1.get("col")] = (
            train_preprocessed[input_dict1.get("col")]
            .map(transform_input_1)
            .astype(input_dict1.get("mapping").values.dtype)
        )

        input_dict2 = dict()
        input_dict2["col"] = "other"
        input_dict2["mapping"] = pd.Series(data=["A", "C"], index=["A-B", "C"])
        input_dict2["data_type"] = "object"

        transform_input_2 = pd.Series(data=input_dict2.get("mapping").values, index=input_dict2.get("mapping").index)
        train_preprocessed[input_dict2.get("col")] = (
            train_preprocessed[input_dict2.get("col")]
            .map(transform_input_2)
            .astype(input_dict2.get("mapping").values.dtype)
        )

        input_dict3 = dict()
        input_dict3["col"] = "state"
        input_dict3["mapping"] = pd.Series(data=["US FR"], index=["US-FR"])
        input_dict3["data_type"] = "object"

        transform_input_3 = pd.Series(data=input_dict3.get("mapping").values, index=input_dict3.get("mapping").index)
        train_preprocessed[input_dict3.get("col")] = (
            train_preprocessed[input_dict3.get("col")]
            .map(transform_input_3)
            .astype(input_dict3.get("mapping").values.dtype)
        )

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["other"]),
            ],
            remainder="passthrough",
        )

        enc.fit(train_preprocessed)
        train_preprocessed = pd.DataFrame(enc.transform(train_preprocessed), index=train.index)
        train_preprocessed.columns = [str(feature) for feature in train_preprocessed.columns]

        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)

        list_dict = [input_dict2, input_dict3]

        test_preprocessing = apply_preprocessing(train, clf, [input_dict1, list_dict, enc])
        pd.testing.assert_frame_equal(train_preprocessed, test_preprocessing)

    def test_transform_ct_5(self):
        """
        Unit test for apply_preprocessing with ColumnTransformer and sklearn model.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"]),
                ("onehot_skp", skp.OneHotEncoder(), ["num1", "num2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = GradientBoostingClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert all(expected.index == result.index)

    def test_transform_ct_6(self):
        """
        Unit test for apply_preprocessing with ColumnTransformer and catboost model.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"]),
                ("onehot_skp", skp.OneHotEncoder(), ["num1", "num2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = cb.CatBoostClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.feature_names_ for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ct_7(self):
        """
        Unit test for apply_preprocessing with ColumnTransformer and lightgbm model.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"]),
                ("onehot_skp", skp.OneHotEncoder(), ["num1", "num2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = lightgbm.sklearn.LGBMClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.booster_.feature_name() for column in result.columns]
        assert all(expected.index == result.index)

    def test_transform_ct_8(self):
        """
        Unit test for apply_preprocessing with ColumnTransformer and xgboost model.
        """
        y = pd.DataFrame(data=[0, 1], columns=["y"])

        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": [1, 0]})

        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["num1", "num2"]),
                ("onehot_skp", skp.OneHotEncoder(), ["num1", "num2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

        train_preprocessed = pd.DataFrame(enc.transform(train))
        clf = xgboost.sklearn.XGBClassifier(n_estimators=1).fit(train_preprocessed, y)
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 0], "other": [1, 0, 0]})

        expected = pd.DataFrame(enc.transform(test), index=test.index)
        result = apply_preprocessing(test, clf, enc)
        assert result.shape == expected.shape
        assert [column in clf.get_booster().feature_names for column in result.columns]
        assert all(expected.index == result.index)

    def test_get_feature_names_1(self):
        """
        Unit test get_feature_names 1
        """
        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc_1 = ColumnTransformer(
            transformers=[("Quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )

        enc_2 = ColumnTransformer(
            transformers=[("Quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])],
            remainder="passthrough",
        )

        enc_1.fit(train)
        enc_2.fit(train)

        feature_enc_1 = get_feature_names(enc_1)
        feature_enc_2 = get_feature_names(enc_2)

        assert len(feature_enc_1) == 2
        assert len(feature_enc_2) == 3

    def test_get_names_1(self):
        """
        Unit test get_names 1
        """
        train = pd.DataFrame({"num1": [0, 1], "num2": [0, 2], "other": ["A", "B"]})

        enc_1 = ColumnTransformer(
            transformers=[("quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])], remainder="drop"
        )

        enc_2 = ColumnTransformer(
            transformers=[("quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])],
            remainder="passthrough",
        )

        enc_3 = ColumnTransformer(transformers=[("onehot", skp.OneHotEncoder(), ["other"])], remainder="drop")

        enc_4 = ColumnTransformer(transformers=[("onehot", skp.OneHotEncoder(), ["other"])], remainder="passthrough")

        enc_1.fit(train)
        enc_2.fit(train)
        enc_3.fit(train)
        enc_4.fit(train)

        feature_names_1 = []
        l_transformers = list(enc_1._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True))

        for name, trans, column, _ in l_transformers:
            feature_names_1.extend(get_names(name, trans, column, enc_1))

        feature_names_2 = []
        l_transformers = list(enc_2._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True))

        for name, trans, column, _ in l_transformers:
            feature_names_2.extend(get_names(name, trans, column, enc_2))

        feature_names_3 = []
        l_transformers = list(enc_3._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True))

        for name, trans, column, _ in l_transformers:
            feature_names_3.extend(get_names(name, trans, column, enc_3))

        feature_names_4 = []
        l_transformers = list(enc_4._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True))

        for name, trans, column, _ in l_transformers:
            feature_names_4.extend(get_names(name, trans, column, enc_4))

        assert len(feature_names_1) == 2
        assert len(feature_names_2) == 3
        assert len(feature_names_3) == 2
        assert len(feature_names_4) == 4

    def test_get_list_features_names_1(self):
        """
        Unit test get_list_features_names 1
        """
        train = pd.DataFrame(
            {"city": ["chicago", "paris"], "state": ["US", "FR"], "other": [5, 10]}, index=["index1", "index2"]
        )

        columns_dict = {i: features for i, features in enumerate(train.columns)}

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

        enc_2.fit(train)
        train_2 = pd.DataFrame(enc_2.transform(train), columns=["city_ce", "state_ce", "city_skp", "state_skp"])
        train_2["y"] = np.array([1, 0])

        enc_3 = ce.OneHotEncoder(cols=["city", "state"])
        enc_3.fit(train)
        train_3 = enc_3.transform(train)
        train_3["y"] = np.array([1, 0])

        dict_4 = {"col": "state", "mapping": pd.Series(data=[1, 2], index=["US", "FR"]), "data_type": "object"}

        dict_5 = {"col": "city", "mapping": pd.Series(data=[1, 2], index=["chicago", "paris"]), "data_type": "object"}

        enc_4 = [enc_3, [dict_4]]

        enc_5 = [enc_3, [dict_4, dict_5]]

        list_1 = get_list_features_names([enc], columns_dict)
        list_2 = get_list_features_names([enc_2], columns_dict)
        list_3 = get_list_features_names([enc_3], columns_dict)
        list_4 = get_list_features_names(enc_4, columns_dict)
        list_5 = get_list_features_names(enc_5, columns_dict)

        assert len(list_1) == 5
        assert len(list_2) == 4
        assert len(list_3) == 5
        assert len(list_4) == 5
        assert len(list_5) == 5

    def test_get_col_mapping_ct_1(self):
        """
        Test ColumnTransformer col mapping with drop option
        """
        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="drop",
        )

        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test))

        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {
            "onehot_ce_city": [0, 1],
            "onehot_ce_state": [2, 3],
            "onehot_skp_city": [4, 5],
            "onehot_skp_state": [6, 7],
        }

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_2(self):
        """
        Test ColumnTransformer col mapping with onehotencoder and passthrough option
        """
        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="passthrough",
        )

        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test))

        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {
            "onehot_ce_city": [0, 1],
            "onehot_ce_state": [2, 3],
            "onehot_skp_city": [4, 5],
            "onehot_skp_state": [6, 7],
            "other": [8],
        }

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_3(self):
        """
        Test ColumnTransformer col mapping with target encoder
        """
        enc = ColumnTransformer(
            transformers=[("target", ce.TargetEncoder(), ["city", "state"])], remainder="passthrough"
        )

        y = pd.DataFrame(data=[0, 1, 1], columns=["y"], index=["index1", "index2", "index3"])
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test, y))

        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"other": [2], "target_city": [0], "target_state": [1]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_4(self):
        """
        Test ColumnTransformer col mapping with Ordinal Category Encoder and drop option
        """
        enc = ColumnTransformer(transformers=[("ordinal", ce.OrdinalEncoder(), ["city", "state"])], remainder="drop")

        y = pd.DataFrame(data=[0, 1, 1], columns=["y"], index=["index1", "index2", "index3"])
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test, y))

        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"ordinal_city": [0], "ordinal_state": [1]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_5(self):
        """
        test get_col_mapping_ct with Binary encoder and drop option
        """
        enc = ColumnTransformer(transformers=[("binary", ce.BinaryEncoder(), ["city", "state"])], remainder="drop")

        y = pd.DataFrame(data=[0, 1, 1], columns=["y"], index=["index1", "index2", "index3"])
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test, y))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"binary_city": [0, 1], "binary_state": [2, 3]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_6(self):
        """
        test get_col_mapping_ct with BaseN Encoder and drop option
        """
        enc = ColumnTransformer(transformers=[("basen", ce.BaseNEncoder(), ["city", "state"])], remainder="drop")

        y = pd.DataFrame(data=[0, 1, 1], columns=["y"], index=["index1", "index2", "index3"])
        test = pd.DataFrame(
            {"city": ["chicago", "new york", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test, y))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"basen_city": [0, 1], "basen_state": [2, 3]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_7(self):
        """
        test get_col_mapping_ct with ordinal Encoder sklearn and drop option
        """
        enc = ColumnTransformer(transformers=[("ordinal", skp.OrdinalEncoder(), ["city", "state"])], remainder="drop")

        y = pd.DataFrame(data=[0, 1, 1], columns=["y"], index=["index1", "index2", "index3"])
        test = pd.DataFrame(
            {"city": ["chicago", "new york", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test, y))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"ordinal_city": [0], "ordinal_state": [1]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_8(self):
        """
        test get_col_mapping_ct with Standardscaler Encoder Sklearn and passthrough option
        """
        enc = ColumnTransformer(transformers=[("std", skp.StandardScaler(), ["num1", "num2"])], remainder="passthrough")
        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        test_encoded = pd.DataFrame(enc.fit_transform(test))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"std_num1": [0], "std_num2": [1], "other": [2]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_9(self):
        """
        test get_col_mapping_ct with QuantileTransformer Encoder Sklearn and passthrough option
        """
        enc = ColumnTransformer(
            transformers=[("quantile", skp.QuantileTransformer(n_quantiles=2), ["num1", "num2"])],
            remainder="passthrough",
        )

        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        test_encoded = pd.DataFrame(enc.fit_transform(test))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"quantile_num1": [0], "quantile_num2": [1], "other": [2]}

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_col_mapping_ct_10(self):
        """
        test get_col_mapping_ct with PowerTransformer Encoder Sklearn and passthrough option
        """
        enc = ColumnTransformer(
            transformers=[("power", skp.PowerTransformer(), ["num1", "num2"])], remainder="passthrough"
        )

        test = pd.DataFrame({"num1": [0, 1, 1], "num2": [0, 2, 3], "other": ["A", "B", "C"]})

        test_encoded = pd.DataFrame(enc.fit_transform(test))
        mapping = get_col_mapping_ct(enc, test_encoded)
        expected_mapping = {"power_num1": [0], "power_num2": [1], "other": [2]}

        self.assertDictEqual(mapping, expected_mapping)
