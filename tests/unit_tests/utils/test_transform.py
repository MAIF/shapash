"""
Unit test of transform module.
"""
import unittest

import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer

from shapash.utils.transform import (
    get_features_transform_mapping,
    get_preprocessing_mapping,
    handle_categorical_missing,
)


class TestInverseTransformCaterogyEncoder(unittest.TestCase):
    def test_get_preprocessing_mapping_1(self):
        """
        test get_preprocessing_mapping with multiple encoding and dictionary
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

        test_encoded.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3_0", "col3_1", "col4_0", "col4_1"]

        mapping = get_preprocessing_mapping(test_encoded, [enc, input_dict1, list_dict])

        expected_mapping = {
            "onehot_ce_city": ["col1_0", "col1_1"],
            "onehot_ce_state": ["col2_0", "col2_1"],
            "onehot_skp_city": ["col3_0", "col3_1"],
            "onehot_skp_state": ["col4_0", "col4_1"],
        }
        self.assertDictEqual(mapping, expected_mapping)

    def test_get_preprocessing_mapping_2(self):
        """
        test get_preprocessing_mapping with multiple encoding and drop option
        """
        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("ordinal_skp", skp.OrdinalEncoder(), ["city"]),
            ],
            remainder="drop",
        )
        test = pd.DataFrame(
            {"city": ["chicago", "chicago", "paris"], "state": ["US", "FR", "FR"], "other": ["A", "B", "C"]},
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test))
        test_encoded.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3"]

        mapping = get_preprocessing_mapping(test_encoded, enc)

        expected_mapping = {
            "onehot_ce_city": ["col1_0", "col1_1"],
            "onehot_ce_state": ["col2_0", "col2_1"],
            "ordinal_skp_city": ["col3"],
        }
        self.assertDictEqual(mapping, expected_mapping)

    def test_get_features_transform_mapping_1(self):
        """
        test get_features_transform_mapping with multiple encoding and dictionary
        """
        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city", "state"]),
                ("onehot_skp", skp.OneHotEncoder(), ["city", "state"]),
            ],
            remainder="drop",
        )

        test = pd.DataFrame(
            {
                "city": ["chicago", "chicago", "paris"],
                "state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
                "other2": ["D", "E", "F"],
            },
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test))

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

        test_encoded.columns = ["col1_0", "col1_1", "col2_0", "col2_1", "col3_0", "col3_1", "col4_0", "col4_1"]
        # Construct expected x_encoded that will be built from test dataframe
        x_encoded = pd.DataFrame(
            {
                "onehot_ce_city": ["chicago", "chicago", "paris"],
                "onehot_ce_state": ["US", "FR", "FR"],
                "onehot_skp_city": ["chicago", "chicago", "paris"],
                "onehot_skp_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
                "other2": ["D", "E", "F"],
            },
            index=["index1", "index2", "index3"],
        )

        mapping = get_features_transform_mapping(x_encoded, test_encoded, [enc, input_dict1, list_dict])

        expected_mapping = {
            "onehot_ce_city": ["col1_0", "col1_1"],
            "onehot_ce_state": ["col2_0", "col2_1"],
            "onehot_skp_city": ["col3_0", "col3_1"],
            "onehot_skp_state": ["col4_0", "col4_1"],
            "other": ["other"],
            "other2": ["other2"],
        }
        self.assertDictEqual(mapping, expected_mapping)

    def test_get_features_transform_mapping_2(self):
        """
        test get_features_transform_mapping with multiple different encoding
        """
        enc = ColumnTransformer(
            transformers=[
                ("onehot_ce", ce.OneHotEncoder(), ["city"]),
                ("ordinal_skp", skp.OrdinalEncoder(), ["state"]),
            ],
            remainder="drop",
        )

        test = pd.DataFrame(
            {
                "city": ["chicago", "chicago", "paris"],
                "state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
                "other2": ["D", "E", "F"],
            },
            index=["index1", "index2", "index3"],
        )

        test_encoded = pd.DataFrame(enc.fit_transform(test))

        test_encoded.columns = ["col1_0", "col1_1", "col2"]
        # Construct expected x_encoded that will be built from test dataframe
        x_encoded = pd.DataFrame(
            {
                "onehot_ce_city": ["chicago", "chicago", "paris"],
                "ordinal_skp_state": ["US", "FR", "FR"],
                "other": ["A", "B", "C"],
                "other2": ["D", "E", "F"],
            },
            index=["index1", "index2", "index3"],
        )

        mapping = get_features_transform_mapping(x_encoded, test_encoded, enc)
        print(mapping)

        expected_mapping = {
            "onehot_ce_city": ["col1_0", "col1_1"],
            "ordinal_skp_state": ["col2"],
            "other": ["other"],
            "other2": ["other2"],
        }
        self.assertDictEqual(mapping, expected_mapping)

    def test_get_features_transform_mapping_3(self):
        """
        test get_features_transform_mapping with category encoders
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["A", "B", "B"]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.OneHotEncoder(cols=["city", "state"], use_cat_names=True)
        test_encoded = pd.DataFrame(enc.fit_transform(test, y))

        x_encoded = pd.DataFrame(
            {"city": ["chicago", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["A", "B", "B"]}
        )

        mapping = get_features_transform_mapping(x_encoded, test_encoded, enc)
        expected_mapping = {
            "city": ["city_chicago", "city_paris"],
            "state": ["state_US", "state_FR"],
            "other": ["other"],
        }

        self.assertDictEqual(mapping, expected_mapping)

    def test_get_features_transform_mapping_4(self):
        """
        test get_features_transform_mapping with list of category encoders
        """
        test = pd.DataFrame(
            {"city": ["chicago", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["A", "B", "B"]}
        )
        y = pd.DataFrame(data=[0, 1, 1], columns=["y"])

        enc = ce.OneHotEncoder(cols=["state"], use_cat_names=True)
        test_encoded = pd.DataFrame(enc.fit_transform(test, y))

        enc2 = ce.OrdinalEncoder(cols=["city"])
        test_encoded = enc2.fit_transform(test_encoded, y)

        x_encoded = pd.DataFrame(
            {"city": ["chicago", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["A", "B", "B"]}
        )

        mapping = get_features_transform_mapping(x_encoded, test_encoded, [enc, enc2])
        expected_mapping = {"city": ["city"], "state": ["state_US", "state_FR"], "other": ["other"]}

        self.assertDictEqual(mapping, expected_mapping)

    def handle_categorical_missing(self):
        """
        test handle_categorical_missing
        """
        df_test = pd.DataFrame(
            {"city": [np.nan, "paris", "chicago"], "state": ["US", "FR", "FR"], "other": [np.nan, "B", "B"]}
        )

        df_test = handle_categorical_missing(df_test)

        df_expected = pd.DataFrame(
            {"city": ["missing", "paris", "chicago"], "state": ["US", "FR", "FR"], "other": ["missing", "B", "B"]}
        )

        assert_frame_equal(df_test, df_expected)
