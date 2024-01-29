"""
Unit test of Inverse Transform
"""
import unittest

import category_encoders as ce
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp
from sklearn.compose import ColumnTransformer

from shapash.decomposition.contributions import inverse_transform_contributions


class TestInverseContribCaterogyEncoder(unittest.TestCase):
    def test_inverse_transform_none(self):
        """
        Test no preprocessing
        """
        contributions = pd.DataFrame(np.random.rand(10, 15))
        preprocessing = None
        original = inverse_transform_contributions(contributions, preprocessing)
        pd.testing.assert_frame_equal(original, contributions)

    def test_multiple_encoding_category_encoder(self):
        """
        Test multiple preprocessing
        """
        train = pd.DataFrame(
            {
                "Onehot1": ["A", "B"],
                "Onehot2": ["C", "D"],
                "Binary1": ["E", "F"],
                "Binary2": ["G", "H"],
                "Ordinal1": ["I", "J"],
                "Ordinal2": ["K", "L"],
                "BaseN1": ["M", "N"],
                "BaseN2": ["O", "P"],
                "Target1": ["Q", "R"],
                "Target2": ["S", "T"],
                "other": ["other", np.nan],
            }
        )

        contributions = pd.DataFrame(
            [
                [1, 0, 1, 1, 3, 0, -3.5, 0, 4, 5, 0, 6, 7, 0, 8, 9, 10],
                [0.5, 0.5, 2, 0, 1.5, 1.5, 5.5, -2, -4, -5, 8.5, -2.5, -7, 14, -8, -9, -10],
            ],
            index=["index1", "index2"],
        )

        expected_contrib = pd.DataFrame(
            {
                "Onehot1": [1.0, 1.0],
                "Onehot2": [2, 2],
                "Binary1": [3.0, 3.0],
                "Binary2": [-3.5, 3.5],
                "Ordinal1": [4, -4],
                "Ordinal2": [5, -5],
                "BaseN1": [6.0, 6.0],
                "BaseN2": [7, 7],
                "Target1": [8, -8],
                "Target2": [9, -9],
                "other": [10, -10],
            },
            index=["index1", "index2"],
        )

        y = pd.DataFrame(data=[0, 1], columns=["y"])

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

        result1 = enc_onehot.transform(train)
        result2 = enc_binary.transform(result1)
        result3 = enc_ordinal.transform(result2)
        result4 = enc_basen.transform(result3)
        result5 = enc_target.transform(result4)

        contributions.columns = result5.columns

        original = inverse_transform_contributions(
            contributions, [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1, list_dict]
        )

        pd.testing.assert_frame_equal(expected_contrib, original)

    def test_multiple_encoding_columntransfomers(self):
        """
        Test multiple preprocessing columntransformers
        """
        train = pd.DataFrame(
            {
                "Onehot1": ["A", "B"],
                "Onehot2": ["C", "D"],
                "Binary1": ["E", "F"],
                "Binary2": ["G", "H"],
                "Ordinal1": ["I", "J"],
                "Ordinal2": ["K", "L"],
                "BaseN1": ["M", "N"],
                "BaseN2": ["O", "P"],
                "Target1": ["Q", "R"],
                "Target2": ["S", "T"],
                "other": ["other", np.nan],
            }
        )

        contributions = pd.DataFrame(
            [
                [1, 0, 1, 1, 1, 0, 1, 1, 3, 0, -3.5, 0, 4, 4, 5, 5, 0, 6, 7, 0, 8, 9, 10],
                [0.5, 0.5, 2, 0, 0.5, 0.5, 2, 0, 1.5, 1.5, 5.5, -2, -4, -4, -5, -5, 8.5, -2.5, -7, 14, -8, -9, -10],
            ],
            index=["index1", "index2"],
        )

        expected_contrib = pd.DataFrame(
            {
                "onehot_skp_Onehot1": [1.0, 1.0],
                "onehot_skp_Onehot2": [2, 2],
                "onehot_ce_Onehot1": [1.0, 1.0],
                "onehot_ce_Onehot2": [2, 2],
                "binary_ce_Binary1": [3.0, 3.0],
                "binary_ce_Binary2": [-3.5, 3.5],
                "ordinal_ce_Ordinal1": [4, -4],
                "ordinal_ce_Ordinal2": [4, -4],
                "ordinal_skp_Ordinal1": [5, -5],
                "ordinal_skp_Ordinal2": [5, -5],
                "basen_ce_BaseN1": [6.0, 6.0],
                "basen_ce_BaseN2": [7, 7],
                "target_ce_Target1": [8, -8],
                "target_ce_Target2": [9, -9],
                22: [10, -10],
            },
            index=["index1", "index2"],
        )

        y = pd.DataFrame(data=[0, 1], columns=["y"])

        enc = ColumnTransformer(
            transformers=[
                ("onehot_skp", skp.OneHotEncoder(), ["Onehot1", "Onehot2"]),
                ("onehot_ce", ce.OneHotEncoder(), ["Onehot1", "Onehot2"]),
                ("binary_ce", ce.BinaryEncoder(), ["Binary1", "Binary2"]),
                ("ordinal_ce", ce.OrdinalEncoder(), ["Ordinal1", "Ordinal2"]),
                ("ordinal_skp", skp.OrdinalEncoder(), ["Ordinal1", "Ordinal2"]),
                ("basen_ce", ce.BaseNEncoder(), ["BaseN1", "BaseN2"]),
                ("target_ce", ce.TargetEncoder(), ["Target1", "Target2"]),
            ],
            remainder="passthrough",
        )
        enc.fit(train, y)

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

        original = inverse_transform_contributions(contributions, [enc, input_dict1, list_dict])

        pd.testing.assert_frame_equal(expected_contrib, original)
