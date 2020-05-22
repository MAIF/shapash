"""
Unit test of Inverse Transform
"""
import unittest
import pandas as pd
from category_encoders import OrdinalEncoder
from category_encoders import OneHotEncoder
from category_encoders import BaseNEncoder
from category_encoders import BinaryEncoder
from shapash.utils.transform import inverse_transform

#DataFrame used for fiting the encoding
X_ENC = pd.DataFrame({'Name': ['Tom', 'nick', 'krish', 'jack'],
                      'Name2': ['B', 'A', 'B', 'B'],
                      'Age': [20, 21, 19, 18]})

X_ENC2 = pd.DataFrame({'Name': ['Tom', 'nick', 'nick', 'jack'],
                       'Name2': ['B', 'A', 'B', 'B'],
                       'Age': [20, 21, 19, 18]})

EXPECTED = pd.DataFrame({
    'Name': ['Tom', 'nick', 'nick', 'jack', 'krish'],
    'Name2': ['A', 'A', 'B', 'B', 'A'],
    'Age': [4, 21, 19, 18, 17]})

EXPECTED2 = pd.DataFrame({
    'Name': ['Tom', 'nick', 'nick', 'unknow', 'unknow'],
    'Name2': ['A', 'A', 'B', 'B', 'A'],
    'Age': [4, 21, 19, 18, 17]})

class TestInverseTransform(unittest.TestCase):
    """
    Test inverse Transform class
    """
    def test_inverse_transform_none(self):
        """
        test inverse transform none
        """
        output_none = inverse_transform(EXPECTED)
        pd.testing.assert_frame_equal(output_none, EXPECTED)

    def test_inverse_transform_ce_ordinal(self):
        """
        test inverse transform ce ordinal
        """
        enc = OrdinalEncoder(cols=['Name', 'Name2']).fit(X_ENC)
        x_ordinal = pd.DataFrame({'Name': [1, 2, 2, 4, 3],
                                  'Name2': [2, 2, 1, 1, 2],
                                  'Age': [4, 21, 19, 18, 17]})
        output_ordinal = inverse_transform(x_ordinal, enc)
        pd.testing.assert_frame_equal(output_ordinal, EXPECTED)

    def test_inverse_transform_ce_onehot(self):
        """
        test inverse transform ce onehot encoder
        """
        enc = OneHotEncoder(cols=['Name', 'Name2']).fit(X_ENC)
        x_onehot = pd.DataFrame({'Name_1': [1, 0, 0, 0, 0],
                                 'Name_2': [0, 1, 1, 0, 0],
                                 'Name_3': [0, 0, 0, 0, 1],
                                 'Name_4': [0, 0, 0, 1, 0],
                                 'Name2_1': [0, 0, 1, 1, 0],
                                 'Name2_2': [1, 1, 0, 0, 1],
                                 'Age': [4, 21, 19, 18, 17]})
        output_onehot = inverse_transform(x_onehot, enc)
        pd.testing.assert_frame_equal(output_onehot, EXPECTED)

    def test_inverse_transform_ce_basen(self):
        """
        unit test of inverse transform ce_basen
        """
        enc = BaseNEncoder(cols=['Name', 'Name2'], base=3).fit(X_ENC)
        x_basen = pd.DataFrame({
            'Name_0': [0, 0, 0, 0, 0],
            'Name_1': [0, 0, 0, 1, 1],
            'Name_2': [1, 2, 2, 1, 0],
            'Name2_0': [0, 0, 0, 0, 0],
            'Name2_1': [2, 2, 1, 1, 2],
            'Age': [4, 21, 19, 18, 17]})
        output_basen = inverse_transform(x_basen, enc)
        pd.testing.assert_frame_equal(output_basen, EXPECTED)

    def test_inverse_transform_ce_binary(self):
        """
        unit test of inverse transform ce_binary
        """
        enc = BinaryEncoder(cols=['Name', 'Name2']).fit(X_ENC)
        x_binary = pd.DataFrame({
            'Name_0': [0, 0, 0, 1, 0],
            'Name_1': [0, 1, 1, 0, 1],
            'Name_2': [1, 0, 0, 0, 1],
            'Name2_0': [1, 1, 0, 0, 1],
            'Name2_1': [0, 0, 1, 1, 0],
            'Age': [4, 21, 19, 18, 17]})
        output_binary = inverse_transform(x_binary, enc)
        pd.testing.assert_frame_equal(output_binary, EXPECTED)

    #test when encoding is not complete
    def test_inverse_transform_ce_ordinal2(self):
        """
        unit test inverse transform ce ordinal 2
        """
        enc2 = OrdinalEncoder(cols=['Name', 'Name2']).fit(X_ENC2)
        x_ordinal2 = pd.DataFrame({'Name': [1, 2, 2, -1, -1],
                                   'Name2': [2, 2, 1, 1, 2],
                                   'Age': [4, 21, 19, 18, 17]})
        output_ordinal2 = inverse_transform(x_ordinal2, enc2)
        pd.testing.assert_frame_equal(output_ordinal2, EXPECTED2)

    def test_inverse_transform_ce_onehot2(self):
        """
        unit test inverse transform ce onehot2
        """
        enc2 = OneHotEncoder(cols=['Name', 'Name2']).fit(X_ENC2)
        x_onehot2 = pd.DataFrame({'Name_1': [1, 0, 0, 0, 0],
                                  'Name_2': [0, 1, 1, 0, 0],
                                  'Name_3': [0, 0, 0, 0, 0],
                                  'Name2_1': [0, 0, 1, 1, 0],
                                  'Name2_2': [1, 1, 0, 0, 1],
                                  'Age': [4, 21, 19, 18, 17]})
        output_onehot2 = inverse_transform(x_onehot2, enc2).fillna('unknow')
        pd.testing.assert_frame_equal(output_onehot2, EXPECTED2)

    def test_inverse_transform_ce_basen2(self):
        """
        unit test inverse transform ce base n 2
        """
        enc2 = BaseNEncoder(cols=['Name', 'Name2'], base=2).fit(X_ENC2)
        x_basen2 = pd.DataFrame({'Name_0': [0, 0, 0, 0, 0],
                                 'Name_1': [0, 1, 1, 0, 0],
                                 'Name_2': [1, 0, 0, 0, 0],
                                 'Name2_0': [1, 1, 0, 0, 1],
                                 'Name2_1': [0, 0, 1, 1, 0],
                                 'Age': [4, 21, 19, 18, 17]})
        output_basen2 = inverse_transform(x_basen2, enc2)
        pd.testing.assert_frame_equal(output_basen2, EXPECTED2)

    def test_inverse_transform_ce_binary2(self):
        """
        unit test inverse transform ce binary 2
        """
        enc2 = BinaryEncoder(cols=['Name', 'Name2']).fit(X_ENC2)
        x_binary2 = pd.DataFrame({
            'Name_0': [0, 0, 0, 0, 0],
            'Name_1': [0, 1, 1, 0, 0],
            'Name_2': [1, 0, 0, 0, 0],
            'Name2_0': [1, 1, 0, 0, 1],
            'Name2_1': [0, 0, 1, 1, 0],
            'Age': [4, 21, 19, 18, 17]})
        output_binary2 = inverse_transform(x_binary2, enc2)
        pd.testing.assert_frame_equal(output_binary2, EXPECTED2)
