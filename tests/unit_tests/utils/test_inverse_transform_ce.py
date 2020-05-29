"""
Unit test of Inverse Transform
"""
import unittest
import pandas as pd
import numpy as np
import category_encoders as ce
from shapash.utils.transform import inverse_transform


class TestInverseTransformCaterogyEncoder(unittest.TestCase):
    def test_inverse_transform_none(self):
        """
        Test no preprocessing
        """
        train = pd.DataFrame({'city': ['chicago', 'paris'],
                              'state': ['US', 'FR']})
        original = inverse_transform(train)
        pd.testing.assert_frame_equal(original, train)

    def test_multiple_encoding(self):
        """
        Test multiple preprocessing
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        test = pd.DataFrame({'Onehot1': ['A', 'B', 'A'], 'Onehot2': ['C', 'D', 'ZZ'],
                             'Binary1': ['E', 'F', 'F'], 'Binary2': ['G', 'H', 'ZZ'],
                             'Ordinal1': ['I', 'J', 'J'], 'Ordinal2': ['K', 'L', 'ZZ'],
                             'BaseN1': ['M', 'N', 'N'], 'BaseN2': ['O', 'P', 'ZZ'],
                             'Target1': ['Q', 'R', 'R'], 'Target2': ['S', 'T', 'ZZ'],
                             'other': ['other', '123', np.nan]})

        expected = pd.DataFrame({'Onehot1': ['A', 'B', 'A'], 'Onehot2': ['C', 'D', 'missing'],
                                 'Binary1': ['E', 'F', 'F'], 'Binary2': ['G', 'H', 'missing'],
                                 'Ordinal1': ['I', 'J', 'J'], 'Ordinal2': ['K', 'L', 'missing'],
                                 'BaseN1': ['M', 'N', 'N'], 'BaseN2': ['O', 'P', np.nan],
                                 'Target1': ['Q', 'R', 'R'], 'Target2': ['S', 'T', 'NaN'],
                                 'other': ['other', '123', np.nan]})

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

        input_dict3 = dict()
        input_dict3['col'] = 'Ordinal2'
        input_dict3['mapping'] = pd.Series(data=['K', 'L', np.nan], index=['K', 'L', 'missing'])
        input_dict3['data_type'] = 'object'
        list_dict = [input_dict2, input_dict3]

        result1 = enc_onehot.transform(test)
        result2 = enc_binary.transform(result1)
        result3 = enc_ordinal.transform(result2)
        result4 = enc_basen.transform(result3)
        result5 = enc_target.transform(result4)

        original = inverse_transform(result5, [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1,
                                               list_dict])

        pd.testing.assert_frame_equal(expected, original)


    def test_target(self):
        """
        Test target encoding
        """
        train = pd.DataFrame({'city': ['chicago', 'paris', 'paris', 'chicago', 'chicago'],
                              'state': ['US', 'FR', 'FR', 'US', 'US'],
                              'other': ['A', 'A', np.nan, 'B', 'B']})
        test = pd.DataFrame({'city': ['chicago', 'paris', 'paris'],
                             'state': ['US', 'FR', 'FR'],
                             'other': ['A', np.nan, np.nan]})
        expected = pd.DataFrame({'city': ['chicago', 'paris', 'paris'],
                                 'state': ['US', 'FR', 'FR'],
                                 'other': ['A', np.nan, np.nan]})
        y = pd.DataFrame(data=[0, 1, 1, 0, 1], columns=['y'])

        enc = ce.TargetEncoder(cols=['city', 'state']).fit(train, y)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_HaveUnknown_ExpectWarning(self):
        """
        Test ordinal encoding
        """
        train = pd.DataFrame({'city': ['chicago', 'st louis']})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan_Ordinal(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OrdinalEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan_Ordinal(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OrdinalEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_BothFieldsAreReturnNanWithNan_ExpectValueError_Ordinal(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        enc = ce.OrdinalEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed_Ordinal(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        enc = ce.OrdinalEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse_Ordinal(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
        enc = ce.OrdinalEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = enc.inverse_transform(result)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_multiple_ordinal(self):
        data = pd.DataFrame({'city': ['chicago', 'paris'],
                             'state': ['US', 'FR'],
                             'other': ['a', 'b']})
        test = pd.DataFrame({'city': [1, 2, 2],
                             'state': [1, 2, 2],
                             'other': ['a', 'b', 'a']})
        expected = pd.DataFrame({'city': ['chicago', 'paris', 'paris'],
                                 'state': ['US', 'FR', 'FR'],
                                 'other': ['a', 'b', 'a']})
        enc = ce.OrdinalEncoder(cols=['city', 'state'])
        enc.fit(data)
        original = inverse_transform(test, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_multiple_binary(self):
        """
        Test binary encoding
        """
        train = pd.DataFrame({'city': ['chicago', 'paris'],
                              'state': ['US', 'FR'],
                              'other': ['A', np.nan]})

        test = pd.DataFrame({'city': ['chicago', 'paris', 'monaco'],
                             'state': ['US', 'FR', 'FR'],
                             'other': ['A', np.nan, 'B']})

        expected = pd.DataFrame({'city': ['chicago', 'paris', np.nan],
                                 'state': ['US', 'FR', 'FR'],
                                 'other': ['A', np.nan, 'B']})

        enc = ce.BinaryEncoder(cols=['city', 'state']).fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inverse_transform_HaveData_ExpectResultReturned(self):
        train = pd.Series(list('abcd')).to_frame('letter')
        enc = ce.BaseNEncoder(base=2)
        result = enc.fit_transform(train)
        inversed_result = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, inversed_result)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan_baseN(self):
        """
        Test basen encoding
        """
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.BaseNEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})

        enc = ce.BaseNEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})

        enc = ce.BaseNEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)

        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
        enc = ce.BaseNEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_inverse_transform_multiple_baseN(self):
        train = pd.DataFrame({'city': ['chicago', 'paris'],
                              'state': ['US', 'FR']})
        test = pd.DataFrame({'city_0': [0, 1],
                             'city_1': [1, 0],
                             'state_0': [0, 1],
                             'state_1': [1, 0]})
        enc = ce.BaseNEncoder(cols=['city', 'state'], handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        original = inverse_transform(test, enc)
        pd.testing.assert_frame_equal(original, train)

    def test_inverse_transform_HaveDedupedColumns_ExpectCorrectInverseTransform(self):
        """
        Test Onehot encoding
        """
        encoder = ce.OneHotEncoder(cols=['match', 'match_box'], use_cat_names=True)
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series(-1)})
        transformed = encoder.fit_transform(value)
        inversed_result = inverse_transform(transformed, encoder)
        pd.testing.assert_frame_equal(value, inversed_result)

    def test_inverse_transform_HaveNoCatNames_ExpectCorrectInverseTransform(self):
        encoder = ce.OneHotEncoder(cols=['match', 'match_box'], use_cat_names=False)
        value = pd.DataFrame({'match': pd.Series('box_-1'), 'match_box': pd.Series(-1)})
        transformed = encoder.fit_transform(value)
        inversed_result = inverse_transform(transformed, encoder)
        pd.testing.assert_frame_equal(value, inversed_result)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingValue_ExpectReturnedWithNan_OneHot(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OneHotEncoder(handle_missing='value', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveNanInTrainAndHandleMissingReturnNan_ExpectReturnedWithNan_OneHot(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OneHotEncoder(handle_missing='return_nan', handle_unknown='value')
        result = enc.fit_transform(train)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_BothFieldsAreReturnNanWithNan_ExpectValueError_Onehot(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan]})
        enc = ce.OneHotEncoder(handle_missing='return_nan', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(original, expected)

    def test_inverse_transform_HaveMissingAndNoUnknown_ExpectInversed_Onehot(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', 'los angeles']})
        enc = ce.OneHotEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(train, original)

    def test_inverse_transform_HaveHandleMissingValueAndHandleUnknownReturnNan_ExpectBestInverse_Onehot(self):
        train = pd.DataFrame({'city': ['chicago', np.nan]})
        test = pd.DataFrame({'city': ['chicago', np.nan, 'los angeles']})
        expected = pd.DataFrame({'city': ['chicago', np.nan, np.nan]})
        enc = ce.OneHotEncoder(handle_missing='value', handle_unknown='return_nan')
        enc.fit(train)
        result = enc.transform(test)
        original = inverse_transform(result, enc)
        pd.testing.assert_frame_equal(expected, original)

    def test_dict(self):
        """
        Test dict encoding
        """
        data = pd.DataFrame({'city': ['chicago', 'paris-1', 'paris-2'],
                             'state': ['US', 'FR-1', 'FR-2'],
                             'other': ['A', 'B', np.nan]})

        expected = pd.DataFrame({'city': ['chicago', 'paris-1', 'paris-2'],
                                 'state': ['US', 'FR', 'FR'],
                                 'other': ['A', 'B', np.nan]})
        input_dict = dict()
        input_dict['col'] = 'state'
        input_dict['mapping'] = pd.Series(data=['US', 'FR-1', 'FR-2'], index=['US', 'FR', 'FR'])
        input_dict['data_type'] = 'object'
        result = inverse_transform(data, input_dict)
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_dict_encoding_withindex(self):
        """
        Test multiple dict encoding
        """
        train = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D'],
                              'Binary1': ['E', 'F', 'E', 'F'], 'Binary2': ['G', 'H', 'G', 'H'],
                              'Ordinal1': ['I', 'J', 'I', 'J'], 'Ordinal2': ['K', 'L', 'K', 'L'],
                              'BaseN1': ['M', 'N', 'M', 'N'], 'BaseN2': ['O', 'P', 'O', 'P'],
                              'Target1': ['Q', 'R', 'Q', 'R'], 'Target2': ['S', 'T', 'S', 'T'],
                              'other': ['other', np.nan, 'other', 'other']})

        test = pd.DataFrame({'Onehot1': ['A', 'B', 'A'], 'Onehot2': ['C', 'D', 'ZZ'],
                             'Binary1': ['E', 'F', 'F'], 'Binary2': ['G', 'H', 'ZZ'],
                             'Ordinal1': ['I', 'J', 'J'], 'Ordinal2': ['K', 'L', 'ZZ'],
                             'BaseN1': ['M', 'N', 'N'], 'BaseN2': ['O', 'P', 'ZZ'],
                             'Target1': ['Q', 'R', 'R'], 'Target2': ['S', 'T', 'ZZ'],
                             'other': ['other', '123', np.nan]},
                             index=['index1', 'index2', 'index3'])

        expected = pd.DataFrame({'Onehot1': ['A', 'B', 'A'], 'Onehot2': ['C', 'D', 'missing'],
                                 'Binary1': ['E', 'F', 'F'], 'Binary2': ['G', 'H', 'missing'],
                                 'Ordinal1': ['I', 'J', 'J'], 'Ordinal2': ['K', 'L', 'missing'],
                                 'BaseN1': ['M', 'N', 'N'], 'BaseN2': ['O', 'P', np.nan],
                                 'Target1': ['Q', 'R', 'R'], 'Target2': ['S', 'T', 'NaN'],
                                 'other': ['other', '123', np.nan]},
                             index=['index1', 'index2', 'index3'])

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

        input_dict3 = dict()
        input_dict3['col'] = 'Ordinal2'
        input_dict3['mapping'] = pd.Series(data=['K', 'L', np.nan], index=['K', 'L', 'missing'])
        input_dict3['data_type'] = 'object'
        list_dict = [input_dict2, input_dict3]

        result1 = enc_onehot.transform(test)
        result2 = enc_binary.transform(result1)
        result3 = enc_ordinal.transform(result2)
        result4 = enc_basen.transform(result3)
        result5 = enc_target.transform(result4)

        original = inverse_transform(result5, [enc_onehot, enc_binary, enc_ordinal, enc_basen, enc_target, input_dict1,
                                               list_dict])

        pd.testing.assert_frame_equal(expected, original)