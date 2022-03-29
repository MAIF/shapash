import unittest
from unittest.mock import patch
import types

import numpy as np
import category_encoders as ce
import pandas as pd
from pandas.testing import assert_frame_equal

from shapash.backend.base_backend import BaseBackend, _needs_preprocessing


class TestBackend(BaseBackend):
    column_aggregation = 'sum'
    name = 'test'

    def _run_explainer(self, x):
        return [[0, 1]]

    def _get_local_contributions(self, x, explain_data, subset=None):
        return [[0, 2]]

    def _get_global_features_importance(self, contributions, explain_data, subset=None):
        if subset is None:
            return [[0, 3]]
        else:
            return [[0, 4]]


class TestBaseBackend(unittest.TestCase):
    def predict(self, arg1, arg2):
        matrx = np.array(
            [12, 3, 7]
        )
        return matrx

    def setUp(self):
        self.model = lambda: None
        self.model.predict = types.MethodType(self.predict, self.model)
        self.preprocessing = ce.OneHotEncoder(cols=['Onehot1', 'Onehot2'])

        self.test_backend = TestBackend(model=self.model, preprocessing=self.preprocessing)

    def test_init(self):
        assert self.test_backend.model == self.model
        assert self.test_backend.preprocessing == self.preprocessing
        assert self.test_backend.explain_data is None
        assert self.test_backend._state is None
        assert self.test_backend._case
        assert self.test_backend._classes is None

    def test_run_explainer(self):
        assert self.test_backend.run_explainer(pd.DataFrame([0])) == [[0, 1]]

    @patch('shapash.backend.base_backend.BaseBackend.format_and_aggregate_local_contributions')
    def test_get_local_contributions(self, mock_format_contrib):
        mock_format_contrib.return_value = [0, 4]
        res = self.test_backend.get_local_contributions(pd.DataFrame([0]), 0)
        assert res == [0, 4]

    def test_get_global_features_importance(self):
        assert self.test_backend.get_global_features_importance(pd.DataFrame([0]), None) == [[0, 3]]

    def test_get_global_features_importance_2(self):
        assert self.test_backend.get_global_features_importance(pd.DataFrame([0]), None, [1]) == [[0, 4]]

    def test_format_and_aggregate_local_contributions(self):
        self.test_backend._case = 'classification'
        self.test_backend._classes = [0, 1]
        self.test_backend.preprocessing = None

        x = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['a', 'b', 'c'])

        contributions = [np.array([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]]), np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])]
        res_contributions = self.test_backend.format_and_aggregate_local_contributions(x, contributions)

        assert len(res_contributions) == 2
        assert_frame_equal(
            res_contributions[0],
            pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]], columns=['a', 'b', 'c'])
        )
        assert_frame_equal(
            res_contributions[1],
            pd.DataFrame([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], columns=['a', 'b', 'c'])
        )

    def test_format_and_aggregate_local_contributions_2(self):
        self.test_backend._case = 'classification'
        self.test_backend._classes = [0, 1]
        self.test_backend.preprocessing = None

        x = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['a', 'b', 'c'])

        contributions = [
            pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]], columns=['a', 'b', 'c']),
            pd.DataFrame([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], columns=['a', 'b', 'c'])
        ]
        res_contributions = self.test_backend.format_and_aggregate_local_contributions(x, contributions)

        assert len(res_contributions) == 2
        assert_frame_equal(
            res_contributions[0],
            pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]], columns=['a', 'b', 'c'])
        )
        assert_frame_equal(
            res_contributions[1],
            pd.DataFrame([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], columns=['a', 'b', 'c'])
        )

    def test_needs_preprocessing(self):
        df = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D']})
        encoder = ce.OneHotEncoder(cols=['Onehot1', 'Onehot2']).fit(df)
        df_ohe = encoder.transform(df)

        res = _needs_preprocessing(df_ohe.columns, df_ohe, preprocessing=encoder)

        assert res is True

    def test_needs_preprocessing_2(self):
        df = pd.DataFrame({'Onehot1': ['A', 'B', 'A', 'B'], 'Onehot2': ['C', 'D', 'C', 'D']})
        encoder = ce.OrdinalEncoder(cols=['Onehot1', 'Onehot2']).fit(df)
        df_ord = encoder.transform(df)

        res = _needs_preprocessing(df_ord.columns, df_ord, preprocessing=encoder)

        assert res is False
