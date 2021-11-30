"""
Unit tests acv backend.
"""

import unittest
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import xgboost as xgb
from shapash.utils.acv_backend import active_shapley_values, compute_features_import_acv

class TestAcvBackend(unittest.TestCase):
    def setUp(self):
        self.modellist = [
            xgb.XGBClassifier(n_estimators=1),            
            ske.RandomForestClassifier(n_estimators=1)
        ]

        df = pd.DataFrame(range(0, 5), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 3 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df = df[['x1', 'x2']]
        self.y_df = df['y'].to_frame()

    def test_shap_contributions_0(self):
        """
        test acv backend 0
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df.values, self.y_df.values)
            active_shapley_values(model, self.x_df, self.x_df, self.x_df)

    def test_shap_contributions_1(self):
        """
        test acv backend 1 with coalitions
        """
        for model in self.modellist:
            print(type(model))
            model.fit(self.x_df.values, self.y_df.values)
            active_shapley_values(model, self.x_df, self.x_df, self.x_df, c=[[0, 1]])

    def test_compute_features_import_acv(self):
        """
        test compute features importance
        """
        sdp_index = np.array([[3,  9,  0,  1,  2, -1, -1, -1, -1, -1],
                              [7,  9, -1, -1, -1, -1, -1, -1, -1, -1],
                              [5,  9,  0,  1,  2, -1, -1, -1, -1, -1]])
        sdp = np.array([0.94, 0.95, 0.92])
        features_mapping1 = {'Pclass': ['Pclass_1', 'Pclass_2', 'Pclass_3'],
                             'Embarked': ['Embarked'],
                             'Title': ['Title'],
                             'Sex': ['Sex'],
                             'Age': ['Age'],
                             'SibSp': ['SibSp'],
                             'Parch': ['Parch'],
                             'Fare': ['Fare']}
        x_init_columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                          'Embarked', 'Title']

        features_imp = compute_features_import_acv(sdp_index, sdp, x_init_columns, features_mapping1)
        np.testing.assert_array_equal(list(features_imp.values),
                                      [0, 0, 0, 1/3, 1/3, 1/3, 2/3, 1])

