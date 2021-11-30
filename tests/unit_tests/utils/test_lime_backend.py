import unittest
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import sklearn.linear_model as skl
import xgboost as xgb
from shapash.utils.lime_backend import lime_contributions
from random import randrange


class TestLimeBackend(unittest.TestCase):
    def setUp(self):
        self.modellist_classif = [
            xgb.XGBClassifier(n_estimators=1),
            ske.GradientBoostingClassifier(n_estimators=1),
            ske.ExtraTreesClassifier(n_estimators=1),
            ske.RandomForestClassifier(n_estimators=1)
        ]

        self.modellist_regression = [
            skl.LinearRegression(),
            ske.GradientBoostingRegressor(n_estimators=1),
            ske.ExtraTreesRegressor(n_estimators=1),
            ske.RandomForestRegressor(n_estimators=1)
        ]

    def test_lime_contributions_1(self):
        """
        test lime backend with binary classification
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 1 if x < 10 else 0)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df = df[['x1', 'x2']]
        self.y_df = df['y'].to_frame()
        for model in self.modellist_classif:
            print(type(model))
            model.fit(self.x_df.values, self.y_df.values)
            lime_contributions(model, self.x_df, self.x_df, classes=[0, 1])

    def test_lime_contributions_2(self):
        """
        test lime backend with multiple classification
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = df['id'].apply(lambda x: 0 if x <= 8 else 1 if x > 8 and x <= 14 else 2)
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df1 = df[['x1', 'x2']]
        self.y_df1 = df['y'].to_frame()

        for model in self.modellist_classif:
            print(type(model))
            model.fit(self.x_df1.values, self.y_df1.values)
            lime_contributions(model, self.x_df1, self.x_df1, classes=[0, 1, 2])

    def test_lime_contributions_3(self):
        """
        test lime backend with regression
        """
        df = pd.DataFrame(range(0, 21), columns=['id'])
        df['y'] = np.random.randint(100, 1000, df.shape[0])
        df['x1'] = np.random.randint(1, 123, df.shape[0])
        df['x2'] = np.random.randint(1, 3, df.shape[0])
        df = df.set_index('id')
        self.x_df2 = df[['x1', 'x2']]
        self.y_df2 = df['y'].to_frame()
        for model in self.modellist_regression:
            print(type(model))
            model.fit(self.x_df2.values, self.y_df2.values)
            lime_contributions(model, self.x_df2, self.x_df2, mode="regression")
