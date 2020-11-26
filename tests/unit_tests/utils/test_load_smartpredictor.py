"""
Unit test smart predictor
"""
import unittest
from os import path
from pathlib import Path
from shapash.utils.load_smartpredictor import load_smartpredictor
from shapash.explainer.smart_explainer import SmartExplainer
import pandas as pd
import numpy as np
import catboost as cb
import shap

class Test_load_smartpredictor(unittest.TestCase):
    def test_load_smartpredictor_1(self):
        """
        Unit test load_smartpredictor 1
        """
        xpl = SmartExplainer(features_dict={})
        y_pred = pd.DataFrame(data=np.array([1, 2]), columns=['pred'])
        dataframe_x = pd.DataFrame([[1, 2, 4], [1, 2, 3]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(dataframe_x, y_pred)
        xpl.compile(x=dataframe_x, y_pred=y_pred, model=clf)
        predictor = xpl.to_smartpredictor()

        current = Path(path.abspath(__file__)).parent.parent.parent
        pkl_file = path.join(current, 'data/predictor_to_load.pkl')
        predictor2 = load_smartpredictor(pkl_file)

        attrib_predictor = [element for element in predictor.__dict__.keys()]
        attrib_predictor2 = [element for element in predictor2.__dict__.keys()]

        assert all(attrib in attrib_predictor2 for attrib in attrib_predictor)
        assert all(attrib2 in attrib_predictor for attrib2 in attrib_predictor2)