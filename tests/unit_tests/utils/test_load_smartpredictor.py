"""
Unit test smart predictor
"""

import sys
import unittest
from os import path
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

from shapash import SmartExplainer
from shapash.utils.load_smartpredictor import load_smartpredictor


class Test_load_smartpredictor(unittest.TestCase):
    def test_load_smartpredictor_1(self):
        """
        Unit test load_smartpredictor 1
        """
        y_pred = pd.DataFrame(data=np.array([1, 2]), columns=["pred"])
        dataframe_x = pd.DataFrame([[1, 2, 4], [1, 2, 3]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(dataframe_x, y_pred)
        xpl = SmartExplainer(model=clf, features_dict={})
        xpl.compile(x=dataframe_x, y_pred=y_pred)
        predictor = xpl.to_smartpredictor()

        current = Path(path.abspath(__file__)).parent.parent.parent
        if str(sys.version)[0:4] == "3.10":
            pkl_file = path.join(current, "data/predictor_to_load_310.pkl")
        elif str(sys.version)[0:3] == "3.9":
            pkl_file = path.join(current, "data/predictor_to_load_39.pkl")
        elif str(sys.version)[0:4] == "3.11":
            pkl_file = path.join(current, "data/predictor_to_load_311.pkl")
        elif str(sys.version)[0:4] == "3.12":
            pkl_file = path.join(current, "data/predictor_to_load_312.pkl")
        else:
            raise NotImplementedError

        predictor.save(pkl_file)
        predictor2 = load_smartpredictor(pkl_file)

        attrib_predictor = [element for element in predictor.__dict__.keys()]
        attrib_predictor2 = [element for element in predictor2.__dict__.keys()]

        assert all(attrib in attrib_predictor2 for attrib in attrib_predictor)
        assert all(attrib2 in attrib_predictor for attrib2 in attrib_predictor2)
