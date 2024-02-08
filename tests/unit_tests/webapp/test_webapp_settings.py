import unittest

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from shapash import SmartExplainer


class TestWebappSettings(unittest.TestCase):
    """
    Unit tests for webapp settings class
    Checks that the webapp settings remain valid whether the user input is valid or not
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor - loads a SmartExplainer object from the appropriate pickle
        """
        contributions = pd.DataFrame([[-0.1, 0.2, -0.3], [0.1, -0.2, 0.3]])
        y_pred = pd.DataFrame(data=np.array([1, 2]), columns=["pred"])
        dataframe_x = pd.DataFrame([[1, 2, 3], [1, 2, 3]])
        model = DecisionTreeRegressor().fit([[0]], [[0]])
        self.xpl = SmartExplainer(model=model)
        self.xpl.compile(contributions=contributions, x=dataframe_x, y_pred=y_pred)
        self.xpl.filter(max_contrib=2)
        super().__init__(*args, **kwargs)

    def test_settings_types(self):
        """
        Test settings dtypes (must be ints)
        """
        settings = {"rows": None, "points": 5200.4, "violin": -1, "features": "oui"}
        self.xpl.init_app(settings)
        print(self.xpl.smartapp.settings)
        assert all(isinstance(attrib, int) for k, attrib in self.xpl.smartapp.settings.items())

    def test_settings_values(self):
        """
        Test settings values (must be >0)
        """
        settings = {"rows": 0, "points": 5200.4, "violin": -1, "features": "oui"}
        self.xpl.init_app(settings)
        assert all(attrib > 0 for k, attrib in self.xpl.smartapp.settings.items())

    def test_settings_keys(self):
        """
        Test settings keys : the expected keys must be in the final settings dict, whatever the user input is
        """
        settings = {"oui": 1, 1: 2, "a": []}
        self.xpl.init_app(settings)
        assert all(k in ["rows", "points", "violin", "features"] for k in self.xpl.smartapp.settings)
