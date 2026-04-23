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
        assert all(k in ["rows", "points", "violin", "features", "toggle_group"] for k in self.xpl.smartapp.settings)

    def test_toggle_group_true(self):
        """
        Test that toggle_group=True is correctly stored
        """
        settings = {"toggle_group": True}
        self.xpl.init_app(settings)
        assert self.xpl.smartapp.settings["toggle_group"] is True

    def test_toggle_group_false(self):
        """
        Test that toggle_group=False is correctly stored
        """
        settings = {"toggle_group": False}
        self.xpl.init_app(settings)
        assert self.xpl.smartapp.settings["toggle_group"] is False

    def test_toggle_group_default(self):
        """
        Test that toggle_group defaults to True when not provided
        """
        settings = {}
        self.xpl.init_app(settings)
        assert self.xpl.smartapp.settings["toggle_group"] is True

    def test_toggle_group_invalid_values(self):
        """
        Test that invalid toggle_group values (non-bool) fall back to the default True
        """
        for invalid in [1, 0, "true", "false", None, 1.0, []]:
            with self.subTest(invalid=invalid):
                settings = {"toggle_group": invalid}
                self.xpl.init_app(settings)
                assert self.xpl.smartapp.settings["toggle_group"] is True
