"""
Tests for the identity card callback in shapash.webapp.smart_app.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from shapash import SmartExplainer
from shapash.webapp.smart_app import SmartApp


class TestUpdateIdCard(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 20
        x_df = pd.DataFrame({"feat": rng.normal(size=n), "other_feat": rng.normal(size=n)})
        y = pd.Series(np.arange(n) % 2, index=x_df.index, name="target")
        # RidgeClassifier has no predict_proba, so the explainer has no probabilities
        model = RidgeClassifier().fit(x_df, y)
        contributions = pd.DataFrame(rng.normal(size=(n, 2)), columns=x_df.columns, index=x_df.index)

        xpl = SmartExplainer(model=model)
        xpl.compile(x=x_df, contributions=contributions, y_pred=pd.DataFrame({"pred": model.predict(x_df)}))
        self.smart_app = SmartApp(xpl.explainer)

    def _callback(self, out_substr):
        # callback_map stores Dash's context-wrapping shim; the raw user function is under __wrapped__.
        for key, spec in self.smart_app.app.callback_map.items():
            if out_substr in key:
                fn = spec["callback"]
                return getattr(fn, "__wrapped__", fn)
        raise KeyError(out_substr)

    def test_title_without_probabilities(self):
        self.assertIsNone(self.smart_app.explainer.proba_values)
        update_id_card = self._callback("id_card_title_contrib.children")
        data = self.smart_app.dataframe.to_dict("records")
        index = data[0]["_index_"]

        style, children, title = update_id_card(1, 1, "feature_name", True, data, index)

        self.assertEqual(style["display"], "flex")
        self.assertTrue(children)
        self.assertRegex(title, r"^Contribution: \S+$")


if __name__ == "__main__":
    unittest.main()
