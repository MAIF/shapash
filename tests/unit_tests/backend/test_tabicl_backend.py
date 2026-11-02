import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from shapash import SmartExplainer
from shapash.backend import TabiclBackend, get_backend_cls_from_name


class _TabiclModel:
    __module__ = "tabicl.testing"

    def predict(self, x):
        return np.zeros(len(x))


class TestTabiclBackend(unittest.TestCase):
    def setUp(self):
        self.model = _TabiclModel()
        self.x = pd.DataFrame({"feature_a": [1, 2], "feature_b": [3, 4]})

    def test_run_explainer_uses_input_data_and_returns_shap_values(self):
        values = np.array([[0.1, 0.2], [0.3, 0.4]])
        base_values = np.array([0.5, 0.6])
        result = SimpleNamespace(values=values, base_values=base_values)
        backend = TabiclBackend(self.model)

        with patch("shapash.backend.tabicl_shap_backend.get_shap_values", return_value=result) as get_shap_values:
            explain_data = backend.run_explainer(self.x)

        get_shap_values.assert_called_once_with(
            estimator=self.model,
            X_test=self.x,
            feature_names=["feature_a", "feature_b"],
        )
        self.assertIs(explain_data["contributions"], values)
        self.assertIs(explain_data["base_values"], base_values)

    def test_run_explainer_uses_masker_and_forwards_additional_parameters(self):
        masker = pd.DataFrame({"feature_a": [5], "feature_b": [6]})
        values = np.array([[0.1, 0.2]])
        result = SimpleNamespace(values=values)
        backend = TabiclBackend(self.model, masker=masker, random_state=42)

        with patch("shapash.backend.tabicl_shap_backend.get_shap_values", return_value=result) as get_shap_values:
            explain_data = backend.run_explainer(self.x)

        get_shap_values.assert_called_once_with(
            estimator=self.model,
            X_test=masker,
            feature_names=["feature_a", "feature_b"],
            random_state=42,
        )
        self.assertIs(explain_data["contributions"], values)
        self.assertIsNone(explain_data["base_values"])

    def test_smart_explainer_auto_selects_tabicl_backend(self):
        smart_explainer = SmartExplainer(self.model)

        self.assertEqual(smart_explainer.explainer.backend_name, "tabicl")
        self.assertIs(get_backend_cls_from_name(smart_explainer.explainer.backend_name), TabiclBackend)
