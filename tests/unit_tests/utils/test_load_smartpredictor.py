"""
Unit test smart predictor
"""

import json
import os
import sys
import tempfile
import unittest
import warnings
from os import path
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import pytest

from shapash import SmartExplainer
from shapash.utils.io import (
    MANIFEST_SUFFIX,
    _build_predictor_manifest,
    _compute_schema_fingerprint,
    save_pickle,
)
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
        if sys.version_info[:2] == (3, 11):
            pkl_file = path.join(current, "data/predictor_to_load_311.pkl")
        elif sys.version_info[:2] == (3, 12):
            pkl_file = path.join(current, "data/predictor_to_load_312.pkl")
        elif sys.version_info[:2] == (3, 13):
            pkl_file = path.join(current, "data/predictor_to_load_313.pkl")
        elif sys.version_info[:2] == (3, 14):
            pkl_file = path.join(current, "data/predictor_to_load_314.pkl")
        else:
            raise NotImplementedError

        predictor.save(pkl_file)
        predictor2 = load_smartpredictor(pkl_file)

        attrib_predictor = [element for element in predictor.__dict__.keys()]
        attrib_predictor2 = [element for element in predictor2.__dict__.keys()]

        assert all(attrib in attrib_predictor2 for attrib in attrib_predictor)
        assert all(attrib2 in attrib_predictor for attrib2 in attrib_predictor2)


class TestSmartPredictorManifest(unittest.TestCase):
    """
    Tests for the SmartPredictor save/load manifest sidecar.
    Regression tests for https://github.com/MAIF/shapash/issues/707 (Gap 1).
    """

    def _make_predictor(self):
        y_pred = pd.DataFrame(data=np.array([1, 2]), columns=["pred"])
        dataframe_x = pd.DataFrame([[1, 2, 4], [1, 2, 3]])
        clf = cb.CatBoostClassifier(n_estimators=1).fit(dataframe_x, y_pred)
        xpl = SmartExplainer(model=clf, features_dict={})
        xpl.compile(x=dataframe_x, y_pred=y_pred)
        return xpl.to_smartpredictor()

    def test_save_writes_manifest_sidecar(self) -> None:
        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            predictor.save(pkl)
            manifest_path = pkl + MANIFEST_SUFFIX
            assert os.path.exists(manifest_path), "manifest sidecar not written"
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for key in (
                "shapash_version",
                "python_version",
                "model_framework",
                "shap_version",
                "schema_fingerprint",
                "saved_at",
            ):
                assert key in manifest, f"manifest missing '{key}'"
            assert manifest["schema_fingerprint"].startswith("sha256:")
            assert manifest["model_framework"].get("name") == "catboost"

    def test_load_with_matching_manifest_is_clean(self) -> None:
        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            predictor.save(pkl)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loaded = load_smartpredictor(pkl)
            assert isinstance(loaded, type(predictor))

    def test_load_without_manifest_warns_and_loads(self) -> None:
        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            save_pickle(predictor, pkl)
            assert not os.path.exists(pkl + MANIFEST_SUFFIX)
            with pytest.warns(DeprecationWarning, match="without a manifest sidecar"):
                loaded = load_smartpredictor(pkl)
            assert isinstance(loaded, type(predictor))

    def test_load_with_schema_fingerprint_mismatch_raises(self) -> None:
        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            predictor.save(pkl)
            manifest_path = pkl + MANIFEST_SUFFIX
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["schema_fingerprint"] = "sha256:" + "0" * 64
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with pytest.raises(ValueError, match="schema fingerprint mismatch"):
                load_smartpredictor(pkl)

    def test_load_with_major_version_mismatch_raises(self) -> None:
        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            predictor.save(pkl)
            manifest_path = pkl + MANIFEST_SUFFIX
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["shapash_version"] = "999.0.0"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with pytest.raises(ValueError, match="Major version mismatch"):
                load_smartpredictor(pkl)

    def test_load_with_minor_version_mismatch_warns(self) -> None:
        from shapash.__version__ import VERSION

        predictor = self._make_predictor()
        with tempfile.TemporaryDirectory() as tmp:
            pkl = os.path.join(tmp, "predictor.pkl")
            predictor.save(pkl)
            manifest_path = pkl + MANIFEST_SUFFIX
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["shapash_version"] = f"{VERSION[0]}.{VERSION[1] + 1}.0"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with pytest.warns(UserWarning, match="Minor version skew"):
                load_smartpredictor(pkl)

    def test_schema_fingerprint_is_deterministic(self) -> None:
        predictor = self._make_predictor()
        fp1 = _compute_schema_fingerprint(predictor)
        fp2 = _compute_schema_fingerprint(predictor)
        assert fp1 == fp2
        assert fp1.startswith("sha256:")
        assert len(fp1) == len("sha256:") + 64

    def test_manifest_round_trip_preserves_fingerprint(self) -> None:
        predictor = self._make_predictor()
        manifest = _build_predictor_manifest(predictor)
        recomputed = _compute_schema_fingerprint(predictor)
        assert manifest["schema_fingerprint"] == recomputed
