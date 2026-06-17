import os
import shutil
import tempfile
import unittest
from pathlib import Path

import catboost as cb
import category_encoders as ce
import numpy as np
import pandas as pd
import yaml
from category_encoders import OrdinalEncoder

from shapash import SmartExplainer

current_path = os.path.dirname(os.path.abspath(__file__))
report_test_cfg_path = os.path.join(current_path, "../data/report_test_config.yml")


class TestGeneration(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(range(0, 21), columns=["id"])
        df["y"] = df["id"].apply(lambda x: 1 if x < 10 else 0)
        df["x1"] = np.random.randint(1, 123, df.shape[0])
        df["x2"] = np.random.randint(1, 3, df.shape[0])
        df["x3"] = np.random.choice(["A", "B", "C", "D"], df.shape[0])
        df["x4"] = np.random.choice(["A", "B", "C", np.nan], df.shape[0])
        df = df.set_index("id")
        encoder = ce.OrdinalEncoder(cols=["x3", "x4"], handle_unknown="None")
        encoder_fitted = encoder.fit(df)
        df_encoded = encoder_fitted.transform(df)
        clf = cb.CatBoostClassifier(n_estimators=1).fit(df_encoded[["x1", "x2", "x3", "x4"]], df_encoded["y"])
        self.xpl = SmartExplainer(model=clf, preprocessing=encoder)
        self.xpl.compile(x=df_encoded[["x1", "x2", "x3", "x4"]])
        self.df = df_encoded

    def test_generate_report_default_config(self):
        tmp_dir_path = tempfile.mkdtemp()
        outfile = os.path.join(tmp_dir_path, "report.html")

        self.xpl.palette_name = "eurybia"
        self.xpl.generate_report(
            output_file=outfile,
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            y_train=self.df["y"],
            y_test=self.df["y"],
            working_dir=tmp_dir_path,
            yaml_path=report_test_cfg_path,
        )
        self.xpl.palette_name = "default"
        assert os.path.exists(outfile)

        shutil.rmtree(tmp_dir_path)

    def test_generate_report_with_custom_yaml_config(self):
        tmp_dir_path = tempfile.mkdtemp()
        cfg_path = Path(tmp_dir_path) / "custom_report_config.yml"
        outfile = str(Path(tmp_dir_path) / "report_custom.html")

        config = {
            "sections": [
                {
                    "type": "header",
                    "params": {"title": "Integration report", "subtitle": "custom yaml"},
                },
                {
                    "type": "project_information",
                    "params": {
                        "title": "Project information",
                        "project_info_file": os.path.join(current_path, "../data/metadata.yaml"),
                    },
                },
            ]
        }
        with cfg_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

        self.xpl.generate_report(
            output_file=outfile,
            yaml_path=str(cfg_path),
        )
        assert os.path.exists(outfile)

        shutil.rmtree(tmp_dir_path)

    def test_generate_report_interactions_enabled(self):
        tmp_dir_path = tempfile.mkdtemp()
        outfile = os.path.join(tmp_dir_path, "report_interactions.html")

        self.xpl.generate_report(
            output_file=outfile,
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            display_interaction_plot=True,
            working_dir=tmp_dir_path,
        )
        assert os.path.exists(outfile)

        shutil.rmtree(tmp_dir_path)
