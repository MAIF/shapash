import os
import shutil
import tempfile
import unittest

import catboost as cb
import category_encoders as ce
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder

from shapash import SmartExplainer
from shapash.report.generation import execute_report, export_and_save_report

current_path = os.path.dirname(os.path.abspath(__file__))


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

    def test_execute_report_1(self):
        tmp_dir_path = tempfile.mkdtemp()

        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
            config=None,
            notebook_path=None,
        )
        assert os.path.exists(os.path.join(tmp_dir_path, "smart_explainer.pickle"))
        assert os.path.exists(os.path.join(tmp_dir_path, "base_report.ipynb"))

        shutil.rmtree(tmp_dir_path)

    def test_execute_report_2(self):
        tmp_dir_path = tempfile.mkdtemp()

        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            config=None,
            notebook_path=None,
        )
        assert os.path.exists(os.path.join(tmp_dir_path, "x_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "smart_explainer.pickle"))
        assert os.path.exists(os.path.join(tmp_dir_path, "base_report.ipynb"))

        shutil.rmtree(tmp_dir_path)

    def test_execute_report_3(self):
        tmp_dir_path = tempfile.mkdtemp()

        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            y_test=self.df["y"],
            config=None,
            notebook_path=None,
        )
        assert os.path.exists(os.path.join(tmp_dir_path, "x_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "y_test.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "smart_explainer.pickle"))
        assert os.path.exists(os.path.join(tmp_dir_path, "base_report.ipynb"))

        shutil.rmtree(tmp_dir_path)

    def test_execute_report_4(self):
        tmp_dir_path = tempfile.mkdtemp()

        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            y_train=self.df["y"],
            y_test=self.df["y"],
            config=None,
            notebook_path=None,
        )
        assert os.path.exists(os.path.join(tmp_dir_path, "x_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "y_test.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "y_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "smart_explainer.pickle"))
        assert os.path.exists(os.path.join(tmp_dir_path, "base_report.ipynb"))

        shutil.rmtree(tmp_dir_path)

    def test_execute_report_5(self):
        tmp_dir_path = tempfile.mkdtemp()

        self.xpl.palette_name = "eurybia"
        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
            x_train=self.df[["x1", "x2", "x3", "x4"]],
            y_train=self.df["y"],
            y_test=self.df["y"],
            notebook_path=None,
        )
        self.xpl.palette_name = "default"
        assert os.path.exists(os.path.join(tmp_dir_path, "x_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "y_test.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "y_train.csv"))
        assert os.path.exists(os.path.join(tmp_dir_path, "smart_explainer.pickle"))
        assert os.path.exists(os.path.join(tmp_dir_path, "base_report.ipynb"))

        shutil.rmtree(tmp_dir_path)

    def test_export_and_save_report_1(self):
        tmp_dir_path = tempfile.mkdtemp()

        execute_report(
            working_dir=tmp_dir_path,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/metadata.yaml"),
        )

        outfile = os.path.join(tmp_dir_path, "report.html")
        export_and_save_report(working_dir=tmp_dir_path, output_file=outfile)
        assert os.path.exists(outfile)
        shutil.rmtree(tmp_dir_path)
