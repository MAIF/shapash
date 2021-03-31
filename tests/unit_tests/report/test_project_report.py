import unittest
from unittest.mock import patch
import os
import numpy as np
import pandas as pd
import catboost as cb

from shapash.explainer.smart_explainer import SmartExplainer
from shapash.report.project_report import ProjectReport

expected_attrs = ['explainer', 'metadata', 'x_train_init', 'y_test', 'x_pred', 'config',
                  'col_names', 'df_train_test', 'title_story', 'title_description']

current_path = os.path.dirname(os.path.abspath(__file__))


class TestProjectReport(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(range(0, 21), columns=['id'])
        self.df['y'] = self.df['id'].apply(lambda x: 1 if x < 10 else 0)
        self.df['x1'] = np.random.randint(1, 123, self.df.shape[0])
        self.df['x2'] = np.random.randint(1, 3, self.df.shape[0])
        self.df = self.df.set_index('id')
        self.clf = cb.CatBoostClassifier(n_estimators=1).fit(self.df[['x1', 'x2']], self.df['y'])
        self.xpl = SmartExplainer()
        self.xpl.compile(model=self.clf, x=self.df[['x1', 'x2']])
        self.report1 = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
        )
        self.report2 = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
        )

    def test_init_1(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
        )
        for attr in expected_attrs:
            assert hasattr(report, attr)

    def test_init_2(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
        )
        for attr in expected_attrs:
            assert hasattr(report, attr)

    def test_init_3(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
            y_test=self.df['y']
        )
        for attr in expected_attrs:
            assert hasattr(report, attr)

    def test_init_4(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
            y_test=self.df['y'],
            config={}
        )
        for attr in expected_attrs:
            assert hasattr(report, attr)

    def test_init_5(self):
        ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
            y_test=self.df['y'],
            config={'metrics': {'mse': 'sklearn.metrics.mean_squared_error'}}
        )

    def test_init_6(self):
        self.assertRaises(ValueError, ProjectReport,
            self.xpl,
            os.path.join(current_path, '../../data/metadata.yaml'),
            self.df[['x1', 'x2']],
            self.df['y'],
            {'metrics': ['sklearn.metrics.mean_squared_error']}
        )

    @patch('shapash.report.project_report.print_html')
    def test_display_title_description_1(self, mock_print_html):
        self.report1.display_title_description()
        mock_print_html.assert_called_once()

    @patch('shapash.report.project_report.print_html')
    def test_display_title_description_2(self, mock_print_html):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
            y_test=self.df['y'],
            config={'title_story': "My project report",
                    'title_description': """This document is a data science project report."""}
        )
        report.display_title_description()
        self.assertEqual(mock_print_html.call_count, 2)

    @patch('shapash.report.project_report.print_md')
    def test_display_general_information_1(self, mock_print_html):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml')
        )
        report.display_metadata_information()
        self.assertTrue(mock_print_html.called)

    @patch('shapash.report.project_report.print_md')
    def test_display_model_information_1(self, mock_print_md):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml')
        )
        report.display_model_information()
        self.assertTrue(mock_print_md.called)

    def test_display_dataset_analysis_1(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            x_train=self.df[['x1', 'x2']],
        )
        report.display_dataset_analysis()

    def test_display_dataset_analysis_2(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
        )
        report.display_dataset_analysis()

    def test_display_model_explainability(self):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
        )
        report.display_model_explainability()

    @patch('shapash.report.project_report.logging')
    def test_display_model_performance_1(self, mock_logging):
        """
        No y_test given
        """
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
        )
        report.display_model_performance()
        mock_logging.info.assert_called_once()

    @patch('shapash.report.project_report.logging')
    def test_display_model_performance_2(self, mock_logging):
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            y_test=self.df['y'],
            config=dict(metrics={'mse': 'sklearn.metrics.mean_squared_error'})
        )
        report.display_model_performance()
        self.assertEqual(mock_logging.call_count, 0)

    @patch('shapash.report.project_report.logging')
    def test_display_model_performance_3(self, mock_logging):
        """
        No metrics given in ProjectReport
        """
        report = ProjectReport(
            explainer=self.xpl,
            metadata_file=os.path.join(current_path, '../../data/metadata.yaml'),
            y_test=self.df['y'],
        )
        report.display_model_performance()
        mock_logging.info.assert_called_once()
