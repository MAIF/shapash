from typing import Optional
import sys
from datetime import date

import numpy as np
import pandas as pd

from shapash.utils.transform import inverse_transform, apply_postprocessing
from shapash.explainer.smart_explainer import SmartExplainer
from shapash.utils.io import load_yml
from shapash.report.visualisation import print_md, print_html, print_css_table, print_df_and_image, print_figure
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.plots import generate_fig_univariate, generate_correlation_matrix_fig, generate_scatter_plot_fig
from shapash.report.common import series_dtype


class ProjectReport:
    """
    The ProjectReport class allows to generate general information about a
    Data Science project.
    It analyzes the data and the model used in order to provide interesting
    insights that can be shared with non technical person.

    Parameters
    ----------
    explainer : shapash.explainer.smart_explainer.SmartExplainer
        A shapash SmartExplainer object that has already be compiled.
    metadata_file : str
        Path to the yml file containing information about the project (author, description, ...).
    config : dict, optional
        Contains configuration options for the report.

    Attributes
    ----------
    explainer : shapash.explainer.smart_explainer.SmartExplainer
         A shapash SmartExplainer object that has already be compiled.
    metadata : dict
        Information about the project (author, description, ...).
    config : dict
        Configuration options for the report.

    """
    def __init__(
            self,
            explainer: SmartExplainer,
            metadata_file: str,
            x_train: Optional[pd.DataFrame] = None,
            config: Optional[dict] = None
    ):
        self.explainer = explainer
        self.metadata = load_yml(path=metadata_file)
        self.x_train_init = x_train
        if x_train is not None:
            self.x_train_pre = inverse_transform(x_train, self.explainer.preprocessing)
            if self.explainer.postprocessing:
                self.x_train_pre = apply_postprocessing(self.x_train_pre, self.explainer.postprocessing)
        else:
            self.x_train_pre = None
        self.x_pred = self.explainer.x_pred
        self.config = config
        self.col_names = list(self.explainer.columns_dict.values())
        self.df_train_test = self._create_train_test_df(x_pred=self.x_pred, x_train_pre=self.x_train_pre)
        print_css_table()

    def display_general_information(self):
        for k, v in self.metadata['general'].items():
            if k.lower() == 'date' and v.lower() == 'auto':
                print_md(f"**{k.title()}** : {date.today()}")
            else:
                print_md(f"**{k.title()}** : {v}")

    def display_dataset_information(self):
        for k, v in self.metadata['dataset'].items():
            print_md(f"**{k.title()}** : {v}")

    def display_model_information(self):
        print_md(f"**Model used :** : {self.explainer.model.__class__.__name__}")

        print_md(f"**Library :** : {self.explainer.model.__class__.__module__}")

        for name, module in sorted(sys.modules.items()):
            if hasattr(module, '__version__') \
                    and self.explainer.model.__class__.__module__.split('.')[0] in module.__name__:
                print_md(f"**Library version :** : {module.__version__}")

        print_md(f"**Model parameters :** {self.explainer.model.__dict__}")

    def display_dataset_analysis(
            self,
            global_analysis: Optional[bool] = True,
            univariate_analysis: Optional[bool] = True,
            multivariate_analysis: Optional[bool] = True
    ):
        if global_analysis:
            print_md("### Global analysis")
            self._display_dataset_analysis_global()

        if univariate_analysis:
            print_md("### Univariate analysis")
            self._display_dataset_analysis_univariate()

        if multivariate_analysis:
            print_md("### Multivariate analysis")
            self._display_dataset_analysis_multivariate()

    def _display_dataset_analysis_global(self):
        df_stats_global = self._stats_to_table(test_stats=perform_global_dataframe_analysis(self.x_pred),
                                               train_stats=perform_global_dataframe_analysis(self.x_train_pre))
        print_html(df_stats_global.to_html(classes="greyGridTable"))

    def _display_dataset_analysis_univariate(self):
        test_stats_univariate = perform_univariate_dataframe_analysis(self.x_pred)
        train_stats_univariate = perform_univariate_dataframe_analysis(self.x_train_pre)

        for col in self.col_names[:5]:
            print_md(f"#### {col} - {str(series_dtype(self.df_train_test[col]))}")
            fig = generate_fig_univariate(df_train_test=self.df_train_test, col=col)
            df_col_stats = self._stats_to_table(
                test_stats=test_stats_univariate[col],
                train_stats=train_stats_univariate[col] if self.x_train_pre is not None else {}
            )
            print_df_and_image(df_col_stats, fig=fig)

    def _display_dataset_analysis_multivariate(self):
        print_md("#### Numerical vs Numerical")
        fig = generate_correlation_matrix_fig(df_train_test=self.df_train_test)
        print_figure(fig=fig)
        if len(self.df_train_test.select_dtypes(include=np.number).columns.to_list()) < 8:
            fig = generate_scatter_plot_fig(df_train_test=self.df_train_test)
            print_figure(fig=fig)
        else:
            fig = generate_scatter_plot_fig(df_train_test=self.df_train_test[self.col_names[:10] + ['data_train_test']])
            print_figure(fig=fig)

    def _stats_to_table(self, test_stats: dict, train_stats: Optional[dict] = None) -> pd.DataFrame:
        if self.x_train_pre is not None:
            return pd.DataFrame({
                    'Training dataset': pd.Series(train_stats),
                    'Prediction dataset': pd.Series(test_stats)
                })
        else:
            return pd.DataFrame({'Prediction dataset': pd.Series(test_stats)})

    @staticmethod
    def _create_train_test_df(x_pred: pd.DataFrame, x_train_pre: Optional[pd.DataFrame]) -> pd.DataFrame:
        if 'data_train_test' in x_pred.columns:
            raise ValueError('"data_train_test" column must be renamed as it is used in ProjectReport')
        return pd.concat([x_pred.assign(data_train_test="test"),
                          x_train_pre.assign(data_train_test="train") if x_train_pre is not None else None])

