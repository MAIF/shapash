from typing import Optional
import logging
import sys
import os
from datetime import date
from IPython.display import HTML, display
from jinja2 import Template
import pandas as pd

from shapash.utils.transform import inverse_transform, apply_postprocessing
from shapash.explainer.smart_explainer import SmartExplainer
from shapash.utils.io import load_yml
from shapash.utils.utils import get_project_root
from shapash.report.visualisation import print_md, print_html, print_css_style, convert_fig_to_html, print_figure, \
    print_javascript_misc
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.plots import generate_fig_univariate, generate_correlation_matrix_fig, generate_scatter_plot_fig
from shapash.report.common import series_dtype, get_callable

logging.basicConfig(level=logging.INFO)


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
    x_train : pd.DataFrame
        DataFrame used for training the model.
    y_test : pd.Series or pd.DataFrame
        Series of labels in the test set.
    config : dict
        Configuration options for the report.

    """
    def __init__(
            self,
            explainer: SmartExplainer,
            metadata_file: str,
            x_train: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.DataFrame] = None,
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
        self.y_test = y_test
        self.x_pred = self.explainer.x_pred
        self.config = config if config is not None else dict()
        self.col_names = list(self.explainer.columns_dict.values())
        self.df_train_test = self._create_train_test_df(x_pred=self.x_pred, x_train_pre=self.x_train_pre)

        if 'title_story' in config.keys():
            self.title_story = config['title_story']
        elif self.explainer.title_story != '':
            self.title_story = self.explainer.title_story
        else:
            self.title_story = 'Shapash report'
        self.title_description = config['title_description'] if 'title_description' in config.keys() else ''

        print_css_style()
        print_javascript_misc()

        if 'metrics' in self.config.keys() and not isinstance(self.config['metrics'], dict):
            raise ValueError(f"The report config dict includes a 'metrics' key but this key expects a dict, "
                             f"but an object of type {type(self.config['metrics'])} was found")

    @staticmethod
    def _create_train_test_df(x_pred: pd.DataFrame, x_train_pre: Optional[pd.DataFrame]) -> pd.DataFrame:
        if 'data_train_test' in x_pred.columns:
            raise ValueError('"data_train_test" column must be renamed as it is used in ProjectReport')
        return pd.concat([x_pred.assign(data_train_test="test"),
                          x_train_pre.assign(data_train_test="train") if x_train_pre is not None else None])

    def display_title_description(self):
        print_html(f"""<h1 style="text-align:center">{self.title_story}</p> """)
        if self.title_description != '':
            print_html(f'<blockquote class="panel-warning text_cell_render">{self.title_description} </blockquote>')

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

        with open(os.path.join(get_project_root(), 'shapash', 'report', 'html', 'univariate.html')) as file_:
            univariate_template = Template(file_.read())

        univariate_features_desc = list()
        for col in self.col_names:
            fig = generate_fig_univariate(df_train_test=self.df_train_test, col=col)
            df_col_stats = self._stats_to_table(
                test_stats=test_stats_univariate[col],
                train_stats=train_stats_univariate[col] if self.x_train_pre is not None else {}
            )
            univariate_features_desc.append({
                'feature_index': int(self.explainer.inv_columns_dict[col]),
                'name': col,
                'type': str(series_dtype(self.df_train_test[col])),
                'description': self.explainer.features_dict[col],
                'table': df_col_stats.to_html(classes="greyGridTable"),
                'image': convert_fig_to_html(fig)
            })
        display(HTML(univariate_template.render(features=univariate_features_desc)))

    def _display_dataset_analysis_multivariate(self):
        print_md("#### Numerical vs Numerical")
        fig = generate_correlation_matrix_fig(df_train_test=self.df_train_test)
        print_figure(fig=fig)

    def _stats_to_table(self, test_stats: dict, train_stats: Optional[dict] = None) -> pd.DataFrame:
        if self.x_train_pre is not None:
            return pd.DataFrame({
                    'Training dataset': pd.Series(train_stats),
                    'Prediction dataset': pd.Series(test_stats)
                })
        else:
            return pd.DataFrame({'Prediction dataset': pd.Series(test_stats)})

    def display_model_explainability(self):
        print_md("*Note : the explainability graphs were generated using the test set only.*")
        print_md("### Global feature importance plot")
        fig = self.explainer.plot.features_importance()
        fig.show()

        print_md("### Top 5 features contribution plot")
        for feature in self.explainer.features_imp.index[::-1][:5]:
            fig = self.explainer.plot.contribution_plot(feature)
            fig.show()

    def display_model_performance(self):
        if self.y_test is None:
            logging.info("No labels given for test set. Skipping model performance part")
            return
        if 'metrics' not in self.config.keys():
            logging.info("No 'metrics' key found in report config dict. Skipping model performance part.")
            return

        y_pred = self.explainer.model.predict(self.explainer.x_init)
        y_true = self.y_test
        for metric_name, metric_path in self.config['metrics'].items():
            metric_fn = get_callable(path=metric_path)
            print_md(f"**{metric_name} :** {metric_fn(y_true, y_pred)}")

