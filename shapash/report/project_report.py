from typing import Optional
import logging
import sys
import os
from datetime import date
import jinja2
import pandas as pd
import plotly

from shapash.utils.transform import inverse_transform, apply_postprocessing
from shapash.explainer.smart_explainer import SmartExplainer
from shapash.utils.io import load_yml
from shapash.utils.utils import get_project_root, truncate_str
from shapash.report.visualisation import print_md, print_html, print_css_style, convert_fig_to_html, print_figure, \
    print_javascript_misc
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.plots import generate_fig_univariate, generate_correlation_matrix_fig
from shapash.report.common import series_dtype, get_callable

logging.basicConfig(level=logging.INFO)

template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(get_project_root(), 'shapash', 'report', 'html'))
template_env = jinja2.Environment(loader=template_loader)


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
        print_md(f"**Model used :** {self.explainer.model.__class__.__name__}")

        print_md(f"**Library :** {self.explainer.model.__class__.__module__}")

        for name, module in sorted(sys.modules.items()):
            if hasattr(module, '__version__') \
                    and self.explainer.model.__class__.__module__.split('.')[0] in module.__name__:
                print_md(f"**Library version :** {module.__version__}")

        print_md("**Model parameters :** ")
        model_params = self.explainer.model.__dict__
        table_template = template_env.get_template("double_table.html")
        print_html(table_template.render(
            columns1=["Parameter key", "Parameter value"],
            rows1=[{"name": truncate_str(str(k), 50), "value": truncate_str(str(v), 300)}
                   for k, v in list(model_params.items())[:len(model_params)//2:]],  # Getting half of the parameters
            columns2=["Parameter key", "Parameter value"],
            rows2=[{"name": truncate_str(str(k), 50), "value": truncate_str(str(v), 300)}
                   for k, v in list(model_params.items())[len(model_params)//2:]]  # Getting 2nd half of the parameters
        ))

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
                                               train_stats=perform_global_dataframe_analysis(self.x_train_pre),
                                               names=["Prediction dataset", "Training dataset"])
        print_html(df_stats_global.to_html(classes="greyGridTable"))

    def _display_dataset_analysis_univariate(self):
        self._perform_and_display_analysis_univariate(
            df=self.df_train_test,
            col_splitter="data_train_test",
            split_values=["test", "train"],
            names=["Prediction dataset", "Training dataset"]
        )

    def _perform_and_display_analysis_univariate(self, df: pd.DataFrame, col_splitter: str, split_values: list, names: list):
        n_splits = df[col_splitter].nunique()
        test_stats_univariate = perform_univariate_dataframe_analysis(df.loc[df[col_splitter] == split_values[0]])
        if n_splits > 1:
            train_stats_univariate = perform_univariate_dataframe_analysis(df.loc[df[col_splitter] == split_values[1]])

        univariate_template = template_env.get_template("univariate.html")
        univariate_features_desc = list()
        for col in df.drop(col_splitter, axis=1).columns:
            fig = generate_fig_univariate(df_all=df, col=col, hue=col_splitter)
            df_col_stats = self._stats_to_table(
                test_stats=test_stats_univariate[col],
                train_stats=train_stats_univariate[col] if n_splits > 1 else None,
                names=names
            )
            univariate_features_desc.append({
                'feature_index': int(self.explainer.inv_columns_dict.get(col, 0)),
                'name': col,
                'type': str(series_dtype(df[col])),
                'description': self.explainer.features_dict.get(col, ''),
                'table': df_col_stats.to_html(classes="greyGridTable"),
                'image': convert_fig_to_html(fig)
            })
        print_html(univariate_template.render(features=univariate_features_desc))

    def _display_dataset_analysis_multivariate(self):
        print_md("#### Numerical vs Numerical")
        fig = generate_correlation_matrix_fig(df_train_test=self.df_train_test)
        print_figure(fig=fig)

    @staticmethod
    def _stats_to_table(test_stats: dict,
                        names: list,
                        train_stats: Optional[dict] = None,
                        ) -> pd.DataFrame:
        if train_stats is not None:
            return pd.DataFrame({
                    names[1]: pd.Series(train_stats),
                    names[0]: pd.Series(test_stats)
                })
        else:
            return pd.DataFrame({names[0]: pd.Series(test_stats)})

    def display_model_explainability(self):
        print_md("*Note : the explainability graphs were generated using the test set only.*")
        print_md("### Global feature importance plot")
        fig = self.explainer.plot.features_importance()
        print_html(plotly.io.to_html(fig, include_plotlyjs=False, full_html=False))

        explainability_contrib_template = template_env.get_template("explainability_contrib.html")

        print_md("### Features contribution plots")
        explain_contrib_data = list()
        for feature in self.col_names:
            fig = self.explainer.plot.contribution_plot(feature)
            explain_contrib_data.append({
                'feature_index': int(self.explainer.inv_columns_dict[feature]),
                'name': feature,
                'description': self.explainer.features_dict[feature],
                'plot': plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)
            })
        print_html(explainability_contrib_template.render(features=explain_contrib_data))

    def display_model_performance(self):
        if self.y_test is None:
            logging.info("No labels given for test set. Skipping model performance part")
            return

        print_md("### Univariate analysis of target variable")
        y_pred = self.explainer.model.predict(self.explainer.x_init)
        if isinstance(self.y_test, pd.DataFrame):
            col_name = self.y_test.columns[0]
            y_true = self.y_test.values[:, 0]
        elif isinstance(self.y_test, pd.Series):
            col_name = self.y_test.name
            y_true = self.y_test.values
        else:
            col_name = "target"
            y_true = self.y_test
        df = pd.concat([pd.DataFrame({col_name: y_pred}).assign(_dataset="pred"),
                        pd.DataFrame({col_name: y_true}).assign(_dataset="true") if y_true is not None else None])
        self._perform_and_display_analysis_univariate(
            df=df,
            col_splitter="_dataset",
            split_values=["pred", "true"],
            names=["Prediction values", "True values"]
        )

        if 'metrics' not in self.config.keys():
            logging.info("No 'metrics' key found in report config dict. Skipping model performance part.")
            return
        print_md("### Metrics")

        for metric_name, metric_path in self.config['metrics'].items():
            metric_fn = get_callable(path=metric_path)
            print_md(f"**{metric_name} :** {round(metric_fn(y_true, y_pred), 2)}")

