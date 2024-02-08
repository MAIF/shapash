import logging
import os
import sys
from datetime import date
from numbers import Number
from typing import Optional, Tuple, Union

import jinja2
import numpy as np
import pandas as pd
import plotly

from shapash import SmartExplainer
from shapash.report.common import compute_col_types, display_value, get_callable, series_dtype
from shapash.report.data_analysis import perform_global_dataframe_analysis, perform_univariate_dataframe_analysis
from shapash.report.plots import generate_confusion_matrix_plot, generate_fig_univariate
from shapash.report.visualisation import (
    convert_fig_to_html,
    print_css_style,
    print_html,
    print_javascript_misc,
    print_md,
)
from shapash.utils.io import load_yml
from shapash.utils.transform import apply_postprocessing, handle_categorical_missing, inverse_transform
from shapash.utils.utils import get_project_root, truncate_str
from shapash.webapp.utils.utils import round_to_k

logging.basicConfig(level=logging.INFO)

template_loader = jinja2.FileSystemLoader(searchpath=os.path.join(get_project_root(), "shapash", "report", "html"))
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
    project_info_file : str
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
        Series of labels in the train set.
    y_test : pd.Series or pd.DataFrame
        Series of labels in the test set.
    config : dict, optional
        Configuration options for the report.

    """

    def __init__(
        self,
        explainer: SmartExplainer,
        project_info_file: str,
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
        config: Optional[dict] = None,
    ):
        self.explainer = explainer
        self.metadata = load_yml(path=project_info_file)
        self.x_train_init = x_train
        if x_train is not None:
            x_train_pre = inverse_transform(x_train, self.explainer.preprocessing)
            self.x_train_pre = handle_categorical_missing(x_train_pre)

            if self.explainer.postprocessing:
                self.x_train_pre = apply_postprocessing(self.x_train_pre, self.explainer.postprocessing)
        else:
            self.x_train_pre = None
        self.x_init = self.explainer.x_init
        self.config = config if config is not None else dict()
        self.col_names = list(self.explainer.columns_dict.values())
        self.df_train_test = self._create_train_test_df(test=self.x_init, train=self.x_train_pre)
        if self.explainer.y_pred is not None:
            self.y_pred = np.array(self.explainer.y_pred.T)[0]
        else:
            self.y_pred = self.explainer.model.predict(self.explainer.x_encoded)
        self.y_test, target_name_test = self._get_values_and_name(y_test, "target")
        self.y_train, target_name_train = self._get_values_and_name(y_train, "target")
        self.target_name = target_name_train or target_name_test

        if "title_story" in self.config.keys():
            self.title_story = config["title_story"]
        elif self.explainer.title_story != "":
            self.title_story = self.explainer.title_story
        else:
            self.title_story = "Shapash report"
        self.title_description = self.config["title_description"] if "title_description" in self.config.keys() else ""

        print_css_style()
        print_javascript_misc()

        if "metrics" in self.config.keys():
            if not isinstance(self.config["metrics"], list) or not isinstance(self.config["metrics"][0], dict):
                raise ValueError("The metrics parameter expects a list of dict.")
            for metric in self.config["metrics"]:
                for key in metric:
                    if key not in ["path", "name", "use_proba_values"]:
                        raise ValueError(f"Unknown key : {key}. Key should be in ['path', 'name', 'use_proba_values']")
                    if key == "use_proba_values" and not isinstance(metric["use_proba_values"], bool):
                        raise ValueError('"use_proba_values" metric key expects a boolean value.')

    @staticmethod
    def _get_values_and_name(
        y: Optional[Union[pd.DataFrame, pd.Series, list]], default_name: str
    ) -> Union[Tuple[list, str], Tuple[None, None]]:
        """
        Extracts vales and column name from a Pandas Series, DataFrame, or assign a default
        name if y is a list of values.

        Parameters
        ----------
        y : list or pd.Series or pd.DataFrame
            Column we want to extract the name and values
        default_name :
            Name assigned if no name was found for y

        Returns
        -------
        values : list
            list of values of y
        name : str
            name of y
        """
        if y is None:
            return None, None
        elif isinstance(y, pd.DataFrame):
            assert len(y.columns) == 1, "Number of columns found is greater than 1"
            name = y.columns[0]
            values = y.values[:, 0]
        elif isinstance(y, pd.Series):
            name = y.name
            values = y.values
        elif isinstance(y, list):
            name = default_name
            values = y
        else:
            raise ValueError(f"Cannot process following type : {type(y)}")
        return values, name

    @staticmethod
    def _create_train_test_df(test: Optional[pd.DataFrame], train: Optional[pd.DataFrame]) -> Union[pd.DataFrame, None]:
        """
        Creates a DataFrame that contains train and test dataset with the column 'data_train_test'
        allowing to distinguish the values.

        Parameters
        ----------
        test : pd.DataFrame, optional
            test dataframe
        train : pd.DataFrame, optional
            train dataframe

        Returns
        -------
        pd.DataFrame
            The concatenation of train and test as a dataframe containing train and test values with
            a new 'data_train_test' column allowing to distinguish the values.
        """
        if (test is not None and "data_train_test" in test.columns) or (
            train is not None and "data_train_test" in train.columns
        ):
            raise ValueError('"data_train_test" column must be renamed as it is used in ProjectReport')
        if test is None and train is None:
            return None
        return pd.concat(
            [
                test.assign(data_train_test="test") if test is not None else None,
                train.assign(data_train_test="train") if train is not None else None,
            ]
        ).reset_index(drop=True)

    def display_title_description(self):
        """
        Displays title of the report and its description if defined.
        """
        print_html(f"""<h1 style="text-align:center">{self.title_story}</p> """)
        if self.title_description != "":
            print_html(f'<blockquote class="panel-warning text_cell_render">{self.title_description} </blockquote>')

    def display_project_information(self):
        """
        Displays general information about the project as defined in the metdata file.
        """
        for section in self.metadata.keys():
            print_md(f"## {section.title()}")
            for k, v in self.metadata[section].items():
                if k.lower() == "date" and v.lower() == "auto":
                    print_md(f"**{k.title()}** : {date.today()}")
                else:
                    print_md(f"**{k.title()}** : {v}")
            print_md("---")

    def display_model_analysis(self):
        """
        Displays information about the model used : class name, library name, library version,
        model parameters, ...
        """
        print_md(f"**Model used :** {self.explainer.model.__class__.__name__}")

        print_md(f"**Library :** {self.explainer.model.__class__.__module__}")

        for _, module in sorted(sys.modules.items()):
            if (
                hasattr(module, "__version__")
                and self.explainer.model.__class__.__module__.split(".")[0] == module.__name__
            ):
                print_md(f"**Library version :** {module.__version__}")

        print_md("**Model parameters :** ")
        model_params = self.explainer.model.__dict__
        table_template = template_env.get_template("double_table.html")
        print_html(
            table_template.render(
                columns1=["Parameter key", "Parameter value"],
                rows1=[
                    {"name": truncate_str(str(k), 50), "value": truncate_str(str(v), 300)}
                    for k, v in list(model_params.items())[: len(model_params) // 2 :]
                ],  # Getting half of the parameters
                columns2=["Parameter key", "Parameter value"],
                rows2=[
                    {"name": truncate_str(str(k), 50), "value": truncate_str(str(v), 300)}
                    for k, v in list(model_params.items())[len(model_params) // 2 :]
                ],  # Getting 2nd half of the parameters
            )
        )
        print_md("---")

    def display_dataset_analysis(
        self,
        global_analysis: bool = True,
        univariate_analysis: bool = True,
        target_analysis: bool = True,
        multivariate_analysis: bool = True,
    ):
        """
        This method performs and displays an exploration of the data given.
        It allows to compare train and test values for each part of the analysis.

        The parameters of the method allow to filter which part to display or not.

        Parameters
        ----------
        global_analysis : bool
            Whether or not to display the global analysis part.
        univariate_analysis : bool
            Whether or not to display the univariate analysis part.
        target_analysis : bool
            Whether or not to display the target analysis part that plots
            the distribution of the target variable.
        multivariate_analysis : bool
            Whether or not to display the multivariate analysis part
        """
        if global_analysis:
            print_md("### Global analysis")
            self._display_dataset_analysis_global()

        if univariate_analysis:
            print_md("### Univariate analysis")
            self._perform_and_display_analysis_univariate(
                df=self.df_train_test,
                col_splitter="data_train_test",
                split_values=["test", "train"],
                names=["Prediction dataset", "Training dataset"],
                group_id="univariate",
            )
        if target_analysis:
            df_target = self._create_train_test_df(
                test=pd.DataFrame({self.target_name: self.y_test}, index=range(len(self.y_test)))
                if self.y_test is not None
                else None,
                train=pd.DataFrame({self.target_name: self.y_train}, index=range(len(self.y_train)))
                if self.y_train is not None
                else None,
            )
            if df_target is not None:
                if target_analysis:
                    print_md("### Target analysis")
                    self._perform_and_display_analysis_univariate(
                        df=df_target,
                        col_splitter="data_train_test",
                        split_values=["test", "train"],
                        names=["Prediction dataset", "Training dataset"],
                        group_id="target",
                    )
        if multivariate_analysis:
            print_md("### Multivariate analysis")
            fig_corr = self.explainer.plot.correlations(
                self.df_train_test,
                facet_col="data_train_test",
                max_features=20,
                width=900 if len(self.df_train_test["data_train_test"].unique()) > 1 else 500,
                height=500,
            )
            print_html(plotly.io.to_html(fig_corr))
        print_md("---")

    def _display_dataset_analysis_global(self):
        df_stats_global = self._stats_to_table(
            test_stats=perform_global_dataframe_analysis(self.x_init),
            train_stats=perform_global_dataframe_analysis(self.x_train_pre),
            names=["Prediction dataset", "Training dataset"],
        )
        print_html(df_stats_global.to_html(classes="greyGridTable"))

    def _perform_and_display_analysis_univariate(
        self, df: pd.DataFrame, col_splitter: str, split_values: list, names: list, group_id: str
    ):
        col_types = compute_col_types(df)
        n_splits = df[col_splitter].nunique()
        inv_columns_dict = {v: k for k, v in self.explainer.columns_dict.items()}
        test_stats_univariate = perform_univariate_dataframe_analysis(
            df.loc[df[col_splitter] == split_values[0]], col_types=col_types
        )
        if n_splits > 1:
            train_stats_univariate = perform_univariate_dataframe_analysis(
                df.loc[df[col_splitter] == split_values[1]], col_types=col_types
            )

        univariate_template = template_env.get_template("univariate.html")
        univariate_features_desc = list()
        list_cols_labels = [
            self.explainer.features_dict.get(col, col) for col in df.drop(col_splitter, axis=1).columns.to_list()
        ]
        for col_label in sorted(list_cols_labels):
            col = self.explainer.inv_features_dict.get(col_label, col_label)
            fig = generate_fig_univariate(
                df_all=df, col=col, hue=col_splitter, type=col_types[col], colors_dict=self.explainer.colors_dict
            )
            df_col_stats = self._stats_to_table(
                test_stats=test_stats_univariate[col],
                train_stats=train_stats_univariate[col] if n_splits > 1 else None,
                names=names,
            )

            univariate_features_desc.append(
                {
                    "feature_index": int(inv_columns_dict.get(col, 0)),
                    "name": col,
                    "type": str(series_dtype(df[col])),
                    "description": col_label,
                    "table": df_col_stats.to_html(classes="greyGridTable"),
                    "image": convert_fig_to_html(fig),
                }
            )
        print_html(univariate_template.render(features=univariate_features_desc, groupId=group_id))

    @staticmethod
    def _stats_to_table(
        test_stats: dict,
        names: list,
        train_stats: Optional[dict] = None,
    ) -> pd.DataFrame:
        if train_stats is not None:
            return pd.DataFrame({names[1]: pd.Series(train_stats), names[0]: pd.Series(test_stats)})
        else:
            return pd.DataFrame({names[0]: pd.Series(test_stats)})

    def display_model_explainability(self):
        """
        Displays explainability of the model as computed in SmartPlotter object
        """
        print_md("*Note : the explainability graphs were generated using the test set only.*")
        explainability_template = template_env.get_template("explainability.html")
        inv_columns_dict = {v: k for k, v in self.explainer.columns_dict.items()}
        explain_data = list()
        multiclass = True if (self.explainer._classes and len(self.explainer._classes) > 2) else False
        c_list = self.explainer._classes if multiclass else [1]  # list just used for multiclass
        for index_label, label in enumerate(c_list):  # Iterating over all labels in multiclass case
            label_value = self.explainer.check_label_name(label)[2] if multiclass else ""
            fig_features_importance = self.explainer.plot.features_importance(label=label)

            explain_contrib_data = list()
            list_cols_labels = [self.explainer.features_dict.get(col, col) for col in self.col_names]
            for feature_label in sorted(list_cols_labels):
                feature = self.explainer.inv_features_dict.get(feature_label, feature_label)
                fig = self.explainer.plot.contribution_plot(feature, label=label, max_points=200)
                explain_contrib_data.append(
                    {
                        "feature_index": int(inv_columns_dict[feature]),
                        "name": feature,
                        "description": self.explainer.features_dict[feature],
                        "plot": plotly.io.to_html(fig, include_plotlyjs=False, full_html=False),
                    }
                )
            explain_data.append(
                {
                    "index": index_label,
                    "name": label_value,
                    "feature_importance_plot": plotly.io.to_html(
                        fig_features_importance, include_plotlyjs=False, full_html=False
                    ),
                    "features": explain_contrib_data,
                }
            )
        print_html(explainability_template.render(labels=explain_data))
        print_md("---")

    def display_model_performance(self):
        """
        Displays the performance of the model. The metrics are computed using the config dict.

        Metrics should be given as a list of dict. Each dict contains they following keys :
        'path' (path to the metric function, ex: 'sklearn.metrics.mean_absolute_error'),
        'name' (optional, name of the metric as displayed in the report),
        and 'use_proba_values' (optional, possible values are False (default) or True
        if the metric uses proba values instead of predicted values).

        For example :
        config['metrics'] = [
                {
                    'path': 'sklearn.metrics.mean_squared_error',
                    'name': 'Mean absolute error',  # Optional : name that will be displayed next to the metric
                    'y_pred': 'predicted_values'  # Optional
                },
                {
                    'path': 'Scoring_AP.utils.lift10',  # Custom function path
                    'name': 'Lift10',
                    'y_pred': 'proba_values'  # Use proba values instead of predicted values
                }
            ]
        """
        if self.y_test is None:
            logging.info("No labels given for test set. Skipping model performance part")
            return

        print_md("### Univariate analysis of target variable")
        df = pd.concat(
            [
                pd.DataFrame({self.target_name: self.y_pred}).assign(_dataset="pred"),
                pd.DataFrame({self.target_name: self.y_test}).assign(_dataset="true")
                if self.y_test is not None
                else None,
            ]
        ).reset_index(drop=True)
        self._perform_and_display_analysis_univariate(
            df=df,
            col_splitter="_dataset",
            split_values=["pred", "true"],
            names=["Prediction values", "True values"],
            group_id="target-distribution",
        )

        if "metrics" not in self.config.keys():
            logging.info("No 'metrics' key found in report config dict. Skipping model performance part.")
            return
        print_md("### Metrics")

        for metric in self.config["metrics"]:
            if "name" not in metric.keys():
                metric["name"] = metric["path"]

            if (
                metric["path"] in ["confusion_matrix", "sklearn.metrics.confusion_matrix"]
                or metric["name"] == "confusion_matrix"
            ):
                print_md(f"**{metric['name']} :**")
                print_html(
                    convert_fig_to_html(
                        generate_confusion_matrix_plot(
                            y_true=self.y_test, y_pred=self.y_pred, colors_dict=self.explainer.colors_dict
                        )
                    )
                )
            else:
                try:
                    metric_fn = get_callable(path=metric["path"])
                    #  Look if we should use proba values instead of predicted values
                    if "use_proba_values" in metric.keys() and metric["use_proba_values"] is True:
                        y_pred = self.explainer.proba_values
                    else:
                        y_pred = self.y_pred
                    res = metric_fn(self.y_test, y_pred)
                except Exception as e:
                    logging.info(f"Could not compute following metric : {metric['path']}. \n{e}")
                    continue
                if isinstance(res, Number):
                    res = display_value(round_to_k(res, 3))
                    print_md(f"**{metric['name']} :** {res}")
                elif isinstance(res, (list, tuple, np.ndarray)):
                    print_md(f"**{metric['name']} :**")
                    print_html(pd.DataFrame(res).to_html(classes="greyGridTable"))
                elif isinstance(res, str):
                    print_md(f"**{metric['name']} :**")
                    print_html(f"<pre>{res}</pre>")
                else:
                    logging.info(
                        f"Could not compute following metric : {metric['path']}. \n"
                        f"Result of type {res} cannot be displayed"
                    )
        print_md("---")
