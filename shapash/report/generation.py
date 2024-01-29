"""
Report generation helper module.
"""
import os
from typing import Optional, Union

import pandas as pd
import papermill as pm
from nbconvert import HTMLExporter

from shapash.utils.utils import get_project_root


def execute_report(
    working_dir: str,
    explainer: object,
    project_info_file: str,
    x_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    y_test: Optional[Union[pd.Series, pd.DataFrame]] = None,
    config: Optional[dict] = None,
    notebook_path: Optional[str] = None,
    kernel_name: Optional[str] = None,
):
    """
    Executes the base_report.ipynb notebook and saves the results in working_dir.

    Parameters
    ----------
    working_dir : str
        Directory in which will be saved the executed notebook.
    explainer : shapash.explainer.smart_explainer.SmartExplainer
        Compiled shapash explainer.
    project_info_file : str
        Path to the file used to display some information about the project in the report.
    x_train : pd.DataFrame
        DataFrame used for training the model.
    y_train : pd.Series or pd.DataFrame
        Series of labels in the training set.
    y_test : pd.Series or pd.DataFrame
        Series of labels in the test set.
    config : dict, optional
        Report configuration options.
    notebook_path : str, optional
        Path to the notebook used to generate the report. If None, the Shapash base report
        notebook will be used.
    kernel_name : str, optional
        Name of the kernel used to generate the report. This parameter can be usefull if
        you have multiple jupyter kernels and that the method does not use the right kernel
        by default.
    """
    if config is None:
        config = {}
    explainer.save(path=os.path.join(working_dir, "smart_explainer.pickle"))
    if x_train is not None:
        x_train.to_csv(os.path.join(working_dir, "x_train.csv"))
    if y_train is not None:
        y_train.to_csv(os.path.join(working_dir, "y_train.csv"))
    if y_test is not None:
        y_test.to_csv(os.path.join(working_dir, "y_test.csv"))
    root_path = get_project_root()
    if notebook_path is None or notebook_path == "":
        notebook_path = os.path.join(root_path, "shapash", "report", "base_report.ipynb")

    pm.execute_notebook(
        notebook_path,
        os.path.join(working_dir, "base_report.ipynb"),
        parameters=dict(dir_path=working_dir, project_info_file=project_info_file, config=config),
        kernel_name=kernel_name,
    )


def export_and_save_report(working_dir: str, output_file: str):
    """
    Exports a previously executed notebook and saves it as a static HTML file.

    Parameters
    ----------
    working_dir : str
        Path to the directory containing the executed notebook.
    output_file : str
        Path to the html file that will be created.
    """

    exporter = HTMLExporter(
        exclude_input=True,
        extra_template_basedirs=[os.path.join(get_project_root(), "shapash", "report", "template")],
        template_name="custom",
        exclude_anchor_links=True,
    )
    (body, resources) = exporter.from_filename(filename=os.path.join(working_dir, "base_report.ipynb"))

    with open(output_file, "w") as file:
        file.write(body)
