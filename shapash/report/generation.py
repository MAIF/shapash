"""
Report generation helper module.
"""
import os
from nbconvert import HTMLExporter
import papermill as pm

from shapash.utils.utils import get_project_root


def execute_report(explainer: object, metadata_file: str, config: dict, working_dir: str):
    """
    Executes the base_report.ipynb notebook and saves the results in working_dir.

    Parameters
    ----------
    explainer : shapash.explainer.smart_explainer.SmartExplainer object
        Compiled shapash explainer.
    metadata_file : str
        Path to the metadata file used o display some information about the project in the report.
    config : dict, optional
        Report configuration options.
    working_dir : str
        Directory in which will be saved the executed notebook.

    """
    explainer.save(path=os.path.join(working_dir, 'smart_explainer.pickle'))
    root_path = get_project_root()

    pm.execute_notebook(
        os.path.join(root_path, 'shapash', 'report', 'base_report.ipynb'),
        os.path.join(working_dir, 'base_report.ipynb'),
        parameters=dict(
            dir_path=working_dir,
            metadata_file=metadata_file,
            config=config
        )
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

    exporter = HTMLExporter(exclude_input=True)
    (body, resources) = exporter.from_filename(filename=os.path.join(working_dir, 'base_report.ipynb'))

    with open(output_file, "w") as file:
        file.write(body)
