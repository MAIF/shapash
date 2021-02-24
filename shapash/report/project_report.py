from typing import Optional

from shapash.explainer.smart_explainer import SmartExplainer
from shapash.utils.io import load_yaml


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
        Path to the yaml file containing information about the project (author, description, ...).
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
    def __init__(self, explainer: SmartExplainer, metadata_file: str, config: Optional[dict] = None):
        self.explainer = explainer
        self.metadata = load_yaml(path=metadata_file)
        self.config = config
