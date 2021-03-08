from typing import Optional
import sys
from datetime import date

from shapash.explainer.smart_explainer import SmartExplainer
from shapash.utils.io import load_yml
from shapash.report.visualisation import print_md


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
    def __init__(self, explainer: SmartExplainer, metadata_file: str, config: Optional[dict] = None):
        self.explainer = explainer
        self.metadata = load_yml(path=metadata_file)
        self.config = config

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
