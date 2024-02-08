import importlib

# This list should be identical to the list in setup.py
report_requirements = ["nbconvert==6.0.7", "papermill", "matplotlib", "seaborn", "notebook", "Jinja2"]


def check_report_requirements():
    """
    Checks that all required packages for the report are installed.
    This function should be called before executing the report.
    """
    for req in report_requirements:
        pkg = req.split("=")[0]
        try:
            importlib.import_module(pkg.lower())
        except ImportError:
            raise ModuleNotFoundError(
                f"The following package is necessary to generate the Shapash Report : {pkg}. "
                f"Try 'pip install shapash[report]' to install all required packages."
            )
