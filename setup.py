#!/usr/bin/env python

"""The setup script."""
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open("README.md", encoding="utf8") as readme_file:
    long_description = readme_file.read()

# Load the package's __version__.py module as a dictionary.
version_d: dict = {}
with open(os.path.join(here, "shapash", "__version__.py")) as f:
    exec(f.read(), version_d)


requirements = [
    "plotly>=5.0.0",
    "matplotlib>=3.2.0",
    "numpy>1.18.0",
    "pandas>1.0.2",
    "shap>=0.38.1,<0.45.0",
    "Flask<2.3.0",
    "dash>=2.3.1",
    "dash-bootstrap-components>=1.1.0",
    "dash-core-components>=2.0.0",
    "dash-daq>=0.5.0",
    "dash-html-components>=2.0.0",
    "dash-renderer==1.8.3",
    "dash-table>=5.0.0",
    "nbformat>4.2.0",
    "numba>=0.53.1",
    "scikit-learn>=1.0.1,<1.4",
    "category_encoders>=2.6.0",
    "scipy>=0.19.1",
]

extras = dict()

# This list should be identical to the list in shapash/report/__init__.py
extras["report"] = [
    "nbconvert>=6.0.7",
    "papermill>=2.0.0",
    "jupyter-client>=7.4.0",
    "seaborn==0.12.2",
    "notebook",
    "Jinja2>=2.11.0",
    "phik",
]

extras["xgboost"] = ["xgboost>=1.0.0"]
extras["lightgbm"] = ["lightgbm>=2.3.0"]
extras["catboost"] = ["catboost>=1.0.1"]
extras["lime"] = ["lime>=0.2.0.0"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
]

setup(
    name="shapash",
    version=version_d["__version__"],
    python_requires=">3.7, <3.12",
    url="https://github.com/MAIF/shapash",
    author="Yann Golhen, Sebastien Bidault, Yann Lagre, Maxime Gendre",
    author_email="yann.golhen@maif.fr",
    description="Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    extras_require=extras,
    license="Apache Software License 2.0",
    keywords="shapash",
    package_dir={
        "shapash": "shapash",
        "shapash.data": "shapash/data",
        "shapash.decomposition": "shapash/decomposition",
        "shapash.explainer": "shapash/explainer",
        "shapash.backend": "shapash/backend",
        "shapash.manipulation": "shapash/manipulation",
        "shapash.report": "shapash/report",
        "shapash.utils": "shapash/utils",
        "shapash.webapp": "shapash/webapp",
        "shapash.webapp.utils": "shapash/webapp/utils",
        "shapash.style": "shapash/style",
    },
    packages=[
        "shapash",
        "shapash.data",
        "shapash.decomposition",
        "shapash.explainer",
        "shapash.backend",
        "shapash.manipulation",
        "shapash.utils",
        "shapash.webapp",
        "shapash.webapp.utils",
        "shapash.report",
        "shapash.style",
    ],
    data_files=[("style", ["shapash/style/colors.json"])],
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    zip_safe=False,
)
