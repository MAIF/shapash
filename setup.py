#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup

with open('README.md') as readme_file:
    long_description = readme_file.read()


requirements = [
        'pip==20.1.1',
        'plotly==4.3.0',
        'numpy==1.17.3',
        'pandas==0.25.0',
        'shap==0.32.1',
        'dash>=0.41.0',
        'dash-bootstrap-components>=0.9.1',
        'dash-core-components>=0.46.0',
        'dash-dangerously-set-inner-html>=0.0.2',
        'dash-daq>=0.1.0',
        'dash-html-components>=0.15.0',
        'dash-renderer>=0.22.0',
        'dash-table>=3.6.0',
        'dash-table-experiments>=0.6.0'
    ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name="shapash", # Replace with your own username
    version="0.0.1",
    url='https://github.com/MAIF/shapash',
    author="Yann Golhen, Sebastien Bidault, Yann Lagre, Maxime Gendre",
    author_email="yann.golhen@maif.fr",
    description="Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    license="Apache Software License 2.0",
    keywords='shapash',
    package_dir={
        'shapash': 'shapash',
        'shapash.data': 'shapash/data',
        'shapash.decomposition': 'shapash/decomposition',
        'shapash.explainer': 'shapash/explainer',
        'shapash.manipulation': 'shapash/manipulation',
        'shapash.utils': 'shapash/utils',
        'shapash.webapp': 'shapash/webapp'
    },
    packages=['shapash', 'shapash.data', 'shapash.decomposition',
              'shapash.explainer', 'shapash.manipulation',
              'shapash.utils', 'shapash.webapp'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)