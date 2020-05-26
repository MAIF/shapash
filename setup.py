#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup

with open('README.md') as readme_file:
    long_description = readme_file.read()


requirements = [
        'plotly==4.5.4',
        'numpy==1.17.3',
        'pandas==1.0.3',
        'shap==0.35.0',
        'dash==1.9.1',
        'dash-bootstrap-components==0.9.1',
        'dash-core-components==1.8.1',
        'dash-daq==0.5.0',
        'dash-html-components==1.0.2',
        'dash-renderer==1.2.4',
        'dash-table==4.6.1',
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