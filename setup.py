#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import setup

with open('README.md') as readme_file:
    long_description = readme_file.read()


requirements = [
        'plotly==4.12.0',
        'numpy==1.18.1',
        'pandas==1.0.3',
        'shap==0.37.0',
        'dash==1.17.0',
        'dash-bootstrap-components==0.9.1',
        'dash-core-components==1.13.0',
        'dash-daq==0.5.0',
        'dash-html-components==1.1.1',
        'dash-renderer==1.8.3',
        'dash-table==4.11.0',
        'nbformat==5.0.8',
        'numba==0.51.2',
    ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name="shapash", # Replace with your own username
    version="1.0.1",
    python_requires='>3.5, < 3.8',
    url='https://github.com/MAIF/shapash',
    author="Yann Golhen, Sebastien Bidault, Yann Lagre, Maxime Gendre",
    author_email="yann.golhen@maif.fr",
    description="Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: Apache Software License",
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
        'shapash.webapp': 'shapash/webapp',
        'shapash.webapp.utils': 'shapash/webapp/utils',
    },
    packages=['shapash', 'shapash.data', 'shapash.decomposition',
              'shapash.explainer', 'shapash.manipulation',
              'shapash.utils', 'shapash.webapp', 'shapash.webapp.utils'],
    data_files=[('data', ['shapash/data/house_prices_dataset.csv']),
                ('data', ['shapash/data/house_prices_labels.json']),
                ('data', ['shapash/data/titanicdata.csv']),
                ('data', ['shapash/data/titaniclabels.json'])],
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
