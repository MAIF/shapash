#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import setup

with open('README.md', encoding='utf8') as readme_file:
    long_description = readme_file.read()


requirements = [
        'plotly==4.12.0',
        'matplotlib>=3.3.0',
        'numpy>1.18.0',
        'pandas>1.0.2',
        'shap>=0.36.0',
        'dash==1.17.0',
        'dash-bootstrap-components==0.9.1',
        'dash-core-components==1.13.0',
        'dash-daq==0.5.0',
        'dash-html-components==1.1.1',
        'dash-renderer==1.8.3',
        'dash-table==4.11.0',
        'nbformat>4.2.0',
        'numba==0.53.1',
        'scikit-learn'
    ]

extras = dict()

# This list should be identical to the list in shapash/report/__init__.py
extras['report'] = [
    'nbconvert==6.0.7',
    'papermill',
    'seaborn<=0.11.2',
    'notebook',
    'Jinja2',
    'phik'
]

extras['xgboost'] = ['xgboost>=1.0.0']
extras['lightgbm'] = ['lightgbm>=2.3.0']
extras['catboost'] = ['catboost>=0.21']
extras['scikit-learn'] = ['scikit-learn>=0.23.0']
extras['category_encoders'] = ['category_encoders==2.2.2']
extras['acv'] = ['acv-exp==1.1.2']
extras['lime'] = ['lime']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    name="shapash",
    version="1.7.1",
    python_requires='>3.5, <3.10',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    extras_require=extras,
    license="Apache Software License 2.0",
    keywords='shapash',
    package_dir={
        'shapash': 'shapash',
        'shapash.data': 'shapash/data',
        'shapash.decomposition': 'shapash/decomposition',
        'shapash.explainer': 'shapash/explainer',
        'shapash.manipulation': 'shapash/manipulation',
        'shapash.report': 'shapash/report',
        'shapash.utils': 'shapash/utils',
        'shapash.webapp': 'shapash/webapp',
        'shapash.webapp.utils': 'shapash/webapp/utils',
        'shapash.style': 'shapash/style',
    },
    packages=['shapash', 'shapash.data', 'shapash.decomposition',
              'shapash.explainer', 'shapash.manipulation',
              'shapash.utils', 'shapash.webapp', 'shapash.webapp.utils',
              'shapash.report','shapash.style'],
    data_files=[('data', ['shapash/data/house_prices_dataset.csv']),
                ('data', ['shapash/data/house_prices_labels.json']),
                ('data', ['shapash/data/titanicdata.csv']),
                ('data', ['shapash/data/titaniclabels.json']),
                ('style', ['shapash/style/colors.json'])],
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    zip_safe=False,
)
