#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup

with open('README.md') as readme_file:
    long_description = readme_file.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements.dev.txt') as f:
    requirements_dev = f.read().splitlines()


setup(
    name="shapash", # Replace with your own username
    version="0.0.1",
    url='https://github.com/MAIF/shapash',
    author="Yann Golhen, Sebastien Bidault, Yann Lagre, Maxime Gendre",
    author_email="yann.golhen@maif.fr",
    description="Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone.",
    long_description=long_description,
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
    setup_requires=requirements_dev,
    test_suite='tests',
    tests_require=requirements_dev,
    zip_safe=False,
)