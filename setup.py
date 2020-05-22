#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
#     history = history_file.read()

requirements = ['pandas>=0.25.0',
                # 'scikit-learn>=0.19.0',
                # 'gensim>=3.3.0',
                # 'keras>=2.2.0',
                # 'tqdm>=4.34',
                # 'streamlit>=0.57.3',
                # 'tensorflow>=1.10.0,<=1.13.1',
                # 'unidecode',
                # 'joblib',
                # 'plotly'
                ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Yann GOLHEN",
    author_email='yanndu79@gmail.com',
    classifiers=[
        'Development Status :: 1 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Shapash is a python package dedicated to understanding and intelligibility of supervised Data Science models. It is an overlay package for libraries dedicated to the interpretability of models.",
    entry_points={},
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='shapash',
    name='shapash',
    # package_dir={
    #     'shapash': 'shapash',
    #     'shapash.config': 'shapash/config',
    #     'shapash.utils': 'shapash/utils',
    #     'shapash.nlp_tools': 'shapash/nlp_tools',
    #     'shapash.prepare_email': 'shapash/prepare_email',
    #     'shapash.summarizer': 'shapash/summarizer',
    #     'shapash.models': 'shapash/models',
    #     'shapash.data': 'shapash/data'
    # },
    # packages=['shapash', 'shapash.config', 'shapash.utils',
    #           'shapash.nlp_tools', 'shapash.prepare_email',
    #           'shapash.summarizer', 'shapash.models',
    #           'shapash.data'],
    # data_files=[('config', ['shapash/config/conf.json']),
    #             ('data', ['shapash/data/emails.csv'])],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    # url='https://github.com/MAIF/shapash',
    # version='1.9.5',
    zip_safe=False,
)