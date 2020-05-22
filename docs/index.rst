.. Shapash documentation master file, created by
   sphinx-quickstart on Mon Apr  2 15:29:28 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: ../_static/shapash-resize.png
   :width: 500px
   :align: center

**Shapash is a Python library which aims to make machine learning interpretable and understandable by everyone It provides several types of visualization that display explicit labels that everyone can understand. Data Scientists can more easily understand their models and share their results. End users can understand the decision proposed by a model using a summary of the most influential criteria.

- Compatible with Shap & Lime</li>
- Uses shap backend to display results in a few lines of code</li>
- Encoders objects and features dictionaries used for clear results</li>
- Compatible with category_encoders & Sklearn ColumnTransformer</li>
- Visualizations of global and local explainability</li>
- Webapp to easily navigate from global to local</li>
- Summarizes local explanation</li>
- Offers several parameters in order to summarize in the most suitable way for your use case</li>
- Exports your local summaries to a Pandas DataFrame</li>
- Usable for Regression, Binary Classification or Multiclass</li>
- Compatible with most of sklearn, lightgbm, catboost, xgboost models</li>

Provide a SmartExplainer class to understand your model and summarize explanation with a simple syntax
Very few arguments are required to display results. But the more you work on cleaning and documenting the data, the clearer the results will be for the end user

.. toctree::
   :maxdepth: 3


   overview
   tutorials/index
   autodocs/index

License is Apache Software License 2.0