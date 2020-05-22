Shapash
====================


**Shapash** is a Python library which aims to make machine learning interpretable and understandable by everyone It provides several types of visualization that display explicit labels that everyone can understand. Data Scientists can more easily understand their models and share their results. End users can understand the decision proposed by a model using a summary of the most influential criteria.

- Compatible with Shap & Lime
- Uses shap backend to display results in a few lines of code
- Encoders objects and features dictionaries used for clear results
- Compatible with category_encoders & Sklearn ColumnTransformer
- Visualizations of global and local explainability
- Webapp to easily navigate from global to local
- Summarizes local explanation
- Offers several parameters in order to summarize in the most suitable way for your use case
- Exports your local summaries to a Pandas DataFrame
- Usable for Regression, Binary Classification or Multiclass
- Compatible with most of sklearn, lightgbm, catboost, xgboost models

Provide a SmartExplainer class to understand your model and summarize explanation with a simple syntax
Very few arguments are required to display results. But the more you work on cleaning and documenting the data, the clearer the results will be for the end user

The objectives of shapash:

To display clear and understandable results: Plots and outputs use explicit labels for each feature and its modalities:

.. image:: ./_static/shapash-contribution_plot-example.png
   :width: 500px
   :align: center
   
.. toctree::
   :maxdepth: 3

   overview
   tutorials/index
   autodocs/index


License is Apache Software License 2.0