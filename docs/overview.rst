Overview
========

Installation
------------

Shapash works in Python 3.6.
You can install shapash using pip::

    pip install shapash

Features
--------

Shapash is an overlay package for libraries dedicated to the interpretability of models. It uses Shap or Lime backend
to compute contributions.
Shapash relies on the different steps necessary to build a Machine Learning model to make the results understandable.

.. image:: _static/shapash-resize.png
   :width: 300px
   :align: center


Shapash works for Regression, Binary Classification or Multiclass problem.
It is compatible with many models: *Catboost*, *Xgboost*, *LightGBM*, *Sklearn Ensemble*, *Linear models*, SVM.
Shapash can use category-encoders object, sklearn ColumnTransformer or simply features dictionary.

You can use Shap or Lime, compute your own local contributions and use **Shapash** to display plot or summarize it.

Using **Shapash** is simple and requires only a few lines of code.
Most parameters are optional, you can displays plots effortlessly.

But you can also tune plots and outputs, specifying labels dict, features dict, encoders, predictions, ... :
The more you specify parameters, options, dictionaries and more the outputs will be understandable