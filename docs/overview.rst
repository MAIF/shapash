Getting started
===============

How Shapash works
-----------------

Shapash sits on top of explainability libraries: it uses SHAP or LIME as a backend to compute the
contribution of each feature to each prediction. It then uses what you know about the model and its
data (encoders, feature names, labels) to present those contributions in a readable form.

.. image:: _static/shapash-diagram.png
   :width: 700px
   :align: center
   :alt: How Shapash works

Shapash needs only a model and a dataset. Every other parameter is optional, but the more you provide
(feature names, label names, encoders), the more understandable the outputs.

Additional scikit-learn models
-----------------------------

The following estimators are registered for feature validation and prediction,
including preprocessing with a supported ``ColumnTransformer``:

- Trees: ``DecisionTreeClassifier``, ``DecisionTreeRegressor``,
  ``ExtraTreeClassifier``, ``ExtraTreeRegressor``,
  ``HistGradientBoostingClassifier`` and ``HistGradientBoostingRegressor``.
- Regularized regression: ``Ridge``, ``RidgeCV``, ``Lasso``, ``LassoCV``,
  ``ElasticNet`` and ``ElasticNetCV``.
- Bayesian regression: ``BayesianRidge`` and ``ARDRegression``.
- Robust and quantile regression: ``HuberRegressor``, ``TheilSenRegressor``
  and ``QuantileRegressor``.
- Incremental regression: ``SGDRegressor`` and ``PassiveAggressiveRegressor``.
  The latter is deprecated in scikit-learn 1.8; prefer ``SGDRegressor`` for new models.
- Other linear regression: ``Lars``, ``LarsCV``, ``LassoLars``, ``LassoLarsCV``,
  ``LassoLarsIC``, ``OrthogonalMatchingPursuit`` and ``OrthogonalMatchingPursuitCV``.
- Classification with cross-validation: ``LogisticRegressionCV``.
- Linear support vector machines: ``LinearSVC`` and ``LinearSVR``.

These additions cover single-output regression and binary or multiclass
classification. They do not add support for multi-target regression or multilabel
classification. The SHAP backend chooses a specialized explainer when supported
by the installed SHAP version, otherwise a model-agnostic explainer, which may be
slower. In particular, multiclass histogram gradient boosting may use this fallback.

For classifiers without ``predict_proba``, such as ``LinearSVC``, the default SHAP
fallback explains ``decision_function`` scores rather than predicted labels.
``SmartPredictor`` can compute predictions and contributions without probabilities;
calling ``predict_proba()`` explicitly still raises an error. Scores and their
contributions must not be interpreted as probabilities.

A compiled ``SmartExplainer`` can be exported using ``to_smartpredictor()``,
including when it uses a ``ColumnTransformer``. The exported predictor accepts
the original raw input column names in ``add_input()``, as a DataFrame or a
dictionary. It retains the transformer-prefixed names internally for feature
labels, groups, postprocessing and explanations. DataFrames using those internal
names remain accepted. Columns explicitly dropped by the transformer are optional
in raw input; unknown columns are rejected.

Quick start
-----------

Step 1: declare a SmartExplainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is the only required parameter.

.. code-block:: python

    from shapash import SmartExplainer

    xpl = SmartExplainer(
        model=regressor,
        features_dict=house_dict,  # Optional: readable feature names
        preprocessing=encoder,  # Optional: used to display the original feature values
        postprocessing=postprocess,  # Optional: see the postprocessing tutorial
    )

Step 2: compile the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset is the only required parameter.

.. code-block:: python

    xpl.compile(
        x=X_test,
        y_pred=y_pred,  # Optional: your own predictions (default: model.predict)
        y_target=y_test,  # Optional: displays true values next to predictions
        additional_data=X_additional,  # Optional: extra features shown in the web app
        additional_features_dict=features_dict_additional,  # Optional: names of the extra features
    )

Step 3: display the results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot the results in a notebook, for example the feature importance:

.. code-block:: python

    xpl.plot.features_importance()

Or launch the web app:

.. code-block:: python

    app = xpl.run_app()

Step 4: generate the Shapash Report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The report is a standalone HTML file. It needs the ``report`` extra (see :doc:`installation-instructions/index`).
A YAML file can customize its sections and blocks.

.. code-block:: python

    xpl.generate_report(
        output_file="path/to/output/report.html",
        x_train=X_train,
        y_train=y_train,
        y_test=y_test,
        yaml_path="path/to/report_config.yml",  # Optional: custom layout
    )

Step 5: deploy with a SmartPredictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``SmartPredictor`` is a lighter object than ``SmartExplainer``, made for deployment, with additional
consistency checks on the input data. It computes predictions and detailed or summarized local
explanations, through an API or in batch mode.

.. code-block:: python

    predictor = xpl.to_smartpredictor()
    predictor.save("path/to/predictor.pkl")

Next steps
----------

The :doc:`tutorials` cover each feature in detail, and the :doc:`autodocs/index` documents every
parameter.
