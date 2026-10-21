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
