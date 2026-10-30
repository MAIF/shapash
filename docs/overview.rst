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

Open ``http://127.0.0.1:8050/`` in a browser on the same machine as Python.
By default, the server only accepts local connections.
For local-only listening addresses, the startup message provides a network URL
to try only after restarting the app with network access enabled.

To access the app from another machine on a trusted network:

.. code:: ipython

    app = xpl.run_app(host="0.0.0.0", port=8050)

In that machine's browser, open ``http://<server-ip-or-hostname>:8050/``.
When listening on all interfaces, Shapash prints both a local browser URL and
a network URL based on the server's hostname.
The server hostname must resolve from the browser's machine, and the firewall
must allow port 8050. ``0.0.0.0`` is an IPv4 listening address, not a browser
destination. For IPv6, listen with ``host="::"`` and use
``http://[<server-ipv6-address>]:8050/``.

If Python runs in a remote notebook, container, or VM, ``localhost`` in your
browser refers to your own machine. Configure port forwarding, publish the
container port, or use your notebook platform's proxy URL as appropriate;
listening on ``0.0.0.0`` alone does not make the app reachable.
Shapash cannot infer the external URL assigned by a proxy or port forwarding.
Only expose this development server on a trusted network.
Stop the previous app with ``app.kill()`` before restarting it on the same port.

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
