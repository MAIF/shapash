Text data (NLP)
===============

Explainability for text classifiers, with its own entry point: ``NlpExplainer``. It needs the ``nlp``
extra (``pip install shapash[nlp]``).

Explainer and explanation
-------------------------

.. autoclass:: shapash.explainer.nlp_explainer.NlpExplainer

.. autoclass:: shapash.explainer.nlp_explanation.NlpExplanation

.. autoclass:: shapash.explainer.nlp_plotter.NlpPlotter

.. autofunction:: shapash.explainer.nlp_comparison.corpus_agreement

.. autoclass:: shapash.compute.embeddings.Embedding

Webapp
------

.. autoclass:: shapash.webapp.nlp_app.NlpWebApp

Text models
-----------

.. autoclass:: shapash.model.HFClassifierModel

.. autoclass:: shapash.model.HFPipelineModel

.. autoclass:: shapash.model.EncoderClassifierModel

.. autoclass:: shapash.model.TorchClassifierModel

.. autoclass:: shapash.model.SentenceTransformerModel

.. autoclass:: shapash.model.TextModel

.. autofunction:: shapash.model.has_capabilities

Backends
--------

.. autoclass:: shapash.backend.NlpBackend

.. autoclass:: shapash.backend.NlpShapBackend

.. autoclass:: shapash.backend.NlpLimeBackend

.. autoclass:: shapash.backend.NlpCaptumLigBackend
