API reference
=============

The public API of Shapash. Each entry links to its full documentation.

Tabular data
------------

.. currentmodule:: shapash

.. autosummary::

   explainer.smart_explainer.SmartExplainer
   explainer.smart_plotter.SmartPlotter
   explainer.consistency.Consistency
   explainer.smart_predictor.SmartPredictor
   utils.load_smartpredictor.load_smartpredictor
   data.data_loader.data_loading

Backends
--------

.. autosummary::

   backend.BaseBackend
   backend.ShapBackend
   backend.LimeBackend

Text data (NLP)
---------------

.. autosummary::

   explainer.nlp_explainer.NlpExplainer
   explainer.nlp_explanation.NlpExplanation
   explainer.nlp_plotter.NlpPlotter
   webapp.nlp_app.NlpWebApp
   model.HFClassifierModel
   model.HFPipelineModel
   backend.NlpShapBackend
   backend.NlpLimeBackend
   backend.NlpCaptumLigBackend

.. toctree::
   :maxdepth: 2
   :hidden:

   shapash.explainer
   shapash.predictor
   shapash.load_smartpredictor
   shapash.data
   shapash.backend
   shapash.nlp
