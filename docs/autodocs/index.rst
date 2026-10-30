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

Tabular deep-learning adapters
------------------------------

.. autoclass:: shapash.model.KerasModelAdapter
   :members:
   :inherited-members:

.. autoclass:: shapash.model.TensorFlowModelAdapter
   :members:
   :inherited-members:

.. autoclass:: shapash.model.PyTorchModelAdapter
   :members:
   :inherited-members:

Report and standalone plots
---------------------------

.. autosummary::

   report.ReportTemplate
   report.export_report_yml
   report.blocks.ReportBlockMixin
   report.blocks.block
   plots.plot_univariate.plot_distribution
   plots.plot_correlations.plot_correlations
   plots.plot_evaluation_metrics.plot_confusion_matrix
   plots.plot_feature_importance.plot_feature_importance

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
   shapash.report
   shapash.plots
   shapash.backend
   shapash.nlp
