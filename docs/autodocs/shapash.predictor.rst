SmartPredictor Object
=========================

The **SmartPredictor** Object from the library, is provided to also use the library for production project.
With several methods, it allows users to get predictions and configured summary of the local explanation
for each new dataset.

It can be used in API mode and batch mode.

.. autoclass:: shapash.explainer.smart_predictor.SmartPredictor
   :members: add_input, detail_contributions, summarize, modify_mask, save
   :undoc-members:
   :show-inheritance:
