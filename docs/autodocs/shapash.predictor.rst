SmartPredictor Object
=========================

The **SmartPredictor** Object allows to:
    - compute predictions
    - configure summary of the local explanation
    - deploy interpretability of your model for operational needs

It can be used in API mode and batch mode.

.. autoclass:: shapash.explainer.smart_predictor.SmartPredictor
   :members: add_input, predict, predict_proba, detail_contributions, summarize, modify_mask, save
   :undoc-members:
   :show-inheritance:
