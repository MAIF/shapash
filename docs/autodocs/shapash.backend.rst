Backends
========

A backend computes the contributions that Shapash displays. Pass one to ``SmartExplainer`` with the
``backend`` argument, as a name (``"shap"``, ``"lime"``) or an instance. Subclass ``BaseBackend`` to
write your own.

.. autoclass:: shapash.backend.BaseBackend

.. autoclass:: shapash.backend.ShapBackend

.. autoclass:: shapash.backend.LimeBackend
