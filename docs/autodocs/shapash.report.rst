Report customization
====================

``SmartExplainer.generate_report`` builds the Shapash Report from a YAML layout of blocks. Export a
default layout with ``export_report_yml`` and edit it, or add your own blocks: subclass
``ReportBlockMixin``, add ``block_<name>`` methods decorated with ``block``, and pass an instance as
``block_instance``. The ``report`` extra is required.

.. autoclass:: shapash.report.ReportTemplate
   :members:
   :undoc-members:

.. autofunction:: shapash.report.export_report_yml

.. autoclass:: shapash.report.blocks.ReportBlockMixin

.. autofunction:: shapash.report.blocks.block
