"""Dataset-quality diagnostics — what is wrong with the *data*, not with the model.

Where ``compute/generators/`` perturbs an input and ``compute/retrieval/`` looks outward to a
reference corpus, this package audits the labelled corpus itself. The first method is
:func:`~shapash.compute.diagnostics.label_noise.detect_label_issues` — confident learning
(Northcutt et al., JAIR 2021) over the predicted probabilities an explainer has already computed —
paired with :class:`~shapash.compute.diagnostics.label_probe.LabelProbe`, the model-independent
second opinion that says whether a flagged row is a label error or just a model error.

This is a peer axis to ``generators`` and ``retrieval``, seeded by the NLP prototype. The master
plan reserves ``compute/analyses/{local,global,diagnostics}`` for the interpreter hierarchy and
assigns the design of that namespace to the tabular modality; when it lands, this package is
expected to move under it unchanged. Sitting here in the meantime keeps NLP from originating a
namespace it does not own.
"""

from __future__ import annotations

from shapash.compute.diagnostics.label_noise import (
    LabelIssue,
    LabelNoiseReport,
    detect_label_issues,
    has_usable_probabilities,
)
from shapash.compute.diagnostics.label_probe import LabelProbe, ProbeVerdict

__all__ = [
    "LabelIssue",
    "LabelNoiseReport",
    "LabelProbe",
    "ProbeVerdict",
    "detect_label_issues",
    "has_usable_probabilities",
]
