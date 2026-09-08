"""Composable webapp components for the NLP what-if tools.

Prototype of the master plan's Phase-5b ``WebappComponent`` contract, applied first to the
interactive what-if panels (data editor + counterfactuals). Each component owns its layout and
callbacks and declares its ``requires`` capabilities, self-disabling when the bound explanation/engine
cannot satisfy them.

Notably this **extends** the Phase-5b ``requires`` idea from *data* capabilities (e.g.
``proba_values``) to *engine/model* capabilities (``engine:predict``, ``engine:counterfactual``),
so a panel can require a live, gradient-capable model — the refinement to fold back into the master
contract.
"""

from __future__ import annotations

from shapash.webapp.nlp_components.base import (
    WebappComponent,
    available_capabilities,
    compose_selection,
    error_mask,
    error_positions,
)
from shapash.webapp.nlp_components.counterfactual import CounterfactualComponent
from shapash.webapp.nlp_components.data_editor import DataEditorComponent
from shapash.webapp.nlp_components.datapoint import (
    datapoint_from_contributions,
    pack_datapoint,
    unpack_datapoint,
)
from shapash.webapp.nlp_components.error_analysis import ErrorAnalysisComponent
from shapash.webapp.nlp_components.label_noise import LabelNoiseComponent
from shapash.webapp.nlp_components.scatter import ScatterComponent
from shapash.webapp.nlp_components.sentence_highlight import SentenceHighlightComponent
from shapash.webapp.nlp_components.similar_examples import SimilarExamplesComponent
from shapash.webapp.nlp_components.waterfall import WaterfallComponent
from shapash.webapp.nlp_components.word_importance import WordImportanceComponent
from shapash.webapp.nlp_components.word_profile import WordProfileComponent

__all__ = [
    "WebappComponent",
    "available_capabilities",
    "compose_selection",
    "error_mask",
    "error_positions",
    "DataEditorComponent",
    "CounterfactualComponent",
    "ErrorAnalysisComponent",
    "LabelNoiseComponent",
    "ScatterComponent",
    "SentenceHighlightComponent",
    "SimilarExamplesComponent",
    "WaterfallComponent",
    "WordImportanceComponent",
    "WordProfileComponent",
    "pack_datapoint",
    "unpack_datapoint",
    "datapoint_from_contributions",
]
