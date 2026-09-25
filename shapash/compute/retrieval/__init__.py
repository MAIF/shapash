"""Reference-example retrieval: find the corpus examples most like a prediction's input.

Where ``compute/generators/`` perturbs an input (counterfactuals), this package looks *outward* to a
reference corpus and answers "which examples most resemble this one?". The first method is
:class:`~shapash.compute.retrieval.similar_examples.SimilarExampleRetriever` — cosine similarity
between a query and a corpus in the model's representation space (a lightweight, in-house take on
Captum's ``SimilarityInfluence``, keyed to the capability-based model layer).
"""

from __future__ import annotations

from shapash.compute.retrieval.similar_examples import Neighbor, SimilarExampleRetriever

__all__ = ["Neighbor", "SimilarExampleRetriever"]
