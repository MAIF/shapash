"""Compute layer — heavy computation behind modality-agnostic interfaces.

Seeded by the NLP prototype. Today it holds the ``generators`` sub-package (counterfactual / what-if
generation), the ``retrieval`` sub-package (similar-example lookup), and
:mod:`~shapash.compute.embedding_store` — the shared cache both retrieval and the 2-D scatter draw
their vectors from. ``generators`` is a *new* sibling axis for generative components (LIT's
``Generator``, as opposed to explanation ``Interpreter``s).
"""
