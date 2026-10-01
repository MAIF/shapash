"""Corpus digests shared by every NLP cache key and by ``corpus_id``.

A dependency-free leaf module, so the persistence layer (``NlpExplanation``) can compute a corpus id
without importing the model or embedding layers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def hash_corpus(texts: Sequence[str], key: str = "") -> str:
    """Stable digest over an identity ``key`` plus the corpus texts (order-sensitive).

    Each string is framed with its own byte length before hashing, so no separator choice can
    let two different inputs collide — a literal ``"\\0"`` inside a text can no longer make
    ``["a\\0b"]`` and ``["a", "b"]`` hash identically. ``key`` carries whatever else changes the
    cached artifact (model identity, representation space, explanation backend).

    With the default empty ``key`` the digest depends on the texts alone: that is
    :attr:`~shapash.explainer.nlp_explanation.NlpExplanation.corpus_id`, the value that pairs an
    embedding or projection with the explanation of the same texts.
    """
    h = hashlib.md5(usedforsecurity=False)
    for s in (key, *texts):
        data = s.encode()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()
