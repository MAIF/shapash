"""Cached embeddings for a fixed corpus in a model's current representation space.

Embedding a corpus is the expensive, repeated step behind more than one feature: similar-example
retrieval needs one vector per reference text, and the 2-D scatter needs one vector per compiled
text. Both are a pure function of *(model identity, effective space, corpus)*, so both belong behind
one cache with one key — otherwise each caller invents its own filename and they drift, which is how
a bank built in the ``"decision"`` space ends up reloaded for a scatter drawn in ``"pooled"``.

Entries are :class:`~shapash.compute.embeddings.Embedding` files, so a cache entry can be copied out
and loaded with :meth:`Embedding.load`. Projections are not cached here: reducing is cheap next to
embedding, and a layout worth keeping is saved explicitly with :meth:`Embedding.save`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from shapash.compute.embeddings import Embedding
from shapash.compute.hashing import hash_corpus
from shapash.model.base import EmbeddingSource

__all__ = ["EmbeddingStore"]

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """One vector per text, computed once and cached to disk.

    Parameters
    ----------
    model : EmbeddingSource
        A model exposing ``model_id``, ``resolve_space`` and ``embed``.
    texts : sequence of str
        The corpus to embed. Fixed for the lifetime of the store — the cache key is derived from it.
    cache_dir : str or Path or None, optional
        Where cached ``.npz`` files live. When ``None`` the store still memoizes in memory but writes
        nothing, so a fresh process recomputes.

    Notes
    -----
    The key is ``model_id | resolve_space() | corpus-hash``. ``model_id`` is the model's own
    declaration of what makes it distinct (checkpoint, pooling, normalization, head weights) and
    ``resolve_space()`` is the single place that knows which space :meth:`embed` will *actually* use,
    so ``None`` (meaning "the model's default") can never collide with the default it stands for.

    Examples
    --------
    >>> store = EmbeddingStore(model, train_texts, cache_dir="cache/")
    >>> store.vectors().shape
    (5000, 384)
    """

    def __init__(
        self,
        model: EmbeddingSource,
        texts: Sequence[str],
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.texts = list(texts)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Keyed by the full key, not a single slot: the key folds in the model's *current* space,
        # which is assignable at runtime (EncoderClassifierModel.embedding_space). A single slot would
        # answer from the old space after such a switch.
        self._memo: dict[str, Embedding] = {}

    @property
    def space_key(self) -> str:
        """Readable ``model_id|space`` half of the key — what gets logged when a lookup misses."""
        return f"{self.model.model_id}|{self.model.resolve_space()}"

    @property
    def key(self) -> str:
        """The cache key: model identity, effective space, and corpus digest."""
        return hash_corpus(self.texts, self.space_key)

    @property
    def path(self) -> Path | None:
        """On-disk location of the cached embeddings, or ``None`` when caching is off."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.key}.emb.npz"

    def vectors(self) -> np.ndarray:
        """Return ``(n_texts, hidden_dim)`` embeddings in the model's current space."""
        return self.embedding().vectors

    def embedding(self) -> Embedding:
        """Return this corpus's embeddings, loading them from cache or computing and caching them."""
        key = self.key
        if key in self._memo:
            return self._memo[key]

        cache_file = self.path
        if cache_file is not None and cache_file.exists():
            logger.info("Embedding store hit — loading %s", cache_file)
            embedding = Embedding.load(cache_file)
        else:
            logger.info("Embedding store miss — computing over %d texts (%s)", len(self.texts), self.space_key)
            embedding = Embedding(
                vectors=np.asarray(self.model.embed(self.texts)),
                model_id=self.model.model_id,
                space=str(self.model.resolve_space()),
                corpus_id=hash_corpus(self.texts),
            )
            if cache_file is not None:
                embedding.save(cache_file)
                logger.info("Embedding store cached to %s", cache_file)
        self._memo[key] = embedding
        return embedding

    def clear(self) -> None:
        """Drop the cached embeddings — every in-memory entry, and the on-disk file for the current space.

        Files written for a space the model has since moved off are left alone: they are unreachable
        by lookup, and re-selecting that space finds them again.
        """
        self._memo.clear()
        cache_file = self.path
        if cache_file is not None and cache_file.exists():
            cache_file.unlink()
            logger.info("Embedding store dropped %s", cache_file)
