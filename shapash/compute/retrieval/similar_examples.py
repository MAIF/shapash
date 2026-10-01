"""Similar-example retrieval: nearest reference examples in a model's representation space.

For a prediction, this retrieves the reference-corpus examples whose internal representation is most
similar to the query's — the examples the model treats most alike. It is the in-house counterpart of
Captum's ``SimilarityInfluence``, reimplemented over the capability-based model layer
(:class:`~shapash.model.base.SupportsEmbeddings`) so it needs no per-model wiring and carries none of
Captum 0.9.0's single-layer / short-final-batch quirks. The cost splits cleanly:

* **Bank** — one vector per reference example, computed once and cached by
  :class:`~shapash.compute.embedding_store.EmbeddingStore` (keyed by the corpus hash, space, and model
  id). This is the expensive, amortizable part, and the same cache the 2-D scatter draws from.
* **Query** — one vector for the query text, then cosine similarity against the bank and a top-k
  selection. Milliseconds, so it runs live per selection in the webapp.

This is a *representation-similarity* (nearest-neighbour) method, **not** a leave-one-out / influence-
function measure: it surfaces the examples most alike in decision space, a cheap and faithful proxy for
"what shaped this prediction", not a causal retraining attribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shapash.compute.embedding_store import EmbeddingStore
from shapash.model.base import EmbeddingSource, SupportsEmbeddings, has_capabilities

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neighbor:
    """One retrieved reference example for a query.

    Attributes
    ----------
    index : int
        Position of the example within the reference corpus.
    score : float
        Cosine similarity to the query in the model's representation space (higher = more similar).
    text : str
        The reference example's text.
    label : str or None
        The reference example's label, when the corpus was built with labels.
    """

    index: int
    score: float
    text: str
    label: str | None = None


class SimilarExampleRetriever:
    """Retrieve the reference examples most similar to a query in the model's representation space.

    Parameters
    ----------
    model : EmbeddingSource
        A model exposing :meth:`~shapash.model.base.SupportsEmbeddings.embed` plus the identity and
        space resolution the embedding cache keys on (see
        :class:`~shapash.model.base.EmbeddingSource`). Admission is still checked by *capability*
        (``SupportsEmbeddings``) below.
    reference_texts : list[str]
        The corpus to retrieve from (typically the model's training set).
    reference_labels : list[str] or None, optional
        Labels aligned with ``reference_texts`` — surfaced on each :class:`Neighbor` when present.
    cache_dir : str or Path or None, optional
        When given, the reference embeddings are persisted here by
        :class:`~shapash.compute.embedding_store.EmbeddingStore` and reloaded on later runs, so only
        the first build pays the cost.

    Notes
    -----
    The comparison space is deliberately **not** a parameter here: it is whatever the model's
    :attr:`~shapash.model.encoder.EncoderClassifierModel.embedding_space` currently is. Letting a
    caller override it per-retriever allowed the scatter and the neighbours to sit in different spaces
    — you would select a cluster in one space and rank neighbours in another, with nothing saying so.
    Change the model's space to move both together.

    This matches the 2-D scatter *when the scatter is built by*
    :meth:`~shapash.explainer.nlp_explainer.NlpExplainer.compute_projection`, which reads the same
    space through the same :class:`~shapash.compute.embedding_store.EmbeddingStore`. Coordinates passed
    in by hand via ``run_app(projection=...)`` are outside that guarantee.

    Examples
    --------
    >>> retriever = SimilarExampleRetriever(model, train_texts, train_labels)
    >>> for n in retriever.query("i feel wonderful today", top_k=5):
    ...     print(n.score, n.label, n.text)
    """

    def __init__(
        self,
        model: EmbeddingSource,
        reference_texts: list[str],
        reference_labels: list[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not has_capabilities(model, SupportsEmbeddings):
            raise TypeError(
                f"{type(model).__name__} does not support embeddings (SupportsEmbeddings); "
                "similar-example retrieval is unavailable."
            )
        if reference_labels is not None and len(reference_labels) != len(reference_texts):
            raise ValueError(
                f"reference_labels length ({len(reference_labels)}) must match "
                f"reference_texts length ({len(reference_texts)})."
            )
        self.model = model
        self.reference_texts = list(reference_texts)
        self.reference_labels = list(reference_labels) if reference_labels is not None else None
        self.store = EmbeddingStore(model, self.reference_texts, cache_dir=cache_dir)
        self._bank: np.ndarray | None = None  # (n_reference, hidden), L2-normalized rows
        self._bank_space: str | None = None  # store.space_key the bank was built under — see build()
        self._positions: dict[str, list[int]] | None = None  # text -> reference rows, for exclude_self

    @property
    def size(self) -> int:
        """Number of reference examples."""
        return len(self.reference_texts)

    def build(self) -> None:
        """Compute (or load) and L2-normalize the reference embedding bank.

        Called automatically on the first :meth:`query`; call it up front to pay the cost eagerly
        (e.g. at app start). The raw vectors come from the shared
        :class:`~shapash.compute.embedding_store.EmbeddingStore`; normalization is applied here rather
        than cached, since it is a single cheap pass and keeps the stored artifact the plain embeddings
        that other consumers (e.g. the scatter projection) can reuse. Rows end up unit-norm so a query
        is a single matrix-vector product.

        Idempotent *while the model stays in one space*. The space is assignable at runtime
        (:attr:`~shapash.model.encoder.EncoderClassifierModel.embedding_space`), and a bank held over
        such a switch would be compared against queries encoded in the new space — a cosine between
        two unrelated spaces, which returns a plausible-looking number rather than failing. So the
        bank records the space it was built under and is rebuilt when that moves. The comparison is
        on ``space_key`` (two string reads) rather than the store's full key (an MD5 over the whole
        corpus), which is sound here because the corpus is fixed at construction: only the space can
        change under a given retriever.
        """
        space = self.store.space_key
        if self._bank is not None and self._bank_space == space:
            return
        self._bank = _l2_normalize(self.store.vectors())
        self._bank_space = space

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` in the model's configured space — the same one the scatter projects."""
        return self.model.embed(texts)

    def query(self, text: str, top_k: int = 5) -> list[Neighbor]:
        """Return the ``top_k`` reference examples most similar to ``text``, most-similar first.

        Parameters
        ----------
        text : str
            The query text (a dataset row, an edited sentence, or a counterfactual).
        top_k : int
            Number of neighbours to return (clamped to the corpus size).

        Returns
        -------
        list[Neighbor]
            Neighbours ordered by descending cosine similarity.
        """
        self.build()
        assert self._bank is not None  # noqa: S101 - build() guarantees this
        query_vec = _l2_normalize(self._encode([text]))[0]
        sims = self._bank @ query_vec  # cosine, rows already unit-norm
        return self._top_k(sims, top_k)

    def query_threshold(self, text: str, threshold: float, limit: int | None = None) -> tuple[list[Neighbor], int]:
        """Return every reference example scoring above ``threshold``, most-similar first.

        Unlike :meth:`query`, the candidate set is not a fixed size: any reference example whose
        cosine similarity to ``text`` exceeds ``threshold`` qualifies, however many that is. This is
        the counterpart the webapp's threshold-filter mode needs — :meth:`query`'s ``top_k`` is an
        ``argpartition`` over a fixed count and cannot tell you "how many clear 0.95", only "the k
        best, whatever their score".

        Parameters
        ----------
        text : str
            The query text.
        threshold : float
            Minimum cosine similarity a reference example must *exceed* (strict) to qualify.
        limit : int or None, optional
            Cap on how many matches to return, most-similar first. ``None`` returns all of them —
            callers rendering a live list should pass a display cap, since a low threshold can match
            most of the corpus.

        Returns
        -------
        tuple[list[Neighbor], int]
            The matching neighbours (capped to ``limit``) and the total number that cleared
            ``threshold`` before capping, so a caller can report e.g. "showing 50 of 138".
        """
        self.build()
        assert self._bank is not None  # noqa: S101 - build() guarantees this
        query_vec = _l2_normalize(self._encode([text]))[0]
        sims = self._bank @ query_vec  # cosine, rows already unit-norm
        return self._above_threshold(sims, threshold, limit)

    def query_many(self, texts: list[str], top_k: int = 5, exclude_self: bool = False) -> list[list[Neighbor]]:
        """Return the ``top_k`` most similar reference examples for **each** of ``texts``.

        The batch counterpart of :meth:`query`, for callers ranking a whole set at once (e.g.
        corroborating a list of suspected label errors). One :meth:`~shapash.model.base.SupportsEmbeddings.embed`
        call — which the model chunks internally — plus a single matrix product against the bank,
        rather than one forward pass per text.

        Parameters
        ----------
        texts : list of str
            The query texts.
        top_k : int
            Number of neighbours per query (clamped to the corpus size).
        exclude_self : bool, optional
            Drop reference examples whose text is *exactly equal* to the query. Set this when the
            queries may themselves be in the reference corpus, where every text would otherwise
            retrieve itself as its own nearest neighbour and crowd out a real one. Matching on text
            equality rather than a similarity cutoff keeps genuine near-duplicates in the results.

        Returns
        -------
        list[list[Neighbor]]
            One neighbour list per query, in the order of ``texts``, each ordered by descending
            cosine similarity.
        """
        if not texts:
            return []
        self.build()
        assert self._bank is not None  # noqa: S101 - build() guarantees this
        query_vecs = _l2_normalize(self._encode(list(texts)))  # (n_queries, hidden)
        sims = query_vecs @ self._bank.T  # (n_queries, n_reference)
        if exclude_self:
            positions = self._text_positions()
            for row, text in enumerate(texts):
                hits = positions.get(text)
                if hits:
                    sims[row, hits] = -np.inf
        return [self._top_k(sims[row], top_k) for row in range(sims.shape[0])]

    def _text_positions(self) -> dict[str, list[int]]:
        """Reference text to the rows holding it, built once (supports ``exclude_self``)."""
        if self._positions is None:
            positions: dict[str, list[int]] = {}
            for i, text in enumerate(self.reference_texts):
                positions.setdefault(text, []).append(i)
            self._positions = positions
        return self._positions

    def _top_k(self, sims: np.ndarray, top_k: int) -> list[Neighbor]:
        """Turn one row of similarities into its ``top_k`` :class:`Neighbor` objects, best first."""
        k = max(1, min(top_k, self.size))
        # argpartition for the top-k, then sort just those descending.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        # An excluded row is -inf; it only surfaces when the corpus has nothing else to offer.
        top_idx = top_idx[np.isfinite(sims[top_idx])]
        return self._neighbors_from_indices(top_idx, sims)

    def _above_threshold(self, sims: np.ndarray, threshold: float, limit: int | None) -> tuple[list[Neighbor], int]:
        """Every finite row exceeding ``threshold``, descending, plus the pre-cap match count."""
        idx = np.flatnonzero(np.isfinite(sims) & (sims > threshold))
        idx = idx[np.argsort(-sims[idx])]
        total = len(idx)
        if limit is not None:
            idx = idx[:limit]
        return self._neighbors_from_indices(idx, sims), total

    def _neighbors_from_indices(self, idx: np.ndarray, sims: np.ndarray) -> list[Neighbor]:
        """Build :class:`Neighbor` objects for ``idx``, in the order given."""
        return [
            Neighbor(
                index=int(i),
                score=float(sims[i]),
                text=self.reference_texts[i],
                label=self.reference_labels[i] if self.reference_labels is not None else None,
            )
            for i in idx
        ]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (zero rows left as zeros)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
