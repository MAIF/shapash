"""``Embedding`` — one vector per text, with the provenance needed to pair it back to its texts.

The companion artifact to :class:`~shapash.explainer.nlp_explanation.NlpExplanation`. The same class
holds raw model output (``reducer_tag=None``) and a 2-D projection of it (``reducer_tag`` names the
reducer): same shape ``(n, k)``, same provenance, same file. It is saved separately from the
explanation because it has a different key — *(texts, model, space)* rather than
*(texts, model, backend)* — and can be replaced without any contribution changing.

Persistence is a single ``.npz`` (the vectors plus a JSON ``meta`` entry), read back with
``allow_pickle=False``.

Lives under ``shapash/compute`` because :class:`~shapash.compute.embedding_store.EmbeddingStore`
returns it, and nothing in ``shapash/compute`` may depend on ``shapash/explainer``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from shapash.__version__ import __version__ as _shapash_version


@dataclass(frozen=True, eq=False, slots=True)
class Embedding:
    """Vectors over a corpus, carrying the model, space and corpus they came from.

    Parameters
    ----------
    vectors : np.ndarray
        ``(n_samples, n_components)``. Sealed read-only on construction, like the arrays of
        ``NlpExplanation``: the store hands the same instance to every caller.
    model_id : str
        The model's own identity string (see :attr:`~shapash.model.base.EmbeddingSource.model_id`).
    space : str
        The resolved representation space the vectors were read from (e.g. ``"pooled"``).
    corpus_id : str
        Digest of the texts alone (:func:`~shapash.compute.hashing.hash_corpus`). Row ``i`` is text
        ``i``, so this is what pairs the embedding with an explanation of the same texts.
    reducer_tag : str or None, optional
        ``None`` for raw model output; otherwise the name of the reducer that produced these vectors.

    Examples
    --------
    >>> emb = xpl.compute_embeddings(explanation)      # doctest: +SKIP
    >>> proj = emb.project()                           # PCA(2) by default  # doctest: +SKIP
    >>> proj.save("amazon.proj.npz")                   # doctest: +SKIP
    """

    vectors: np.ndarray
    model_id: str
    space: str
    corpus_id: str
    reducer_tag: str | None = None

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D (n_samples, n_components), got shape {vectors.shape}")
        vectors.setflags(write=False)
        object.__setattr__(self, "vectors", vectors)

    @property
    def n_samples(self) -> int:
        """Number of texts — rows of :attr:`vectors`."""
        return int(self.vectors.shape[0])

    @property
    def n_components(self) -> int:
        """Dimensionality — columns of :attr:`vectors`. 2 for a plottable projection."""
        return int(self.vectors.shape[1])

    def project(self, reducer: Any = None, **fit_transform_kwargs: Any) -> Embedding:
        """Reduce these vectors with *reducer*, returning another :class:`Embedding`.

        Model-free and cheap compared with embedding: it only reads :attr:`vectors`. Most reducers
        other than PCA are stochastic, so :meth:`save` a layout you want to keep.

        Parameters
        ----------
        reducer : object, optional
            Anything with ``fit_transform`` (sklearn's ``PCA``/``TSNE``, ``pacmap.PaCMAP``,
            ``umap.UMAP``). Defaults to ``PCA(n_components=2)``.
        **fit_transform_kwargs
            Passed through to ``reducer.fit_transform`` (e.g. PaCMAP's ``init="pca"``).

        Returns
        -------
        Embedding
            Same model, space and corpus; reduced vectors, ``reducer_tag`` set to the reducer's
            class name.

        Raises
        ------
        ValueError
            If the reducer does not return one row per text.
        """
        if reducer is None:
            reducer = PCA(n_components=2)
        coords = np.asarray(reducer.fit_transform(self.vectors, **fit_transform_kwargs))
        if coords.ndim != 2 or coords.shape[0] != self.n_samples:
            raise ValueError(
                f"a reduction must keep one row per text: got shape {coords.shape} from {self.n_samples} samples."
            )
        return Embedding(
            vectors=coords,
            model_id=self.model_id,
            space=self.space,
            corpus_id=self.corpus_id,
            reducer_tag=type(reducer).__name__.lower(),
        )

    def save(self, path: str | Path) -> None:
        """Write this embedding to a single ``.npz`` file, exactly at *path* (no suffix appended)."""
        meta = {
            "shapash_version": _shapash_version,
            "model_id": self.model_id,
            "space": self.space,
            "corpus_id": self.corpus_id,
            "reducer_tag": self.reducer_tag,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Through an open handle: np.savez appends ".npz" to a path given by name that lacks it.
        with path.open("wb") as fh:
            np.savez(fh, vectors=self.vectors, meta=json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> Embedding:
        """Read an embedding written by :meth:`save` (or cached by ``EmbeddingStore``)."""
        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
            vectors = archive["vectors"]
        return cls(
            vectors=vectors,
            model_id=meta["model_id"],
            space=meta["space"],
            corpus_id=meta["corpus_id"],
            reducer_tag=meta.get("reducer_tag"),
        )

    def __repr__(self) -> str:
        # model_id and corpus_id are identity strings for cache lookups (a long fingerprint, an
        # md5 digest) — useful to compare, not to read. reducer_tag says whether this is a
        # plottable 2-D layout or raw model output, which is what you'd actually check by eye;
        # space is shown only for raw vectors, since a projection's reducer already implies one.
        if self.reducer_tag:
            return (
                f"Embedding(n_samples={self.n_samples}, n_components={self.n_components}, reducer={self.reducer_tag!r})"
            )
        return f"Embedding(n_samples={self.n_samples}, n_components={self.n_components}, space={self.space!r})"


def projection_coords(projection: Embedding | np.ndarray, explanation: Any) -> np.ndarray:
    """Check *projection* can be drawn for *explanation* and return its ``(n_samples, 2)`` coordinates.

    The single check shared by ``explanation.plot.scatter`` and the webapp. An :class:`Embedding` must
    come from the same texts (``corpus_id``) — otherwise every point is labelled from the wrong
    sample. A bare array is accepted on its shape alone, as the escape hatch for coordinates produced
    outside shapash.

    Parameters
    ----------
    projection : Embedding or np.ndarray
        The 2-D projection.
    explanation : NlpExplanation
        The explanation it is drawn with (anything with ``n_samples`` and ``corpus_id``).

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError
        If the projection is over different texts or is not ``(n_samples, 2)``.
    """
    if isinstance(projection, Embedding):
        if projection.corpus_id != explanation.corpus_id:
            raise ValueError(
                "projection was built from different texts than the explanation "
                f"(projection.corpus_id={projection.corpus_id[:12]}…, "
                f"explanation.corpus_id={explanation.corpus_id[:12]}…)."
            )
        coords = projection.vectors
    else:
        coords = np.asarray(projection)
    expected = (explanation.n_samples, 2)
    if coords.shape != expected:
        raise ValueError(
            f"projection must have shape {expected}, got {coords.shape}. "
            "Reduce an embedding to 2-D first: embedding.project(reducer)."
        )
    return coords
