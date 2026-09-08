"""Prototype NLP explainer for text classification models.

This module is a **bridge prototype** toward Phase 6 of the refactoring plan,
where token-level NLP explanations will be integrated into ``SmartExplainer``
via ``TextDataset`` and ``ExplanationSession``. Once those are in place, users
will call ``SmartExplainer.compile(x=TextDataset(texts))`` and this class will
be removed.

``NlpExplainer`` follows the ``fit``/``explain`` seam (Amendment A5 in
``docs/architecture/refactoring-plan.md``): ``fit(X_reference, y=None)`` learns
reference state (the similar-example / label-noise-probe corpus) and
``explain(X, y=None)`` runs the backend and returns an immutable
:class:`~shapash.explainer.nlp_explanation.NlpExplanation` — the explainer
itself keeps no compiled batch.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_shap_backend import NlpShapBackend
from shapash.compute.diagnostics.label_noise import (
    LabelNoiseReport,
    detect_label_issues,
    has_usable_probabilities,
)
from shapash.compute.diagnostics.label_probe import LabelProbe
from shapash.compute.embedding_store import EmbeddingStore, hash_corpus
from shapash.compute.generators.ablation_flip import AblationFlipGenerator
from shapash.compute.generators.base import Counterfactual, CounterfactualGenerator, Field
from shapash.compute.generators.hotflip import HotFlipGenerator
from shapash.compute.retrieval.similar_examples import Neighbor, SimilarExampleRetriever
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.model.base import SupportsEmbeddings, SupportsTokenization, TextModel, has_capabilities
from shapash.model.hf import HFPipelineModel
from shapash.webapp.nlp_app import NlpWebApp

# Built-in counterfactual generators, in preference order: HotFlip (gradient-based, richer
# substitutions) first, AblationFlip (forward-pass-only removal) as the broader fallback. Every entry
# compatible with the bound model is offered in the webapp's method selector.
_BUILTIN_CF_GENERATORS: tuple[type[CounterfactualGenerator], ...] = (HotFlipGenerator, AblationFlipGenerator)


def _looks_like_pipeline(model: object) -> bool:
    """Heuristic: a HuggingFace ``text-classification`` pipeline is callable and has a tokenizer."""
    return callable(model) and hasattr(model, "tokenizer")


def _cache_file(data_hash: str, cache_dir: Path) -> Path:
    return cache_dir / f"{data_hash}.xpl"


def _reducer_tag(reducer: object) -> str:
    """Name a reducer for a cache tag: its class plus a digest of its parameters when it exposes them.

    Two runs of the same reducer class with different settings produce different coordinates, so the
    class name alone would silently reload the wrong scatter. ``get_params()`` (sklearn's convention,
    which ``pacmap`` also follows) makes the settings part of the tag. A reducer without it is keyed by
    class name only — documented on :meth:`NlpExplainer.compute_projection` as needing ``recompute``.
    """
    name = type(reducer).__name__.lower()
    get_params = getattr(reducer, "get_params", None)
    if get_params is None:
        return name
    digest = hashlib.md5(repr(sorted(get_params().items())).encode(), usedforsecurity=False).hexdigest()
    return f"{name}-{digest[:8]}"


class NlpExplainer:
    """Minimal explainer for text classification — token-level contributions.

    Uses ``NlpShapBackend`` by default.  Pass a pre-built backend instance to
    switch to a different explainer (e.g. ``NlpLimeBackend``).

    Parameters
    ----------
    model : callable
        Text pipeline or callable accepted by the chosen backend.
    label_names : list[str], optional
        Class names in the same order as the model output columns.
        Used in plot titles and the webapp class selector.
    backend : NlpBackend, optional
        Pre-built backend instance.  When provided, ``explainer_args`` and
        ``explainer_compute_args`` are ignored — configure the backend directly
        before passing it in.  Defaults to ``NlpShapBackend``.
    explainer_args : dict, optional
        Forwarded to ``NlpShapBackend.__init__`` when no ``backend`` is given.
    explainer_compute_args : dict, optional
        Forwarded to ``NlpShapBackend.__init__`` when no ``backend`` is given.

    Examples
    --------
    >>> import transformers
    >>> pipe = transformers.pipeline("text-classification", model="...", return_all_scores=True)
    >>> xpl = NlpExplainer(pipe, label_names=["sadness", "joy", "love", "anger", "fear", "surprise"])
    >>> xpl.fit(reference_texts, y=reference_labels)  # optional — see fit()
    >>> explanation = xpl.explain(texts)
    >>> xpl.run_app(explanation, port=8050)

    >>> from shapash.backend.nlp_lime_backend import NlpLimeBackend
    >>> lime_backend = NlpLimeBackend(classifier_fn, label_names=[...], explainer_compute_args={"num_features": 15})
    >>> xpl = NlpExplainer(classifier_fn, label_names=[...], backend=lime_backend)
    """

    def __init__(
        self,
        model,
        label_names: list[str] | None = None,
        backend: NlpBackend | None = None,
        cf_generator: CounterfactualGenerator | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
    ) -> None:
        self.model = model
        self.label_names = label_names

        # Wrap the model in a capability-aware TextModel adapter when possible. A raw HuggingFace
        # pipeline becomes an HFPipelineModel (predict-only); a TextModel is used as-is; anything
        # else (e.g. a LIME classifier_fn) leaves _text_model as None and the legacy path is kept.
        if isinstance(model, TextModel):
            self._text_model: TextModel | None = model
            if label_names is None:
                self.label_names = model.label_names
        elif _looks_like_pipeline(model):
            self._text_model = HFPipelineModel(model, label_names)
        else:
            self._text_model = None

        # Default backend only when the caller brought none. Building it reads the model's SHAP surface
        # (``shap_callable`` + its companion ``shap_masker``: ``None`` for a pipeline-backed model, which
        # SHAP can infer a Text masker from; explicit when the callable is a bare scoring function) —
        # which must not happen when an explicit backend makes those values unused, or an adapter that
        # never intends to be explained by SHAP could not be used at all.
        if backend is not None:
            self.backend: NlpBackend = backend
        else:
            self.backend = NlpShapBackend(
                model=self._text_model.shap_callable if self._text_model is not None else model,
                label_names=self.label_names,
                masker=self._text_model.shap_masker if self._text_model is not None else None,
                explainer_args=explainer_args or {},
                explainer_compute_args=explainer_compute_args or {},
            )

        # Counterfactual generators: explicit > auto-discovered > none. An explicit ``cf_generator``
        # is used verbatim (no surprise additions). Otherwise every built-in compatible with the model
        # is offered — so the webapp can switch methods live — with HotFlip preferred and AblationFlip
        # (forward-pass-only) covering prediction-only models such as a plain pipeline. ``cf_generator``
        # stays the *active/default* generator for callers that don't select one.
        if cf_generator is not None:
            generators: list[CounterfactualGenerator] = [cf_generator]
        elif self._text_model is not None:
            generators = [
                cls(self._text_model) for cls in _BUILTIN_CF_GENERATORS if cls.is_compatible(self._text_model)
            ]
        else:
            generators = []
        self.cf_generators: dict[str, CounterfactualGenerator] = {g.name: g for g in generators}
        self.cf_generator: CounterfactualGenerator | None = generators[0] if generators else None

        # Reference state (similar-example retrieval / label-noise probe corpus) is *fit-time*, not
        # constructor-time — see fit(). Not initialized here at all: every reader already goes through
        # getattr(self, "_retriever"/"reference_"/"_label_probe", None) (see can_find_similar,
        # can_probe_labels, _attach_probe), which is also what keeps the object.__new__ bypass some
        # tests use working without ever calling __init__ or fit().

        # Private memoization for explain() — (hash, NlpExplanation), never assigned to public state
        # otherwise (the compiled batch is only ever returned, never stored on self beyond this key).
        # Excludes anything not determined by (texts, model, backend) — ground truth, class names —
        # so a keyed cache entry can never carry a stale y (explain() overrides texts/y_true on read).
        self._computed_cache: tuple[str, NlpExplanation] | None = None

        # Serializes the live compute ops (predict / explain_text / find_similar / generate) — the
        # webapp runs Dash callbacks on Werkzeug's threaded server, and the shared HF fast tokenizer
        # (used directly *and* via the SHAP pipeline) is not thread-safe: concurrent calls raise
        # "Already borrowed". Re-entrant so explain_text can nest predict on the same thread.
        self._compute_lock = threading.RLock()

    def fit(
        self,
        X_reference: list[str] | pd.Series | None = None,
        y: list[str] | pd.Series | None = None,
        cache_dir: str | Path | None = None,
        precompute: bool = True,
    ) -> NlpExplainer:
        """Learn reference state: the similar-example / label-noise-probe corpus.

        None of the three built-in backends (SHAP, LIME, Captum LIG) have fit-time reference needs
        of their own — see ``NlpBackend.reference_kind`` on each (``"none"``/``"point"``: a masker
        that infers itself, or a point constructed by the tokenizer, neither learned from data). So
        calling ``fit`` is optional; :meth:`explain` works standalone, exactly as ``compile`` did.
        What *is* fit-time reference state for text — the two pieces the plan calls out (see the
        refactoring plan's Context 4a) — is the corpus this method learns:

        Parameters
        ----------
        X_reference : list[str] or pd.Series, optional
            Reference pool for similar-example retrieval — typically the model's training set.
            When given *and* the model supports embeddings, the webapp gains a "Similar Examples"
            panel that retrieves the most similar reference examples for the selected/edited text.
            Ignored when the model cannot embed (e.g. a prediction-only pipeline).
            Neighbours are compared in the model's own ``embedding_space``. To compare in a
            different space, set the model's ``embedding_space`` — that moves the neighbours and
            any scatter built by :meth:`compute_projection` together, since both read the space
            through the same store.
        y : list[str] or pd.Series, optional
            Labels for ``X_reference``. When given, this corpus additionally trains the label-noise
            panel's independent probe (:meth:`can_probe_labels`), which needs no model and so is
            available even when retrieval is not.
        cache_dir : str or Path, optional
            When given, the reference embedding bank is persisted here and reloaded on later runs,
            so only the first call pays the (one-off) cost of embedding the reference corpus.
        precompute : bool, optional
            Embed the reference corpus here rather than on first use (default). The bank is the
            expensive, amortizable half of retrieval — one forward pass per reference example — and
            paying it in ``fit`` is what makes ``fit`` actually *fit*: the cost, and any failure
            (a corpus the model cannot encode, an unwritable ``cache_dir``), surface at the call
            that asked for them instead of inside the first webapp callback that happens to touch
            the panel. Set ``False`` to keep the old lazy behaviour — worth it for a large corpus
            behind a feature you may never open. No effect when no retriever was built.

        Returns
        -------
        NlpExplainer
            ``self``, following the scikit-learn ``fit`` convention.

        Notes
        -----
        Sets ``self.reference_`` (the ``(texts, labels)`` pair, or ``None``) and ``self.classes_``
        (label names — from the model when known, else derived from ``y``) with the sklearn
        trailing-underscore convention for fitted state. Does not subclass ``BaseEstimator``: text
        models here are HuggingFace pipelines / torch models, not sklearn estimators, so there is no
        ``Pipeline``/skrub interop need to justify full conformance (``get_params``/``clone``) the
        way there is for the tabular explainer — see the refactoring plan's Phase 1.
        """
        ref_texts = list(X_reference) if X_reference is not None else None
        ref_labels = list(y) if y is not None else None
        if ref_texts is not None and ref_labels is not None and len(ref_texts) != len(ref_labels):
            raise ValueError(f"y length ({len(ref_labels)}) must match X_reference length ({len(ref_texts)}).")

        self._retriever: SimilarExampleRetriever | None = None
        if ref_texts is not None and has_capabilities(self._text_model, SupportsEmbeddings):
            self._retriever = SimilarExampleRetriever(
                self._text_model,  # type: ignore[arg-type]  # has_capabilities narrows the capability
                reference_texts=ref_texts,
                reference_labels=ref_labels,
                cache_dir=cache_dir,
            )
        self._label_probe: LabelProbe | None = None  # fit lazily on first use, then reused

        # Kept as plain data, not inside the retriever: the label probe (can_probe_labels,
        # _attach_probe) needs only labelled text and no model at all, so it must stay available for
        # a prediction-only pipeline that cannot embed — exactly the case the retriever above
        # declines to build for.
        self.reference_ = (ref_texts, ref_labels) if ref_texts is not None else None
        self.classes_ = self._resolve_classes(ref_labels)

        # Deliberately last: building the bank runs the model over the whole corpus and may raise
        # (OOM, an unencodable corpus, an unwritable cache_dir). Assigning reference_/classes_ first
        # means such a failure costs only retrieval — the label probe, which needs no model at all,
        # is already usable on the half-fitted object.
        if precompute and self._retriever is not None:
            self._retriever.build()
        return self

    def _resolve_classes(self, y: list[str] | None) -> list[str] | None:
        """Label names known from the model/constructor, else derived from ``y``."""
        text_model = getattr(self, "_text_model", None)
        if text_model is not None and text_model.label_names is not None:
            return list(text_model.label_names)
        if self.label_names is not None:
            return list(self.label_names)
        return sorted(set(y)) if y else None

    def explain(
        self,
        X: list[str] | pd.Series,
        y: list | pd.Series | None = None,
        cache_dir: str | Path | None = None,
    ) -> NlpExplanation:
        """Compute token-level contributions and model predictions, and return them as an artifact.

        Results are memoized in memory, keyed by the input texts *together with the model and
        backend that score them* (see :meth:`_compute_key`), so calling ``explain`` again with the
        same data on the same instance skips the expensive explainer run — while changing the
        model, the backend or ``label_names`` correctly recomputes. The explainer keeps nothing
        else: every call returns a fresh
        :class:`~shapash.explainer.nlp_explanation.NlpExplanation`; ``y`` (ground truth) travels
        only on that returned artifact, never on ``self``. "Fresh" means a new object, not a deep
        copy — it is derived from the memoized artifact via
        :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.relabelled` and shares its
        contribution arrays with it. Those arrays are sealed read-only by
        :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.__post_init__`, so the sharing is
        safe: an in-place edit raises rather than quietly rewriting the memo. Derive variants with
        ``dataclasses.replace``. For persistence across kernel restarts
        (or across processes), pass a ``cache_dir`` path to enable an opt-in disk cache — it is
        keyed by hash automatically rather than by an explicit path, but on disk it is exactly a
        :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.save` file, so it stays one format
        whether you let ``explain`` manage it or call ``save``/``load`` yourself.

        Parameters
        ----------
        X : list[str] or pd.Series
            Text samples to explain.
        y : list or pd.Series, optional
            Ground-truth labels for each sample, carried on the returned artifact
            (``NlpExplanation.y_true``). When provided, the webapp will show a "Ground Truth"
            column alongside "Prediction" in the dataset table, making it easy to spot
            misclassifications.
        cache_dir : str or Path, optional
            If provided, the computed explanation is also persisted to
            ``<cache_dir>/<hash>.xpl`` (an :class:`~shapash.explainer.nlp_explanation.NlpExplanation`
            file) and reloaded on subsequent calls — even after a kernel restart. Disabled by
            default. One directory can be shared across models and backends: the hash identifies
            them, so entries cannot collide.

        Returns
        -------
        NlpExplanation
            The computed contributions, predictions and ground truth for this batch.
        """
        texts = X if isinstance(X, pd.Series) else pd.Series(X)
        text_list = texts.tolist()
        new_hash = self._compute_key(text_list)

        # ``computed`` is an NlpExplanation because that is the one on-disk format, but only its
        # ``_COMPUTED_FIELDS`` half is meaningful here — its own texts/index/y_true belong to
        # whoever computed it first. See ``relabelled`` at the return below.
        cached = getattr(self, "_computed_cache", None)
        if cached is not None and cached[0] == new_hash:
            computed = cached[1]
        else:
            cache_path = _cache_file(new_hash, Path(cache_dir)) if cache_dir is not None else None

            if cache_path is not None and cache_path.exists():
                computed, _ = NlpExplanation.load(cache_path)
            else:
                contributions = self.backend.run_explainer(text_list)
                pred_df = self._predict(text_list, texts.index)
                y_pred, y_prob = pred_df["prediction"], pred_df.drop(columns=["prediction"])
                backend_cls = type(self.backend)
                computed = NlpExplanation(
                    texts=texts,
                    token_strings=contributions.token_strings,
                    values=contributions.values,
                    base_values=contributions.base_values,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    y_true=None,
                    label_names=self.label_names,
                    folds_case=self._folds_case(),
                    backend_name=backend_cls.name,
                    is_additive=backend_cls.is_additive,
                    reference_kind=backend_cls.reference_kind,
                    output_space=backend_cls.output_space,
                    model_id=getattr(self._text_model, "checkpoint", None),
                    architecture=getattr(self._text_model, "architecture", None),
                )
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    computed.save(cache_path)

            self._computed_cache = (new_hash, computed)

        if y is None:
            y_series = None
        elif isinstance(y, pd.Series):
            # Aligned to the texts, or refused. Silently reindexing a labelled Series onto
            # texts.index would pair rows positionally under the appearance of index alignment,
            # which is exactly how ground truth ends up attached to the wrong sample.
            if not y.index.equals(texts.index):
                raise ValueError(
                    "y is a Series indexed differently from X, so its labels cannot be matched to "
                    f"the texts: X.index starts {list(texts.index[:3])!r} while y.index starts "
                    f"{list(y.index[:3])!r} (lengths {len(texts.index)} and {len(y.index)}). "
                    "Reindex y onto X.index, or pass it as a plain list to be taken in order."
                )
            y_series = y
        else:
            y_series = pd.Series(y, index=texts.index, name="ground_truth")

        # ``computed`` is whatever the cache key covers, so on a hit it still carries the index and
        # the ground truth of whichever call populated it. ``relabelled`` swaps in this caller's:
        # NlpExplanation owns the split between the fields a content key determines and the fields
        # it cannot, so that list stays next to the fields rather than being restated here.
        return computed.relabelled(texts, y_true=y_series)

    def cache_path(self, x: list[str] | pd.Series, cache_dir: str | Path) -> Path:
        """Return the file :meth:`explain` would read/write for ``x`` under ``cache_dir``.

        Lets a caller report a cache hit or drop a stale entry without reconstructing the key, which
        depends on the bound model and backend and is the explainer's to own (see :meth:`_compute_key`).

        Parameters
        ----------
        x : list[str] or pd.Series
            The same texts that would be passed to :meth:`explain`.
        cache_dir : str or Path
            The same directory that would be passed to :meth:`explain`.

        Returns
        -------
        Path
            The cache file's location (an :class:`~shapash.explainer.nlp_explanation.NlpExplanation`
            file). It need not exist.
        """
        text_list = x.tolist() if isinstance(x, pd.Series) else list(x)
        return _cache_file(self._compute_key(text_list), Path(cache_dir))

    def clear_cache(self, x: list[str] | pd.Series, cache_dir: str | Path) -> None:
        """Drop the cached :meth:`explain` result for ``x``, in memory and on disk.

        The counterpart of :meth:`~shapash.compute.embedding_store.EmbeddingStore.clear` for
        contributions: use it to force a fresh explainer run. Only *this* model+backend's entry is
        removed — another backend's cache for the same texts is left intact.
        """
        self.cache_path(x, cache_dir).unlink(missing_ok=True)
        self._computed_cache = None

    def compute_projection(
        self,
        explanation: NlpExplanation,
        reducer=None,
        cache_dir: str | Path | None = None,
        recompute: bool = False,
    ):
        """Return a 2-D projection of ``explanation``'s texts, ready to pass to :meth:`run_app`.

        The library owns the parts that must stay consistent — *which* space the texts are embedded in
        and how that is cached — while the caller injects the dimensionality reducer, which is a
        modelling choice shapash has no business picking for you. Embedding goes through the model's
        current ``embedding_space``, so the scatter and the similar-example neighbours are guaranteed
        to sit in the same space; they even share the cached vectors.

        Parameters
        ----------
        explanation : NlpExplanation
            The result of :meth:`explain` for the batch to project.
        reducer : object, optional
            Anything with ``fit_transform(X) -> (n_samples, 2)`` — ``sklearn`` PCA/TSNE, ``pacmap``,
            ``umap``. Defaults to :class:`sklearn.decomposition.PCA` with two components (sklearn is
            already a core dependency, so the default costs no extra install).
        cache_dir : str or Path, optional
            When given, both the embeddings and the projected coordinates are persisted here and
            reloaded on later runs, so only the first call pays the cost. The key covers the model, the
            effective space, the texts, and the reducer's class + parameters.
        recompute : bool, optional
            Drop this text set's cached artifacts first, forcing a fresh embed + fit.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Coordinates aligned with the compiled texts.

        Notes
        -----
        The reducer's contribution to the cache key is its class name plus a digest of ``get_params()``
        when it exposes one (sklearn and pacmap both do). A reducer with neither is keyed by class name
        alone, so re-tuning such a reducer needs ``recompute=True`` to take effect.

        Examples
        --------
        >>> explanation = xpl.explain(texts)
        >>> xy = xpl.compute_projection(explanation, reducer=pacmap.PaCMAP(n_components=2), cache_dir="cache/")
        >>> xpl.run_app(explanation, scatter_xy=xy)
        """
        text_model = self._require_text_model()
        if not has_capabilities(text_model, SupportsEmbeddings):
            raise TypeError(
                f"{type(text_model).__name__} does not support embeddings (SupportsEmbeddings); "
                "pass pre-computed coordinates to run_app(scatter_xy=...) instead."
            )
        if reducer is None:
            reducer = PCA(n_components=2)

        store = EmbeddingStore(
            text_model,  # type: ignore[arg-type]  # has_capabilities narrows the capability
            explanation.texts.tolist(),
            cache_dir=cache_dir,
        )
        if recompute:
            store.clear()
        with self._compute_guard():  # embed() tokenizes — serialize against the live compute ops
            return store.cached_array(
                f"{_reducer_tag(reducer)}.proj",
                lambda: np.asarray(reducer.fit_transform(store.vectors())),
            )

    def run_app(
        self,
        explanation: NlpExplanation,
        port: int = 8050,
        debug: bool = False,
        host: str = "127.0.0.1",
        scatter_xy=None,
        url_base_pathname: str | None = None,
        palette_name: str = "default",
        colors_dict: dict[str, str] | None = None,
        info: dict[str, str] | None = None,
    ) -> None:
        """Launch the NLP explanation webapp for ``explanation``.

        The webapp reads ``explanation`` directly — it never writes to it, so the artifact it renders
        stays the one that was computed — and reaches back to ``self`` (as an :class:`~shapash.explainer.interactive.InteractiveEngine`)
        only for live what-if actions (re-predicting edited text, generating counterfactuals,
        similar-example retrieval, label-noise detection) — a snapshot loaded via
        :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.load` has no such engine and serves
        this same method with ``engine=None`` passed to :class:`~shapash.webapp.nlp_app.NlpWebApp`
        directly, self-disabling those panels.

        Parameters
        ----------
        explanation : NlpExplanation
            The result of :meth:`explain` (or a loaded one — see
            :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.load`) to serve.
        port : int
            Port for the Dash development server.
        debug : bool
            Enable Dash debug mode (hot reload, error overlay).
        scatter_xy : np.ndarray, optional
            Pre-computed 2-D projection of the text samples, shape ``(n_samples, 2)``. When provided, a
            scatter panel appears in the webapp that can be used to lasso/box-select a subset of
            samples, filtering both the dataset table and the global word importance plot to that
            subset.

            Prefer :meth:`compute_projection`, whose output goes here and which guarantees the scatter
            is drawn in the same space the similar-example neighbours are ranked in. This parameter is
            an **unverified escape hatch**: coordinates from any source are accepted as-is, so a
            projection built from a different space (or a different model entirely) renders happily
            next to neighbours computed in another — selecting a cluster then means something other
            than it appears to. Use it when you want coordinates the library cannot produce, and
            accept that keeping them consistent is yours to manage.
        url_base_pathname : str, optional
            Mount the app under a URL prefix instead of the server root — for a reverse proxy that
            routes a subpath (e.g. ``"/shapash-nlp-explainer/"``) to this process. Must match the
            proxied path exactly; see :class:`~shapash.webapp.nlp_app.NlpWebApp`. ``None`` serves
            at ``/``.
        palette_name, colors_dict
            Theme the webapp's header band — see :class:`~shapash.webapp.nlp_app.NlpWebApp`.
        info : dict[str, str], optional
            Extra facts shown in the header's "ⓘ" popover, layered on top of what this call already
            knows: ``explanation`` carries the model checkpoint/architecture
            (:attr:`~shapash.explainer.nlp_explanation.NlpExplanation.model_id`/``architecture`` —
            see :meth:`explain`) and sample/class counts, and — when :meth:`fit` was called with a
            reference corpus — this adds "Train samples" from it. Pass *info* for anything neither
            source can know — e.g. ``{"Dataset": "dair-ai/emotion"}`` for a caller-supplied dataset
            label — or to override a value from either.
        """
        # fit() is the one place that knows the reference corpus, so its size is added here rather
        # than living on `explanation` (which is model-free and never sees the training set at all).
        reference = getattr(self, "reference_", None)
        engine_info = {"Train samples": str(len(reference[0]))} if reference is not None else {}
        NlpWebApp(
            explanation,
            engine=self,
            scatter_xy=scatter_xy,
            url_base_pathname=url_base_pathname,
            palette_name=palette_name,
            colors_dict=colors_dict,
            info={**engine_info, **(info or {})},
        ).run(port=port, debug=debug, host=host)

    # ------------------------------------------------------------------
    # InteractiveEngine — live what-if surface (see explainer/interactive.py)
    # ------------------------------------------------------------------

    def _compute_guard(self) -> threading.RLock:
        """Return the lock serializing live model compute (lazily created for ``__new__``/snapshot paths)."""
        lock = getattr(self, "_compute_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._compute_lock = lock
        return lock

    def can_edit(self) -> bool:
        """Whether edited text can be re-predicted and re-explained live.

        Requires a capability-aware model adapter (``False`` for a bare LIME ``classifier_fn`` with
        no ``TextModel`` wrapper). An :class:`~shapash.explainer.nlp_explanation.NlpExplanation`
        loaded via :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.load` carries no engine
        at all — :class:`~shapash.webapp.nlp_app.NlpWebApp` is served with ``engine=None`` in that
        case, and this method is never reached.
        """
        return getattr(self, "_text_model", None) is not None and getattr(self, "backend", None) is not None

    def can_counterfactual(self) -> bool:
        """Whether a counterfactual generator is bound and ready."""
        return getattr(self, "cf_generator", None) is not None

    def can_find_similar(self) -> bool:
        """Whether similar-example retrieval is available (a reference corpus + embedding-capable model).

        ``False`` when :meth:`fit` was never called with an ``X_reference``, or the model cannot embed.
        """
        return getattr(self, "_retriever", None) is not None

    def find_similar(self, text: str, top_k: int = 5) -> list[Neighbor]:
        """Return the reference examples most similar to ``text`` in the model's embedding space.

        Parameters
        ----------
        text : str
            Query text (a selected dataset row, an edited sentence, or a counterfactual).
        top_k : int
            Number of neighbours to return.

        Returns
        -------
        list[Neighbor]
            Reference examples ordered by descending similarity.
        """
        retriever = getattr(self, "_retriever", None)
        if retriever is None:
            raise RuntimeError(
                "find_similar() requires a reference_corpus and an embedding-capable model (unavailable on a snapshot)."
            )
        with self._compute_guard():  # embed() tokenizes — serialize against explain_text/predict
            return retriever.query(text, top_k=top_k)

    def find_similar_threshold(self, text: str, threshold: float = 0.95, limit: int = 50) -> tuple[list[Neighbor], int]:
        """Return reference examples scoring above ``threshold``, in the model's embedding space.

        Companion to :meth:`find_similar` for the webapp's threshold-filter mode: rather than a fixed
        count, every reference example that exceeds ``threshold`` qualifies. ``limit`` caps how many
        are actually returned (a low threshold can otherwise match most of the corpus); the second
        return value is the *total* count that cleared the threshold, so a caller can report e.g.
        "showing 50 of 138".

        Parameters
        ----------
        text : str
            Query text (a selected dataset row, an edited sentence, or a counterfactual).
        threshold : float
            Minimum cosine similarity a reference example must exceed to qualify.
        limit : int
            Maximum neighbours to return, most-similar first.

        Returns
        -------
        tuple[list[Neighbor], int]
            Matching neighbours (capped to ``limit``) and the total number that cleared ``threshold``.
        """
        retriever = getattr(self, "_retriever", None)
        if retriever is None:
            raise RuntimeError(
                "find_similar_threshold() requires a reference_corpus and an embedding-capable model "
                "(unavailable on a snapshot)."
            )
        with self._compute_guard():  # embed() tokenizes — serialize against explain_text/predict
            return retriever.query_threshold(text, threshold=threshold, limit=limit)

    def can_detect_label_noise(self, explanation: NlpExplanation) -> bool:
        """Whether label-noise detection can run on ``explanation``.

        Needs ground truth plus one probability column per class — nothing else. In particular it
        needs **no model**, so unlike the other capability flags this stays usable on an
        :class:`~shapash.explainer.nlp_explanation.NlpExplanation` loaded from disk (no live model);
        only the optional neighbour corroboration in :meth:`detect_label_noise` requires a live
        embedding-capable model.

        ``False`` when ``explanation`` carries no ``y_true``, and when the bound model is a raw
        pipeline that reports only the winning class's confidence (a single ``probability`` column).
        """
        return explanation.y_true is not None and has_usable_probabilities(explanation.y_prob)

    def can_probe_labels(self) -> bool:
        """Whether the model-independent label probe can run (needs a *labelled* reference corpus).

        The probe is fit on the reference corpus, so unlike :meth:`can_find_similar` it needs no
        model — a prediction-only pipeline still gets it. It is ``False`` before :meth:`fit` is
        called with a labelled ``X_reference``.
        """
        reference = getattr(self, "reference_", None)
        return reference is not None and reference[1] is not None

    def detect_label_noise(
        self, explanation: NlpExplanation, top_n: int = 50, score: str = "self_confidence", probe: bool = True
    ) -> LabelNoiseReport:
        """Rank ``explanation``'s samples whose ground-truth label is probably wrong.

        Runs confident learning (see :mod:`shapash.compute.diagnostics.label_noise`) over the
        probabilities and labels carried on ``explanation``, then attaches a model-independent
        second opinion to each flagged sample.

        Parameters
        ----------
        explanation : NlpExplanation
            The result of :meth:`explain` for the batch to audit — must carry ``y_true`` and
            per-class ``y_prob`` (see :meth:`can_detect_label_noise`).
        top_n : int
            Maximum number of issues to return, worst first.
        score : {"self_confidence", "normalized_margin"}
            Ranking method for the flagged samples.
        probe : bool, optional
            Attach a :class:`~shapash.compute.diagnostics.label_probe.ProbeVerdict` to each returned
            issue — a second opinion from a bag-of-words classifier fit on the reference corpus,
            which owes nothing to the audited model. Silently skipped when
            :meth:`can_probe_labels` is ``False``. The probe judges only the issues actually
            returned, so cost scales with ``top_n``, not the corpus.

        Returns
        -------
        LabelNoiseReport
            The ranked issues plus the estimated class-to-class noise matrix.

        Raises
        ------
        RuntimeError
            If the batch carries no ground truth or no per-class probabilities.

        Notes
        -----
        Confident learning assumes the probabilities are **out-of-sample** — compiled on data the
        model did not train on. On a model's own training split it will under-report. See the module
        docstring of :mod:`shapash.compute.diagnostics.label_noise`.

        Confident learning cannot tell a wrong *label* from a confidently wrong *model* — both look
        like "the model disagrees with the label". The probe verdict is what separates them, and it
        is reported alongside the ranking rather than folded into it: a high
        ``issue.probe.given_prob`` means the reference corpus backs the label, so the row is
        probably the model's mistake, not the corpus's. Only the probe can say this, because it is
        the one signal here that does not come from the audited model.

        Examples
        --------
        >>> explanation = xpl.explain(texts, y=labels)
        >>> report = xpl.detect_label_noise(explanation, top_n=20)
        >>> report.noise_rate
        0.062
        >>> report.issues[0].probe.backs_given  # corpus sides with the label -> suspect the model
        True
        """
        if not self.can_detect_label_noise(explanation):
            raise RuntimeError(
                "detect_label_noise() needs ground-truth labels (pass y= to explain()) and "
                "per-class probabilities (a model reporting every class's score, not just the "
                "predicted one)."
            )
        # id() is safe here because the cache only needs to survive while the caller keeps holding
        # this exact explanation (the webapp's typical pattern — NlpWebApp holds one for its lifetime);
        # a garbage-collected explanation can't collide because nothing still holds its id then.
        cache_key = (id(explanation), top_n, score, probe)
        cached = getattr(self, "_label_noise_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cast("LabelNoiseReport", cached[1])

        y_prob = cast("pd.DataFrame", explanation.y_prob)
        y_true = cast("pd.Series", explanation.y_true)
        texts = explanation.texts
        report = detect_label_issues(
            y_prob.to_numpy(dtype=float),
            [str(v) for v in y_true.tolist()],
            [str(t) for t in texts.tolist()],
            [str(c) for c in y_prob.columns],
            top_n=top_n,
            score=score,
        )
        if probe and report.issues and self.can_probe_labels():
            report = replace(report, issues=self._attach_probe(report.issues))

        self._label_noise_cache = (cache_key, report)
        return report

    def _attach_probe(self, issues: list) -> list:
        """Return ``issues`` with each carrying the independent probe's verdict on its given label.

        The probe is fit once and kept: it costs seconds over a few thousand reference texts, and a
        panel re-detecting on every control change would otherwise pay that each time. No compute
        guard is taken — this is scikit-learn over plain strings, touching neither the shared
        tokenizer nor the model.
        """
        probe = getattr(self, "_label_probe", None)
        if probe is None:
            ref_texts, ref_labels = cast("tuple[list[str], list[str]]", self.reference_)
            probe = LabelProbe(ref_texts, ref_labels)
            self._label_probe = probe
        verdicts = probe.verdicts([iss.text for iss in issues], [iss.given_label for iss in issues])
        return [replace(issue, probe=verdict) for issue, verdict in zip(issues, verdicts, strict=True)]

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict a single text, returning ``(label, {label: probability})``."""
        text_model = self._require_text_model()
        with self._compute_guard():
            probs = text_model.predict([text])[0]
        names = text_model.label_names or self.label_names or [str(i) for i in range(len(probs))]
        idx = int(probs.argmax())
        return names[idx], {name: float(p) for name, p in zip(names, probs, strict=False)}

    def explain_text(self, text: str) -> tuple[NlpContributions, str, dict[str, float]]:
        """Re-explain one (possibly edited) text: contributions + prediction + probabilities.

        Bypasses the batch hash cache — this is a single, on-demand explanation of new input.
        """
        if getattr(self, "backend", None) is None:
            raise RuntimeError("explain_text() requires a live backend (unavailable on a snapshot).")
        # Hold the lock across the whole op: run_explainer drives the SHAP pipeline (which tokenizes)
        # and predict tokenizes again — both must be serialized against a concurrent find_similar.
        with self._compute_guard():
            contributions = self.backend.run_explainer([text])
            label, probabilities = self.predict(text)  # re-entrant: same thread re-acquires the RLock
        return contributions, label, probabilities

    def available_cf_generators(self) -> list[tuple[str, str]]:
        """Return ``(name, display_name)`` for each bound generator, in preference order.

        Drives the webapp's counterfactual-method selector; empty when no generator is configured
        (e.g. a snapshot-restored explainer).
        """
        return [(g.name, g.display_name) for g in getattr(self, "cf_generators", {}).values()]

    def generate_counterfactuals(
        self, text: str, config: dict | None = None, generator: str | None = None
    ) -> list[Counterfactual]:
        """Generate counterfactuals for ``text`` via the selected (or active) generator.

        Parameters
        ----------
        text : str
            Text to perturb.
        config : dict, optional
            Overrides for the generator's config spec.
        generator : str, optional
            ``name`` of one of :meth:`available_cf_generators`. Defaults to the active generator.
        """
        gen = self._select_cf_generator(generator)
        if gen is None:
            raise RuntimeError("No counterfactual generator is configured for this explainer.")
        with self._compute_guard():  # generators tokenize / run the model — same serialization
            return gen.generate(text, config=config)

    def cf_config_spec(self, generator: str | None = None) -> dict[str, Field]:
        """Return the tunable config spec of the selected (or active) generator, empty when none."""
        gen = self._select_cf_generator(generator)
        return gen.config_spec() if gen is not None else {}

    def _select_cf_generator(self, generator: str | None) -> CounterfactualGenerator | None:
        """Resolve a generator by ``name`` (or the active one when ``generator`` is ``None``)."""
        generators = getattr(self, "cf_generators", {})
        if generator is not None:
            if generator not in generators:
                raise KeyError(f"Unknown counterfactual generator {generator!r}; available: {list(generators)}.")
            return generators[generator]
        return getattr(self, "cf_generator", None)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_key(self, text_list: list[str]) -> str:
        """Return the cache key for explaining ``text_list`` with *this* model and backend.

        The cached explanation is a function of the texts **and** of everything that scores them,
        so all of it belongs in the key: the model's own identity declaration
        (:attr:`~shapash.model.base.TextModel.model_id` — checkpoint, pooling, normalization, head
        weights), the backend's registered ``name``, its explainer settings, and ``label_names`` (which
        fixes the column order of ``y_prob``).

        Keying on the texts alone — as this once did — means swapping the backend (SHAP -> LIG),
        the model, or the label order and pointing at the same ``cache_dir`` silently reloads the
        previous run's contributions. It also defeats the in-memory guard in :meth:`explain`, so even
        without a ``cache_dir`` a re-``explain`` after changing the backend was a no-op. Callers
        currently work around this by hand-partitioning ``cache_dir`` per model and per backend; with
        the identity in the key that is no longer necessary or possible to get wrong.
        """
        text_model = getattr(self, "_text_model", None)
        model_id = text_model.model_id if text_model is not None else type(self.model).__name__
        backend = getattr(self, "backend", None)
        backend_id = "none"
        if backend is not None:
            # Sorted repr, so two logically identical configs written in a different order agree.
            args = sorted(getattr(backend, "explainer_args", {}).items())
            compute_args = sorted(getattr(backend, "explainer_compute_args", {}).items())
            backend_id = f"{type(backend).name}:{args!r}:{compute_args!r}"
        return hash_corpus(text_list, f"{model_id}|{backend_id}|{self.label_names!r}")

    def _require_text_model(self) -> TextModel:
        text_model = getattr(self, "_text_model", None)
        if text_model is None:
            raise RuntimeError("Live prediction requires a TextModel-backed explainer (unavailable on a snapshot).")
        return text_model

    def _folds_case(self) -> bool | None:
        """Whether this model's tokenizer normalises case away, or ``None`` when it cannot be asked.

        ``None`` for a model with no tokenization capability (a bare LIME ``classifier_fn``, or a
        prediction-only adapter) — the honest answer, which
        :meth:`~shapash.backend.nlp_backend.NlpContributions.resolve_lowercase` turns into a default.
        """
        text_model = getattr(self, "_text_model", None)
        if text_model is None or not has_capabilities(text_model, SupportsTokenization):
            return None
        return cast("SupportsTokenization", text_model).folds_case()

    def _predict(self, text_list: list[str], index: pd.Index) -> pd.DataFrame:
        """Run the pipeline and return a unified DataFrame of predictions and probabilities.

        The first column is always ``"prediction"`` (the argmax label).
        Subsequent columns hold class probabilities: one per class when the
        pipeline returns all scores (``return_all_scores=True``), or a single
        ``"probability"`` column (the winning class confidence) otherwise.

        Handles both ``return_all_scores=True`` (list of lists of dicts) and
        single-prediction (list of dicts) pipeline output formats.
        """
        text_model = getattr(self, "_text_model", None)
        if text_model is not None:
            probs = text_model.predict(text_list)
            names = text_model.label_names or self.label_names or [str(i) for i in range(probs.shape[1])]
            result = pd.DataFrame(probs, index=index, columns=list(names))
            labels = [names[int(i)] for i in probs.argmax(axis=1)]
            result.insert(0, "prediction", pd.Series(labels, index=index))
            return result

        raw = self.model(text_list)
        if raw and isinstance(raw[0], list):
            labels = [max(preds, key=lambda p: p["score"])["label"] for preds in raw]
            col_labels = [d["label"] for d in raw[0]]
            if self.label_names and len(self.label_names) == len(col_labels):
                col_labels = self.label_names
            probs = [[d["score"] for d in preds] for preds in raw]
            result = pd.DataFrame(probs, index=index, columns=col_labels)
        else:
            labels = [p["label"] for p in raw]
            result = pd.DataFrame(
                {"probability": [p["score"] for p in raw]},
                index=index,
            )
        result.insert(0, "prediction", pd.Series(labels, index=index))
        return result
