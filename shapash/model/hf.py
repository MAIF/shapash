"""HuggingFace text-model adapters.

Two concrete :class:`~shapash.model.base.TextModel` implementations:

* :class:`HFPipelineModel` — wraps a ``transformers.pipeline`` and exposes **prediction only**
  (plus tokenization). This is the backward-compatible default: everything the existing prototype
  passes to ``NlpExplainer`` is a pipeline, so it is wrapped here transparently.
* :class:`HFClassifierModel` — wraps a raw ``AutoModelForSequenceClassification`` + tokenizer and
  implements **all** capabilities, which gradient-based counterfactual methods (HotFlip) and the
  Captum LIG backend need. It is a thin preset over
  :class:`~shapash.model.encoder.EncoderClassifierModel`: the classifier itself is the backbone
  (pooling and head are internal to it), so no fusing is required.

For models whose encoder body and classification head are *separate* modules (sentence-transformers,
hand-rolled PyTorch classifiers), see :class:`~shapash.model.torch_models.SentenceTransformerModel` and
:class:`~shapash.model.torch_models.TorchClassifierModel` — they share the same capability spine via a
fused backbone. Generators and the webapp need no changes because everything depends on *capabilities*,
not concrete classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from shapash._optional import import_optional_module
from shapash.model.base import (
    SupportsTokenization,
    TextModel,
)
from shapash.model.encoder import _DECISION_SPACE, EncoderClassifierModel, _position_capacity

logger = logging.getLogger(__name__)

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# Files whose presence in a local checkpoint directory means it ships its own tokenizer. A directory
# saved with ``model.save_pretrained(...)`` holds weights + ``config.json`` but *none* of these — the
# tokenizer is saved separately (often archived elsewhere entirely), which is what makes the base-model
# fallback in :func:`_resolve_tokenizer_source` necessary.
_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json", "vocab.txt")

# ``transformers`` reports this "unset" sentinel (~1e30, stored as a huge int) for a tokenizer whose
# config never set ``model_max_length``. It means "no configured limit", not a real length: passing it
# straight to ``truncation=True`` makes truncation a silent no-op. Any ``model_max_length`` above this
# is treated as unset. See :func:`_resolve_max_length`.
_UNSET_TOKENIZER_MAX_LENGTH = 100_000

# Last-resort truncation length when a checkpoint reports no usable ``model_max_length`` *and* its
# backbone exposes no absolute-position capacity to fall back on (a rotary/ALiBi model). A conservative
# encoder default that every standard architecture can index.
_FALLBACK_MAX_LENGTH = 512


def _resolve_tokenizer_source(name_or_path: str, tokenizer_name: str | None = None) -> str:
    """Decide *where* the tokenizer is loaded from — an explicit override, the checkpoint, or its base.

    ``tokenizer_name`` wins when given. Otherwise the checkpoint at ``name_or_path`` is used, except in
    the one case it cannot be: a local directory saved with ``model.save_pretrained(...)`` holds weights
    and ``config.json`` but no tokenizer files. Since a fine-tune keeps its base model's tokenizer unless
    it was deliberately customised, the base id recorded in the checkpoint's own ``config.json``
    (``_name_or_path``, transformers' record of what it was fine-tuned from) is the right fallback — it
    needs no extra argument and no access to wherever the tokenizer artifacts were archived.

    A *custom* tokenizer therefore always needs an explicit ``tokenizer_name``: nothing here can detect
    that the base tokenizer is the wrong one, so pass it whenever the vocabulary was extended.

    Parameters
    ----------
    name_or_path : str
        A HuggingFace hub id or a local checkpoint directory.
    tokenizer_name : str or None, optional
        Explicit tokenizer id/path; short-circuits everything below when given.

    Returns
    -------
    str
        The id or path to load the tokenizer from.

    Raises
    ------
    RuntimeError
        If ``name_or_path`` is a local directory that ships no tokenizer files and whose ``config.json``
        records no base model to fall back on.
    """
    if tokenizer_name:
        return tokenizer_name

    path = Path(name_or_path).expanduser()
    if not path.is_dir():
        return str(name_or_path)  # a hub id: the repo always ships its own tokenizer files
    if any((path / name).is_file() for name in _TOKENIZER_FILES):
        return str(path)

    config_file = path / "config.json"
    # JSON is UTF-8 by specification; read_text()'s locale default would fail on a config carrying
    # non-ASCII label names (e.g. accented id2label values) under a non-UTF-8 locale.
    base_model = (
        json.loads(config_file.read_text(encoding="utf-8")).get("_name_or_path") if config_file.is_file() else None
    )
    if not base_model:
        raise RuntimeError(
            f"{path} ships no tokenizer files, and its config.json records no base model to fall back on. "
            "Pass tokenizer= with the HF id (or local path) of the tokenizer this checkpoint was trained "
            "with."
        )
    logger.info(
        "%s ships no tokenizer — falling back to its base model %r (from config.json's _name_or_path). "
        "Pass tokenizer= if this checkpoint uses a customised tokenizer.",
        path,
        base_model,
    )
    return base_model


def _load_tokenizer(source: str, transformers: Any, **tokenizer_kwargs: Any) -> Any:
    """Load the tokenizer, preferring a fast one but falling back to the slow implementation.

    A fast tokenizer is best — it unlocks the exact ``word_ids()`` word-alignment the LIG highlights
    use — but some checkpoints ship no ``tokenizer.json``, so the fast load raises. We then retry with
    ``use_fast=False``; a slow tokenizer still works (gradient attribution degrades to a scheme-aware
    string merge). When *both* fail the checkpoint has no tokenizer this call can build, and the error
    says what to do about it rather than dumping a deep tokenizer stack trace.

    Parameters
    ----------
    source : str
        Hub id or local path to load the tokenizer from.
    transformers : module
        The imported ``transformers`` module (injected so this stays unit-testable).
    **tokenizer_kwargs
        Forwarded to both attempts — notably ``trust_remote_code``, which a checkpoint shipping its own
        tokenizer implementation needs. ``use_fast`` is set here and must not be passed.
    """
    try:
        return transformers.AutoTokenizer.from_pretrained(source, use_fast=True, **tokenizer_kwargs)
    except Exception as fast_err:  # noqa: BLE001 — retry slow, then re-raise with a usable hint
        logger.warning(
            "Fast tokenizer unavailable for %s (%s) — retrying with use_fast=False.",
            source,
            type(fast_err).__name__,
        )
        try:
            return transformers.AutoTokenizer.from_pretrained(source, use_fast=False, **tokenizer_kwargs)
        except Exception as slow_err:  # noqa: BLE001 — no loadable tokenizer; give an actionable message
            # The hint depends on what was already tried: suggesting trust_remote_code=True to someone
            # who just passed it is noise, and would read as "this checkpoint is unsupported" when the
            # real problem is something else (a bad path, a missing sentencepiece/protobuf, no network).
            hint = (
                "If this checkpoint ships a custom tokenizer implementation, retry with "
                "trust_remote_code=True (this executes code from the checkpoint — only do so for a "
                "source you trust)."
                if not tokenizer_kwargs.get("trust_remote_code")
                else "Custom code was already allowed, so this is not a trust_remote_code problem — "
                "check the path/hub id, network access, and that any extra tokenizer dependency "
                "(e.g. sentencepiece, protobuf) is installed."
            )
            raise RuntimeError(
                f"Could not load a tokenizer for {source!r} in fast or slow mode. {hint} You can also "
                "pass an already-loaded tokenizer via tokenizer=, or build HFClassifierModel(classifier, "
                "tokenizer) directly."
            ) from slow_err


def _resolve_max_length(tokenizer: Any, backbone: Any) -> int | None:
    """Resolve the truncation length for a checkpoint, defeating the ``model_max_length`` no-op trap.

    A checkpoint whose tokenizer config never set ``model_max_length`` reports the ``~1e30`` sentinel, so
    ``truncation=True`` with no explicit ``max_length`` silently does nothing and a long text tokenizes
    unbounded — which is not merely slow: once a sequence outruns the backbone's position buffer the
    out-of-bounds gather aborts the CUDA context (see
    :meth:`~shapash.model.encoder.EncoderClassifierModel._check_encoded_length`). When the tokenizer
    reports a real limit we trust it; otherwise we fall back to the backbone's *own* absolute-position
    capacity (the exactly-correct bound), and only to :data:`_FALLBACK_MAX_LENGTH` when even that is
    unknowable (a rotary/ALiBi backbone).
    """
    model_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max, int) and 0 < model_max <= _UNSET_TOKENIZER_MAX_LENGTH:
        return model_max
    return _position_capacity(backbone) or _FALLBACK_MAX_LENGTH


def _probs_from_pipeline_output(raw: list, order: list[str]) -> np.ndarray:
    """Convert ``pipeline`` output to a probability matrix aligned to ``order``.

    Handles both ``top_k=None`` output (list of per-class dict lists) and single-prediction output
    (list of dicts). Aligning by label name makes the result robust to per-row score sorting.
    """
    rows = raw if raw and isinstance(raw[0], list) else [[r] for r in raw]
    matrix = np.zeros((len(rows), len(order)), dtype=np.float64)
    for i, row in enumerate(rows):
        scores = {d["label"]: float(d["score"]) for d in row}
        for j, label in enumerate(order):
            matrix[i, j] = scores.get(label, 0.0)
    return matrix


class HFPipelineModel(TextModel, SupportsTokenization):
    """Prediction-only adapter over a HuggingFace ``text-classification`` pipeline.

    Parameters
    ----------
    pipeline : transformers.Pipeline
        A ``text-classification`` pipeline. ``top_k=None`` is requested at call time so all class
        scores are returned.
    label_names : list[str] or None
        Class names in output-column order. When ``None``, inferred from the pipeline's first
        prediction (the labels it emits). When given, validated against those labels on the first
        prediction — a mismatched set raises ``ValueError`` rather than silently misaligning columns.
    """

    def __init__(self, pipeline, label_names: list[str] | None = None) -> None:
        super().__init__(label_names=label_names)
        self.pipeline = pipeline
        self.tokenizer = getattr(pipeline, "tokenizer", None)
        self._order: list[str] | None = None

    @property
    def shap_callable(self):
        """Callable SHAP's text explainer can wrap directly (the pipeline itself)."""
        return self.pipeline

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return ``(n_texts, n_classes)`` probabilities via the pipeline."""
        raw = self.pipeline(list(texts), top_k=None)
        if self._order is None:
            first = raw[0] if raw and isinstance(raw[0], list) else raw
            labels = [d["label"] for d in first]
            if self.label_names is not None:
                if set(self.label_names) != set(labels):
                    raise ValueError(
                        f"label_names {self.label_names!r} do not match the labels the pipeline "
                        f"emits {labels!r}; pass the exact label set (any order), or leave "
                        "label_names=None to infer it from the pipeline."
                    )
                self._order = list(self.label_names)
            else:
                self._order = labels
                self.label_names = list(self._order)
        return _probs_from_pipeline_output(raw, self._order)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize with the pipeline's tokenizer."""
        if self.tokenizer is None:
            raise RuntimeError("Pipeline has no tokenizer; cannot tokenize.")
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from tokens."""
        if self.tokenizer is None:
            raise RuntimeError("Pipeline has no tokenizer; cannot detokenize.")
        return self.tokenizer.convert_tokens_to_string(tokens)


class HFClassifierModel(EncoderClassifierModel):
    """Full-capability adapter over a raw ``AutoModelForSequenceClassification`` + tokenizer.

    A thin preset over :class:`~shapash.model.encoder.EncoderClassifierModel`: the classifier *is* the
    backbone (it maps ids/embeds to ``.logits`` with pooling and head internal), so this class only
    resolves ``label_names`` from the model config and delegates the entire capability surface —
    prediction, tokenization, the input-embedding table, sentence embeddings in any representation
    space, per-token gradients and the Captum layer-attribution surface.

    Parameters
    ----------
    classifier : transformers.PreTrainedModel
        A sequence-classification model whose ``forward`` accepts ``inputs_embeds`` and supports
        ``output_hidden_states=True``.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching the classifier.
    label_names : list[str] or None
        Class names in class-index (id) order. When ``None``, read from ``classifier.config.id2label``.
    batch_size : int, optional
        Batch size for ``predict``, ``embed`` and the SHAP pipeline. Default 32.
    device : int or str or torch.device or None, optional
        Device for the SHAP pipeline (e.g. ``0``, ``"cuda"``, ``"cpu"``). When ``None`` (default),
        the pipeline inherits the device the classifier already lives on, so CPU-only environments
        work with no configuration. An explicit CUDA request is silently downgraded to CPU (with a
        warning) when no GPU is available.
    embedding_space : str, optional
        Representation :meth:`embed` returns (see :class:`~shapash.model.encoder.EncoderClassifierModel`).
        Default ``"decision"`` (input to the final classification linear); ``"pooled"`` restores the
        historical mask-pooled last hidden state.
    pool : {"mean", "cls", "max"} or callable, optional
        How a token-level output is reduced to one vector per text in :meth:`embed`. Default ``"mean"``.
        Note this affects the *analysis* representation only — an ``AutoModelForSequenceClassification``
        does its own pooling internally, so :meth:`predict` is unaffected.
    max_length : int or None, optional
        Truncation length applied to every tokenizer call — see
        :class:`~shapash.model.encoder.EncoderClassifierModel`. When ``None`` (default) the tokenizer's
        own ``model_max_length`` applies; note some checkpoints ship a tokenizer whose config never sets
        this, so ``transformers`` reports its "unset" sentinel (~1e30) instead of a real limit — with
        ``max_length=None`` that makes truncation a no-op, so a gradient-based attribution method (e.g.
        ``NlpCaptumLigBackend``) can exhaust GPU memory on a long enough text. Pass an explicit
        ``max_length`` for such checkpoints.
    """

    def __init__(
        self,
        classifier,
        tokenizer,
        label_names: list[str] | None = None,
        batch_size: int = 32,
        device: int | str | object | None = None,
        *,
        embedding_space: str = _DECISION_SPACE,
        pool: Any = "mean",
        max_length: int | None = None,
    ) -> None:
        if label_names is None:
            id2label = getattr(getattr(classifier, "config", None), "id2label", None)
            if id2label:
                label_names = [id2label[i] for i in sorted(id2label)]
        else:
            # Only an *explicit* list can be wrong: names read from ``config.id2label`` above always
            # match ``num_labels`` by construction. A wrong-length list would otherwise misalign every
            # class name, or fail much later inside a webapp — check it at the point of the mistake.
            num_labels = getattr(getattr(classifier, "config", None), "num_labels", None)
            if isinstance(num_labels, int) and len(label_names) != num_labels:
                raise ValueError(
                    f"{len(label_names)} label name(s) {label_names!r} for a {num_labels}-class model. "
                    "Pass exactly one label name per output column, in class-index order."
                )
        super().__init__(
            classifier,
            tokenizer,
            label_names=label_names,
            batch_size=batch_size,
            device=device,
            pool=pool,
            embedding_space=embedding_space,
            max_length=max_length,
        )
        # ``classifier`` is the backbone; keep the historical alias for introspection and tests.
        self.classifier = classifier

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        tokenizer: Any = None,
        label_names: list[str] | None = None,
        device: int | str | object | None = None,
        max_length: int | str | None = "auto",
        trust_remote_code: bool = False,
        load_kwargs: dict[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> HFClassifierModel:
        """Build a full-capability adapter straight from a checkpoint id or a local directory.

        The one-call entry point for serving a classifier: it loads the weights, resolves a matching
        tokenizer, and picks a safe truncation length — the three steps that are easy to get subtly
        wrong (a checkpoint saved without its tokenizer; a tokenizer reporting no ``model_max_length``,
        which silently disables truncation and can abort the CUDA context on a long input). Both a
        HuggingFace hub id (``"distilbert-base-uncased-finetuned-sst-2-english"``) and a local
        ``save_pretrained`` directory work with no extra arguments.

        Parameters
        ----------
        name_or_path : str
            A HuggingFace hub id, or a path to a local checkpoint directory.
        tokenizer : str or transformers.PreTrainedTokenizerBase or None, optional
            An already-loaded tokenizer (used as-is), or an id/path to load one from. When ``None``
            (default) the tokenizer is resolved from the checkpoint, falling back to the base model
            recorded in its ``config.json`` for a directory that ships no tokenizer files — see
            :func:`_resolve_tokenizer_source`. Pass this explicitly for a customised tokenizer.
        label_names : list[str] or None, optional
            Class names in class-index order. When ``None`` (default) they are read from the model's
            ``config.id2label``; pass this to override a fine-tune saved without label names (its config
            reports ``LABEL_0``/``LABEL_1``). Validated against the model's ``num_labels``.
        device : int or str or torch.device or None, optional
            Device to move the classifier onto (e.g. ``0``, ``"cuda"``, ``"cpu"``). When ``None``
            (default) the checkpoint loads wherever ``transformers`` places it (CPU), and the adapter
            inherits that device — so CPU-only environments need no configuration.
        max_length : int, "auto", or None, optional
            Truncation length. ``"auto"`` (default) resolves the tokenizer's own ``model_max_length``,
            substituting the backbone's true position capacity when the tokenizer reports the "unset"
            sentinel (see :func:`_resolve_max_length`). ``None`` trusts the tokenizer verbatim (the
            length guard in :class:`~shapash.model.encoder.EncoderClassifierModel` still applies). An int
            forces an explicit length.
        trust_remote_code : bool, optional
            Allow the checkpoint to define its own model/tokenizer classes (``AutoModel``'s flag of the
            same name), which is what a custom architecture such as ``Alibaba-NLP/gte-base-en-v1.5``,
            Jina or Nomic needs. Default ``False``. Applied to the tokenizer *and* the model, since a
            checkpoint that ships one usually ships both.

            **This executes Python code downloaded from the checkpoint**, so pass it only for a source
            you trust. It is always forwarded explicitly (``False`` included) rather than left unset:
            unset, ``transformers`` asks for confirmation on an interactive terminal, which would hang
            a webapp or a batch job on a prompt nobody sees.

            A loaded custom architecture is not automatically *fully* capable — the capability surface
            still needs a backbone that accepts ``inputs_embeds`` and exposes ``get_input_embeddings()``
            (see :class:`~shapash.model.encoder.EncoderClassifierModel`). Prediction and SHAP work
            regardless; gradients and Captum LIG depend on the checkpoint's own implementation.
        load_kwargs : dict, optional
            Extra keyword arguments forwarded to ``AutoModelForSequenceClassification.from_pretrained``
            and to the tokenizer load (e.g. ``revision``, ``dtype``, ``token``, ``cache_dir``). Kept
            separate from ``**model_kwargs`` below, which configures *this adapter* — one dict per
            destination, so no argument is ambiguous about which of the two it reaches.
        **model_kwargs
            Forwarded to the constructor (e.g. ``batch_size``, ``embedding_space``, ``pool``).

        Returns
        -------
        HFClassifierModel
            The constructed adapter, with the classifier on ``device`` and truncation resolved.

        Raises
        ------
        ValueError
            If ``max_length`` is not an int, ``None``, or ``"auto"`` (or ``label_names`` has the wrong
            length — see :meth:`__init__`).
        """
        transformers = import_optional_module("transformers", extra=_NLP_EXTRA)
        loader_kwargs = {"trust_remote_code": trust_remote_code, **(load_kwargs or {})}
        if tokenizer is None or isinstance(tokenizer, str):
            tokenizer = _load_tokenizer(
                _resolve_tokenizer_source(name_or_path, tokenizer), transformers, **loader_kwargs
            )

        classifier = transformers.AutoModelForSequenceClassification.from_pretrained(str(name_or_path), **loader_kwargs)
        if device is not None:
            classifier = classifier.to(device)
        logger.info("Loaded %s as HFClassifierModel on device: %s", name_or_path, next(classifier.parameters()).device)

        if max_length == "auto":
            resolved_max_length = _resolve_max_length(tokenizer, classifier)
        elif max_length is None or isinstance(max_length, int):
            resolved_max_length = max_length
        else:
            raise ValueError(f"max_length must be an int, None, or 'auto'; got {max_length!r}.")

        return cls(classifier, tokenizer, label_names=label_names, max_length=resolved_max_length, **model_kwargs)
