"""External-head presets: wrap a separate encoder body + pooling + classification head.

:class:`~shapash.model.encoder.EncoderClassifierModel` needs a *backbone* that maps input ids/embeds to
``.logits`` with pooling and the head baked in. A raw ``AutoModelForSequenceClassification`` already is
such a backbone (see :class:`~shapash.model.hf.HFClassifierModel`). Sentence-transformer classifiers and
hand-rolled PyTorch classifiers instead keep the encoder body, the pooling, and the head as *separate*
pieces. :func:`build_encoder_head_backbone` fuses those pieces into one backbone honouring the contract,
so both get the full capability surface with no bespoke per-model code.

* :class:`TorchClassifierModel` — the general, framework-agnostic preset: ``(body, head, tokenizer)``.
* :class:`SentenceTransformerModel` — extracts ``body``/``tokenizer``/pooling-mode from a
  ``sentence_transformers`` model, then defers to the same machinery with a caller-supplied head.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from types import SimpleNamespace
from typing import Any

# This module is imported lazily by ``shapash.model`` (see its ``__getattr__``) precisely so that torch
# can be imported here at module level: ``_EncoderHeadBackbone`` subclasses ``nn.Module``, which cannot
# be done inside a lazy helper without minting a new class per call. Core installs never load this file.
import torch

from shapash._optional import import_optional_module
from shapash.model.encoder import _DECISION_SPACE, EncoderClassifierModel, _pool_hidden

logger = logging.getLogger(__name__)


def _extract_hidden(body_output: Any) -> Any:
    """Return the token-level ``(batch, seq, hidden)`` tensor from an encoder body's output.

    Handles the common return shapes: a HuggingFace output with ``.last_hidden_state``, a bare tensor,
    or a tuple/list whose first element is the hidden state.
    """
    hidden = getattr(body_output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(body_output, (tuple, list)):
        return body_output[0]
    return body_output


class _EncoderHeadBackbone(torch.nn.Module):
    """``body -> pool -> [normalize] -> head`` fused into one HF-model-like backbone.

    Built by :func:`build_encoder_head_backbone`; see that function for the contract it honours.
    """

    def __init__(self, body, head, pool, normalize):
        super().__init__()
        self.body = body
        self.head = head
        self.pool = pool
        self.normalize = normalize

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        **_kwargs,
    ):
        # Only pass the id/embed argument the caller actually supplied; some bodies reject a None
        # inputs_embeds alongside input_ids. token_type_ids etc. are intentionally not threaded
        # through — the historical HF gradient/logits paths did not either.
        if inputs_embeds is not None:
            body_out = self.body(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            body_out = self.body(input_ids=input_ids, attention_mask=attention_mask)
        hidden = _extract_hidden(body_out)
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], device=hidden.device)
        pooled = _pool_hidden(hidden, attention_mask, self.pool)
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        logits = self.head(pooled)
        out = SimpleNamespace(logits=logits)
        if output_hidden_states:
            out.hidden_states = (hidden,)
        return out

    def get_input_embeddings(self):
        """Return the body's word-embedding module (what LIG attributes through)."""
        return self.body.get_input_embeddings()

    @property
    def config(self):
        """Expose the body's HF config, so backbone-contract consumers see the real checkpoint.

        Without this the fused backbone looks anonymous and everything that identifies a model by
        ``config._name_or_path`` (notably ``TextModel.model_id``, which keys downstream caches)
        would read the same value for every external-head model.
        """
        return getattr(self.body, "config", None)

    @property
    def device(self):
        """Device the fused module's parameters live on."""
        return next(self.parameters()).device


def build_encoder_head_backbone(body, head, pool: Any, normalize: bool = False):
    """Fuse an encoder ``body``, a ``pool`` strategy, and a classification ``head`` into one backbone.

    The returned ``nn.Module`` honours the backbone contract
    (:mod:`shapash.model.encoder`): it accepts ``input_ids``/``inputs_embeds`` (+ ``attention_mask``),
    runs ``body -> pool -> [normalize] -> head`` to produce ``.logits``, exposes ``.hidden_states`` on
    request, and forwards ``get_input_embeddings`` / ``.device`` to the body. Pooling (and the optional
    L2 normalization) happen *inside* the differentiated path, so Captum ``LayerIntegratedGradients``
    attributes embeddings -> logits through them.

    Parameters
    ----------
    body : nn.Module
        Encoder returning a token-level last hidden state (e.g. a HF ``AutoModel``). Must accept
        ``inputs_embeds`` for the gradient/Captum paths and expose ``get_input_embeddings()``.
    head : nn.Module
        Maps the pooled ``(batch, hidden)`` vector to ``(batch, n_classes)`` logits.
    pool : {"mean", "cls", "max"} or callable
        Pooling strategy applied between body and head; see :func:`~shapash.model.encoder._pool_hidden`.
    normalize : bool, optional
        L2-normalize the pooled vector before the head. Default ``False``. Reproduces a
        sentence-transformer ``Normalize`` module so a head trained on unit-length embeddings receives
        them. L2 normalization is differentiable, so gradients/LIG still flow embeddings -> logits.

    Returns
    -------
    _EncoderHeadBackbone
        A backbone instance in **eval mode**, ready to pass to
        :class:`~shapash.model.encoder.EncoderClassifierModel`.

    Notes
    -----
    The returned module is switched to eval mode here. Assigning a submodule to an ``nn.Module`` does
    not change that submodule's ``training`` flag, so a head the caller built fresh (``nn.Sequential``
    with a ``Dropout``/``BatchNorm``) would otherwise stay in *training* mode and randomise every
    forward pass — silently, since nothing raises: predictions, contributions, embeddings and
    counterfactual flips would all differ run to run. ``eval()`` recurses into body and head, so one
    call pins the whole path. These adapters are inference wrappers; there is no path that wants
    training mode.
    """
    backbone = _EncoderHeadBackbone(body, head, pool, normalize)
    return backbone.eval()


class TorchClassifierModel(EncoderClassifierModel):
    """Full-capability adapter over a hand-rolled ``(encoder body, classification head)`` pair.

    Use this for any PyTorch text classifier whose encoder and head are separate modules (i.e. the head
    is not baked into a single ``AutoModelForSequenceClassification``). The body must expose a
    token-embedding layer to attribute through (``get_input_embeddings()``) and accept ``inputs_embeds``;
    given that, every capability — gradients, Captum LIG, similar-example retrieval — is available.

    Parameters
    ----------
    body : nn.Module
        Encoder returning a token-level last hidden state; accepts ``inputs_embeds`` and exposes
        ``get_input_embeddings()``.
    head : nn.Module
        Maps the pooled ``(batch, hidden)`` vector to ``(batch, n_classes)`` logits.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching ``body``. A fast tokenizer enables exact word alignment.
    label_names : list[str] or None
        Class names in class-index order.
    pool : {"mean", "cls", "max"} or callable, optional
        Pooling between body and head. Default ``"mean"``.
    normalize : bool, optional
        L2-normalize the pooled vector before the head (and in :meth:`embed`). Default ``False``. Set
        this when the head was trained on unit-length sentence embeddings (a sentence-transformer
        ``Normalize`` step); :class:`SentenceTransformerModel` detects and sets it automatically.
    embedding_space : str, optional
        Representation :meth:`embed` returns (see :class:`~shapash.model.encoder.EncoderClassifierModel`).
        Default ``"decision"`` (input to the final classification linear) — for an external head this
        reaches into it, so the projection stays class-discriminative even with a frozen encoder.
    batch_size : int, optional
        Batch size. Default 32.
    device : int or str or torch.device or None, optional
        Device for the SHAP pipeline; inherits the body's device when ``None``.
    max_length : int or None, optional
        Truncation length for every tokenizer call. When ``None`` (default) the tokenizer's own
        ``model_max_length`` applies — **which is often longer than the length the body is actually
        served at**. Loading ``sentence-transformers/all-MiniLM-L6-v2`` via ``AutoTokenizer`` reports 512,
        while the same model loaded through ``SentenceTransformer`` truncates at 256; on a 400-token text
        the resulting sentence embeddings differ by ~0.02 per component on unit-length vectors, and the
        head then scores an input it was never trained on. Pass the length the body is served at (here
        ``max_length=256``) whenever the tokenizer's limit does not already reflect it.

    Notes
    -----
    The SHAP backend cannot build a ``transformers`` pipeline here (the fused backbone is not a
    ``PreTrainedModel``), so :attr:`shap_callable` is the plain :meth:`predict` function and
    :attr:`shap_masker` supplies an explicit ``shap.maskers.Text`` over the tokenizer — the two pieces
    SHAP's text explainer needs. The Captum LIG backend works unchanged.
    """

    def __init__(
        self,
        body,
        head,
        tokenizer,
        label_names: list[str] | None = None,
        *,
        pool: Any = "mean",
        normalize: bool = False,
        embedding_space: str = _DECISION_SPACE,
        batch_size: int = 32,
        device: int | str | object | None = None,
        max_length: int | None = None,
    ) -> None:
        backbone = build_encoder_head_backbone(body, head, pool, normalize=normalize)
        super().__init__(
            backbone,
            tokenizer,
            label_names=label_names,
            batch_size=batch_size,
            device=device,
            pool=pool,
            normalize=normalize,
            embedding_space=embedding_space,
            max_length=max_length,
        )
        self.body = body
        self.head = head
        self._head_fingerprint: str | None = None

    @property
    def model_id(self) -> str:
        """Identity including a fingerprint of the **head's weights**, on top of the base identity.

        The base :attr:`~shapash.model.base.TextModel.model_id` identifies the body checkpoint, pooling
        and normalization — but here the head is a separate, independently-trained module, and two
        models sharing a frozen body (the common sentence-transformers case) differ *only* in it.
        Without the head in the key they would silently share a cached embedding/neighbour bank. Heads
        are small, so the fingerprint hashes their actual parameter bytes rather than just the
        architecture; it is computed once and memoized.
        """
        if self._head_fingerprint is None:
            self._head_fingerprint = self._fingerprint_head()
        return f"{super().model_id}:head={self._head_fingerprint}"

    def _fingerprint_head(self) -> str:
        """Return a short stable hash of the head's parameter values (architecture + weights)."""
        digest = hashlib.sha256()
        try:
            for name, tensor in sorted(self.head.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
        except (AttributeError, RuntimeError, TypeError):
            # A head without a usable state_dict: fall back to its architecture repr. Weaker (two
            # identically-shaped heads collide) but never wrong-by-crash, and still separates shapes.
            logger.warning("Could not hash head weights for the cache key; falling back to its repr().")
            digest.update(repr(self.head).encode())
        return digest.hexdigest()[:16]

    @property
    def shap_callable(self):
        """The plain scoring callable SHAP's text explainer wraps (``predict``).

        Unlike :class:`~shapash.model.hf.HFClassifierModel` (whose backbone is a ``PreTrainedModel`` that
        SHAP wraps as a pipeline, auto-inferring a masker), an external-head model exposes only
        ``predict``; the companion :attr:`shap_masker` provides the text masker SHAP would otherwise
        infer from a pipeline.
        """
        return self.predict

    @property
    def shap_masker(self):
        """A ``shap.maskers.Text`` over the tokenizer, so SHAP can segment/mask a plain callable.

        Overrides :attr:`~shapash.model.base.TextModel.shap_masker` (which returns ``None``, letting
        SHAP infer a masker from a pipeline) because :attr:`shap_callable` here is a bare function with
        no tokenizer for SHAP to find. ``NlpExplainer`` forwards it to ``NlpShapBackend``.
        """
        shap = import_optional_module("shap")
        return shap.maskers.Text(self.tokenizer)


# Sentence-transformers pooling modes this module can reproduce, mapped onto ``_pool_hidden`` strategies.
# Modern ST (>=2.x) exposes a single ``pooling_mode`` string on its ``Pooling`` module; older releases
# set one of several ``pooling_mode_*_tokens`` booleans instead. Both are handled below.
_ST_MODE_MAP = {"cls": "cls", "max": "max", "mean": "mean"}
_ST_POOLING_FLAGS = (
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_mean_tokens", "mean"),
)


def _sentence_transformer_pool(st_model) -> str:
    """Read the pooling mode from a ``sentence_transformers`` model's ``Pooling`` module.

    Returns one of ``"cls"``/``"max"``/``"mean"``. Raises when the model uses an exotic ST pooling mode
    (``mean_sqrt_len_tokens``, ``weightedmean``, ``lasttoken``) that :func:`~shapash.model.encoder._pool_hidden`
    does not reproduce — silently mean-pooling there would diverge from how the head was trained, so the
    caller must pass an explicit ``pool`` instead.
    """
    for module in st_model:
        mode = getattr(module, "pooling_mode", None)
        if isinstance(mode, str):
            if mode in _ST_MODE_MAP:
                return _ST_MODE_MAP[mode]
            raise ValueError(
                f"sentence-transformers pooling mode {mode!r} is not reproduced by "
                f"SentenceTransformerModel (supported: {sorted(_ST_MODE_MAP)}). Pass an explicit "
                "`pool=` (mode name or callable) matching how the classification head was trained."
            )
        for flag, mapped in _ST_POOLING_FLAGS:  # legacy ST boolean flags
            if getattr(module, flag, False):
                return mapped
    return "mean"


def _sentence_transformer_max_length(st_model) -> int | None:
    """Return the sentence-transformers model's ``max_seq_length``, or ``None`` when unset.

    Loading *through* ``SentenceTransformer`` already syncs this onto the tokenizer's
    ``model_max_length`` (both 256 for ``all-MiniLM-L6-v2``), so for this preset ``truncation=True``
    alone would agree with ``st_model.encode`` anyway — reading it here is belt-and-braces.

    It is worth carrying for two reasons. The sync is an ST implementation detail, not a documented
    guarantee; and it makes the served length explicit and inspectable as ``model.max_length``, matching
    :class:`TorchClassifierModel`, where the same value must be passed by hand because a raw
    ``AutoTokenizer`` for the *same checkpoint* reports 512 rather than 256.
    """
    max_length = getattr(st_model, "max_seq_length", None)
    if max_length is None:
        first = st_model[0]
        max_length = getattr(first, "max_seq_length", None)
    return int(max_length) if max_length else None


def _sentence_transformer_normalize(st_model) -> bool:
    """Return whether the sentence-transformers model ends its forward in an L2 ``Normalize`` module.

    Many ST embedding models (e.g. ``all-MiniLM-L6-v2``) append a ``Normalize`` module after pooling, so
    ``st_model.encode`` returns unit-length vectors and a downstream head is trained on them. When present
    we reproduce it (see :func:`build_encoder_head_backbone`) so ``predict`` stays faithful. Post-pooling
    modules other than ``Normalize`` (e.g. a ``Dense`` projection) are *not* reproduced — warn so the user
    can pass an explicit head/pooling that accounts for them rather than get silently divergent scores.
    """
    normalizes = False
    for idx, module in enumerate(st_model):
        name = type(module).__name__
        if name == "Normalize":
            normalizes = True
        elif idx >= 1 and name not in ("Pooling", "Normalize"):
            # Anything after the transformer body that isn't pooling/normalize changes the sentence
            # vector in a way this adapter does not replicate.
            warnings.warn(
                f"sentence-transformers module {name!r} after pooling is not reproduced by "
                "SentenceTransformerModel; embeddings/predictions may diverge from `st_model.encode`. "
                "Pass a TorchClassifierModel with an explicit body/head if this module matters.",
                stacklevel=3,
            )
    return normalizes


class SentenceTransformerModel(TorchClassifierModel):
    """Full-capability adapter for a ``sentence_transformers`` encoder + a classification ``head``.

    Extracts the underlying HuggingFace transformer body, its tokenizer, and the configured pooling
    mode from ``st_model``, then reuses :class:`TorchClassifierModel`'s ``body -> pool -> head`` machinery.
    The head is supplied by the caller (sentence-transformers keeps the classifier separate from the
    embedding model).

    Parameters
    ----------
    st_model : sentence_transformers.SentenceTransformer
        A sentence-transformers model whose first module wraps a HF transformer (``.auto_model``) and
        exposes a ``.tokenizer``, and which contains a ``Pooling`` module.
    head : nn.Module
        Maps the pooled ``(batch, hidden)`` sentence embedding to ``(batch, n_classes)`` logits.
    label_names : list[str] or None
        Class names in class-index order.
    pool : {"mean", "cls", "max"} or callable or None, optional
        Override the pooling mode. When ``None`` (default), read from ``st_model``'s ``Pooling`` module.
    normalize : bool or None, optional
        Whether to L2-normalize the sentence embedding before the head. When ``None`` (default),
        detected from ``st_model`` — set to ``True`` if it ends in a ``Normalize`` module (e.g.
        ``all-MiniLM-L6-v2``), so a head trained on ``st_model.encode`` output receives unit-length
        vectors. Pass ``True``/``False`` to override.
    embedding_space : str, optional
        Representation :meth:`embed` returns (see :class:`~shapash.model.encoder.EncoderClassifierModel`).
        Default ``"decision"`` (input to the final classification linear). Because the encoder is
        typically frozen here, this is what keeps the projection scatter class-discriminative; pass
        ``"pooled"`` to project the raw sentence embedding instead.
    batch_size : int, optional
        Batch size. Default 32.
    device : int or str or torch.device or None, optional
        Device for the SHAP pipeline; inherits the body's device when ``None``.
    max_length : int or None, optional
        Truncation length. When ``None`` (default), read from the model's ``max_seq_length`` so the
        length this adapter uses is explicit rather than implicit in the tokenizer. Sentence-transformers
        normally syncs the two already, so this rarely changes behaviour; pass a value to override.
    """

    def __init__(
        self,
        st_model,
        head,
        label_names: list[str] | None = None,
        *,
        pool: Any = None,
        normalize: bool | None = None,
        embedding_space: str = _DECISION_SPACE,
        batch_size: int = 32,
        device: int | str | object | None = None,
        max_length: int | None = None,
    ) -> None:
        body = self._extract_body(st_model)
        tokenizer = getattr(st_model, "tokenizer", None) or getattr(st_model[0], "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "Could not find a tokenizer on the sentence-transformers model. Pass a "
                "TorchClassifierModel with an explicit tokenizer instead."
            )
        resolved_pool = pool if pool is not None else _sentence_transformer_pool(st_model)
        resolved_normalize = normalize if normalize is not None else _sentence_transformer_normalize(st_model)
        resolved_max_length = max_length if max_length is not None else _sentence_transformer_max_length(st_model)
        super().__init__(
            body,
            head,
            tokenizer,
            label_names=label_names,
            pool=resolved_pool,
            normalize=resolved_normalize,
            embedding_space=embedding_space,
            batch_size=batch_size,
            device=device,
            max_length=resolved_max_length,
        )
        self.st_model = st_model

    @staticmethod
    def _extract_body(st_model):
        """Return the underlying HF transformer module (``auto_model``) from a ST model's first module."""
        try:
            first = st_model[0]
        except (TypeError, KeyError, IndexError) as exc:
            raise ValueError(
                "sentence-transformers model is not indexable as expected; cannot locate its "
                "transformer body. Use TorchClassifierModel with an explicit body instead."
            ) from exc
        body = getattr(first, "auto_model", None)
        if body is None:
            raise ValueError(
                "The sentence-transformers model's first module exposes no `.auto_model` HF transformer; "
                "cannot attribute through its embeddings. Use TorchClassifierModel with an explicit body."
            )
        return body
