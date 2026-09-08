"""General encoder-classifier adapter — the full-capability text-model spine.

Every text model in the *full-capability* regime shares one structure::

    text -> [tokenizer] -> input_ids -> [embedding layer] -> inputs_embeds
         -> [ body -> pool -> head ] -> logits

`EncoderClassifierModel` implements **all** capability mixins (tokenization, embeddings, gradients and
the Captum attribution surface) once, in terms of a single *backbone* object that honours a small
HuggingFace-model-like contract. Concrete models are then thin **presets** that build such a backbone:

* :class:`~shapash.model.hf.HFClassifierModel` — the backbone *is* a
  ``AutoModelForSequenceClassification`` (pooling + head are internal to it).
* :class:`SentenceTransformerModel` / :class:`TorchClassifierModel` — wrap a separate encoder body,
  a pooling mode, and a classification head into a backbone via :class:`_EncoderHeadBackbone`.

The backbone contract (what every method here relies on):

* ``backbone(input_ids=?, inputs_embeds=?, attention_mask=?, output_hidden_states=?, **extra)`` returns
  an object with ``.logits``; with ``output_hidden_states=True`` it also exposes ``.hidden_states`` whose
  last element is the token-level ``(batch, seq, hidden)`` last hidden state.
* ``backbone.get_input_embeddings()`` returns the word-embedding ``nn.Module`` (with a ``.weight``).
* ``backbone.device`` is the device its parameters live on.
* ``backbone.named_modules()`` / ``backbone.zero_grad()`` — it is an ``nn.Module``.

Because the whole engine gates on *declared capabilities* (``has_capabilities``), never on a concrete
class, adding a new preset needs no change to any backend, generator, or webapp component.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np

from shapash._optional import import_optional_module
from shapash.model.base import (
    SupportsCaptumIG,
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
)

logger = logging.getLogger(__name__)

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# ``embedding_space`` keywords selecting the representation :meth:`EncoderClassifierModel.embed` returns
# (any other value is treated as a named backbone submodule):
#   "decision" — the input to the final classification ``nn.Linear`` (the features the model linearly
#                separates into logits). Reaches *into* the head, so a frozen-encoder + trained-head
#                model still projects in a class-discriminative space; ``pre_classifier``-like for HF.
#   "pooled"   — the pooled last hidden state of the encoder (mask-aware :attr:`pool`), the historical
#                behaviour. Best when the encoder itself was fine-tuned end-to-end for the task.
_DECISION_SPACE = "decision"
_POOLED_SPACE = "pooled"
_KEYWORD_SPACES = (_DECISION_SPACE, _POOLED_SPACE)

# Supported string pooling modes for turning a ``(batch, seq, hidden)`` token-level output into one
# vector per text. A callable ``pool`` may be passed instead (see ``EncoderClassifierModel``).
_POOL_MODES = ("mean", "cls", "max")

# Attribute names under which a backbone exposes its classification head, tried in order. ``head`` is
# what ``build_encoder_head_backbone`` names it; ``classifier`` is the universal HuggingFace
# sequence-classification convention (BERT, DistilBERT, RoBERTa, XLM-R, DeBERTa-v2), with ``score`` used
# by a few newer checkpoints. See ``EncoderClassifierModel._head_module``.
_HEAD_ATTRS = ("head", "classifier", "score")


def _encoded_length(encoding: Any) -> int | None:
    """Return the sequence length of a tokenizer ``encoding``, or ``None`` if it cannot be read.

    Handles every shape :meth:`EncoderClassifierModel._tokenize` can produce: a ``return_tensors="pt"``
    ``(batch, seq)`` tensor, a batched list of id lists, and a single un-tensorised list of ids.
    """
    ids = encoding["input_ids"]
    shape = getattr(ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    if not len(ids):
        return 0
    if isinstance(ids[0], (list, tuple)):
        return max(len(row) for row in ids)
    return len(ids)


def _position_capacity(backbone: Any) -> int | None:
    """Return how many *absolute* position ids ``backbone`` can index, or ``None`` when not knowable.

    ``None`` means "do not check" and is returned generously — this feeds a hard error, so a missed
    check (status quo) is far cheaper than a wrong one. Two cases yield ``None``:

    * no ``config.max_position_embeddings`` — a rotary/ALiBi model has no such table;
    * ``config.position_biased_input`` is ``False`` — DeBERTa v1/v2/v3 add no absolute position
      embeddings at all, so their ``max_position_embeddings`` is not an indexing bound and comparing
      against it would reject sequences the model handles fine.
    """
    config = getattr(backbone, "config", None)
    if config is None or getattr(config, "position_biased_input", True) is False:
        return None
    capacity = getattr(config, "max_position_embeddings", None)
    return capacity if isinstance(capacity, int) and not isinstance(capacity, bool) and capacity > 0 else None


def _pool_hidden(hidden: Any, attention_mask: Any, pool: Any) -> Any:
    """Reduce a token-level ``(batch, seq, hidden)`` tensor to ``(batch, hidden)``, mask-aware.

    Parameters
    ----------
    hidden : torch.Tensor
        Token-level output, shape ``(batch, seq, hidden)``.
    attention_mask : torch.Tensor
        Shape ``(batch, seq)``; padding positions (0) are excluded from ``mean``/``max``.
    pool : {"mean", "cls", "max"} or callable
        Pooling strategy. ``"mean"`` mask-averages over tokens (the default, matching the historical
        ``HFClassifierModel`` behaviour), ``"cls"`` takes position 0, ``"max"`` mask-maxes. A callable
        is invoked as ``pool(hidden, attention_mask)`` and must return ``(batch, hidden)``.

    Returns
    -------
    torch.Tensor
        Pooled representation, shape ``(batch, hidden)``.
    """
    if callable(pool):
        return pool(hidden, attention_mask)
    if pool == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1)
    if pool == "max":
        return hidden.masked_fill(mask == 0, float("-inf")).max(dim=1).values
    # "mean" — mask-aware average over non-padding tokens.
    return (hidden * mask).sum(1) / mask.sum(1)


class EncoderClassifierModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients, SupportsCaptumIG):
    """Full-capability text adapter over any *encoder + classification head* backbone.

    Implements prediction, tokenization, the input-embedding table, sentence embeddings in any
    representation space, per-token gradients and the Captum ``LayerIntegratedGradients`` surface —
    everything gradient-based counterfactual generators, LIG, and similar-example retrieval require —
    in terms of the backbone contract documented in this module. Concrete model classes subclass this
    as thin presets (see :class:`~shapash.model.hf.HFClassifierModel`,
    :class:`SentenceTransformerModel`, :class:`TorchClassifierModel`).

    Parameters
    ----------
    backbone : nn.Module
        An object honouring the HuggingFace-model-like backbone contract (see module docstring): it
        maps ``input_ids``/``inputs_embeds`` (+ ``attention_mask``) to ``.logits`` and, on request,
        ``.hidden_states``; and exposes ``get_input_embeddings()`` and ``.device``.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer matching the backbone. A *fast* tokenizer additionally enables exact
        subword-to-word grouping via :meth:`word_alignment`.
    label_names : list[str] or None
        Class names in class-index order. Presets typically resolve this from the model config.
    batch_size : int, optional
        Batch size for ``predict``, ``embed`` and the SHAP pipeline. Default 32.
    device : int or str or torch.device or None, optional
        Device for the SHAP pipeline. When ``None``, inherits the backbone's device.
    pool : {"mean", "cls", "max"} or callable, optional
        How a token-level layer output is reduced to one vector per text in :meth:`embed`. Default
        ``"mean"`` (the historical behaviour).
    normalize : bool, optional
        Whether the pooled sentence vector is L2-normalized before the classification head (and in
        :meth:`embed`). Default ``False``. Sentence-transformer pipelines that end in a ``Normalize``
        module train their head on unit-length embeddings; setting this reproduces that step so
        ``predict`` stays faithful and :meth:`embed` lives in the head's true decision space. Presets
        that own the pooling→head path (:class:`~shapash.model.torch_models.TorchClassifierModel` and
        subclasses) also apply it *inside* the backbone; it is a no-op for
        :class:`~shapash.model.hf.HFClassifierModel` (its head does its own pooling).
    max_length : int or None, optional
        Truncation length applied to every tokenizer call. When ``None`` (default) the tokenizer's own
        ``model_max_length`` applies. Set this when the model is served at a shorter length than its
        tokenizer allows, so long texts are not scored on more context than the model was trained with.
        :class:`~shapash.model.torch_models.SentenceTransformerModel` resolves it from the
        sentence-transformers ``max_seq_length``.
    embedding_space : str, optional
        Default representation :meth:`embed` returns — the space used for the 2-D projection scatter and
        similar-example retrieval. Default ``"decision"``: the input to the final classification linear
        (the model's decision space), which stays class-discriminative even when the encoder is frozen
        and only a head was trained. ``"pooled"`` returns the mask-pooled last hidden state
        (:attr:`pool`) — the historical behaviour, best when the encoder was fine-tuned end-to-end. Any
        other value must name a backbone submodule, whose output is used. Validated eagerly: an unknown
        name raises here rather than deep inside the first :meth:`embed` call.

    Raises
    ------
    ValueError
        If ``pool`` is not a supported mode or callable, or ``embedding_space`` is neither a keyword
        space nor the name of an existing backbone submodule.
    """

    def __init__(
        self,
        backbone,
        tokenizer,
        label_names: list[str] | None = None,
        batch_size: int = 32,
        device: int | str | object | None = None,
        *,
        pool: Any = "mean",
        normalize: bool = False,
        embedding_space: str = _DECISION_SPACE,
        max_length: int | None = None,
    ) -> None:
        if not callable(pool) and pool not in _POOL_MODES:
            raise ValueError(f"pool must be one of {_POOL_MODES} or a callable; got {pool!r}.")
        super().__init__(label_names=label_names)
        # Pin the backbone to inference mode. Every method here is a forward (or a backward *through* a
        # forward) for analysis — none trains — so a live Dropout/BatchNorm can only randomise results,
        # and it does so silently. ``from_pretrained`` happens to return eval-mode models, but a
        # hand-assembled backbone (see ``build_encoder_head_backbone``) does not; relying on the caller
        # makes correctness a convention. ``eval()`` recurses, so this covers body and head at once.
        eval_mode = getattr(backbone, "eval", None)
        if callable(eval_mode):
            eval_mode()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.pool = pool
        self.normalize = bool(normalize)
        self.max_length = max_length
        self._pipeline = None  # lazily built for SHAP
        self._decision_linear: Any = None  # lazily resolved final classification linear
        self._decision_resolved = False
        self.embedding_space = embedding_space  # validated by the property setter

    @property
    def embedding_space(self) -> str:
        """Default representation space :meth:`embed` returns when called without one.

        Assignable after construction — a caller switches the whole app between a semantic and a
        decision view this way, without reloading the model — and validated on assignment, so a bad
        name raises at the point of the mistake rather than on the next forward pass.
        """
        return self._embedding_space

    @embedding_space.setter
    def embedding_space(self, space: str) -> None:
        self._validate_space(space)
        self._embedding_space = space

    def resolve_space(self, space: str | None = None) -> str:
        """Return the concrete space name :meth:`embed` will use — :attr:`embedding_space` for ``None``.

        Validates an explicit ``space`` so cache-key derivation and :meth:`embed` reject the same names.
        See :meth:`~shapash.model.base.SupportsEmbeddings.resolve_space`.
        """
        if space is None:
            return self.embedding_space
        self._validate_space(space)
        return space

    def _validate_space(self, space: str) -> None:
        """Raise if ``space`` is neither a keyword space nor an existing backbone submodule name.

        Checked eagerly at construction (and on every explicit ``space=`` passed to :meth:`embed`) so a
        typo surfaces immediately with the available names, instead of as a ``KeyError`` from the first
        forward pass — which, in the webapp, happens at start-up after the model has already loaded.
        """
        if space in _KEYWORD_SPACES:
            return
        modules = dict(self.backbone.named_modules())
        if space not in modules:
            top_level = [name for name, _ in self.backbone.named_children()]
            raise ValueError(
                f"embedding space {space!r} is neither a keyword space {_KEYWORD_SPACES} nor a submodule "
                f"of the backbone. Available top-level modules: {top_level}."
            )

    @property
    def checkpoint(self) -> str:
        """Human-readable checkpoint identity — the backbone's ``config._name_or_path``.

        Falls back to the backbone's class name for a backbone with no HF-style config (e.g. a
        hand-rolled ``nn.Module`` body). Display-oriented: for a *cache* key see :attr:`model_id`,
        which adds the settings that change the vectors this checkpoint alone does not determine.
        """
        config = getattr(self.backbone, "config", None)
        return getattr(config, "_name_or_path", None) or type(self.backbone).__name__

    @property
    def architecture(self) -> str:
        """Best-effort architecture tag (e.g. ``"distilbert"``), from the backbone's HF config.

        Reads ``config.model_type``, transformers' own architecture tag; falls back to the
        backbone's class name when the backbone carries no such config (same fallback as
        :attr:`checkpoint`, for the same reason).
        """
        config = getattr(self.backbone, "config", None)
        return getattr(config, "model_type", None) or type(self.backbone).__name__

    @property
    def model_id(self) -> str:
        """Stable identity for this adapter *and every setting that changes its vectors/scores*.

        Downstream caches (the similar-example bank, the projection cache) key on this. It therefore
        includes :attr:`checkpoint` **and** ``pool`` / ``normalize`` — two models that differ only in
        pooling produce different embeddings, and keying on the checkpoint alone would make them
        silently share a cache entry.
        """
        pool = self.pool if isinstance(self.pool, str) else getattr(self.pool, "__name__", "callable")
        return f"{type(self).__name__}:{self.checkpoint}:pool={pool}:norm={int(self.normalize)}"

    # ------------------------------------------------------------------
    # Batched forward (shared by predict / embed)
    # ------------------------------------------------------------------

    def _tokenize(self, texts: str | list[str], **overrides: Any):
        """Tokenize with this model's truncation settings — the single place they are applied.

        Every tokenizer call in this class goes through here so they truncate *identically*. That is not
        just tidiness: :meth:`word_alignment` returns positions into the same subword axis as
        :meth:`encode`, so if the two truncated differently the attribution highlights would silently
        misalign on any text long enough to be cut.

        Also the single place the result is checked against what the backbone can actually index — see
        :meth:`_check_encoded_length`.
        """
        kwargs: dict[str, Any] = {"truncation": True}
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        kwargs.update(overrides)
        encoding = self.tokenizer(texts, **kwargs)
        self._check_encoded_length(encoding)
        return encoding

    def _check_encoded_length(self, encoding: Any) -> None:
        """Raise if ``encoding`` is longer than the backbone can index absolute positions for.

        This is a *diagnostic*, not a limiter: it never truncates and never fires for input the model
        could have handled. It exists because the failure it pre-empts is close to undebuggable. A
        sequence longer than ``max_position_embeddings`` makes the model's own embedding layer gather
        position ids past the end of its buffer, which trips a device-side assert; on CUDA that abort is
        *asynchronous*, so it surfaces at whatever unrelated line next synchronises — with a traceback
        pointing at innocent code, often thousands of samples into a batch job. Failing here instead
        names the real cause and the fix.

        The usual trigger is a checkpoint whose tokenizer config never set ``model_max_length``:
        ``transformers`` then reports a ~1e30 sentinel, ``truncation=True`` becomes a silent no-op, and a
        long text tokenizes unbounded. Passing an explicit ``max_length`` is the fix.

        Deliberately conservative — it compares against ``max_position_embeddings`` itself, so it cannot
        reject a sequence the model would have accepted. The cost is that it under-catches RoBERTa-style
        models by a couple of tokens: those offset position ids by ``pad_token_id + 1`` (which is why
        roberta-base pairs a 514-wide buffer with a real 512-token limit), so a sequence in that narrow
        band still reaches the original CUDA abort. Catching most cases with no false positives is worth
        more here than catching all of them at the risk of rejecting valid input.

        Raises
        ------
        ValueError
            If the encoded sequence exceeds the backbone's absolute-position capacity.
        """
        capacity = _position_capacity(self.backbone)
        if capacity is None:
            return
        length = _encoded_length(encoding)
        if length is None or length <= capacity:
            return
        raise ValueError(
            f"Text tokenized to {length} tokens, but {type(self.backbone).__name__} can only index "
            f"{capacity} absolute positions — running it would abort the CUDA context with a "
            f"device-side assert far from this call. Pass an explicit max_length (<= {capacity}) to "
            f"{type(self).__name__}. This usually means the tokenizer reports no model_max_length "
            f"(currently {getattr(self.tokenizer, 'model_max_length', None)!r}), which silently makes "
            "truncation=True a no-op."
        )

    def _encode_one(self, text: str):
        """Return ``(input_ids, attention_mask)`` batch-size-1 tensors on the backbone device."""
        enc = self._tokenize(text, return_tensors="pt")
        device = self.backbone.device
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _batches(self, texts: list[str], **forward_kwargs: Any):
        """Yield ``(batch, output)`` per chunk of ``texts``, tokenized, on-device and under ``no_grad``.

        The single place the tokenize -> to(device) -> forward loop lives; ``predict`` and every
        :meth:`embed` space differ only in what they pull out of each ``(batch, output)`` pair.
        ``forward_kwargs`` are passed through to the backbone (e.g. ``output_hidden_states=True``).
        """
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        device = self.backbone.device
        for i in range(0, len(texts), self.batch_size):
            batch = self._tokenize(texts[i : i + self.batch_size], padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.backbone(**batch, **forward_kwargs)
            yield batch, output

    def _pool_batch(self, tensor: Any, batch: dict) -> Any:
        """Reduce a captured tensor to ``(batch, hidden)``, pooling only if it is still token-level."""
        if tensor.dim() == 3:  # token-level (batch, seq, hidden) -> one vector per text
            return _pool_hidden(tensor, batch["attention_mask"], self.pool)
        return tensor

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return ``(n_texts, n_classes)`` softmax probabilities in class-index order."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        out = [torch.softmax(output.logits, dim=-1).cpu().numpy() for _, output in self._batches(list(texts))]
        return np.vstack(out)

    def _resolve_pipeline_device(self):
        """Pick the pipeline device, degrading an unavailable-GPU request to CPU with a warning."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        if self.device is None:
            # Inherit whatever device the backbone already lives on (CPU-safe by construction).
            return self.backbone.device
        wants_cuda = (isinstance(self.device, str) and self.device.startswith("cuda")) or (
            isinstance(self.device, int) and self.device >= 0
        )
        if wants_cuda and not torch.cuda.is_available():
            warnings.warn(
                f"Requested device {self.device!r} but no CUDA GPU is available; "
                "falling back to CPU for the SHAP pipeline.",
                stacklevel=2,
            )
            return torch.device("cpu")
        return self.device

    @property
    def shap_callable(self):
        """A ``text-classification`` pipeline built from the backbone, for SHAP's TextMasker.

        Requires the backbone to be a ``transformers`` model ``pipeline`` accepts. Presets whose
        backbone is not a ``PreTrainedModel`` (e.g. an external-head wrapper) override this.
        """
        if self._pipeline is None:
            transformers = import_optional_module("transformers", extra=_NLP_EXTRA)
            self._pipeline = transformers.pipeline(
                "text-classification",
                model=self.backbone,
                tokenizer=self.tokenizer,
                top_k=None,
                device=self._resolve_pipeline_device(),
                batch_size=self.batch_size,
                truncation=True,
            )
        return self._pipeline

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Return the tokenizer's tokens for ``text``."""
        return self.tokenizer.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Rebuild a display string from tokens (merges word-pieces)."""
        return self.tokenizer.convert_tokens_to_string(tokens)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embedding_table(self) -> tuple[list[str], np.ndarray]:
        """Return ``(vocab, matrix)`` from the backbone's input-embedding layer."""
        weight = self.backbone.get_input_embeddings().weight
        matrix = weight.detach().cpu().numpy()
        vocab = self.tokenizer.convert_ids_to_tokens(range(matrix.shape[0]))
        return vocab, matrix

    def embed(self, texts: list[str], space: str | None = None) -> np.ndarray:
        """Return one representation vector per text, shape ``(n_texts, hidden_dim)``.

        This is the single entry point for every representation space: the model's ``"decision"`` space
        (input to the final classification linear), the ``"pooled"`` last hidden state, or any named
        backbone submodule whose output is captured. The scatter projection and similar-example
        retrieval both go through it, so they always compare texts in the same space.

        Parameters
        ----------
        texts : list of str
            Input strings.
        space : str or None, optional
            Representation to return. When ``None`` (default), :attr:`embedding_space` is used. An
            explicit value is validated the same way (unknown names raise :class:`ValueError`).

        Returns
        -------
        np.ndarray, shape (n_texts, hidden_dim)
            Dense vectors aligned to ``texts``. The dimension depends on ``space``.
        """
        texts = list(texts)
        space = self.resolve_space(space)
        if space == _POOLED_SPACE:
            return self._pooled_embed(texts)
        if space == _DECISION_SPACE:
            return self._decision_embed(texts)
        # A named backbone submodule: capture its output on the forward pass.
        module = dict(self.backbone.named_modules())[space]
        return self._hooked_embed(texts, module, pre=False, space=space)

    def _pooled_embed(self, texts: list[str]) -> np.ndarray:
        """Return the mask-pooled last hidden state, shape ``(n_texts, hidden_dim)``.

        The token-level last hidden state is reduced with the configured :attr:`pool` (mask-aware for
        ``"mean"``/``"max"``), so padding tokens never contribute. When :attr:`normalize` is set the
        pooled vector is L2-normalized — matching a sentence-transformer ``Normalize`` module, so the
        embedding lives in the same space the classification head reads.
        """
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        out = []
        for batch, output in self._batches(texts, output_hidden_states=True):
            hidden = getattr(output, "hidden_states", None)
            if not hidden:
                # Part of the backbone contract (see the module docstring). Say so plainly — the raw
                # AttributeError names only ``hidden_states`` and not what the caller has to fix.
                raise AttributeError(
                    f"{type(self.backbone).__name__} returned no `hidden_states` for "
                    f"output_hidden_states=True, so the {_POOLED_SPACE!r} embedding space is "
                    f"unavailable. Use embedding_space={_DECISION_SPACE!r} or a named submodule, or "
                    "make the backbone honour output_hidden_states."
                )
            emb = _pool_hidden(hidden[-1], batch["attention_mask"], self.pool)
            if self.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.cpu().numpy())
        return np.vstack(out)

    def _hooked_embed(self, texts: list[str], module: Any, pre: bool, space: str) -> np.ndarray:
        """Return one vector per text captured at ``module``, shape ``(n_texts, hidden_dim)``.

        With ``pre=False`` a forward hook captures the module's *output*; with ``pre=True`` a forward
        pre-hook captures its first *input* (how the decision space is read off the final linear). A
        pooled ``(batch, hidden)`` capture is used as-is; a token-level ``(batch, seq, hidden)`` one is
        reduced with the configured :attr:`pool`. The backbone runs under ``no_grad`` — inference only.

        ``space`` names the requested space for error reporting only. It is needed because a module can
        exist on the backbone without lying on the path a forward pass takes — :meth:`_validate_space`
        checks a submodule name *exists*, not that it *runs* — and the resulting empty capture must say
        which space is unreachable rather than surface as a bare ``IndexError``.
        """
        captured: list = []
        if pre:
            handle = module.register_forward_pre_hook(lambda _mod, args: captured.append(args[0].detach()))
        else:
            handle = module.register_forward_hook(lambda _mod, _inp, out: captured.append(out.detach()))
        out = []
        try:
            for batch, _ in self._batches(texts):
                # The hook fires during the forward pass above; take the first capture of this batch and
                # clear so a module invoked more than once cannot leak captures across batches.
                if not captured:
                    raise RuntimeError(
                        f"embedding space {space!r} resolved to a {type(module).__name__} that did not "
                        f"run during {type(self.backbone).__name__}'s forward pass, so no representation "
                        "was captured. The module exists on the backbone but is not on the path this "
                        f"input takes. Use a keyword space {_KEYWORD_SPACES}, or name a submodule the "
                        "forward pass actually calls."
                    )
                vec = self._pool_batch(captured[0], batch)
                out.append(vec.cpu().numpy())
                captured.clear()
        finally:
            handle.remove()
        return np.vstack(out)

    def _head_module(self):
        """Return the classification head module, or ``None`` when it cannot be located.

        The head is found by attribute name (:data:`_HEAD_ATTRS`), which covers every preset without any
        of them needing to declare their layout: the fused backbone built by
        :func:`~shapash.model.torch_models.build_encoder_head_backbone` names it ``head``, and HuggingFace
        sequence-classification models name it ``classifier`` (a few newer ones ``score``). A preset whose
        head is named otherwise overrides this method.
        """
        for attr in _HEAD_ATTRS:
            module = getattr(self.backbone, attr, None)
            if module is not None:
                return module
        return None

    def _resolve_decision_linear(self):
        """Return the final classification ``nn.Linear`` (its input is the decision space), or ``None``.

        Located structurally: find the classification head (:meth:`_head_module`), then take the last
        ``nn.Linear`` inside it. That last linear is the one producing logits whether the head is a bare
        ``nn.Linear`` (BERT, DistilBERT, DeBERTa-v2) or a small compound module (``RobertaClassificationHead``
        = ``dense`` -> ``dropout`` -> ``out_proj``).

        Scoping the search to the head is what makes it reliable. Scanning the *whole* backbone for a
        linear whose ``out_features`` matched the class count — as this once did — searched dozens of
        candidates in registration rather than execution order, so it could silently select the wrong
        layer and return a plausible embedding from the wrong space. Inside a 1-3 linear head the two
        orders agree and there is nothing else to collide with. It also drops the dependency on
        ``label_names``, which is a display concern and should never have decided which representation a
        caller gets.

        Returns ``None`` when no head can be located, in which case :meth:`_decision_embed` warns and
        falls back to the pooled space.
        """
        if self._decision_resolved:
            return self._decision_linear
        self._decision_resolved = True
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        head = self._head_module()
        if head is None:
            logger.debug("Decision space: no head module found among attributes %s.", (_HEAD_ATTRS,))
            return self._decision_linear
        linears = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
        self._decision_linear = linears[-1] if linears else None
        logger.debug(
            "Decision space: head %s exposes %d linear(s); resolved=%s",
            type(head).__name__,
            len(linears),
            self._decision_linear is not None,
        )
        return self._decision_linear

    def _decision_embed(self, texts: list[str]) -> np.ndarray:
        """Return the input to the final classification linear, shape ``(n_texts, n_features)``.

        A forward pre-hook captures the vector the final linear reads (the model's decision space). When
        no final linear can be resolved (no head found under :data:`_HEAD_ATTRS`, or a head containing no
        ``nn.Linear``), falls back to :meth:`_pooled_embed` — and warns, because the caller asked for the
        decision space and is getting a different one, which would otherwise be invisible.
        """
        linear = self._resolve_decision_linear()
        if linear is None:
            warnings.warn(
                f"{type(self).__name__}: could not locate a final classification nn.Linear — no head "
                f"found under {_HEAD_ATTRS} on {type(self.backbone).__name__}, or it contains no linear. "
                f"Embedding space {_DECISION_SPACE!r} falls back to {_POOLED_SPACE!r} (the pooled last "
                f"hidden state). Pass embedding_space={_POOLED_SPACE!r} to silence this, a submodule name "
                "to pick the representation explicitly, or override _head_module() on your adapter.",
                UserWarning,
                stacklevel=3,
            )
            logger.info("Decision space unavailable — falling back to the pooled last hidden state.")
            return self._pooled_embed(texts)
        return self._hooked_embed(texts, linear, pre=True, space=_DECISION_SPACE)

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------

    def token_gradients(self, text: str, target_class: int) -> tuple[list[str], np.ndarray]:
        """Return ``(tokens, grads)`` of the target-class logit w.r.t. input embeddings.

        Special tokens (``[CLS]``/``[SEP]``/padding) are dropped so the returned tokens and gradient
        rows are aligned to substitutable content tokens only.
        """
        input_ids, attention_mask = self._encode_one(text)
        embed_layer = self.backbone.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()

        self.backbone.zero_grad()
        logits = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        logits[0, target_class].backward()

        grads = inputs_embeds.grad[0].detach().cpu().numpy()  # (seq, hidden)
        ids = input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        special_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

        keep = [i for i, is_special in enumerate(special_mask) if not is_special]
        kept_tokens = [tokens[i] for i in keep]
        kept_grads = grads[keep]
        return kept_tokens, kept_grads

    # ------------------------------------------------------------------
    # Captum layer-attribution surface
    # ------------------------------------------------------------------

    @property
    def embedding_layer(self):
        """Return the word-embedding ``nn.Module`` (what ``LayerIntegratedGradients`` attributes through)."""
        return self.backbone.get_input_embeddings()

    def encode(self, text: str):
        """Return ``(input_ids, attention_mask, tokens)`` — batch-size-1 tensors + aligned token strings."""
        input_ids, attention_mask = self._encode_one(text)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return input_ids, attention_mask, tokens

    def reference_ids(self, input_ids):
        """Return baseline ids: content tokens replaced by the pad/mask reference, special tokens kept."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        ids = input_ids[0].tolist()
        special_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        ref_id = self.tokenizer.pad_token_id
        if ref_id is None:
            ref_id = self.tokenizer.mask_token_id or self.tokenizer.unk_token_id or 0
        ref = [tid if is_special else ref_id for tid, is_special in zip(ids, special_mask, strict=True)]
        return torch.tensor([ref], device=input_ids.device)

    def logits(self, input_ids, attention_mask):
        """Return raw classification logits ``(batch, n_classes)`` (the Captum forward func)."""
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits

    def word_alignment(self, text: str) -> tuple[list[str], list[list[int]], list[int]] | None:
        """Group subwords into whole words via the fast tokenizer's ``word_ids()`` (see base ABC).

        Re-encodes ``text`` with the same ``truncation=True`` as :meth:`encode`, so the returned
        positions align with the token/attribution axis. ``word_ids()`` maps each subword to its word
        index (``None`` for special tokens), giving exact, scheme-independent grouping — no ``##`` / ``Ġ``
        / ``▁`` string-guessing. Display strings come from ``convert_tokens_to_string`` so each word is
        rebuilt correctly whatever the tokenization scheme. Returns ``None`` for a slow tokenizer (no
        ``word_ids()``), letting the caller fall back to a token-string heuristic.
        """
        if not getattr(self.tokenizer, "is_fast", False):
            return None
        enc = self._tokenize(text)
        word_ids = enc.word_ids()
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"])

        # Special tokens (and any tokenless position) carry word_id None: fold their attribution into
        # the baseline. Content subwords sharing a word_id compose one word.
        special_positions = [i for i, wid in enumerate(word_ids) if wid is None]
        word_positions: list[list[int]] = []
        prev_wid: int | None = None
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != prev_wid:
                word_positions.append([i])
                prev_wid = wid
            else:
                word_positions[-1].append(i)
        words = [self.tokenizer.convert_tokens_to_string([tokens[i] for i in pos]).strip() for pos in word_positions]
        return words, word_positions, special_positions
