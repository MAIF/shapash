"""NLP SHAP backend — token-level SHAP contributions for text classification models.

``NlpShapBackend`` wraps ``shap.Explainer`` for text inputs and implements
``run_explainer``, returning an ``NlpContributions``.  All shared infrastructure
(``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).

It masks text one of two ways: in the model's token ids when the model exposes them, as a string
otherwise.

Token-id masking
----------------
``shap.maskers.Text`` masks *strings*: it cuts the text into segments, glues the kept ones back
together with ``" [MASK]"`` for the hidden ones, and lets the model tokenize the result again. Two
things go wrong on the way, and both change what the model is asked:

- with nothing masked, the glued string is not the text. Tokens sharing a character (a byte-level
  BPE emoji or ``’``, a SentencePiece ``▁`` holding the next character's offsets) put it in two
  segments, so ``était`` becomes ``éétait``; runs of whitespace collapse to one space. SHAP stays
  additive, but for that other string;
- hiding one token changes its neighbours. Tokenizing ``"<mask>love"`` again gives ``love``
  where the text had ``Ġlove``, and ``"[MASK]believably"`` turns ``##bel`` into ``bel``, so a
  coalition is not "these tokens hidden, the others as they were".

A model exposing its token ids (:class:`~shapash.model.base.SupportsCaptumIG`, with a fast
tokenizer) is masked in its ids instead, by :class:`_TokenIdMasker`. The text is encoded once, by
the model (its truncation, its special tokens); a masked input is that encoding with the hidden
positions set to the model's :meth:`~shapash.model.base.SupportsCaptumIG.reference_ids`, the
``[MASK]`` reference LIG integrates from. With nothing hidden the model sees exactly its encoding of
the text, every masked input has the same length, and with everything hidden it sees LIG's
reference.

What SHAP perturbs are the pieces of the text's :class:`~shapash.compute.word_layout.WordLayout`: a
word's tokens, with tokens sharing a character as one piece, so a masked input never holds half a
character. Special tokens stay as they are in every input. The hierarchy SHAP's Partition explainer
walks keeps each word one branch (its pieces first, then words pairwise), so a word's value, the sum
of its pieces', is its Owen value as a single player. Words are then reported as LIG reports them,
through the same layout.

String masking
--------------
Any other model (a pipeline, a bare callable, or an explicit ``masker=`` / ``explainer_args``) goes
through ``shap.maskers.Text`` and keeps the defects above. The rest of this docstring is about that
path.

``shap.maskers.Text.token_segments`` emits *segments* of the source string rather than the
tokenizer's raw subword strings, and it does so in three different regimes: the offset-mapping path
slices each token up to the start of the next (so a segment carries the *trailing* gap text), the
slow-tokenizer fallback prepends a *leading* space to each token instead, and ``SimpleTokenizer``
splits on a regex. ``_aggregate_subwords`` merges those segments back into whole words — matching
the word-level highlights ``nlp_captum_lig_backend`` produces — and folds special-token attribution
into the baseline so ``base + Σ(word contributions)`` keeps SHAP's additive guarantee.

Word boundaries come from :func:`_merges`: flush on source-text whitespace *or* on a word/non-word
transition — except that tokens sharing a source character always merge (see below). A whitespace-only rule (what this module used previously) is wrong in two ways — it
glues punctuation onto its neighbours whenever the source has no space around it (``"superb!!!"``,
``"enjoy.Overall,I"``), and under the leading-space fallback regime it never fires at all, so an
entire sample collapses into a single "word".

Each word also records the character span it covers (``NlpContributions.token_spans``), read off the
masker tokenizer's offset mapping — the same call ``token_segments`` makes and then discards — and
its display string is that span of the source text. Concatenating segments instead is wrong when
two tokens share a character, which offsets make visible: byte-level BPE splits a multi-byte
character (``é``, an emoji) across tokens that all map to it, and SentencePiece can give a bare
``▁`` the offsets of the character after it. ``token_segments`` then hands each of those tokens the
same character, so concatenation printed ``éétait`` and an emoji became two "words". Tokens whose
spans overlap are one word, and the word's text comes from the source, never from the segments.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import shap

from shapash._optional import import_optional_module
from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.compute.spans import Span, is_unsegmented_script, locate_spans, span_from_offsets, trim_span
from shapash.compute.word_layout import WordLayout, word_layout
from shapash.model.base import SupportsCaptumIG, SupportsLogits, TextModel, has_capabilities

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# SHAP's masker reports special tokens as blank segments on its offset-mapping path; their
# attribution is folded into the baseline during word aggregation.
_BLANK_RE = re.compile(r"^\s*$")

# Last-ditch special-token detector, used *only* when the caller cannot supply the tokenizer's own
# special set (a bare scoring callable, or SHAP's tokenizer-less ``SimpleTokenizer``). It is
# deliberately not applied otherwise: ``[...]`` also matches literal source text such as
# ``[LAUGHTER]``, which would then be silently folded into the baseline instead of shown as a word.
_BRACKET_SPECIAL_RE = re.compile(r"^\[.*\]$")


def _is_special(segment: str, special_tokens: frozenset[str] | None) -> bool:
    """True when ``segment`` is a special token whose attribution belongs in the baseline."""
    stripped = segment.strip()
    if _BLANK_RE.match(stripped):
        return True
    if special_tokens is not None:
        return stripped in special_tokens
    return bool(_BRACKET_SPECIAL_RE.match(stripped))


def _merges(buffer: str, following: str) -> bool:
    """Whether the segment ``following`` continues the word currently held in ``buffer``.

    Merge only when ``buffer`` ends in a word character and ``following`` starts with one — i.e.
    flush on whitespace *or* on a word/non-word transition, so ``"superb"`` + ``"!"`` splits while
    ``"up"`` + ``"dating"`` merges. It reads only the segment strings, so it works for a bare
    callable with a custom masker as well as for a HuggingFace pipeline, and it holds across all
    three ``token_segments`` regimes.
    """
    if not buffer or not following:
        return False
    if buffer != buffer.rstrip():  # source-text whitespace — a hard word boundary
        return False
    last, first = buffer[-1], following[0]
    if not (last.isalnum() or last == "_") or not (first.isalnum() or first == "_"):
        return False
    # A script written without spaces (CJK, Thai, ...) would otherwise collapse into one "word";
    # breaking between its characters keeps units at the character level instead — the only
    # tokenizer-free option, and what ``BertPreTokenizer`` does anyway.
    return not (is_unsegmented_script(last) or is_unsegmented_script(first))


def _masker_special_tokens(explainer) -> frozenset[str] | None:
    """The masker tokenizer's ``all_special_tokens``, or ``None`` when no tokenizer is reachable."""
    tokenizer = getattr(getattr(explainer, "masker", None), "tokenizer", None)
    specials = getattr(tokenizer, "all_special_tokens", None)
    return frozenset(specials) if specials else None


def _segment_offsets(explainer, text: str, n_segments: int) -> list[Span] | None:
    """Character offsets of the tokens behind ``text``'s SHAP segments, or ``None`` when unavailable.

    Repeats the call ``shap.maskers.Text.token_segments`` makes on its offset-mapping path, whose
    offsets it uses and discards. ``None`` — so spans fall back to locating the words in the text —
    when no tokenizer is reachable, it cannot return offsets (a slow tokenizer: SHAP is then on its
    leading-space regime), or the count does not match the segments.
    """
    tokenizer = getattr(getattr(explainer, "masker", None), "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    except (NotImplementedError, TypeError, KeyError, ValueError):
        return None
    spans = [(0, 0) if o is None else (int(o[0]), int(o[1])) for o in offsets]
    return spans if len(spans) == n_segments else None


def _resolve_text_model(
    model: TextModel, masker, output_space: Literal["probability", "logit"]
) -> tuple[object, object]:
    """Pick the scoring callable and masker a ``TextModel`` offers for ``output_space``.

    Probability space is the model's own SHAP surface (``shap_callable`` + ``shap_masker``: a
    pipeline SHAP infers a ``Text`` masker from, or a bare ``predict`` with an explicit one). Logit
    space wraps :meth:`~shapash.model.base.SupportsLogits.predict_logits`, a bare function, so it
    always needs an explicit masker — built over the model's tokenizer, which is exactly the masker
    SHAP would infer from the pipeline, so the two spaces mask identically.
    """
    if output_space == "probability":
        return model.shap_callable, masker if masker is not None else model.shap_masker
    if not has_capabilities(model, SupportsLogits):
        raise TypeError(
            f"output_space='logit' needs a model implementing SupportsLogits; {type(model).__name__} can "
            "only return probabilities. Use output_space='probability', or an adapter with logit access "
            "(e.g. HFClassifierModel)."
        )
    if masker is None:
        masker = model.shap_masker
    if masker is None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise TypeError(
                f"output_space='logit' needs a tokenizer to mask text with, and {type(model).__name__} "
                "exposes none; pass masker= explicitly."
            )
        masker = shap.maskers.Text(tokenizer)
    return cast(SupportsLogits, model).predict_logits, masker


def _aggregate_subwords(
    tokens: list[str],
    contributions: np.ndarray,
    base_values: np.ndarray,
    special_tokens: frozenset[str] | None = None,
    text: str | None = None,
    offsets: Sequence[Span] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, list[tuple[Span, ...]]]:
    """Merge SHAP's segments into whole words, fold specials into the baseline, and locate each word.

    Word boundaries come from :func:`_merges`, so punctuation becomes its own unit while genuine
    subword pieces still merge; with ``offsets``, tokens covering a shared character merge too (see
    the module docstring). Special-token attribution is added to ``base_values`` rather than
    discarded, so ``base + Σ(word contributions)`` still equals the model output.

    Parameters
    ----------
    tokens : list[str]
        Segment strings for one sample (length ``seq``), as produced by SHAP's text masker.
    contributions : np.ndarray
        Per-segment contributions, shape ``(seq, n_classes)``.
    base_values : np.ndarray
        Baseline SHAP values for the sample, shape ``(n_classes,)``.
    special_tokens : frozenset[str] or None
        The masker tokenizer's ``all_special_tokens``. When given, non-blank specials are detected
        by membership — model-derived, not guessed. When ``None`` (a bare scoring callable, or a
        tokenizer-less masker), the bracket regex :data:`_BRACKET_SPECIAL_RE` stands in.
    text : str or None
        The explained text. Needed for ``offsets`` to be used, and to locate words without them.
    offsets : sequence of (int, int) or None
        Each segment's token offsets into ``text`` (:func:`_segment_offsets`). When given, word
        spans and strings come from them; otherwise words are located in ``text`` by search, the
        approximate path left for a masker that exposes no offsets.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray, list[tuple[tuple[int, int], ...]]]
        Word strings, per-word contributions ``(n_words, n_classes)``, the adjusted baseline
        ``(n_classes,)`` with special-token attribution folded in, and each word's character spans
        (``()`` where it could not be placed; all ``()`` without ``text``).
    """
    offs: Sequence[Span] = offsets if text is not None and offsets is not None and len(offsets) == len(tokens) else ()
    use_offsets = bool(offs)
    words: list[str] = []
    word_rows: list[np.ndarray] = []
    word_spans: list[tuple[Span, ...]] = []
    base = base_values.astype(float).copy()
    buffer_text = ""
    buffer_row: np.ndarray | None = None
    buffer_positions: list[int] = []

    def _flush() -> None:
        nonlocal buffer_text, buffer_row, buffer_positions
        if buffer_row is not None:
            span = trim_span(text, span_from_offsets(offs, buffer_positions)) if use_offsets and text else None
            words.append(text[span[0] : span[1]] if span is not None and text is not None else buffer_text.strip())
            word_rows.append(buffer_row)
            word_spans.append((span,) if span is not None else ())
            buffer_text, buffer_row, buffer_positions = "", None, []

    def _shares_a_character(i: int) -> bool:
        """Whether token ``i + 1`` covers a character the current word already does."""
        if not use_offsets or i + 1 >= len(tokens):
            return False
        span = span_from_offsets(offs, buffer_positions)
        start, end = offs[i + 1]
        return span is not None and end > start and start < span[1]

    for i, (tok, row) in enumerate(zip(tokens, contributions, strict=True)):
        if _is_special(tok, special_tokens):
            _flush()
            base = base + row
            continue
        buffer_text += tok
        buffer_row = row.astype(float).copy() if buffer_row is None else buffer_row + row
        buffer_positions.append(i)
        # A special token never continues a word (it is blank, or bracketed), so the raw next
        # segment is lookahead enough — no need to skip over specials to find the next content one.
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if not (_merges(buffer_text, following) or _shares_a_character(i)):
            _flush()
    _flush()

    if not use_offsets and text is not None:
        word_spans = locate_spans(text, words)
    stacked = np.stack(word_rows, axis=0) if word_rows else np.zeros((0, contributions.shape[-1]))
    return words, stacked, base, word_spans


# ----------------------------------------------------------------------------------------------------
# Token-id masking (see the module docstring)
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Encoding:
    """One text as :class:`_TokenIdMasker` masks it: the model's encoding, its reference, its words.

    Attributes
    ----------
    input_ids : np.ndarray
        The model's encoding of the text, ``(seq,)``.
    reference_ids : np.ndarray
        ``input_ids`` with every piece's tokens set to the model's reference id, ``(seq,)``.
    layout : WordLayout
        The text's words; their pieces are the features SHAP masks.
    feature_of_token : np.ndarray
        Each token's feature index, ``-1`` for a token in no feature (a special token).
    """

    input_ids: np.ndarray
    reference_ids: np.ndarray
    layout: WordLayout
    feature_of_token: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        owner = np.full(len(self.input_ids), -1, dtype=int)
        for f, positions in enumerate(self.layout.pieces):
            owner[positions] = f
        object.__setattr__(self, "feature_of_token", owner)


def _supports_token_ids(model: Any) -> bool:
    """Whether ``model`` can be masked in its token ids.

    It needs the id-level surface of :class:`~shapash.model.base.SupportsCaptumIG` (``encode``,
    ``reference_ids``, ``logits``) and a tokenizer that reports word ids and offsets — a fast one;
    a slow tokenizer reports neither, so pieces could not be grouped into words or placed.
    """
    if not has_capabilities(model, SupportsCaptumIG):
        return False
    try:
        return model.word_alignment("a") is not None and model.token_offsets("a") is not None
    except Exception:  # noqa: BLE001 — a probe: any failure means "not supported", never a crash
        return False


def _encode(model: SupportsCaptumIG, text: str) -> _Encoding:
    """Encode ``text`` with ``model`` and cut the encoding into words and pieces.

    Raises
    ------
    ValueError
        When the model's word alignment or offsets do not cover its encoding (they come from
        separate tokenizer calls, so a tokenizer disagreeing with itself would misplace every value).
    """
    input_ids, _, _ = model.encode(text)
    ids = np.asarray(input_ids[0].tolist(), dtype=np.int64)
    alignment = model.word_alignment(text)
    offsets = model.token_offsets(text)
    if alignment is None or offsets is None or len(offsets) != len(ids):
        raise ValueError(f"{type(model).__name__} gave no word alignment or offsets matching its encoding of {text!r}.")
    layout = word_layout(text, alignment[1], offsets)
    reference = np.asarray(model.reference_ids(input_ids)[0].tolist(), dtype=np.int64)
    masked = np.zeros(len(ids), dtype=bool)
    masked[[p for piece in layout.pieces for p in piece]] = True
    return _Encoding(input_ids=ids, reference_ids=np.where(masked, reference, ids), layout=layout)


def _balanced(nodes: list[int], rows: list[list[float]], sizes: dict[int, int], next_id: int) -> tuple[int, int]:
    """Merge ``nodes`` pairwise, level by level, into one; append the merges to ``rows``."""
    while len(nodes) > 1:
        merged = []
        for a, b in zip(nodes[::2], nodes[1::2], strict=False):
            sizes[next_id] = sizes[a] + sizes[b]
            rows.append([a, b, 0.0, sizes[next_id]])
            merged.append(next_id)
            next_id += 1
        if len(nodes) % 2:
            merged.append(nodes[-1])
        nodes = merged
    return nodes[0], next_id


def _clustering(layout: WordLayout) -> np.ndarray:
    """The feature hierarchy, as the linkage matrix SHAP's Partition explainer reads.

    Each word's pieces merge first, so every word is one branch; the words then merge pairwise with
    their neighbours. Heights are the cluster sizes, rescaled to ``(0, 1]``, which is also what
    ``shap.maskers.Text`` uses.
    """
    n = layout.n_pieces
    rows: list[list[float]] = []
    sizes = dict.fromkeys(range(n), 1)
    next_id, first = n, 0
    word_nodes = []
    for word in layout.words:
        node, next_id = _balanced(list(range(first, first + len(word))), rows, sizes, next_id)
        word_nodes.append(node)
        first += len(word)
    if word_nodes:
        _balanced(word_nodes, rows, sizes, next_id)
    tree = np.array(rows, dtype=float).reshape(-1, 4)
    if len(tree):
        tree[:, 2] = tree[:, 3] / tree[:, 3].max()
    return tree


class _TokenIdMasker(shap.maskers.Masker):
    """SHAP masker hiding a text's pieces in the model's own encoding (see the module docstring).

    Parameters
    ----------
    model : SupportsCaptumIG
        The model whose encoding is masked; :func:`_supports_token_ids` must hold.
    """

    # The ``Text`` masker's default: SHAP's batch size moves which nodes the Owen loop expands under
    # a ``max_evals`` budget, so keeping it keeps the documented trade-offs of that knob.
    default_batch_size = 5

    def __init__(self, model: SupportsCaptumIG) -> None:
        self.model = model
        self.immutable_outputs = True
        # What a hidden token becomes, read like ``shap.maskers.Text.mask_token``.
        self.mask_token = model.baseline_token
        self._text: str | None = None
        self._encoding: _Encoding | None = None

    def encoding(self, s: str) -> _Encoding:
        """The :class:`_Encoding` of ``s``, cached for the text being explained."""
        if self._encoding is None or s != self._text:
            self._encoding, self._text = _encode(self.model, s), s
        return self._encoding

    def __call__(self, mask: np.ndarray, s: str) -> tuple[np.ndarray]:
        """The encoding of ``s`` with every feature ``mask`` hides set to its reference id."""
        encoding = self.encoding(s)
        owner = encoding.feature_of_token
        hidden = owner >= 0
        hidden[hidden] = ~np.asarray(mask, dtype=bool)[owner[hidden]]
        return (np.where(hidden, encoding.reference_ids, encoding.input_ids)[None, :],)

    def shape(self, s: str) -> tuple[int, int]:
        """One masked input per mask, over the text's features."""
        return (1, self.encoding(s).layout.n_pieces)

    def mask_shapes(self, s: str) -> list[tuple[int]]:
        """One boolean per feature."""
        return [(self.encoding(s).layout.n_pieces,)]

    def feature_names(self, s: str) -> list[list[str]]:
        """Each feature's tokens, as the tokenizer writes them."""
        encoding = self.encoding(s)
        convert = self.model.tokenizer.convert_ids_to_tokens  # type: ignore[attr-defined]
        return [["".join(convert(encoding.input_ids[piece].tolist())) for piece in encoding.layout.pieces]]

    def clustering(self, s: str) -> np.ndarray:
        """See :func:`_clustering`."""
        return _clustering(self.encoding(s).layout)


class _TokenIdScorer:
    """The model scored on token ids, as SHAP calls it with :class:`_TokenIdMasker`'s inputs.

    Parameters
    ----------
    model : SupportsCaptumIG
        The model behind the masker.
    output_space : {"probability", "logit"}
        ``"logit"`` returns :meth:`~shapash.model.base.SupportsCaptumIG.logits`; ``"probability"``
        their softmax, which is what :meth:`~shapash.model.base.TextModel.predict` returns for a
        model exposing logits.
    """

    def __init__(self, model: SupportsCaptumIG, output_space: Literal["probability", "logit"]) -> None:
        self.model = model
        self.output_space = output_space

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        """``(n, n_classes)`` scores for ``n`` same-length id rows (one text's masked inputs)."""
        torch = import_optional_module("torch", extra=_NLP_EXTRA)
        device = self.model.embedding_layer.weight.device
        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            logits = self.model.logits(ids, torch.ones_like(ids)).float()
        if self.output_space == "probability":
            logits = torch.softmax(logits, dim=-1)
        return logits.cpu().numpy()


class NlpShapBackend(NlpBackend):
    """SHAP backend for text classification models (HuggingFace pipelines, etc.).

    Wraps ``shap.Explainer`` for text inputs and returns ``NlpContributions``
    via the shared ``get_local_contributions`` in ``NlpBackend``.

    Parameters
    ----------
    model : TextModel or callable
        A :class:`~shapash.model.base.TextModel`, whose scoring surface is chosen by
        ``output_space``; or a text callable accepted by ``shap.Explainer`` (e.g. a
        ``transformers.pipeline`` with ``return_all_scores=True``), which is taken to return
        probabilities. A ``TextModel`` exposing token ids (``SupportsCaptumIG`` with a fast tokenizer)
        is masked in its own encoding (see the module docstring) unless ``masker`` or
        ``explainer_args`` is given.
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    masker : any, optional
        Forwarded to ``shap.Explainer`` when ``explainer_args`` is not given. Typically ``None``: a
        model exposing token ids is then masked in its encoding, and SHAP infers a ``Text`` masker
        for a pipeline. Passing one (e.g. ``shap.maskers.Text(tokenizer)``) always takes the string
        path.
    explainer_args : dict, optional
        Keyword arguments forwarded to ``shap.Explainer.__init__``.
        Use ``{"explainer": SomeExplainerClass, ...}`` to inject a custom
        explainer class (the ``"explainer"`` key selects the class; all other
        keys are forwarded as its constructor arguments).
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to the explainer call (``__call__``). Two matter for text:

        ``max_evals`` (SHAP default 500) is the accuracy knob, and that default is far from
        converged: any imdb review over ~40 words exhausts it and lands 8-45% (max|Δ| / max|value|)
        from the fully-traversed explanation, which costs 3k-23k evals.

        ``batch_size`` (default 5, on both masking paths) caps how many masked variants reach
        ``model`` per call, and is *not* free. The Owen loop drains a whole batch from its
        best-first queue before pushing any children, so a larger batch expands different nodes
        under the same ``max_evals`` and deterministically shifts the values, within the error band
        above. Worth ~1.6x on short texts; 0.98x and 2.2x peak memory on imdb-length ones, where a
        512-token forward pass already saturates the GPU.
    batch_size : int or None, optional
        Batch size applied to ``model`` when it is a ``transformers`` pipeline that does not
        already have one (string path only). Default 64. Pass ``None`` to leave the pipeline untouched. Distinct from
        the explainer's ``batch_size`` above: this one only regroups the strings SHAP has already
        chosen, so it never changes the explanation.

        A pipeline built without ``batch_size`` scores a list of strings *one at a time*, so an
        explanation degenerates into ``max_evals`` (default 500) sequential single-sample forward
        passes. Batching lets them pad into one pass — ~1.9x on distilbert-imdb/GPU — for a
        numerically identical explanation (max|Δ| 1.1e-07), since the explainer's tree traversal
        is untouched.
    output_space : {"probability", "logit"}, default "probability"
        Which model output to explain. ``"logit"`` explains the raw pre-softmax scores, needs a
        ``TextModel`` implementing :class:`~shapash.model.base.SupportsLogits`, and is incompatible
        with ``explainer_args`` (which bring their own model). Same runtime and masking; the
        contributions stop cancelling across classes and stop saturating near 0/1, sum exactly to
        ``logits(x) - base``, and share their scale with ``nlp_captum_lig``.

    Raises
    ------
    TypeError
        If ``output_space="logit"`` and ``model`` cannot return logits.
    ValueError
        If ``output_space`` is not a known space, or ``"logit"`` is combined with ``explainer_args``.
    """

    name = "nlp_shap"
    # No reference is learned from data: a hidden token becomes the model's reference id (token-id
    # masking) or the tokenizer's mask token (``Text`` masker), so ``fit`` has nothing
    # backend-specific to learn here.
    reference_kind = "none"
    # Shapley values satisfy the efficiency axiom by construction, and stay additive
    # even under the Partition/Owen path SHAP silently takes for text. Owen values satisfy efficiency too.
    is_additive = True
    # The default: ``shap_callable`` resolves to a ``text-classification`` pipeline (softmax output),
    # not ``model.logits``. This is why per-token attributions cancel across classes: the explained
    # quantity sums to 1 for every masked variant, which is a constant-payoff game whose Shapley
    # values are all zero. ``output_space="logit"`` overrides it per instance.
    output_space: Literal["probability", "logit"] = "probability"
    requires_model_capabilities = ()  # a plain scoring callable is enough

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        masker=None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
        batch_size: int | None = 64,
        output_space: Literal["probability", "logit"] = "probability",
    ) -> None:
        if output_space not in ("probability", "logit"):
            raise ValueError(f"output_space must be 'probability' or 'logit', got {output_space!r}.")
        if output_space == "logit" and explainer_args:
            raise ValueError(
                "output_space='logit' cannot be combined with explainer_args: those build the explainer "
                "around their own model, whose output space this backend cannot check."
            )
        # How the text is masked: in the model's own encoding whenever it allows it and the caller
        # brought no masker or explainer of their own (see the module docstring).
        self.masking: Literal["token_ids", "text"] = "text"
        if isinstance(model, TextModel) and masker is None and not explainer_args and _supports_token_ids(model):
            masker = _TokenIdMasker(model)  # type: ignore[arg-type]
            model = _TokenIdScorer(model, output_space)  # type: ignore[arg-type]
            self.masking = "token_ids"
        elif isinstance(model, TextModel):
            model, masker = _resolve_text_model(model, masker, output_space)
        elif output_space == "logit":
            raise TypeError(
                "output_space='logit' needs a TextModel implementing SupportsLogits; a bare callable's "
                "output space cannot be checked."
            )
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.masker = masker
        self.output_space = output_space

        # ``_batch_size`` is what ``transformers.Pipeline.__call__`` reads (``None`` means 1) and
        # there is no public setter, so a release that renames it makes this a silent no-op rather
        # than an error. A pipeline the caller already configured is never overridden.
        #
        # transformers logs "using the pipelines sequentially on GPU ... please use a dataset" during
        # every GPU explanation. It is a false positive: the check is only ``call_count > 10 and
        # device == cuda`` and never inspects the input. We already pass lists, which transformers
        # wraps in a ``PipelineDataset`` and feeds to the same ``DataLoader`` a real Dataset would
        # get (measured: 1.009x, i.e. noise), and SHAP would flatten any dataset back to a list anyway.
        if batch_size is not None and getattr(model, "_batch_size", "absent") in (None, 1):
            model._batch_size = batch_size

        if "explainer" in self.explainer_args:
            shap_params = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
            self.explainer = self.explainer_args["explainer"](**shap_params)
        elif self.explainer_args:
            self.explainer = shap.Explainer(**self.explainer_args)
        elif self.masking == "token_ids":
            self.explainer = shap.Explainer(model, masker=self.masker, algorithm="partition")
        else:
            # ``masker=None`` lets SHAP auto-infer a Text masker from a transformers pipeline (the
            # HFClassifierModel/pipeline path); an explicit masker is required when ``model`` is a plain
            # scoring callable (external-head models expose one via ``TextModel.shap_masker``).
            self.explainer = shap.Explainer(model, masker=self.masker)

        # Resolved once: the masker's tokenizer is what decides which segments are special. ``None``
        # when no tokenizer is reachable (bare callable / ``SimpleTokenizer``) — see ``_is_special``.
        self._special_tokens = _masker_special_tokens(self.explainer)
        # What the masker substitutes for a hidden token: the tokenizer's mask token, or "..." when
        # it has none (SHAP's own fallback). ``None`` for a custom masker exposing no such attribute.
        self.baseline_token = getattr(getattr(self.explainer, "masker", None), "mask_token", None)

    def run_explainer(self, x) -> NlpContributions:
        """Run the SHAP text explainer and return all explanation components.

        Subword tokens are merged into whole words via ``_aggregate_subwords`` before being
        returned, so callers (word importance, sentence/token highlight plots) always see
        word-level contributions.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpContributions
            Ragged list of value arrays, baseline predictions, and token
            strings per sample.
        """
        if getattr(self, "masking", "text") == "token_ids":
            return self._run_token_ids(list(x))
        shap_explanation = self.explainer(x, **self.explainer_compute_args)

        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []
        spans: list[list[tuple[Span, ...]]] = []
        for text, tokens, values, base in zip(
            list(x), shap_explanation.data, shap_explanation.values, shap_explanation.base_values, strict=True
        ):
            words, word_contribs, word_base, word_spans = _aggregate_subwords(
                list(tokens),
                np.asarray(values),
                np.asarray(base),
                special_tokens=getattr(self, "_special_tokens", None),
                text=str(text),
                offsets=_segment_offsets(self.explainer, str(text), len(tokens)),
            )
            data.append(words)
            contributions.append(word_contribs)
            base_values.append(word_base)
            spans.append(word_spans)

        return NlpContributions(
            token_strings=data,
            values=contributions,
            base_values=np.stack(base_values, axis=0),
            token_spans=spans,
        )

    def _run_token_ids(self, texts: list[str]) -> NlpContributions:
        """Explain each text on its token-id features, then report its words (see the module docstring)."""
        masker: _TokenIdMasker = self.masker
        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []
        spans: list[list[tuple[Span, ...]]] = []
        for text in map(str, texts):
            encoding = masker.encoding(text)
            if encoding.layout.n_pieces == 0:  # nothing to hide: the text's own score is the baseline
                values = np.zeros((0, 0))
                base = self.model(encoding.input_ids[None, :])[0]
            else:
                explanation = self.explainer([text], **self.explainer_compute_args)
                values, base = explanation.values[0], explanation.base_values[0]
            words, word_values, word_base, word_spans = encoding.layout.aggregate(values, base)
            data.append(words)
            contributions.append(word_values)
            base_values.append(word_base)
            spans.append(word_spans)
        return NlpContributions(
            token_strings=data,
            values=contributions,
            base_values=np.stack(base_values, axis=0),
            token_spans=spans,
        )
