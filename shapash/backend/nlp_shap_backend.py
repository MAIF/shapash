"""NLP SHAP backend — token-level SHAP contributions for text classification models.

``NlpShapBackend`` wraps ``shap.Explainer`` for text inputs and implements
``run_explainer``, returning an ``NlpContributions``.  All shared infrastructure
(``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).

``shap.maskers.Text.token_segments`` emits *segments* of the source string rather than the
tokenizer's raw subword strings, and it does so in three different regimes: the offset-mapping path
slices each token up to the start of the next (so a segment carries the *trailing* gap text), the
slow-tokenizer fallback prepends a *leading* space to each token instead, and ``SimpleTokenizer``
splits on a regex. ``_aggregate_subwords`` merges those segments back into whole words — matching
the word-level highlights ``nlp_captum_lig_backend`` produces — and folds special-token attribution
into the baseline so ``base + Σ(word contributions)`` keeps SHAP's additive guarantee.

Word boundaries come from :func:`_merges`: flush on source-text whitespace *or* on a word/non-word
transition. A whitespace-only rule (what this module used previously) is wrong in two ways — it
glues punctuation onto its neighbours whenever the source has no space around it (``"superb!!!"``,
``"enjoy.Overall,I"``), and under the leading-space fallback regime it never fires at all, so an
entire sample collapses into a single "word".
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import shap

from shapash.backend.nlp_backend import NlpBackend, NlpContributions

# SHAP's masker reports special tokens as blank segments on its offset-mapping path; their
# attribution is folded into the baseline during word aggregation.
_BLANK_RE = re.compile(r"^\s*$")

# Last-ditch special-token detector, used *only* when the caller cannot supply the tokenizer's own
# special set (a bare scoring callable, or SHAP's tokenizer-less ``SimpleTokenizer``). It is
# deliberately not applied otherwise: ``[...]`` also matches literal source text such as
# ``[LAUGHTER]``, which would then be silently folded into the baseline instead of shown as a word.
_BRACKET_SPECIAL_RE = re.compile(r"^\[.*\]$")

# Scripts written without inter-word spaces. A character-class boundary test sees such a sentence as
# one uninterrupted run of word characters, so :func:`_merges` would collapse it into a single
# "word"; breaking between these characters keeps units at the character level instead — the only
# tokenizer-free option, and what ``BertPreTokenizer`` does anyway (it inserts whitespace around
# CJK). Hangul is deliberately absent: Korean *is* space-segmented, so its subword pieces must merge
# like any other alphabetic script's.
_UNSEGMENTED_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "THAI", "LAO", "KHMER", "MYANMAR")


def _is_special(segment: str, special_tokens: frozenset[str] | None) -> bool:
    """True when ``segment`` is a special token whose attribution belongs in the baseline."""
    stripped = segment.strip()
    if _BLANK_RE.match(stripped):
        return True
    if special_tokens is not None:
        return stripped in special_tokens
    return bool(_BRACKET_SPECIAL_RE.match(stripped))


def _is_unsegmented_script(char: str) -> bool:
    """True when ``char`` belongs to a script written without spaces between words."""
    return unicodedata.name(char, "").startswith(_UNSEGMENTED_SCRIPTS)


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
    return not (_is_unsegmented_script(last) or _is_unsegmented_script(first))


def _masker_special_tokens(explainer) -> frozenset[str] | None:
    """The masker tokenizer's ``all_special_tokens``, or ``None`` when no tokenizer is reachable."""
    tokenizer = getattr(getattr(explainer, "masker", None), "tokenizer", None)
    specials = getattr(tokenizer, "all_special_tokens", None)
    return frozenset(specials) if specials else None


def _aggregate_subwords(
    tokens: list[str],
    contributions: np.ndarray,
    base_values: np.ndarray,
    special_tokens: frozenset[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Merge SHAP's segments into whole words and fold specials into the baseline.

    Word boundaries come from :func:`_merges`, so punctuation becomes its own unit while genuine
    subword pieces still merge. Special-token attribution is added to ``base_values`` rather than
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

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        Word strings, per-word contributions ``(n_words, n_classes)``, and the adjusted baseline
        ``(n_classes,)`` with special-token attribution folded in.
    """
    words: list[str] = []
    word_rows: list[np.ndarray] = []
    base = base_values.astype(float).copy()
    buffer_text = ""
    buffer_row: np.ndarray | None = None

    def _flush() -> None:
        nonlocal buffer_text, buffer_row
        if buffer_row is not None:
            words.append(buffer_text.strip())
            word_rows.append(buffer_row)
            buffer_text, buffer_row = "", None

    for i, (tok, row) in enumerate(zip(tokens, contributions, strict=True)):
        if _is_special(tok, special_tokens):
            _flush()
            base = base + row
            continue
        buffer_text += tok
        buffer_row = row.astype(float).copy() if buffer_row is None else buffer_row + row
        # A special token never continues a word (it is blank, or bracketed), so the raw next
        # segment is lookahead enough — no need to skip over specials to find the next content one.
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if not _merges(buffer_text, following):
            _flush()
    _flush()

    stacked = np.stack(word_rows, axis=0) if word_rows else np.zeros((0, contributions.shape[-1]))
    return words, stacked, base


class NlpShapBackend(NlpBackend):
    """SHAP backend for text classification models (HuggingFace pipelines, etc.).

    Wraps ``shap.Explainer`` for text inputs and returns ``NlpContributions``
    via the shared ``get_local_contributions`` in ``NlpBackend``.

    Parameters
    ----------
    model : callable
        A text pipeline callable accepted by ``shap.Explainer`` (e.g. a
        ``transformers.pipeline`` with ``return_all_scores=True``).
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    masker : any, optional
        Forwarded to ``shap.Explainer`` when ``explainer_args`` is not given.
        Typically ``None`` for text (SHAP auto-selects a ``TextMasker``).
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

        ``batch_size`` (default: the ``Text`` masker's 5) caps how many masked variants reach
        ``model`` per call, and is *not* free. The Owen loop drains a whole batch from its
        best-first queue before pushing any children, so a larger batch expands different nodes
        under the same ``max_evals`` and deterministically shifts the values, within the error band
        above. Worth ~1.6x on short texts; 0.98x and 2.2x peak memory on imdb-length ones, where a
        512-token forward pass already saturates the GPU.
    batch_size : int or None, optional
        Batch size applied to ``model`` when it is a ``transformers`` pipeline that does not
        already have one. Default 64. Pass ``None`` to leave the pipeline untouched. Distinct from
        the explainer's ``batch_size`` above: this one only regroups the strings SHAP has already
        chosen, so it never changes the explanation.

        A pipeline built without ``batch_size`` scores a list of strings *one at a time*, so an
        explanation degenerates into ``max_evals`` (default 500) sequential single-sample forward
        passes. Batching lets them pad into one pass — ~1.9x on distilbert-imdb/GPU — for a
        numerically identical explanation (max|Δ| 1.1e-07), since the explainer's tree traversal
        is untouched.
    """

    name = "nlp_shap"
    # No reference is learned from data: the ``Text`` masker infers its own masking
    # scheme from the pipeline/tokenizer (``masker=None`` below), so ``fit`` has
    # nothing backend-specific to learn here.
    reference_kind = "none"
    # Shapley values satisfy the efficiency axiom by construction, and stay additive
    # even under the Partition/Owen path SHAP silently takes for text. Owen values satisfy efficiency too.
    is_additive = True
    # ``shap_callable`` resolves to a ``text-classification`` pipeline (softmax output), not
    # ``model.logits``. This is why per-token attributions cancel across classes: the explained
    # quantity sums to 1 for every masked variant, which is a constant-payoff game whose Shapley
    # values are all zero.
    output_space = "probability"
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
    ) -> None:
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.masker = masker

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
        else:
            # ``masker=None`` lets SHAP auto-infer a Text masker from a transformers pipeline (the
            # HFClassifierModel/pipeline path); an explicit masker is required when ``model`` is a plain
            # scoring callable (external-head models expose one via ``TextModel.shap_masker``).
            self.explainer = shap.Explainer(model, masker=self.masker)

        # Resolved once: the masker's tokenizer is what decides which segments are special. ``None``
        # when no tokenizer is reachable (bare callable / ``SimpleTokenizer``) — see ``_is_special``.
        self._special_tokens = _masker_special_tokens(self.explainer)

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
        shap_explanation = self.explainer(x, **self.explainer_compute_args)

        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []
        for tokens, values, base in zip(
            shap_explanation.data, shap_explanation.values, shap_explanation.base_values, strict=True
        ):
            words, word_contribs, word_base = _aggregate_subwords(
                list(tokens),
                np.asarray(values),
                np.asarray(base),
                special_tokens=getattr(self, "_special_tokens", None),
            )
            data.append(words)
            contributions.append(word_contribs)
            base_values.append(word_base)

        return NlpContributions(
            token_strings=data,
            values=contributions,
            base_values=np.stack(base_values, axis=0),
        )
