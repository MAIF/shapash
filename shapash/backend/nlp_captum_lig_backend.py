"""NLP Captum backend — token-level Layer Integrated Gradients contributions.

``NlpCaptumLigBackend`` is a second attribution method alongside ``NlpShapBackend``: it attributes a
target-class logit through the model's word-embedding layer with Captum's
``LayerIntegratedGradients`` (LIG), then sums over the embedding dimension to obtain one scalar per
token — the same per-token, per-class shape SHAP produces, so the webapp and the
``plot_sentence_highlight`` / ``plot_token_highlight`` renderers consume it unchanged.

LIG attributes at the tokenizer's *subword* granularity (``[CLS]``, ``[SEP]``, ``##`` continuation
pieces). To match the word-level highlights the SHAP backend produces, ``_aggregate_subwords`` merges
each word's subwords into a single contribution and folds the special tokens' attribution into the
baseline (see below) — so the output carries whole words, not ``##`` fragments.

LIG is a *completion* method: for each class ``c`` the token attributions sum to
``logits(x)[c] - logits(baseline)[c]``. We therefore report ``base_values[c] = logits(baseline)[c]``
so the additive ``base + Σ = total`` summary in ``plot_sentence_highlight`` stays consistent. These
values live in raw logit space — **not** the same space ``NlpShapBackend`` reports: SHAP explains
the pipeline's softmax probability output, not ``model.logits``, so the two backends' numbers are on
different scales and are not directly comparable.

Unlike the SHAP/LIME backends (which wrap a plain text callable), this backend needs the embedding
module and a logits forward pass, so it consumes a :class:`~shapash.model.base.TextModel` that
implements :class:`~shapash.model.base.SupportsCaptumIG` (e.g. ``HFClassifierModel``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

from shapash._optional import import_optional_module
from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.model.base import SupportsCaptumIG

_NLP_EXTRA = 'Install the NLP extra: pip install "shapash[nlp]".'

# Last-ditch special-token detector for the string fallback *only when the caller passes no explicit
# special set* (the pure-numpy unit tests, and the degenerate case). Covers bracket specials
# (BERT/DistilBERT/DeBERTa ``[CLS]``/``[SEP]``/``[PAD]``) and angle-bracket specials (RoBERTa/XLM-R
# ``<s>``/``</s>``/``<pad>``/``<mask>``), plus blanks. In real use the backend passes the model's own
# ``all_special_tokens`` instead, so specials are model-derived rather than guessed here.
_SPECIAL_RE = re.compile(r"^\[.*\]$|^<.*>$|^\s*$")

# The one irreducibly hard-coded bit of the *fallback* path: the subword marker convention. Byte-level
# BPE (``Ġ``, RoBERTa/XLM-R) and SentencePiece (``▁``, DeBERTa-v2/v3, T5) mark word *starts*; WordPiece
# marks *continuations* with ``##``. A slow tokenizer exposes no ``word_ids()`` to derive grouping from,
# so this convention is the only signal left. (Fast tokenizers never reach here — see word_alignment.)
_WORD_START_MARKERS = ("Ġ", "▁")  # "Ġ", "▁"


def _fallback_uses_word_start_markers(tokens: list[str]) -> bool:
    """True when the tokens use word-*start* markers (byte-BPE/SentencePiece) vs. WordPiece ``##``.

    Decides how an *unmarked* content token is treated: a continuation of the previous word under
    byte-BPE/SentencePiece (e.g. ``feel`` + ``ing``), but a brand-new word under WordPiece (``i``,
    ``am``). The very first content token is always a new word regardless (no previous word to join).
    """
    return any(t.strip().startswith(_WORD_START_MARKERS) for t in tokens)


def _aggregate_subwords(
    tokens: list[str],
    contributions: np.ndarray,
    base_values: np.ndarray,
    special_tokens: set[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Token-string fallback: merge subwords into words and fold specials into the baseline.

    Used only when the tokenizer exposes no ``word_ids()`` (a slow tokenizer); the exact, fully
    model-driven grouping lives in :meth:`~shapash.model.base.SupportsCaptumIG.word_alignment`. This
    collapses each word's subword attributions into a single value (matching the word-level highlights
    the SHAP backend produces) while **preserving LIG's completeness relation**: special-token
    attributions are added to ``base_values`` rather than discarded, so ``base + Σ(word contributions)``
    still equals ``logits(x)``.

    Handles every subword scheme via the marker convention (``##`` continuation vs ``Ġ``/``▁`` word-start
    — see :func:`_fallback_uses_word_start_markers`), so RoBERTa/XLM-R (byte-BPE) and DeBERTa/T5
    (SentencePiece) merge correctly, not just WordPiece.

    Parameters
    ----------
    tokens : list[str]
        Subword token strings for one sample (length ``seq``).
    contributions : np.ndarray
        Per-subword contributions, shape ``(seq, n_classes)``.
    base_values : np.ndarray
        Baseline logits for the sample, shape ``(n_classes,)``.
    special_tokens : set[str] or None
        The model's own special tokens (``tokenizer.all_special_tokens``). When given, specials are
        detected by membership — model-derived, not guessed. When ``None``, a last-ditch bracket /
        angle-bracket regex (:data:`_SPECIAL_RE`) is used instead.

    Returns
    -------
    tuple[list[str], np.ndarray, np.ndarray]
        Word strings, per-word contributions ``(n_words, n_classes)``, and the adjusted baseline
        ``(n_classes,)`` with special-token attribution folded in.
    """
    word_start_markers = _fallback_uses_word_start_markers(tokens)

    def _is_special(tok: str) -> bool:
        return tok in special_tokens if special_tokens is not None else bool(_SPECIAL_RE.match(tok))

    words: list[str] = []
    word_rows: list[np.ndarray] = []
    base = base_values.astype(float).copy()
    for tok, row in zip(tokens, contributions, strict=True):
        stripped = tok.strip()
        if _is_special(stripped) or stripped == "":
            base = base + row  # fold special-token attribution into the baseline (keep completeness)
        elif stripped.startswith("##"):  # WordPiece continuation
            _extend_or_start(words, word_rows, stripped[2:], row, continuation=bool(words))
        elif stripped.startswith(_WORD_START_MARKERS):  # byte-BPE/SentencePiece word start
            _extend_or_start(words, word_rows, stripped[1:], row, continuation=False)
        else:
            # Unmarked: a mid-word piece under byte-BPE/SentencePiece (join), else a whole word.
            _extend_or_start(words, word_rows, stripped, row, continuation=word_start_markers and bool(words))

    stacked = np.stack(word_rows, axis=0) if word_rows else np.zeros((0, contributions.shape[-1]))
    return words, stacked, base


def _extend_or_start(
    words: list[str], word_rows: list[np.ndarray], piece: str, row: np.ndarray, *, continuation: bool
) -> None:
    """Append ``piece``/``row`` to the current word (``continuation``) or start a new one, in place."""
    if continuation and words:
        words[-1] += piece
        word_rows[-1] = word_rows[-1] + row
    else:
        words.append(piece)
        word_rows.append(row.astype(float).copy())


def _aggregate_by_alignment(
    contributions: np.ndarray,
    base_values: np.ndarray,
    alignment: tuple[list[str], list[list[int]], list[int]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Exact aggregation from the tokenizer's own word grouping (:meth:`TextModel.word_alignment`).

    Sums each word's subword rows and folds special-token rows into the baseline — same completeness
    guarantee as :func:`_aggregate_subwords`, but with no string-marker guessing.
    """
    words, word_positions, special_positions = alignment
    base = base_values.astype(float).copy()
    if special_positions:
        base = base + contributions[special_positions].sum(axis=0)
    if word_positions:
        word_rows = np.stack([contributions[pos].sum(axis=0) for pos in word_positions], axis=0)
    else:
        word_rows = np.zeros((0, contributions.shape[-1]))
    return words, word_rows, base


def _valid_alignment(
    alignment: tuple[list[str], list[list[int]], list[int]] | None, n_tokens: int
) -> tuple[list[str], list[list[int]], list[int]] | None:
    """Return ``alignment`` only when its positions index within ``n_tokens``, else ``None``.

    A defensive guard: ``word_alignment`` re-encodes the text, so a tokenizer whose second encoding
    disagreed on length would misalign against the attribution rows — fall back to the string path.
    """
    if alignment is None:
        return None
    _, word_positions, special_positions = alignment
    indices = [i for pos in word_positions for i in pos] + list(special_positions)
    if indices and max(indices) >= n_tokens:
        return None
    return alignment


def _model_special_tokens(model: SupportsCaptumIG) -> set[str] | None:
    """Return the model tokenizer's ``all_special_tokens`` as a set, or ``None`` when unavailable."""
    specials = getattr(getattr(model, "tokenizer", None), "all_special_tokens", None)
    return set(specials) if specials else None


class NlpCaptumLigBackend(NlpBackend):
    """Layer Integrated Gradients backend for text classification models.

    Parameters
    ----------
    model : SupportsCaptumIG
        A text model exposing the Captum attribution surface (embedding module, ``encode``,
        ``reference_ids``, ``logits``) — typically an ``HFClassifierModel``.
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns.
    explainer_args : dict, optional
        Unused; accepted for interface parity with the other NLP backends.
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to ``LayerIntegratedGradients.attribute`` for every sample and
        class (e.g. ``{"n_steps": 50, "internal_batch_size": 16}``). ``n_steps`` defaults to 50.
    show_progress : bool, default False
        When True, wrap the per-sample attribution loop in a ``tqdm`` progress bar (LIG runs one
        integration per class per sample, so a batch is slow). Best-effort: if ``tqdm`` is not
        installed the loop runs silently. Left off by default so the library emits no stdout.

    Raises
    ------
    TypeError
        If ``model`` does not implement :class:`~shapash.model.base.SupportsCaptumIG`.
    """

    name = "nlp_captum_lig"
    # A constructed point from the tokenizer (model.reference_ids() -> pad/mask ids),
    # not learned from data.
    reference_kind = "point"
    # Integrated Gradients satisfies the completeness axiom (Sundararajan, Taly & Yan,
    # 2017, "Axiomatic Attribution for Deep Networks"): attributions sum exactly to
    # logits(x) - logits(baseline).
    is_additive = True
    # Attributes model.logits directly (see module docstring) — raw pre-softmax output, not the
    # probability nlp_shap explains.
    output_space = "logit"
    # Checked once by NlpBackend.__init__ instead of here — see its docstring.
    requires_model_capabilities = (SupportsCaptumIG,)

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
        show_progress: bool = False,
    ) -> None:
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.show_progress = show_progress
        captum_attr = import_optional_module("captum.attr", extra=_NLP_EXTRA)
        self.explainer = captum_attr.LayerIntegratedGradients(model.logits, model.embedding_layer)

    def run_explainer(self, x) -> NlpContributions:
        """Attribute each text with LIG, one pass per class, and return per-token contributions.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpContributions
            Ragged list of ``(n_tokens_i, n_classes)`` value arrays, per-sample baseline logits, and
            token strings per sample.
        """
        model: SupportsCaptumIG = self.model
        attribute_args = {"n_steps": 50, **self.explainer_compute_args}

        contributions: list[np.ndarray] = []
        base_values: list[np.ndarray] = []
        data: list[list[str]] = []

        for text in self._progress_iter(list(x)):
            input_ids, attention_mask, tokens = model.encode(text)
            ref_ids = model.reference_ids(input_ids)
            base_logits = model.logits(ref_ids, attention_mask)[0].detach().cpu().numpy()
            n_classes = int(base_logits.shape[-1])

            per_class = []
            for target in range(n_classes):
                attributions = self.explainer.attribute(
                    input_ids,
                    baselines=ref_ids,
                    target=target,
                    additional_forward_args=(attention_mask,),
                    **attribute_args,
                )
                # (1, seq, hidden) -> (seq,): sum the attribution over the embedding dimension.
                token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
                per_class.append(token_scores)

            # Collapse subwords to whole words (dropping special tokens) so LIG highlights read like
            # the SHAP backend's; special-token attribution is folded into base_logits to stay additive.
            # Prefer the model's exact, tokenizer-driven grouping; fall back to the token-string
            # heuristic (with the model's own special set) only for a slow tokenizer.
            stacked = np.stack(per_class, axis=-1)  # (seq, n_classes)
            alignment = _valid_alignment(model.word_alignment(text), n_tokens=stacked.shape[0])
            if alignment is not None:
                word_tokens, word_contribs, base_logits = _aggregate_by_alignment(stacked, base_logits, alignment)
            else:
                word_tokens, word_contribs, base_logits = _aggregate_subwords(
                    list(tokens), stacked, base_logits, special_tokens=_model_special_tokens(model)
                )
            contributions.append(word_contribs)  # (n_words, n_classes)
            base_values.append(base_logits)
            data.append(word_tokens)

        return NlpContributions(
            token_strings=data,
            values=contributions,
            base_values=np.stack(base_values, axis=0),
        )

    def _progress_iter(self, items: list[str]) -> Iterable[str]:
        """Wrap ``items`` in a ``tqdm`` bar when ``show_progress`` is set, else return it unchanged.

        Best-effort and dependency-free: ``tqdm`` is imported with ``errors="ignore"`` so a missing
        install simply yields the plain list rather than raising.
        """
        if not self.show_progress:
            return items
        tqdm_mod = import_optional_module("tqdm", errors="ignore")
        if tqdm_mod is None:
            return items
        return tqdm_mod.tqdm(items, desc="LIG attribution", unit="text")
