"""NLP LIME backend — word-level LIME contributions for text classification models.

``NlpLimeBackend`` wraps ``LimeTextExplainer`` and implements ``run_explainer``,
returning an ``NlpContributions``. All shared infrastructure
(``get_local_contributions``, common ``__init__`` skeleton) lives in
``NlpBackend`` (see ``nlp_backend.py``).

LIME works at word level (bag-of-words by default) rather than at the subword
or token level used by SHAP.  Each sample's ``token_strings`` is therefore the
list of unique vocabulary words found by the ``split_expression`` tokeniser, not
HuggingFace subword tokens.  Every word is scored by default (``num_features="all"``), so
a zero weight means LIME found no effect — not that the word fell outside a top-k.
Shapash's plots pick their own top-k at display time.
"""

from __future__ import annotations

import numpy as np

try:
    from lime.lime_text import IndexedCharacters, IndexedString, LimeTextExplainer

    _lime_available = True
except ImportError:
    _lime_available = False

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.model.base import TextModel


class NlpLimeBackend(NlpBackend):
    """LIME backend for text classification models.

    Wraps ``LimeTextExplainer`` and returns ``NlpContributions`` via the shared
    ``get_local_contributions`` in ``NlpBackend``.

    Parameters
    ----------
    model : TextModel or callable
        A ``TextModel`` adapter (scored through its ``predict``, like the other NLP
        backends), or a scoring function ``f(texts: list[str]) -> np.ndarray`` of shape
        ``(n_texts, n_classes)`` returning class probabilities, columns in the order of
        ``label_names`` (e.g. an sklearn pipeline's ``predict_proba``). A HuggingFace
        pipeline built with ``top_k=None`` (or ``return_all_scores=True``) is also
        accepted as-is, but then ``label_names`` is required so the scores can be
        matched to columns by label.
    preprocessing : None
        Unused; accepted for interface compatibility with ``BaseBackend``.
    label_names : list[str] or None
        Class names in the same order as the model output columns. Defaults to
        ``model.label_names`` for a ``TextModel``. Forwarded to ``LimeTextExplainer``
        as ``class_names``. When unknown, every output column is still explained,
        named by position.
    mask_string : str or None
        Token used to replace masked words when ``bow=False``.  Mirrors the
        ``masker`` parameter of ``NlpShapBackend``.  Defaults to
        ``'UNKWORDZ'`` inside LIME.
    explainer_args : dict, optional
        Keyword arguments forwarded to ``LimeTextExplainer.__init__``.
        Supported keys: ``kernel_width``, ``kernel``, ``verbose``,
        ``feature_selection``, ``split_expression``, ``bow``,
        ``random_state`` (pass as an int), ``char_level``.
        Use ``{"explainer": SubclassOfLimeTextExplainer, ...rest...}`` to
        inject a custom explainer class (mirrors the SHAP escape hatch).
    explainer_compute_args : dict, optional
        Keyword arguments forwarded to ``LimeTextExplainer.explain_instance``
        on every call.  Supported keys: ``num_features`` (default ``"all"``:
        every word of the text; an int keeps LIME's top-k, zero-filling the
        rest, which then reads as "no effect"), ``num_samples`` (default 5000), ``distance_metric`` (default
        ``'cosine'``), ``model_regressor``, ``labels``, ``top_labels``.
        If neither ``labels`` nor ``top_labels`` is provided, ``labels`` is
        automatically set to ``range(len(label_names))``.
    show_progress : bool, default False
        When True, show a ``tqdm`` progress bar over the texts (LIME scores
        ``num_samples`` perturbations per text, so a batch is slow). Best-effort:
        without ``tqdm`` installed the loop runs silently.

    Examples
    --------
    LIME samples random perturbations, so every example fixes ``random_state``: without it the
    weights change from run to run.

    With a ``TextModel`` adapter — the same object ``NlpShapBackend`` and
    ``NlpCaptumLigBackend`` take; ``label_names`` comes from the model:

    >>> from shapash.model.hf import HFClassifierModel
    >>> model = HFClassifierModel.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
    >>> backend = NlpLimeBackend(model, explainer_args={"random_state": 0})

    With a scikit-learn text classifier, whose ``predict_proba`` already maps a list of
    strings to an ``(n_texts, n_classes)`` array:

    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import make_pipeline
    >>> clf = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(train_texts, train_labels)
    >>> backend = NlpLimeBackend(
    ...     clf.predict_proba,
    ...     label_names=list(clf.classes_),
    ...     explainer_args={"random_state": 0},
    ...     explainer_compute_args={"num_samples": 3000},
    ... )

    With a HuggingFace pipeline, which must return every class score:

    >>> from transformers import pipeline
    >>> pipe = pipeline("text-classification", model="...", top_k=None)
    >>> backend = NlpLimeBackend(pipe, label_names=["NEGATIVE", "POSITIVE"], explainer_args={"random_state": 0})

    Then hand the backend to ``NlpExplainer`` along with the same model (the ``TextModel`` itself
    when you have one — that keeps the What-if Lab available):

    >>> from shapash.explainer.nlp_explainer import NlpExplainer
    >>> xpl = NlpExplainer(model, backend=backend)
    >>> explanation = xpl.explain(texts)
    """

    name = "nlp_lime"
    # Unlike tabular LIME (``LimeTabularExplainer(training_data=...)``, which needs a
    # background corpus to build its discretizer/feature statistics), LimeTextExplainer
    # takes no training-data-like constructor argument at all (see __init__ below) —
    # explain_instance perturbs the instance text itself (word removal / mask_string
    # substitution). There is nothing backend-specific for ``fit`` to learn here.
    reference_kind = "none"
    # LIME fits a locally-weighted linear surrogate (Ribeiro, Singh & Guestrin, 2016,
    # "Why Should I Trust You?") whose objective is local fidelity, not exact
    # reconstruction — there is no efficiency/completeness axiom forcing the weights to
    # sum to f(x) - f(baseline). Matches tabular LimeBackend.support_groups=False, for
    # the same algorithmic reason (not because tabular already decided it).
    is_additive = False
    # ``model`` returns class probabilities (see its docstring above) and the surrogate is fit
    # against that output directly, so this is a probability-space explanation like nlp_shap's —
    # not the raw-logit space nlp_captum_lig reports.
    output_space = "probability"
    requires_model_capabilities = ()  # a plain scoring callable is enough

    def __init__(
        self,
        model,
        preprocessing=None,
        label_names: list[str] | None = None,
        mask_string: str | None = None,
        explainer_args: dict | None = None,
        explainer_compute_args: dict | None = None,
        show_progress: bool = False,
    ) -> None:
        if not _lime_available:
            raise ImportError("lime is required for NlpLimeBackend — pip install lime")

        # LIME only needs probabilities, so a TextModel is scored through ``predict`` — no capability
        # beyond the base contract, which is why ``requires_model_capabilities`` stays empty.
        if isinstance(model, TextModel):
            if label_names is None:
                label_names = model.label_names
            model = model.predict
        super().__init__(model, preprocessing, label_names, explainer_args, explainer_compute_args)
        self.mask_string = mask_string
        self.show_progress = show_progress
        # Written into the settings rather than applied silently, so the choice is visible on the
        # backend and part of NlpExplainer's cache key — a result computed under LIME's own default
        # (10) is never served for "every word".
        # A new dict, not ``setdefault``: the caller's own dict must not be modified.
        self.explainer_compute_args = {"num_features": "all", **self.explainer_compute_args}

        if "explainer" in self.explainer_args:
            lime_params = {k: v for k, v in self.explainer_args.items() if k != "explainer"}
            self.explainer = self.explainer_args["explainer"](**lime_params)
        else:
            self.explainer = LimeTextExplainer(
                class_names=label_names or None,
                mask_string=self.mask_string,
                **self.explainer_args,
            )
        # ``bow=True`` (LIME's default) removes words outright — no substitute token at all.
        if not getattr(self.explainer, "bow", True):
            # ``mask_string=None`` makes LIME fall back to its own "UNKWORDZ".
            self.baseline_token = getattr(self.explainer, "mask_string", None) or "UNKWORDZ"

    def _classifier_fn(self, texts: list[str]) -> np.ndarray:
        """Wrap self.model to guarantee a float numpy array of shape (n, n_classes).

        Handles three output formats:

        * ``np.ndarray`` — returned as-is.
        * ``list[list[dict]]`` — HuggingFace pipeline with ``return_all_scores=True``.
          Scores are extracted by label name using ``self._classes`` as the column
          order, so ``label_names`` must be provided when the model uses this format.
        * Anything else — coerced with ``np.array(..., dtype=float)``.

        LIME's internals index the result with ``[:, label_idx]``, so the array
        must be 2-D and numeric.
        """
        result = self.model(texts)
        if isinstance(result, np.ndarray):
            return result
        # HuggingFace pipeline with return_all_scores=True → list[list[dict]]
        if result and isinstance(result[0], list) and isinstance(result[0][0], dict):
            label_to_idx = {name: i for i, name in enumerate(self._classes)}
            matrix = np.zeros((len(result), len(self._classes)), dtype=np.float64)
            for i, preds in enumerate(result):
                for pred in preds:
                    idx = label_to_idx.get(pred["label"])
                    if idx is not None:
                        matrix[i, idx] = pred["score"]
            return matrix
        return np.array(result, dtype=np.float64)

    def _count_features(self, text: str) -> int:
        """Number of features LIME will see in ``text``: its distinct words (or characters).

        Splits the text exactly as ``explain_instance`` is about to — same class, same settings read
        off the explainer — so ``num_features`` covers every word and LIME zero-fills none. An upper
        bound would not do: LIME's sparse ``highest_weights`` path pads the selection up to
        ``num_features`` instead of capping it.
        """
        explainer = self.explainer
        bow = getattr(explainer, "bow", True)
        mask_string = getattr(explainer, "mask_string", None)
        if getattr(explainer, "char_level", False):
            indexed = IndexedCharacters(text, bow=bow, mask_string=mask_string)
        else:
            split_expression = getattr(explainer, "split_expression", r"\W+")
            indexed = IndexedString(text, bow=bow, split_expression=split_expression, mask_string=mask_string)
        return max(1, indexed.num_words())

    def run_explainer(self, x) -> NlpContributions:
        """Run LimeTextExplainer on each sample and normalise output.

        Converts LIME's sparse per-label ``{label_id: [(word_id, weight)]}``
        representation into a dense ``(n_words, n_classes)`` array so that the
        returned ``NlpContributions`` matches the same field shapes as
        ``NlpShapBackend``.

        Parameters
        ----------
        x : list[str] or pd.Series
            Text samples to explain.

        Returns
        -------
        NlpContributions
            Dense weight arrays, LIME intercepts as base values, and unique
            vocabulary words per sample.
        """
        texts = list(x)
        # Without label names, ask the model for its output width: LIME's own default (``labels=(1,)``)
        # would explain class 1 only and leave every other column silently zero.
        n_classes = len(self._classes) or (self._classifier_fn(texts[:1]).shape[1] if texts else 0)

        compute_args = dict(self.explainer_compute_args)
        if "labels" not in compute_args and "top_labels" not in compute_args:
            compute_args["labels"] = list(range(n_classes))

        contributions: list[np.ndarray] = []
        base_values_list: list[list[float]] = []
        data: list[list[str]] = []

        for text in self._progress_iter(texts):
            text_args = compute_args
            if compute_args.get("num_features") == "all":
                text_args = {**compute_args, "num_features": self._count_features(text)}
            exp = self.explainer.explain_instance(text, self._classifier_fn, **text_args)

            indexed_string = exp.domain_mapper.indexed_string
            # Plain ``str``: LIME hands back ``np.str_``, which leaks into reprs and serialisation.
            vocab: list[str] = [str(word) for word in indexed_string.inverse_vocab]
            n_words = len(vocab)

            # Dense weight matrix — same shape contract as NlpShapBackend values. Classes LIME did not
            # explain (outside ``top_labels``) stay zero.
            weight_matrix = np.zeros((n_words, n_classes), dtype=float)
            for label_idx in range(n_classes):
                if label_idx in exp.local_exp:
                    for word_id, weight in exp.local_exp[label_idx]:
                        weight_matrix[word_id, label_idx] = weight

            contributions.append(weight_matrix)
            base_values_list.append([exp.intercept.get(i, 0.0) for i in range(n_classes)])
            data.append(vocab)

        return NlpContributions(
            token_strings=data,
            values=contributions,
            base_values=np.array(base_values_list),
        )
